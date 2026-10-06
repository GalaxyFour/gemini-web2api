"""HTTP server: OpenAI-compatible API endpoints."""
from __future__ import annotations

import threading

import base64
import itertools
import json
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from . import __version__
from .config import (
    CONFIG,
    get_all_api_keys,
    get_current_account,
    resolve_account_from_config,
    set_current_account,
)
from .gemini import (
    extract_response_text,
    generate,
    generate_image_structured,
    generate_stream,
    get_full_size_image,
    log,
)
from .generated_image import download_generated_image, resolve_generated_image_url
from .models import (
    MODELS,
    TOOL_CANVAS,
    TOOL_IMAGE,
    TOOL_MUSIC,
    TOOL_VIDEO,
    resolve_model,
)
from .media import (
    MediaArtifact,
    append_artifact_markdown,
    delete_conversation,
    extract_canvas_doc,
    extract_conversation_id,
    fetch_media_artifacts,
)
from .multimodal import detect_image_mime, fetch_image_bytes, upload_image
from .tools import (
    google_contents_to_prompt,
    messages_to_prompt,
    parse_google_function_calls,
    parse_tool_calls,
    tool_names,
)

_CHAT_IMAGE_REQUEST = re.compile(
    r"\b(?:generate|create|make|draw|render|paint)\s+"
    r"(?:(?:me|us)\s+)?(?:(?:an?|the)\s+)?"
    r"(?:image|picture|photo|illustration|artwork|icon|logo|portrait)\b",
    re.IGNORECASE,
)


def _latest_user_text(messages) -> str:
    """Extract only the latest user turn for intent-sensitive routing."""
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
                and part.get("type") in ("text", "input_text")
                and isinstance(part.get("text"), str)
            )
        return ""
    return ""


def _chat_image_prompt(request: dict) -> str | None:
    """Return an explicit image-generation prompt from the latest user turn."""
    text = _latest_user_text(request.get("messages"))
    modalities = request.get("modalities")
    explicitly_requested = isinstance(modalities, list) and "image" in modalities
    if explicitly_requested or _CHAT_IMAGE_REQUEST.search(text):
        return text.strip() or None
    return None


def _usage(prompt: str, text: str) -> dict:
    p = len(prompt) // 4
    c = len(text or "") // 4
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _generated_image_output(prompt: str, response_format: str):
    """Generate one image and return its optional text plus OpenAI output data."""
    result = generate_image_structured(prompt)
    if not result.images:
        details = []
        if result.error:
            details.append(result.error)
        if result.text:
            details.append(result.text)
        if details:
            raise RuntimeError(f"Gemini rejected image generation: {' - '.join(details)}")
        raise RuntimeError("Gemini returned no generated image metadata")
    source_url = get_full_size_image(result.images[0]) or result.images[0].url
    if response_format == "url":
        data = {"url": resolve_generated_image_url(source_url)}
    else:
        image_bytes, _mime = download_generated_image(source_url)
        data = {"b64_json": base64.b64encode(image_bytes).decode("ascii")}
    return result.text, data


def _upload_images(images: list) -> list:
    """Upload images and return list of file references. Returns None if no images."""
    if not images:
        return None
    file_refs = []
    for item in images:
        if not (isinstance(item, tuple) and len(item) == 2):
            continue
        data, mime = item
        if isinstance(data, str):
            data = fetch_image_bytes(data)
            mime = mime or "image/png"
        if not data:
            raise RuntimeError("image fetch failed")
        mime = detect_image_mime(data, mime or "image/png")
        filename = "image.png"
        try:
            ref = upload_image(data, filename, mime or "image/png")
            # Gemini's current attachment format requires both the uploaded
            # reference and its filename; retain both through generation.
            file_refs.append((ref, filename))
        except Exception as e:
            raise RuntimeError(f"image upload failed: {e}") from e
    return file_refs if file_refs else None



# ─── Video Generation Jobs (Async Sora/OpenAI style) ──────────────────────────

_VIDEO_JOBS: dict[str, dict] = {}
_VIDEO_JOBS_LOCK = threading.Lock()


def _put_video_job(job: dict) -> None:
    with _VIDEO_JOBS_LOCK:
        cutoff = time.time() - 2 * 3600
        expired = [jid for jid, old in _VIDEO_JOBS.items() if old.get("created_at", 0) < cutoff]
        for jid in expired:
            _VIDEO_JOBS.pop(jid, None)
        _VIDEO_JOBS[job["id"]] = job


def _get_video_job(job_id: str) -> Optional[dict]:
    with _VIDEO_JOBS_LOCK:
        return _VIDEO_JOBS.get(job_id)


def _run_video_job(job: dict) -> None:
    job["status"] = "in_progress"
    model_name = job.get("model", "gemini-video")
    prompt = job.get("prompt", "")
    _, model_id, think_mode, err, extra_fields = resolve_model(model_name)
    if err:
        job["status"] = "failed"
        job["error"] = err
        return

    try:
        from .gemini import _generate_raw, load_cookie
        from .multimodal import _cached_page_tokens
        cookie_str, sapisid = load_cookie()
        page_tokens = _cached_page_tokens(max_age=0)
        xsrf = page_tokens.get("at", "")

        raw_resp = _generate_raw(prompt, model_id, think_mode, extra_fields=extra_fields)
        cid = extract_conversation_id(raw_resp)
        arts = fetch_media_artifacts(
            TOOL_VIDEO, raw_resp, cid, cookie_str, sapisid, xsrf, "video/mp4"
        )
        if not arts:
            job["status"] = "failed"
            job["error"] = "no video produced"
            return
        a = arts[0]
        job["mp4"] = a.data
        job["mime"] = a.mime or "video/mp4"
        job["status"] = "completed"
        if CONFIG.get("temporary_chats", False) and cid:
            threading.Thread(target=delete_conversation, args=(cid, cookie_str, sapisid, xsrf), daemon=True).start()
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)


class GeminiHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        client_ip = self.client_address[0] if self.client_address else "-"
        log(f"{client_ip} {fmt % args}")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _parse_body(self, body: bytes) -> dict:
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return None

    def _read_request_body(self) -> bytes:
        transfer_encoding = self.headers.get("Transfer-Encoding", "")
        if "chunked" in transfer_encoding.lower():
            chunks = []
            while True:
                size_line = self.rfile.readline()
                if not size_line:
                    break
                size_text = size_line.split(b";", 1)[0].strip()
                try:
                    size = int(size_text, 16)
                except ValueError:
                    raise ValueError("invalid chunked request body")
                if size == 0:
                    while True:
                        trailer = self.rfile.readline()
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.read(2)
            return b"".join(chunks)

        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _extract_api_key(self):
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        for h in ("x-api-key", "x-goog-api-key"):
            val = self.headers.get(h)
            if val:
                return val.strip()
        if "?" in self.path:
            for pair in self.path.split("?", 1)[1].split("&"):
                if pair.startswith("key="):
                    return pair[4:].strip()
        return None

    def _authorized(self):
        all_keys = get_all_api_keys(CONFIG)
        if not all_keys:
            return True
        key = self._extract_api_key()
        if not key or key not in all_keys:
            return False
        if get_current_account() is None:
            account = resolve_account_from_config(CONFIG, key)
            if account:
                set_current_account(account)
        return True

    def parse_request(self):
        if not super().parse_request():
            return False
        key = self._extract_api_key()
        account = resolve_account_from_config(CONFIG, key)
        set_current_account(account)
        return True

    def handle_one_request(self):
        try:
            super().handle_one_request()
        finally:
            set_current_account(None)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        try:
            if self.path.startswith("/v1") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            if self.path == "/v1/models":
                self.send_json({"object": "list", "data": [
                    {"id": n, "object": "model", "created": 1700000000,
                     "owned_by": "google", "description": c["desc"]}
                    for n, c in MODELS.items()
                ]})
            elif self.path.startswith("/v1beta/models"):
                self.send_json({"models": [
                    {"name": f"models/{n}", "displayName": n, "description": c["desc"],
                     "supportedGenerationMethods": ["generateContent", "streamGenerateContent"]}
                    for n, c in MODELS.items()
                ]})
            elif self.path.startswith("/v1/videos"):
                self._handle_video_get()
            elif self.path == "/":
                self.send_json({"status": "ok", "version": __version__, "models": list(MODELS.keys())})
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            if self.path.startswith("/v1") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            body = self._read_request_body()
            if self.path == "/v1/chat/completions":
                self._handle_chat(body)
            elif self.path == "/v1/images/generations":
                self._handle_image_generation(body)
            elif self.path in ("/v1/videos", "/v1/videos/generations"):
                self._handle_create_video(body)
            elif self.path == "/v1/responses":
                self._handle_responses(body)
            elif ":streamGenerateContent" in self.path:
                self._handle_google_generate(body, stream=True)
            elif ":generateContent" in self.path:
                self._handle_google_generate(body, stream=False)
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"POST error: {e}")
            try:
                self.send_json({"error": {"message": str(e)}}, 500)
            except:
                pass

    # ─── /v1/videos & /v1/videos/generations ─────────────────────────────────

    def _handle_create_video(self, body: bytes):
        req = self._parse_body(body)
        if not isinstance(req, dict):
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        prompt = req.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self.send_json({"error": {"message": "missing prompt", "type": "invalid_request_error"}}, 400)
            return
        model = req.get("model", "gemini-video")
        model_cfg = MODELS.get(model)
        if not model_cfg or model_cfg.get("tool") != TOOL_VIDEO:
            self.send_json({"error": {"message": "model must be a video model (gemini-video)", "type": "invalid_request_error"}}, 400)
            return

        job_id = f"video_{uuid.uuid4().hex}"
        job = {
            "id": job_id,
            "object": "video",
            "model": model,
            "status": "queued",
            "created_at": int(time.time()),
            "prompt": prompt,
        }
        _put_video_job(job)
        threading.Thread(target=_run_video_job, args=(job,), daemon=True).start()
        self.send_json(job)

    def _handle_video_get(self):
        clean_path = self.path.split("?")[0]
        rest = clean_path[len("/v1/videos"):].strip("/")
        if not rest:
            with _VIDEO_JOBS_LOCK:
                jobs = [
                    {"id": j["id"], "object": j.get("object", "video"), "model": j.get("model"),
                     "status": j.get("status"), "created_at": j.get("created_at"), "prompt": j.get("prompt")}
                    for j in _VIDEO_JOBS.values()
                ]
            self.send_json({"object": "list", "data": jobs})
            return

        parts = rest.split("/", 1)
        job_id = parts[0]
        job = _get_video_job(job_id)
        if not job:
            self.send_json({"error": {"message": "video job not found", "type": "not_found"}}, 404)
            return

        if len(parts) == 2 and parts[1] == "content":
            status = job.get("status")
            mp4_bytes = job.get("mp4")
            mime = job.get("mime", "video/mp4")
            if status != "completed" or not mp4_bytes:
                self.send_json({"error": {"message": f"video not ready (status={status})", "type": "not_found"}}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(mp4_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(mp4_bytes)
            return

        resp_job = {
            "id": job["id"],
            "object": job.get("object", "video"),
            "model": job.get("model"),
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "prompt": job.get("prompt"),
        }
        if job.get("error"):
            resp_job["error"] = job["error"]
        self.send_json(resp_job)

    # ─── /v1/images/generations ───────────────────────────────────────────────

    def _handle_image_generation(self, body: bytes):
        req = self._parse_body(body)
        if not isinstance(req, dict):
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        unsupported = [name for name in ("stream", "size", "quality", "style") if name in req]
        prompt = req.get("prompt")
        if unsupported or not isinstance(prompt, str) or not prompt.strip():
            self.send_json({"error": {"message": "invalid image generation request"}}, 400)
            return
        if "n" in req and (not isinstance(req["n"], int) or isinstance(req["n"], bool) or req["n"] != 1):
            self.send_json({"error": {"message": "only n=1 is supported"}}, 400)
            return
        response_format = req.get("response_format", "b64_json")
        if response_format not in ("b64_json", "url"):
            self.send_json({"error": {"message": "response_format must be b64_json or url"}}, 400)
            return
        model_value = req.get("model")
        if model_value is not None and not isinstance(model_value, str):
            self.send_json({"error": {"message": "invalid model"}}, 400)
            return
        try:
            # Gemini Web selects its image route independently of text models.
            _text, data = _generated_image_output(prompt, response_format)
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return
        self.send_json({"created": int(time.time()), "data": [data]})

    # ─── /v1/chat/completions ─────────────────────────────────────────────────

    def _chunk(self, cid, model_name, delta, finish_reason=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

    def _stream_tool_calls(self, cid, model_name, tool_calls, arg_slice=120):
        """Emit tool calls as OpenAI-spec streaming deltas.

        Each call gets an `index` (required by clients to assemble split
        arguments), followed by argument slices, then a `tool_calls`
        finish chunk.
        """
        self.wfile.write(
            f"data: {json.dumps(self._chunk(cid, model_name, {'role': 'assistant'}), ensure_ascii=False)}\n\n".encode())
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            head = {"role": "assistant",
                    "tool_calls": [{"index": i, "id": tc.get("id"), "type": "function",
                                    "function": {"name": fn.get("name", ""), "arguments": ""}}]}
            self.wfile.write(
                f"data: {json.dumps(self._chunk(cid, model_name, head), ensure_ascii=False)}\n\n".encode())
            args = fn.get("arguments", "") or ""
            for j in range(0, len(args), arg_slice):
                piece = {"tool_calls": [{"index": i, "function": {"arguments": args[j:j + arg_slice]}}]}
                self.wfile.write(
                    f"data: {json.dumps(self._chunk(cid, model_name, piece), ensure_ascii=False)}\n\n".encode())
        self.wfile.write(
            f"data: {json.dumps(self._chunk(cid, model_name, {}, 'tool_calls'))}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _handle_chat(self, body: bytes):
        req = self._parse_body(body)
        if req is None:
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        model_name, model_id, think_mode, err, extra_fields = resolve_model(
            req.get("model", CONFIG["default_model"]))
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        tools = req.get("tools")
        tool_choice = req.get("tool_choice", "auto")
        image_prompt = _chat_image_prompt(req)
        prompt, images = messages_to_prompt(req.get("messages", []), tools, tool_choice)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty prompt"}}, 400)
            return

        stream = req.get("stream", False)
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        precomputed_text = None
        if image_prompt:
            try:
                generated_text, image_data = _generated_image_output(image_prompt, "url")
                image_markdown = f"![Generated image]({image_data['url']})"
                precomputed_text = "\n\n".join(
                    part for part in (generated_text, image_markdown) if part
                )
                # Image generation is a native route, not a function call.
                tools = None
                tool_choice = "none"
                images = []
            except Exception as e:
                self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
                return
        try:
            file_refs = _upload_images(images)
        except RuntimeError as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        if stream and (not tools or tool_choice == "none"):
            # Prime the iterator before committing HTTP 200/SSE headers so an
            # immediate upstream rejection remains a normal JSON 502.
            try:
                if precomputed_text is not None:
                    deltas = iter([precomputed_text])
                elif file_refs:
                    deltas = iter([
                        generate(prompt, model_id, think_mode, file_refs, extra_fields)
                    ])
                else:
                    deltas = iter(
                        generate_stream(prompt, model_id, think_mode, None, extra_fields)
                    )
                first_delta = next(deltas, None)
            except Exception as e:
                self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
                return
            try:
                self._start_sse()
                first_chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }],
                }
                self.wfile.write(f"data: {json.dumps(first_chunk)}\n\n".encode())
                self.wfile.flush()
                for delta in itertools.chain(
                    [first_delta] if first_delta else [], deltas
                ):
                    if not delta:
                        continue
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                             "model": model_name, "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                end = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                       "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(end)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"Stream error: {e}")
                error = {"error": {"message": f"upstream error: {e}",
                                   "type": "upstream_error"}}
                try:
                    self.wfile.write(f"data: {json.dumps(error)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return

        try:
            model_cfg = MODELS.get(model_name, {})
            media_tool = model_cfg.get("tool")
            if precomputed_text is not None:
                text = precomputed_text
            elif media_tool is not None:
                from .gemini import _generate_raw, load_cookie
                from .multimodal import _cached_page_tokens
                tool_extra = dict(extra_fields or {})
                tool_extra[49] = media_tool
                if media_tool == TOOL_VIDEO:
                    tool_extra[55] = [[16]]
                raw_resp = _generate_raw(prompt, model_id, think_mode, file_refs or [], tool_extra)
                if media_tool == TOOL_CANVAS:
                    doc = extract_canvas_doc(raw_resp)
                    raw_text = extract_response_text(raw_resp)
                    if not doc:
                        text = raw_text
                    elif raw_text and raw_text not in doc:
                        text = f"{raw_text}\n\n{doc}"
                    else:
                        text = doc
                else:
                    cookie_str, sapisid = load_cookie()
                    page_tokens = _cached_page_tokens(max_age=0)
                    xsrf = page_tokens.get("at", "")
                    cid_raw = extract_conversation_id(raw_resp)
                    default_mime = "image/png"
                    if media_tool == TOOL_MUSIC:
                        default_mime = "audio/mpeg"
                    elif media_tool == TOOL_VIDEO:
                        default_mime = "video/mp4"
                    arts = fetch_media_artifacts(media_tool, raw_resp, cid_raw, cookie_str, sapisid, xsrf, default_mime)
                    raw_text = extract_response_text(raw_resp)
                    text = append_artifact_markdown(raw_text, arts)
                    if CONFIG.get("temporary_chats", False) and cid_raw:
                        threading.Thread(target=delete_conversation, args=(cid_raw, cookie_str, sapisid, xsrf), daemon=True).start()
            else:
                text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        tool_calls = None
        if tools and text and tool_choice != "none":
            text, tool_calls = parse_tool_calls(text, tool_names(tools))
        msg = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else "stop"

        if stream:
            self._start_sse()
            if tool_calls:
                self._stream_tool_calls(cid, model_name, tool_calls)
            else:
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": model_name, "choices": [{"index": 0, "delta": msg, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        else:
            self.send_json({
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": len(prompt)//4, "completion_tokens": len(text or "")//4,
                          "total_tokens": (len(prompt)+len(text or ""))//4},
            })

    # ─── /v1/responses (Codex CLI) ───────────────────────────────────────────

    def _handle_responses(self, body: bytes):
        req = self._parse_body(body)
        if req is None:
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        model_name, model_id, think_mode, err, extra_fields = resolve_model(
            req.get("model", CONFIG["default_model"]))
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        input_items = req.get("input", [])
        raw_tools = req.get("tools")
        image_generation_requested = isinstance(raw_tools, list) and any(
            isinstance(tool, dict) and tool.get("type") == "image_generation"
            for tool in raw_tools
        )
        # Image generation is a native request signal, not an emulated function.
        tools = ([tool for tool in raw_tools
                  if isinstance(tool, dict) and tool.get("type") != "image_generation"]
                 if isinstance(raw_tools, list) else raw_tools)
        messages = []
        if req.get("instructions"):
            messages.append({"role": "system", "content": req["instructions"]})
        if isinstance(input_items, str):
            messages.append({"role": "user", "content": input_items})
        elif isinstance(input_items, list):
            for item in input_items:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    if item.get("type") == "function_call_output":
                        messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""),
                                         "name": item.get("name", ""), "content": item.get("output", "")})
                    elif item.get("type") in ("input_text", "input_image", "image"):
                        messages.append({"role": "user", "content": [item]})
                    elif item.get("role") == "assistant" or (item.get("type") == "message" and item.get("role") == "assistant"):
                        cp = item.get("content", [])
                        text_acc, tc_list = "", []
                        if isinstance(cp, list):
                            for c in cp:
                                if isinstance(c, dict):
                                    if c.get("type") == "output_text":
                                        text_acc += c.get("text", "")
                                    elif c.get("type") == "function_call":
                                        tc_list.append(c)
                        elif isinstance(cp, str):
                            text_acc = cp
                        m = {"role": "assistant", "content": text_acc or None}
                        if tc_list:
                            m["tool_calls"] = [{"id": tc.get("call_id", f"call_{i}"), "type": "function",
                                                "function": {"name": tc.get("name",""), "arguments": tc.get("arguments","{}")}}
                                               for i, tc in enumerate(tc_list)]
                        messages.append(m)
                    else:
                        role = item.get("role", "user")
                        messages.append({"role": role, "content": item.get("content", "")})

        if tools:
            tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}}
                     if t.get("type") == "function" and "function" not in t else t for t in tools]

        tool_choice = req.get("tool_choice", "auto")
        prompt, images = messages_to_prompt(messages, tools, tool_choice)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty input"}}, 400)
            return

        if image_generation_requested and images:
            self.send_json({
                "error": {"message": "image generation with input images is not supported"}
            }, 400)
            return

        generated_image_call = None
        try:
            file_refs = _upload_images(images)
            if image_generation_requested:
                text, image_data = _generated_image_output(prompt, "b64_json")
                generated_image_call = {
                    "type": "image_generation_call",
                    "id": f"imggen_{uuid.uuid4().hex[:12]}",
                    "status": "completed",
                    "result": image_data["b64_json"],
                }
            else:
                text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        tool_calls = None
        if tools and text and tool_choice != "none":
            text, tool_calls = parse_tool_calls(text, tool_names(tools))

        rid = f"resp_{uuid.uuid4().hex[:16]}"
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        output = []
        if tool_calls:
            for tc in tool_calls:
                output.append({"type": "function_call", "id": tc["id"], "call_id": tc["id"],
                               "name": tc["function"]["name"], "arguments": tc["function"]["arguments"], "status": "completed"})
        if text or (not tool_calls and not generated_image_call):
            output.append({"type": "message", "id": mid, "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": text or "", "annotations": []}]})
        if generated_image_call:
            output.append(generated_image_call)

        if req.get("stream"):
            self._start_sse()
            sequence_number = 0

            def emit(event_type, **fields):
                nonlocal sequence_number
                sequence_number += 1
                event = {
                    "type": event_type,
                    "sequence_number": sequence_number,
                    **fields,
                }
                self.wfile.write(
                    f"event: {event_type}\ndata: {json.dumps(event)}\n\n".encode()
                )

            usage = {
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(text or "") // 4,
                "total_tokens": (len(prompt) + len(text or "")) // 4,
            }
            base_response = {
                "id": rid,
                "object": "response",
                "created_at": int(time.time()),
                "model": model_name,
            }
            emit(
                "response.created",
                response={
                    **base_response,
                    "status": "in_progress",
                    "output": [],
                    "usage": None,
                },
            )
            emit(
                "response.in_progress",
                response={
                    **base_response,
                    "status": "in_progress",
                    "output": [],
                    "usage": None,
                },
            )
            for output_index, item in enumerate(output):
                if item["type"] == "function_call":
                    pending_item = {
                        "type": "function_call",
                        "id": item["id"],
                        "call_id": item["call_id"],
                        "name": item["name"],
                        "arguments": "",
                        "status": "in_progress",
                    }
                    emit(
                        "response.output_item.added",
                        output_index=output_index,
                        item=pending_item,
                    )
                    emit(
                        "response.function_call_arguments.delta",
                        item_id=item["id"],
                        output_index=output_index,
                        delta=item["arguments"],
                    )
                    emit(
                        "response.function_call_arguments.done",
                        item_id=item["id"],
                        output_index=output_index,
                        arguments=item["arguments"],
                    )
                    emit(
                        "response.output_item.done",
                        output_index=output_index,
                        item=item,
                    )
                elif item["type"] == "image_generation_call":
                    # The image is already downloaded and validated before SSE headers.
                    emit(
                        "response.output_item.added",
                        output_index=output_index,
                        item={"type": "image_generation_call", "id": item["id"], "status": "in_progress"},
                    )
                    emit(
                        "response.output_item.done",
                        output_index=output_index,
                        item=item,
                    )
                elif item["type"] == "message":
                    pending_item = {
                        "type": "message",
                        "id": item["id"],
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    }
                    emit(
                        "response.output_item.added",
                        output_index=output_index,
                        item=pending_item,
                    )
                    for content_index, content_part in enumerate(item["content"]):
                        event_fields = {
                            "item_id": item["id"],
                            "output_index": output_index,
                            "content_index": content_index,
                        }
                        emit(
                            "response.content_part.added",
                            **event_fields,
                            part={
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            },
                        )
                        emit(
                            "response.output_text.delta",
                            **event_fields,
                            delta=content_part["text"],
                        )
                        emit(
                            "response.output_text.done",
                            **event_fields,
                            text=content_part["text"],
                        )
                        emit(
                            "response.content_part.done",
                            **event_fields,
                            part=content_part,
                        )
                    emit(
                        "response.output_item.done",
                        output_index=output_index,
                        item=item,
                    )
            emit(
                "response.completed",
                response={
                    **base_response,
                    "status": "completed",
                    "output": output,
                    "usage": usage,
                },
            )
            self.wfile.flush()
        else:
            self.send_json({"id": rid, "object": "response", "created_at": int(time.time()), "status": "completed",
                            "model": model_name, "output": output,
                            "usage": {"input_tokens": len(prompt)//4, "output_tokens": len(text or "")//4, "total_tokens": (len(prompt)+len(text or ""))//4}})

    # ─── /v1beta/models (Google Gemini CLI) ──────────────────────────────────

    def _handle_google_generate(self, body: bytes, stream: bool):
        req = self._parse_body(body)
        if req is None:
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        m = re.match(r'/v1beta/models/([^:?]+)', self.path)
        model_name = m.group(1) if m else CONFIG["default_model"]
        model_name, model_id, think_mode, err, extra_fields = resolve_model(model_name)
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        tool_config = req.get("toolConfig", {})
        fc_mode = tool_config.get("functionCallingConfig", {}).get("mode", "AUTO")
        has_tools = bool(req.get("tools")) and fc_mode != "NONE"
        prompt, images = google_contents_to_prompt(req)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty content"}}, 400)
            return

        try:
            file_refs = _upload_images(images)
        except RuntimeError as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return
        log(f"Google API: model={model_name} stream={stream} tools={has_tools} prompt_len={len(prompt)}")

        if stream and not has_tools:
            try:
                deltas = iter(
                    [generate(prompt, model_id, think_mode, file_refs, extra_fields)]
                    if file_refs else
                    generate_stream(prompt, model_id, think_mode, None, extra_fields)
                )
                first_delta = next(deltas, None)
            except Exception as e:
                self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
                return
            try:
                self._start_sse()
                full_text = ""
                for delta in itertools.chain(
                    [first_delta] if first_delta else [], deltas
                ):
                    if not delta:
                        continue
                    full_text += delta
                    chunk_obj = {
                        "candidates": [{"content": {"parts": [{"text": delta}], "role": "model"}, "index": 0}],
                        "modelVersion": model_name,
                    }
                    self.wfile.write(f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                final_chunk = {
                    "candidates": [{"finishReason": "STOP", "index": 0}],
                    "usageMetadata": {
                        "promptTokenCount": len(prompt) // 4,
                        "candidatesTokenCount": len(full_text) // 4,
                        "totalTokenCount": (len(prompt) + len(full_text)) // 4,
                    },
                    "modelVersion": model_name,
                }
                self.wfile.write(f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"Google stream error: {e}")
                error = {"error": {"code": 502, "message": f"upstream error: {e}",
                                   "status": "UNAVAILABLE"}}
                try:
                    self.wfile.write(f"data: {json.dumps(error)}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return

        try:
            text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        if not text:
            log("Warning: empty response from Gemini")

        response_parts = []
        if has_tools and text:
            clean_text, function_calls = parse_google_function_calls(text)
            if function_calls:
                if clean_text:
                    response_parts.append({"text": clean_text})
                for fc in function_calls:
                    response_parts.append({"functionCall": {"name": fc["name"], "args": fc["args"]}})
            else:
                response_parts.append({"text": text})
        else:
            response_parts.append({"text": text or "I apologize, but I was unable to generate a response. Please try again."})

        candidate = {
            "content": {"parts": response_parts, "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }
        usage = {
            "promptTokenCount": len(prompt) // 4,
            "candidatesTokenCount": len(text or "") // 4,
            "totalTokenCount": (len(prompt) + len(text or "")) // 4,
        }
        response_obj = {
            "candidates": [candidate],
            "usageMetadata": usage,
            "modelVersion": model_name,
        }

        if stream:
            self._start_sse()
            self.wfile.write(f"data: {json.dumps(response_obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        else:
            self.send_json(response_obj)


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
