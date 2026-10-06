"""Media generation and artifact handling (video, image, music, canvas).

Reverse-engineered Gemini Web media protocol:
- toolImage  = 14 (Nano Banana)
- toolMusic  = 21 (Lyria)
- toolCanvas = 2  (Interactive HTML document)
- toolVideo  = 11 (Veo async video)
"""

import base64
import json
import logging
import re
import secrets
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from curl_cffi import requests as curl_requests
    HAS_CURL_CFFI = True
except ImportError:
    import requests as curl_requests
    HAS_CURL_CFFI = False

from .config import CONFIG
from .gemini import (
    load_cookie,
    _account_prefix,
    _get_url,
    _build_headers,
    make_sapisidhash,
)
from .multimodal import _cached_page_tokens

log = logging.getLogger("gemini_web2api.media")

TOOL_IMAGE = 14
TOOL_MUSIC = 21
TOOL_CANVAS = 2
TOOL_VIDEO = 11

DOWNLOAD_OPI = "103135050"

DOWNLOAD_COOKIE_NAMES = {
    "HSID", "SSID", "APISID", "SAPISID",
    "__Secure-1PAPISID", "__Secure-3PAPISID",
    "SID", "__Secure-1PSID", "__Secure-3PSID",
    "GOOGLE_ABUSE_EXEMPTION", "NID",
    "__Secure-1PSIDTS", "__Secure-1PSIDRTS",
    "__Secure-3PSIDTS", "__Secure-3PSIDRTS",
    "SIDCC", "__Secure-1PSIDCC", "__Secure-3PSIDCC",
}

DATA_URL_RE = re.compile(r"data:([-\w.+/]+);base64,[A-Za-z0-9+/=]+")


@dataclass
class MediaArtifact:
    mime: str
    data: bytes


def filter_download_cookies(cookie: str) -> str:
    """Filter cookies down to the 18 specific cookies accepted by Google download host."""
    kept = []
    for p in cookie.split(";"):
        p = p.strip()
        if "=" in p:
            name = p.split("=", 1)[0].strip()
            if name in DOWNLOAD_COOKIE_NAMES:
                kept.append(p)
    return "; ".join(kept)


def extract_canvas_doc(raw: str) -> str:
    """Extract interactive HTML document from Canvas (tool=2) response frames."""
    best = ""

    def walk(o):
        nonlocal best
        if isinstance(o, str):
            if len(o) > len(best) and "DOCTYPE" in o:
                best = o
        elif isinstance(o, list):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for x in o.values():
                walk(x)

    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("[["):
            continue
        try:
            arr = json.loads(line)
        except Exception:
            continue
        for it in arr:
            if not isinstance(it, list) or len(it) < 3:
                continue
            payload = it[2]
            if not isinstance(payload, str):
                continue
            try:
                inner = json.loads(payload)
                walk(inner)
            except Exception:
                pass
    return best


def extract_conversation_id(raw: str) -> str:
    """Extract conversation ID from StreamGenerate wrb.fr frame."""
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("[["):
            continue
        try:
            arr = json.loads(line)
        except Exception:
            continue
        for it in arr:
            if not isinstance(it, list) or len(it) < 3 or it[0] != "wrb.fr":
                continue
            payload = it[2]
            if not isinstance(payload, str):
                continue
            try:
                inner = json.loads(payload)
                if isinstance(inner, list) and len(inner) >= 2:
                    meta = inner[1]
                    if isinstance(meta, list) and len(meta) > 0 and isinstance(meta[0], str) and meta[0]:
                        return meta[0]
            except Exception:
                pass
    return ""


def walk_frames_for_urls(raw: str, want_fn) -> List[str]:
    """Recursively search nested JSON frames for target URLs."""
    seen = set()
    out = []

    def walk(o):
        if isinstance(o, str):
            if want_fn(o) and o not in seen:
                seen.add(o)
                out.append(o)
        elif isinstance(o, list):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for x in o.values():
                walk(x)

    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("[["):
            continue
        try:
            arr = json.loads(line)
        except Exception:
            continue
        for it in arr:
            if not isinstance(it, list) or len(it) < 3:
                continue
            payload = it[2]
            if not isinstance(payload, str):
                continue
            try:
                inner = json.loads(payload)
                walk(inner)
            except Exception:
                pass
    return out


def collect_download_urls(raw: str) -> List[str]:
    return walk_frames_for_urls(
        raw, lambda s: "contribution.usercontent.google.com/download" in s
    )


def collect_image_urls(raw: str) -> List[str]:
    return walk_frames_for_urls(
        raw,
        lambda s: "lh3.googleusercontent.com/gg-dl/" in s or "lh3.googleusercontent.com/gg/" in s,
    )


def download_is_response_data(raw_url: str) -> bool:
    try:
        u = urllib.parse.urlparse(raw_url)
        qs = urllib.parse.parse_qs(u.query)
        c = qs.get("c", [""])[0]
        if not c:
            return False
        pad = len(c) % 4
        if pad:
            c += "=" * (4 - pad)
        try:
            dec = base64.urlsafe_b64decode(c)
        except Exception:
            dec = base64.b64decode(c)
        return b"response_data" in dec
    except Exception:
        return False


def pick_response_data_urls(urls: List[str]) -> List[str]:
    return [u for u in urls if download_is_response_data(u)]


def image_full_res_url(u: str) -> str:
    i = u.rfind("/")
    if i < 0:
        return u + "=s0"
    seg = u[i + 1:]
    j = seg.find("=")
    if j >= 0:
        u = u[:i + 1] + seg[:j]
    return u + "=s0"


def poll_history_raw(cid: str, cookie: str, sapisid: str, xsrf: str) -> str:
    inner = [cid, 10, None, 1, [0], [4], None, 1]
    freq = [[["hNvQHb", json.dumps(inner), None, "generic"]]]
    data = {"f.req": json.dumps(freq)}
    if xsrf:
        data["at"] = xsrf
    endpoint = f"https://gemini.google.com{_account_prefix()}/_/BardChatUi/data/batchexecute"
    params = {
        "rpcids": "hNvQHb",
        "bl": _cached_page_tokens().get("gemini_bl", "boq_gemini-web-uiserver_20261005.02_p0"),
        "hl": "en",
        "_reqid": int(time.time() * 1000) % 1000000,
        "rt": "c",
    }
    fsid = _cached_page_tokens().get("f_sid")
    if fsid:
        params["f.sid"] = fsid

    headers = {
        "Cookie": cookie,
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{_account_prefix()}/app",
        "X-Same-Domain": "1",
        "X-Goog-AuthUser": str(CONFIG.get("auth_user", 0)),
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)

    kwargs = {"headers": headers, "params": params, "data": data, "timeout": 30}
    if HAS_CURL_CFFI:
        kwargs["impersonate"] = "chrome120"

    resp = curl_requests.post(endpoint, **kwargs)
    if resp.status_code != 200:
        raise RuntimeError(f"hNvQHb HTTP {resp.status_code}")
    return resp.text


def delete_conversation(cid: str, cookie: str, sapisid: str, xsrf: str) -> None:
    """Best-effort deletion of conversation on Gemini Web."""
    try:
        inner = [cid]
        freq = [[["GzXR5e", json.dumps(inner), None, "generic"]]]
        data = {"f.req": json.dumps(freq)}
        if xsrf:
            data["at"] = xsrf
        endpoint = f"https://gemini.google.com{_account_prefix()}/_/BardChatUi/data/batchexecute"
        params = {
            "rpcids": "GzXR5e",
            "bl": _cached_page_tokens().get("gemini_bl", "boq_gemini-web-uiserver_20261005.02_p0"),
            "hl": "en",
            "_reqid": int(time.time() * 1000) % 1000000,
            "rt": "c",
        }
        fsid = _cached_page_tokens().get("f_sid")
        if fsid:
            params["f.sid"] = fsid
        headers = {
            "Cookie": cookie,
            "Origin": "https://gemini.google.com",
            "Referer": f"https://gemini.google.com{_account_prefix()}/app",
            "X-Same-Domain": "1",
            "X-Goog-AuthUser": str(CONFIG.get("auth_user", 0)),
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        if sapisid:
            headers["Authorization"] = make_sapisidhash(sapisid)
        kwargs = {"headers": headers, "params": params, "data": data, "timeout": 15}
        if HAS_CURL_CFFI:
            kwargs["impersonate"] = "chrome120"
        curl_requests.post(endpoint, **kwargs)
    except Exception as e:
        log.warning(f"delete_conversation {cid} failed: {e}")


def download_bytes_with_follow(
    raw_url: str, cookie: str, default_mime: str = "application/octet-stream"
) -> Tuple[str, bytes]:
    """Download media bytes with up to 6 manual hops resending filtered cookies on redirect."""
    cur = raw_url
    for hop in range(6):
        headers = {
            "Cookie": filter_download_cookies(cookie),
            "Origin": "https://gemini.google.com",
            "Referer": "https://gemini.google.com/",
            "Accept": "image/avif,image/webp,image/apng,video/*,audio/*,*/*;q=0.8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        kwargs = {
            "headers": headers,
            "allow_redirects": False,
            "timeout": 60,
        }
        if HAS_CURL_CFFI:
            kwargs["impersonate"] = "chrome120"

        resp = curl_requests.get(cur, **kwargs)
        if 300 <= resp.status_code < 400 and resp.headers.get("location"):
            loc = resp.headers["location"]
            cur = urllib.parse.urljoin(cur, loc)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Download HTTP {resp.status_code} ({len(resp.content)} bytes)")

        mime = resp.headers.get("content-type", default_mime).split(";")[0].strip()
        if not mime or mime.startswith("text/html"):
            mime = default_mime
        return mime, resp.content

    raise RuntimeError("Download exceeded maximum redirects (6)")


def fetch_image_artifacts(
    raw: str, cid: str, cookie: str, sapisid: str, xsrf: str, default_mime: str = "image/png"
) -> List[MediaArtifact]:
    urls = collect_image_urls(raw)
    if not urls and cid:
        for _ in range(6):
            try:
                body = poll_history_raw(cid, cookie, sapisid, xsrf)
                u = collect_image_urls(body)
                if u:
                    urls = u
                    break
            except Exception:
                pass
            time.sleep(2)

    if not urls:
        raise RuntimeError("No image CDN URLs found in response")

    arts = []
    seen_hashes = set()
    for u in urls:
        full_url = image_full_res_url(u)
        mime, data = download_bytes_with_follow(full_url, cookie, default_mime)
        h = hash(data)
        if h not in seen_hashes:
            seen_hashes.add(h)
            arts.append(MediaArtifact(mime=mime, data=data))
    return arts


def fetch_download_artifacts(
    cid: str,
    cookie: str,
    sapisid: str,
    xsrf: str,
    default_mime: str,
    max_polls: int,
    interval: float,
) -> List[MediaArtifact]:
    if not cid:
        raise RuntimeError("Conversation ID missing, cannot locate media artifact")

    dl_urls = []
    for _ in range(max_polls):
        try:
            body = poll_history_raw(cid, cookie, sapisid, xsrf)
            if "can't generate that video" in body:
                raise RuntimeError("Video rejected by Google content policy")
            picked = pick_response_data_urls(collect_download_urls(body))
            if picked:
                dl_urls = picked
                break
        except Exception as e:
            if "content policy" in str(e):
                raise
        time.sleep(interval)

    if not dl_urls:
        raise RuntimeError("Timed out waiting for response_data download URL in hNvQHb")

    arts = []
    seen_hashes = set()
    for u in dl_urls:
        if "opi=" not in u:
            sep = "&" if "?" in u else "?"
            u += f"{sep}filename=artifact&opi={DOWNLOAD_OPI}"
        mime, data = download_bytes_with_follow(u, cookie, default_mime)
        h = hash(data)
        if h not in seen_hashes:
            seen_hashes.add(h)
            arts.append(MediaArtifact(mime=mime, data=data))
    return arts


def fetch_media_artifacts(
    tool: int,
    raw: str,
    cid: str,
    cookie: str,
    sapisid: str,
    xsrf: str,
    default_mime: str,
) -> List[MediaArtifact]:
    """Retrieve binary media artifacts based on tool type."""
    if tool == TOOL_IMAGE:
        return fetch_image_artifacts(raw, cid, cookie, sapisid, xsrf, default_mime)

    max_polls, interval = 6, 2.0
    if tool == TOOL_VIDEO:
        max_polls, interval = 45, 8.0  # up to 6 minutes for Veo video

    arts = fetch_download_artifacts(
        cid, cookie, sapisid, xsrf, default_mime, max_polls, interval
    )
    if tool == TOOL_VIDEO and len(arts) > 1:
        # Keep largest video file (best quality)
        largest = max(arts, key=lambda a: len(a.data))
        arts = [largest]
    return arts


def append_artifact_markdown(text: str, arts: List[MediaArtifact]) -> str:
    """Format artifacts as data URLs in Markdown for client consumption."""
    items = []
    for a in arts:
        data_url = f"data:{a.mime};base64,{base64.b64encode(a.data).decode('ascii')}"
        if a.mime.startswith("image/"):
            items.append(f"![image]({data_url})")
        elif a.mime.startswith("video/"):
            items.append(f"[video]({data_url})")
        else:
            items.append(f"[audio]({data_url})")
    sep = "\n\n" if text.strip() else ""
    return text + sep + "\n\n".join(items)


def strip_data_urls(text: str) -> str:
    return DATA_URL_RE.sub(r"data:\1;base64,<omitted>", text)
