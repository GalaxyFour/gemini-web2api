import json
import unittest
from unittest import mock
import http.client
import threading
import time

from gemini_web2api.config import CONFIG
from gemini_web2api.models import MODELS, TOOL_IMAGE, TOOL_MUSIC, TOOL_CANVAS, TOOL_VIDEO, resolve_model
from gemini_web2api.media import (
    MediaArtifact,
    extract_canvas_doc,
    extract_conversation_id,
    collect_download_urls,
    collect_image_urls,
    image_full_res_url,
    filter_download_cookies,
    append_artifact_markdown,
)
from gemini_web2api.server import GeminiHandler, ThreadedServer, _put_video_job, _get_video_job


class MediaModelsTests(unittest.TestCase):
    def test_media_models_registered(self):
        self.assertIn("gemini-image", MODELS)
        self.assertIn("gemini-music", MODELS)
        self.assertIn("gemini-video", MODELS)
        self.assertIn("gemini-canvas", MODELS)

        self.assertEqual(MODELS["gemini-image"]["tool"], TOOL_IMAGE)
        self.assertEqual(MODELS["gemini-music"]["tool"], TOOL_MUSIC)
        self.assertEqual(MODELS["gemini-video"]["tool"], TOOL_VIDEO)
        self.assertEqual(MODELS["gemini-canvas"]["tool"], TOOL_CANVAS)

    def test_resolve_media_model(self):
        name, mode, think, err, extra = resolve_model("gemini-canvas")
        self.assertEqual(name, "gemini-canvas")
        self.assertIsNone(err)


class MediaParsingTests(unittest.TestCase):
    def test_filter_download_cookies(self):
        cookie = "HSID=123; OTHER_COOKIE=abc; SID=xyz; COMPASS=foo"
        filtered = filter_download_cookies(cookie)
        self.assertIn("HSID=123", filtered)
        self.assertIn("SID=xyz", filtered)
        self.assertNotIn("OTHER_COOKIE", filtered)
        self.assertNotIn("COMPASS", filtered)

    def test_image_full_res_url(self):
        url = "https://lh3.googleusercontent.com/gg-dl/abcdef=s512-rj"
        self.assertEqual(image_full_res_url(url), "https://lh3.googleusercontent.com/gg-dl/abcdef=s0")

    def test_extract_canvas_doc(self):
        raw_frame = json.dumps([["wrb.fr", None, json.dumps([None, None, "```html\\n<!DOCTYPE html><html><body>Test</body></html>\\n```"])]])
        doc = extract_canvas_doc(raw_frame)
        self.assertIn("<!DOCTYPE html>", doc)

    def test_extract_conversation_id(self):
        raw_frame = json.dumps([["wrb.fr", None, json.dumps([None, ["c_1234567890", "r_abc"]])]])
        self.assertEqual(extract_conversation_id(raw_frame), "c_1234567890")

    def test_append_artifact_markdown(self):
        arts = [
            MediaArtifact(mime="image/png", data=b"pngdata"),
            MediaArtifact(mime="audio/mpeg", data=b"mp3data"),
        ]
        res = append_artifact_markdown("Here is your output:", arts)
        self.assertIn("![image](data:image/png;base64,", res)
        self.assertIn("[audio](data:audio/mpeg;base64,", res)


class VideoEndpointsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadedServer(("127.0.0.1", 0), GeminiHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.original_config = dict(CONFIG)
        CONFIG["api_keys"] = []
        CONFIG["log_requests"] = False

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def post_json(self, path, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            connection.request(
                "POST",
                path,
                body=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            body = response.read().decode()
            headers = dict(response.getheaders())
            return response.status, headers, body
        finally:
            connection.close()

    def get(self, path):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            body = response.read()
            headers = dict(response.getheaders())
            return response.status, headers, body
        finally:
            connection.close()

    @mock.patch("gemini_web2api.server._run_video_job")
    def test_create_video_job(self, mock_run):
        status, _, body = self.post_json(
            "/v1/videos",
            {"prompt": "A cat running in rain", "model": "gemini-video"},
        )
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertTrue(data["id"].startswith("video_"))
        self.assertEqual(data["status"], "queued")
        self.assertEqual(data["model"], "gemini-video")

    @mock.patch("gemini_web2api.server._run_video_job")
    def test_create_video_generations_alias(self, mock_run):
        status, _, body = self.post_json(
            "/v1/videos/generations",
            {"prompt": "A sunset timelapse", "model": "gemini-video"},
        )
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertTrue(data["id"].startswith("video_"))

    def test_get_video_job_status_and_content(self):
        job_id = "video_status_test_1"
        job = {
            "id": job_id,
            "object": "video",
            "model": "gemini-video",
            "status": "completed",
            "created_at": int(time.time()),
            "prompt": "test prompt",
            "mp4": b"\x00\x00\x00\x18ftypmp42fake",
            "mime": "video/mp4",
        }
        _put_video_job(job)

        status, _, body = self.get(f"/v1/videos/{job_id}")
        self.assertEqual(status, 200)
        data = json.loads(body.decode())
        self.assertEqual(data["id"], job_id)
        self.assertEqual(data["status"], "completed")

        status, headers, body = self.get(f"/v1/videos/{job_id}/content")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "video/mp4")
        self.assertEqual(body, b"\x00\x00\x00\x18ftypmp42fake")


if __name__ == "__main__":
    unittest.main()
