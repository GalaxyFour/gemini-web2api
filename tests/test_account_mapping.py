import http.client
import json
import os
import threading
import unittest

from gemini_web2api.config import CONFIG, get_all_api_keys, resolve_account_from_config
from gemini_web2api.server import GeminiHandler, ThreadedServer


class AccountMappingConfigTests(unittest.TestCase):
    def test_get_all_api_keys_empty(self):
        cfg = {}
        self.assertEqual(get_all_api_keys(cfg), [])

    def test_get_all_api_keys_dict_and_list(self):
        cfg = {
            "api_keys": ["sk-global"],
            "accounts": {
                "sk-acc1": {"cookie_file": "/tmp/c1.json"},
                "account_two": {
                    "api_keys": ["sk-acc2-a", "sk-acc2-b"],
                    "cookie_file": "/tmp/c2.json",
                },
                "account_three": {
                    "api_key": "sk-acc3",
                    "auth_user": "1",
                },
            },
        }
        keys = get_all_api_keys(cfg)
        self.assertEqual(keys, ["sk-global", "sk-acc1", "sk-acc2-a", "sk-acc2-b", "sk-acc3"])

    def test_resolve_account_from_config(self):
        c1 = {"cookie_file": "/tmp/c1.json"}
        c2 = {"api_keys": ["sk-acc2"], "cookie_file": "/tmp/c2.json", "auth_user": "1"}
        cfg = {
            "accounts": {
                "sk-acc1": c1,
                "account_two": c2,
            }
        }
        self.assertIs(resolve_account_from_config(cfg, "sk-acc1"), c1)
        self.assertIs(resolve_account_from_config(cfg, "sk-acc2"), c2)
        self.assertIsNone(resolve_account_from_config(cfg, "unknown"))


class AccountMappingServerTests(unittest.TestCase):
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
        cls.thread.join(timeout=5)

    def setUp(self):
        self.original_config = dict(CONFIG)
        CONFIG.clear()
        CONFIG.update(self.original_config)
        CONFIG["log_requests"] = False
        CONFIG["accounts"] = {
            "sk-user-1": {
                "cookie_file": "/tmp/fake-cookie-1.json",
                "auth_user": None,
                "xsrf_token": "token-1",
            },
            "sk-user-2": {
                "cookie_file": "/tmp/fake-cookie-2.json",
                "auth_user": "1",
                "xsrf_token": "token-2",
            },
        }

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def get(self, path, key=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        conn.request("GET", path, headers=headers)
        res = conn.getresponse()
        body = res.read().decode()
        conn.close()
        return res.status, body

    def test_authorized_with_account_key(self):
        status, body = self.get("/v1/models", key="sk-user-1")
        self.assertEqual(status, 200)

        status, body = self.get("/v1/models", key="sk-user-2")
        self.assertEqual(status, 200)

        status, body = self.get("/v1/models", key="invalid-key")
        self.assertEqual(status, 401)


if __name__ == "__main__":
    unittest.main()
