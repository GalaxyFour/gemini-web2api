"""Configuration management."""
import json
import os
import threading

_request_context = threading.local()


def set_current_account(account=None):
    """Set the active account mapping for the current request thread."""
    _request_context.account = account


def get_current_account():
    """Get the active account mapping for the current request thread."""
    return getattr(_request_context, "account", None)


class ContextConfig(dict):
    """Dict wrapper that dynamically resolves account overrides for the active request thread."""

    def get(self, key, default=None):
        acc = get_current_account()
        if acc is not None and key in acc:
            return acc[key]
        return super().get(key, default)

    def __getitem__(self, key):
        acc = get_current_account()
        if acc is not None and key in acc:
            return acc[key]
        return super().__getitem__(key)

    def __contains__(self, key):
        acc = get_current_account()
        if acc is not None and key in acc:
            return True
        return super().__contains__(key)


DEFAULT_CONFIG = {
    "port": 8081,
    "host": "0.0.0.0",
    "retry_attempts": 3,
    "retry_delay_sec": 2,
    "request_timeout_sec": 180,
    "gemini_bl": "boq_assistant-bard-web-server_20260716.08_p0",
    "auth_user": None,
    "xsrf_token": None,
    "default_model": "gemini-3.6-flash",
    "log_requests": True,
    "cookie_file": None,
    "proxy": None,
    "api_keys": [],
    "accounts": None,
    "temporary_chats": False,
    # Generated image output only; values above the hard safety caps are ignored.
    "generated_image_max_bytes": 10 * 1024 * 1024,
    "generated_image_max_redirects": 3,
}

CONFIG = ContextConfig(DEFAULT_CONFIG)


def resolve_account_from_config(config: dict, api_key: str):
    """Find the account config associated with a given API key."""
    if not api_key:
        return None
    accounts = config.get("accounts")
    if not accounts:
        return None

    if isinstance(accounts, dict):
        if api_key in accounts and isinstance(accounts[api_key], dict):
            return accounts[api_key]
        for name, acc in accounts.items():
            if isinstance(acc, dict):
                keys = acc.get("api_keys")
                if isinstance(keys, list) and api_key in keys:
                    return acc
                if isinstance(keys, str) and api_key == keys:
                    return acc
                if acc.get("api_key") == api_key:
                    return acc
    elif isinstance(accounts, list):
        for acc in accounts:
            if isinstance(acc, dict):
                keys = acc.get("api_keys")
                if isinstance(keys, list) and api_key in keys:
                    return acc
                if isinstance(keys, str) and api_key == keys:
                    return acc
                if acc.get("api_key") == api_key:
                    return acc
    return None


def get_all_api_keys(config: dict) -> list:
    """Return all valid API keys accepted by config and accounts."""
    keys = []
    base_keys = config.get("api_keys")
    if isinstance(base_keys, list):
        keys.extend(base_keys)
    elif isinstance(base_keys, str):
        keys.append(base_keys)

    accounts = config.get("accounts")
    if accounts:
        if isinstance(accounts, dict):
            for k, v in accounts.items():
                if isinstance(v, dict):
                    acc_keys = v.get("api_keys")
                    if isinstance(acc_keys, list):
                        keys.extend(acc_keys)
                    elif isinstance(acc_keys, str):
                        keys.append(acc_keys)
                    if v.get("api_key"):
                        keys.append(v["api_key"])
                    if not v.get("api_keys") and not v.get("api_key"):
                        keys.append(k)
                else:
                    keys.append(k)
        elif isinstance(accounts, list):
            for acc in accounts:
                if isinstance(acc, dict):
                    acc_keys = acc.get("api_keys")
                    if isinstance(acc_keys, list):
                        keys.extend(acc_keys)
                    elif isinstance(acc_keys, str):
                        keys.append(acc_keys)
                    if acc.get("api_key"):
                        keys.append(acc["api_key"])
    return list(dict.fromkeys(keys))


def load_config(path: str = None):
    """Load config from JSON file."""
    if path and os.path.exists(path):
        with open(path) as f:
            CONFIG.update(json.load(f))
    return CONFIG


def find_config():
    """Search for config file in standard locations."""
    for p in ["./config.json", os.path.expanduser("~/.config/gemini-web2api/config.json")]:
        if os.path.exists(p):
            return p
    return None

