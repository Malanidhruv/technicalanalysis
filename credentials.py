"""Load AliceBlue credentials from env, Streamlit secrets, or local secrets file."""

import os
from pathlib import Path

_SECRETS_CACHE = None


def _load_local_secrets():
    global _SECRETS_CACHE
    if _SECRETS_CACHE is not None:
        return _SECRETS_CACHE

    _SECRETS_CACHE = {}
    secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if not secrets_path.is_file():
        return _SECRETS_CACHE

    try:
        try:
            import tomllib
            with secrets_path.open("rb") as f:
                _SECRETS_CACHE = tomllib.load(f)
        except ImportError:
            import toml
            with secrets_path.open("r", encoding="utf-8") as f:
                _SECRETS_CACHE = toml.load(f)
    except Exception:
        _SECRETS_CACHE = {}

    return _SECRETS_CACHE


def _from_streamlit_secrets(section, *keys):
    try:
        import streamlit as st
        data = st.secrets.get(section, {})
        for key in keys:
            value = data.get(key)
            if value:
                return str(value).strip()
    except Exception:
        pass
    return None


def _from_local_secrets(section, *keys):
    data = _load_local_secrets().get(section, {})
    for key in keys:
        value = data.get(key)
        if value:
            return str(value).strip()
    return None


def get_app_key():
    if key := os.environ.get("ALICEBLUE_APP_KEY"):
        return key.strip()
    return (
        _from_streamlit_secrets("aliceblue", "app_key", "appcode", "app_code")
        or _from_local_secrets("aliceblue", "app_key", "appcode", "app_code")
    )


def get_api_secret():
    if secret := os.environ.get("ALICEBLUE_API_SECRET"):
        return secret.strip()
    return (
        _from_streamlit_secrets("aliceblue", "api_secret", "api_key", "apiSecret")
        or _from_local_secrets("aliceblue", "api_secret", "api_key", "apiSecret")
    )


def credentials_configured():
    return bool(get_app_key() and get_api_secret())
