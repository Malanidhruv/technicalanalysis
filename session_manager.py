import hashlib
import requests
import streamlit as st
import os

def _get_api_secret():
    if secret := os.environ.get("ALICEBLUE_API_SECRET"):
        return secret
    try:
        return st.secrets["aliceblue"]["api_secret"]
    except Exception:
        return None


def generate_session(auth_code, user_id):
    """
    Generate AliceBlue session from OAuth auth_code + user_id.
    Returns session token string on success, None on failure.
    """
    api_secret = _get_api_secret()
    if not api_secret:
        st.error("AliceBlue API secret not configured. Set ALICEBLUE_API_SECRET or add to .streamlit/secrets.toml.")
        return None

    raw = user_id + auth_code + api_secret
    checksum = hashlib.sha256(raw.encode()).hexdigest()

    url = "https://ant.aliceblueonline.com/open-api/od/v1/vendor/getUserDetails"

    try:
        res = requests.post(
            url,
            json={"checkSum": checksum},
            timeout=15
        )
        res.raise_for_status()
        data = res.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Network error during login: {e}")
        return None
    except Exception as e:
        st.error(f"Login failed: {e}")
        return None

    if data.get("stat") != "Ok":
        st.error(f"Login error: {data.get('emsg', 'Unknown error')}")
        return None

    session = data.get("userSession")
    if not session:
        st.error("No session token received from AliceBlue.")
        return None

    return session
