import hashlib
import requests
import streamlit as st
import os

# ✅ Load API secret from environment variable (never hardcode secrets)
API_SECRET = os.environ.get(
    "ALICEBLUE_API_SECRET",
    "W5X0oyuQJQLpvY68rRhugYSv4QypU9HjS2dgSFAkpkZMec1RW1ag4qiXUp5ipnTxt64wRlaJcaWOXarWWHsw9UmkXBKFoXeU4nUm"
)


def generate_session(auth_code, user_id):
    """
    Generate AliceBlue session from OAuth auth_code + user_id.
    Returns session token string on success, None on failure.
    """
    raw = user_id + auth_code + API_SECRET
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
