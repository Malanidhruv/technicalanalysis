import hashlib
import requests
import streamlit as st

from credentials import get_api_secret


def generate_session(auth_code, user_id):
    """
    Generate AliceBlue session from OAuth auth_code + user_id.
    Returns session token string on success, None on failure.
    """
    api_secret = get_api_secret()
    if not api_secret:
        st.error(
            "AliceBlue API secret not configured. Add credentials to "
            "`.streamlit/secrets.toml`, set `ALICEBLUE_API_SECRET`, or "
            "configure Streamlit Cloud secrets."
        )
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
