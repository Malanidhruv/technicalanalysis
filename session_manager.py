import hashlib
import requests

from credentials import get_api_secret

# Current ANT docs use a3; keep older endpoints as fallback.
SESSION_URLS = [
    "https://a3.aliceblueonline.com/open-api/od/v1/vendor/getUserDetails",
    "https://ant.aliceblueonline.com/open-api/od/v1/vendor/getUserDetails",
    "https://ant.aliceblueonline.com/rest/AliceBlueAPIService/sso/getUserDetails",
]


def generate_session_core(auth_code, user_id):
    """
    Exchange AliceBlue auth_code + user_id for a session token.
    Returns (session_token, error_message).
    """
    api_secret = get_api_secret()
    if not api_secret:
        return None, "API secret not configured in .streamlit/secrets.toml"

    if not auth_code or not user_id:
        return None, "authCode and userId are required"

    raw = user_id + auth_code + api_secret
    checksum = hashlib.sha256(raw.encode()).hexdigest()
    payload = {"checkSum": checksum}
    last_error = "Unknown error"

    for url in SESSION_URLS:
        try:
            res = requests.post(url, json=payload, timeout=15)
            res.raise_for_status()
            data = res.json()
        except requests.exceptions.RequestException as exc:
            last_error = f"{url}: network error - {exc}"
            continue
        except Exception as exc:
            last_error = f"{url}: invalid response - {exc}"
            continue

        if data.get("stat") == "Ok" and data.get("userSession"):
            return data["userSession"], None

        last_error = data.get("emsg") or data.get("message") or str(data)

    return None, last_error


def generate_session(auth_code, user_id):
    """Streamlit wrapper around generate_session_core."""
    import streamlit as st

    session, error = generate_session_core(auth_code, user_id)
    if session:
        return session

    if error and "not configured" in error:
        st.error(
            "AliceBlue API secret not configured. Add credentials to "
            "`.streamlit/secrets.toml`, set `ALICEBLUE_API_SECRET`, or "
            "configure Streamlit Cloud secrets."
        )
    elif error:
        st.error(f"Login error: {error}")
    else:
        st.error("No session token received from AliceBlue.")
    return None
