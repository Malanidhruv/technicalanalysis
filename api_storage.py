import json
import os

SESSION_FILE = "temp_session.json"


def save_session(session, user_id=None):
    """Save user session and AliceBlue user id to local file."""
    data = {"session": session}
    if user_id:
        data["user_id"] = user_id
    try:
        with open(SESSION_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"⚠️ Could not save session: {e}")


def get_session():
    """Get stored session token. Returns None if not found."""
    creds = get_credentials()
    return creds.get("session") if creds else None


def get_user_id():
    """Get stored AliceBlue user id. Returns None if not found."""
    creds = get_credentials()
    return creds.get("user_id") if creds else None


def get_credentials():
    """Load session credentials from local file."""
    if not os.path.exists(SESSION_FILE):
        return None
    try:
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Could not read session file: {e}")
        return None


def clear_session():
    """Clear session file (logout)."""
    if os.path.exists(SESSION_FILE):
        try:
            os.remove(SESSION_FILE)
        except Exception as e:
            print(f"⚠️ Could not delete session file: {e}")
