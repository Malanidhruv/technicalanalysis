import json
import os

SESSION_FILE = "temp_session.json"


def save_session(session):
    """Save user session (JWT) to local file."""
    data = {"session": session}
    try:
        with open(SESSION_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"⚠️ Could not save session: {e}")


def get_session():
    """Get stored session from local file. Returns None if not found."""
    if not os.path.exists(SESSION_FILE):
        return None
    try:
        with open(SESSION_FILE, "r") as f:
            data = json.load(f)
            return data.get("session")
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
