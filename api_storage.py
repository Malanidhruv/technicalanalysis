import json
import os

SESSION_FILE = "temp_session.json"


def save_session(session):
    """Save user session (JWT)"""
    data = {
        "session": session
    }
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f)


def get_session():
    """Get stored session"""
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            try:
                data = json.load(f)
                return data.get("session")
            except json.JSONDecodeError:
                pass
    return None


def clear_session():
    """Clear session (logout)"""
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
