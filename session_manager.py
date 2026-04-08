import hashlib
import requests

API_SECRET = "YOUR_SECRET"
SESSION_FILE = "session.txt"

def generate_session(auth_code, user_id):
    raw = user_id + auth_code + API_SECRET
    checksum = hashlib.sha256(raw.encode()).hexdigest()

    url = "https://ant.aliceblueonline.com/open-api/od/v1/vendor/getUserDetails"

    res = requests.post(url, json={"checkSum": checksum})
    data = res.json()

    if data["stat"] != "Ok":
        raise Exception(data["emsg"])

    save_session(data["userSession"])
    return data["userSession"]


def save_session(session):
    with open(SESSION_FILE, "w") as f:
        f.write(session)


def load_session():
    try:
        with open(SESSION_FILE, "r") as f:
            return f.read()
    except:
        return None
