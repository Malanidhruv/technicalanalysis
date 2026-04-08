from flask import Flask, request
from session_manager import generate_session

app = Flask(__name__)

@app.route("/callback")
def callback():
    auth_code = request.args.get("authCode")
    user_id = request.args.get("userId")

    session = generate_session(auth_code, user_id)

    return f"✅ Login successful. Session created."

if __name__ == "__main__":
    app.run(port=8000)
