import hashlib
import requests
import streamlit as st

API_SECRET = "W5X0oyuQJQLpvY68rRhugYSv4QypU9HjS2dgSFAkpkZMec1RW1ag4qiXUp5ipnTxt64wRlaJcaWOXarWWHsw9UmkXBKFoXeU4nUm"

def generate_session(auth_code, user_id):
    raw = user_id + auth_code + API_SECRET
    checksum = hashlib.sha256(raw.encode()).hexdigest()

    url = "https://ant.aliceblueonline.com/open-api/od/v1/vendor/getUserDetails"

    res = requests.post(url, json={"checkSum": checksum})
    data = res.json()

    if data["stat"] != "Ok":
        st.error(data["emsg"])
        return None

    return data["userSession"]
