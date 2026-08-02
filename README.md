# Stock Screener Application

A Streamlit stock screener for NSE/BSE with AliceBlue API integration.

## AliceBlue auth (updated ANT API)

AliceBlue no longer uses a static `user_id` + `api_key` login for vendor apps.

Current flow ([docs](https://ant.aliceblueonline.com/productdocumentation/Authentication/)):

1. Register an app at [AliceBlue Developer Portal](https://a3.aliceblueonline.com/)
2. Set **Redirect URL** to your Streamlit app, e.g. `https://learninglab.streamlit.app/`
3. Use **App Code** (`app_key`) + **API Secret** (`api_secret`) in Streamlit secrets
4. User opens `https://ant.aliceblueonline.com/?appcode=YOUR_APP_CODE`
5. After login, AliceBlue redirects to your URL with `authCode` and `userId`
6. App builds SHA-256 checksum: `userId + authCode + apiSecret`
7. POST checksum to `getUserDetails` → receive `userSession`
8. API calls use header: `Authorization: Bearer {userId} {userSession}`

Historical chart data is usually available **weekdays 5:30 PM–8:00 AM IST**, and on weekends/holidays. It is often unavailable during market hours.

## Streamlit Cloud secrets

In your app → **Settings → Secrets**, use:

```toml
[aliceblue]
app_key = "YOUR_APP_CODE"
api_secret = "YOUR_API_SECRET"
```

Do **not** use the old format:

```toml
# outdated — will not work with current AliceBlue auth
[aliceblue]
user_id = "..."
api_key = "..."
```

## Local Development

1. Clone the repository
2. Copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` and fill credentials
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   streamlit run app.py
   ```
   Or double-click `run_local.bat` on Windows.

For local OAuth, set AliceBlue Redirect URL to `http://localhost:8501` (or keep the Cloud URL and paste the redirect URL into the app).

## Requirements

- Python 3.8+
- Streamlit 1.30.0+
- Dependencies in `requirements.txt`

## Security

Never commit `.streamlit/secrets.toml` or API secrets to GitHub. Use Streamlit Cloud Secrets for deployment.
