"""CLI test: authorize AliceBlue and run a small scanner check."""

import argparse
import sys
from datetime import datetime, timedelta

from alice_client import Aliceblue, clear_cache
from api_storage import save_session, get_session, get_user_id
from credentials import credentials_configured, get_app_key
from session_manager import generate_session_core
from stock_analysis import analyze_all_tokens
from stock_lists import STOCK_LISTS


def main():
    parser = argparse.ArgumentParser(description="Test AliceBlue auth and scanner")
    parser.add_argument("--auth-code", help="authCode from AliceBlue redirect URL")
    parser.add_argument("--user-id", help="userId from AliceBlue redirect URL")
    parser.add_argument(
        "--strategy",
        default="Consolidation Breakout",
        help="Strategy to test (default: Consolidation Breakout)",
    )
    args = parser.parse_args()

    print("=== Harion Research Scanner Test ===\n")

    print(f"[1/4] Credentials configured: {credentials_configured()}")
    if not credentials_configured():
        print("FAIL: Add app_key and api_secret to .streamlit/secrets.toml")
        return 1

    session = get_session()
    auth_code = args.auth_code
    user_id = args.user_id or get_user_id()

    if auth_code and args.user_id:
        print("[2/4] Logging in with provided authCode/userId...")
        session, error = generate_session_core(auth_code, args.user_id)
        if not session:
            print(f"FAIL: Login failed - {error}")
            return 1
        user_id = args.user_id
        save_session(session, user_id)
        print("OK: Login successful, session saved.")
    elif session and user_id:
        print("[2/4] Using saved session from temp_session.json")
    elif session:
        print("[2/4] Saved session missing user_id — log in again.")
        return 1
    else:
        app_key = get_app_key()
        print("[2/4] No session found.")
        print(f"      Open: https://ant.aliceblueonline.com/?appcode={app_key}")
        print("      Then run:")
        print("      python test_scanner.py --auth-code YOUR_CODE --user-id YOUR_ID")
        return 1

    print("[3/4] Fetching sample NIFTY 50 data...")
    alice = Aliceblue(session, user_id)
    tokens = STOCK_LISTS["NIFTY 50"][:10]
    clear_cache()

    try:
        instrument = alice.get_instrument_by_token("NSE", tokens[0])
        df = alice.get_historical(
            instrument,
            datetime.now() - timedelta(days=365),
            datetime.now(),
            "D",
            "NSE",
        )
        bars = len(df) if df is not None else 0
        print(f"      Token {tokens[0]}: {bars} daily bars returned")
        if bars < 50:
            print("WARN: Low/no historical data (API may be unavailable during market hours)")
    except Exception as exc:
        print(f"FAIL: Historical data error - {exc}")
        return 1

    print(f"[4/4] Running '{args.strategy}' on NIFTY 50 (50 stocks)...")
    results, stats = analyze_all_tokens(
        alice, STOCK_LISTS["NIFTY 50"], args.strategy, exchange="NSE"
    )
    print(f"      Scan: {stats['tokens']} tokens, {stats['with_data']} with data, "
          f"{stats['matched']} matched, {stats['errors']} errors")

    if stats["with_data"] == 0:
        print("\nRESULT: Scanner cannot run — no price data from AliceBlue.")
        print("        Try again after 5:30 PM IST or on weekends.")
        return 2

    if results:
        print(f"\nRESULT: Scanner working — {len(results)} matches found.")
        for row in results[:5]:
            print(f"  - Token {row['Name']}: strength {row.get('Strength')}, "
                  f"close {row.get('Close')}")
    else:
        print("\nRESULT: Auth OK, data OK, but no matches for this strategy right now.")
        print("        Try 'Market Structure Analysis' or 'Strong Uptrend Scanner'.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
