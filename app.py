import streamlit as st
import pandas as pd
import datetime
from urllib.parse import parse_qs, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from alice_client import initialize_alice
from session_manager import generate_session
from api_storage import save_session, get_session, get_user_id
from advanced_analysis import (
    analyze_all_tokens_advanced,
    analyze_all_tokens_custom
)
from stock_analysis import analyze_all_tokens
from educational_scorer import (
    compare_stocks_educational,
    generate_daily_lesson
)
from stock_lists import STOCK_LISTS
from credentials import get_app_key, credentials_configured
from utils import generate_tradingview_link


# ===== PAGE CONFIG (must be first Streamlit call) =====
st.set_page_config(
    page_title="Harion Research Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)


def _qp_get(params, name):
    """Read a query param case-insensitively."""
    for key in params:
        if key.lower() == name.lower():
            val = params[key]
            return val[0] if isinstance(val, list) else val
    return None


def _parse_auth_from_text(text):
    """Extract authCode and userId from a redirect URL or query string."""
    text = (text or "").strip()
    if not text:
        return None, None

    query = text
    if "?" in text:
        query = urlparse(text).query
    elif "=" in text:
        query = text.lstrip("?")

    params = parse_qs(query)
    auth_code = user_id = None
    for key, values in params.items():
        if not values:
            continue
        if key.lower() == "authcode":
            auth_code = values[0].strip()
        elif key.lower() == "userid":
            user_id = values[0].strip()
    return auth_code, user_id


def _complete_login(auth_code, user_id):
    """Exchange AliceBlue auth code for a session and persist it."""
    if not auth_code or not user_id:
        st.error("Both authCode and userId are required.")
        return False

    session = generate_session(auth_code, user_id)
    if session:
        st.session_state["session"] = session
        st.session_state["user_id"] = user_id
        save_session(session, user_id)
        st.query_params.clear()
        st.rerun()
    return bool(session)


# ===== LOGIN HANDLER =====
params = st.query_params
auth_code = _qp_get(params, "authCode")
user_id = _qp_get(params, "userId")

if auth_code and user_id:
    if "session" not in st.session_state:
        if not _complete_login(auth_code, user_id):
            st.error("❌ Login failed. Please try again.")
            st.query_params.clear()
    else:
        st.query_params.clear()

# ===== LOAD SAVED SESSION =====
if "session" not in st.session_state:
    stored_session = get_session()
    stored_user_id = get_user_id()
    if stored_session:
        st.session_state["session"] = stored_session
    if stored_user_id:
        st.session_state["user_id"] = stored_user_id

# ===== STYLES =====
st.markdown("""
    <style>
        .button-box { margin-bottom: 1rem; }

        /* NSE Selected - Aqua */
        .nse-btn button {
            background-color: #A7D6D6 !important;
            color: black !important;
        }
        .nse-btn button:hover { background-color: #94CACA !important; }

        /* BSE Selected - Peach */
        .bse-btn button {
            background-color: #F9D5C2 !important;
            color: black !important;
        }
        .bse-btn button:hover { background-color: #F2C0AC !important; }

        /* Inactive Button */
        .default-btn button {
            background-color: #DDEBF1 !important;
            color: black !important;
        }
        .default-btn button:hover { background-color: #CFE0E8 !important; }

        /* Strength Score Badge */
        .strength-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 14px;
        }
        .strength-high  { background-color: #28a745; color: white; }
        .strength-medium { background-color: #ffc107; color: black; }
        .strength-low   { background-color: #dc3545; color: white; }

        /* Educational note box */
        .edu-note {
            background-color: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 12px;
            margin: 8px 0;
            border-radius: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# ===== SESSION STATE DEFAULTS =====
if 'selected_exchange' not in st.session_state:
    st.session_state.selected_exchange = 'NSE'
if 'show_educational' not in st.session_state:
    st.session_state.show_educational = True
if 'user_tier' not in st.session_state:
    st.session_state.user_tier = 'Free'


def get_stock_lists_for_exchange(exchange):
    if exchange == 'NSE':
        return {k: v for k, v in STOCK_LISTS.items() if k in [
            'NIFTY FNO', 'NIFTY 50', 'NIFTY 200', 'NIFTY 500', 'ALL STOCKS'
        ]}
    else:
        return {k: v for k, v in STOCK_LISTS.items() if k in [
            'BSE 500', 'BSE Large Cap Index', 'BSE Mid Cap Index',
            'BSE Small Cap Index', 'BSE 400 MidSmallCap',
            'BSE 250 LargeMidCap', 'BSE ALL STOCKS'
        ]}


# ===== HEADER =====
st.markdown("""
    <div class="header">
        <h1>📈 Harion Research Stock Screener</h1>
        <p>Learn Technical Analysis While Screening NSE &amp; BSE Stocks</p>
    </div>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### Authentication")
    if "session" not in st.session_state:
        app_key = get_app_key()
        login_url = f"https://ant.aliceblueonline.com/?appcode={app_key}" if app_key else None

        if not credentials_configured():
            st.error(
                "AliceBlue credentials missing. Create `.streamlit/secrets.toml` locally "
                "or add secrets in Streamlit Cloud (see `.streamlit/secrets.toml.example`)."
            )

        if login_url:
            # Streamlit Cloud cannot open the user's browser via webbrowser.open().
            st.link_button("🔐 Login to AliceBlue", login_url, use_container_width=True)
            st.caption(
                "After login you return to your Redirect URL "
                "(`https://learninglab.streamlit.app/`) with `authCode` + `userId`. "
                "This app reads those params automatically."
            )
        else:
            st.error(
                "AliceBlue app key not configured. On Streamlit Cloud add Secrets "
                "(see `.streamlit/secrets.toml.example`)."
            )

        with st.expander("Manual login (if auto-redirect fails)", expanded=False):
            st.markdown(
                "Paste the full redirect URL, or enter `authCode` and `userId`:"
            )
            pasted_url = st.text_area(
                "Paste redirect URL",
                placeholder="https://learninglab.streamlit.app/?authCode=XXXX&userId=YYYY",
                key="manual_redirect_url",
                height=80,
            )
            col1, col2 = st.columns(2)
            with col1:
                manual_auth = st.text_input("authCode", key="manual_auth_code")
            with col2:
                manual_user = st.text_input("userId", key="manual_user_id")

            if st.button("Connect", use_container_width=True, type="primary"):
                parsed_auth, parsed_user = _parse_auth_from_text(pasted_url)
                auth_val = manual_auth.strip() or (parsed_auth or "")
                user_val = manual_user.strip() or (parsed_user or "")
                if not _complete_login(auth_val, user_val):
                    st.error("❌ Login failed. Get a fresh authCode — it expires quickly.")

        st.warning("Please login to continue")
        st.stop()  # Stop rendering until logged in
    else:
        st.success("✅ Logged in")
        if "user_id" not in st.session_state:
            st.warning("Session outdated — please log out and log in again.")
        if st.button("🚪 Logout", use_container_width=True):
            from api_storage import clear_session
            from alice_client import clear_cache
            clear_session()
            clear_cache()
            del st.session_state["session"]
            if "user_id" in st.session_state:
                del st.session_state["user_id"]
            st.rerun()

    st.markdown("---")

    st.markdown("### Account Type")
    st.session_state.user_tier = st.selectbox(
        "Select Tier",
        ["Free", "Premium", "Pro"],
        help="Different tiers unlock different features"
    )

    tier_info = {
        "Free":    "• 2 strategies\n• Top 10 results\n• Basic view",
        "Premium": "• All strategies\n• Top 50 results\n• Educational features\n• Daily lessons",
        "Pro":     "• Unlimited results\n• All features\n• Priority support\n• API access"
    }
    st.info(tier_info[st.session_state.user_tier])

    st.markdown("---")

    st.markdown("### Settings")
    st.session_state.show_educational = st.checkbox(
        "Show Educational Features",
        value=st.session_state.user_tier in ["Premium", "Pro"],
        disabled=st.session_state.user_tier == "Free",
        help="Educational analysis and learning content"
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
        This screener uses professional technical analysis to identify opportunities
        while teaching you technical analysis concepts.

        ⚠️ Educational tool only - not investment advice.
    """)

# ===== INITIALIZE ALICE (shared by both tabs) =====
try:
    alice = initialize_alice()
except Exception as e:
    st.error(f"❌ Could not initialize AliceBlue: {e}")
    st.stop()

tab_tech, tab_swing, tab_breakout = st.tabs([
    "Technical Screener",
    "Swing Setup (Technical + Fundamental)",
    "Breakout Watch",
])

with tab_swing:
    try:
        from swing_setup import render_swing_setup_tab
        render_swing_setup_tab(alice)
    except ImportError as exc:
        st.error(
            f"Swing Setup dependencies missing: {exc}. "
            "Ensure `beautifulsoup4` is in requirements.txt, then "
            "Manage app → Reboot so Streamlit Cloud reinstalls packages."
        )

with tab_breakout:
    try:
        from breakout_watch import render_breakout_watch_tab
        render_breakout_watch_tab(alice)
    except ImportError as exc:
        st.error(f"Breakout Watch dependencies missing: {exc}")

with tab_tech:
    # ===== EXCHANGE TOGGLE =====
    col1, col2 = st.columns(2)

    with col1:
        btn_class = "nse-btn" if st.session_state.selected_exchange == 'NSE' else "default-btn"
        st.markdown(f'<div class="button-box {btn_class}">', unsafe_allow_html=True)
        if st.button("NSE", key="nse_btn", help="Switch to NSE stocks", use_container_width=True):
            st.session_state.selected_exchange = 'NSE'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        btn_class = "bse-btn" if st.session_state.selected_exchange == 'BSE' else "default-btn"
        st.markdown(f'<div class="button-box {btn_class}">', unsafe_allow_html=True)
        if st.button("BSE", key="bse_btn", help="Switch to BSE stocks", use_container_width=True):
            st.session_state.selected_exchange = 'BSE'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ===== STRATEGY SELECTION =====
    col1, col2 = st.columns(2)

    with col1:
        available_lists = get_stock_lists_for_exchange(st.session_state.selected_exchange)
        list_options = list(available_lists.keys())
        ms_key = f"tech_multiselect_{st.session_state.selected_exchange}"
        if ms_key not in st.session_state:
            st.session_state[ms_key] = [list_options[0]] if list_options else []
        else:
            # Drop stale names if exchange lists change
            st.session_state[ms_key] = [
                n for n in st.session_state[ms_key] if n in available_lists
            ] or ([list_options[0]] if list_options else [])

        selected_lists = st.multiselect(
            "Stock lists (one or more)",
            options=list_options,
            key=ms_key,
            help="Universe is the union of selected lists (duplicates removed).",
            format_func=lambda n: f"{n} ({len(available_lists.get(n, []))})",
        )

        # De-dupe tokens while preserving order
        seen_tokens = set()
        tokens = []
        for name in selected_lists:
            for t in available_lists.get(name, []):
                key = str(t)
                if key in seen_tokens:
                    continue
                seen_tokens.add(key)
                tokens.append(t)
        if selected_lists:
            st.caption(
                f"{', '.join(selected_lists)} → **{len(tokens)}** unique symbols"
            )
        else:
            st.caption("Select at least one list to scan.")

    with col2:
        if st.session_state.user_tier == "Free":
            available_strategies = [
                "Strong Uptrend Scanner",
                "Price Action Breakout"
            ]
        elif st.session_state.user_tier == "Premium":
            available_strategies = [
                "Strong Uptrend Scanner",
                "Pullback to Support",
                "Volume Breakout",
                "Price Action Breakout",
                "Volume Profile Analysis"
            ]
        else:  # Pro
            available_strategies = [
                "Strong Uptrend Scanner",
                "Pullback to Support",
                "Volume Breakout",
                "Market Leaders",
                "Consolidation Breakout",
                "Price Action Breakout",
                "Volume Profile Analysis",
                "Market Structure Analysis",
                "Multi-Factor Analysis",
                "Custom Price Movement"
            ]

        strategy = st.selectbox(
            "Select Strategy",
            available_strategies,
            help="Choose a technical analysis strategy"
        )

    # ===== STRATEGY DESCRIPTIONS =====
    strategy_descriptions = {
        "Strong Uptrend Scanner": {
            "description": """
                📈 **Strong Uptrend Scanner**

                Identifies stocks in powerful uptrends with all technical factors aligned.

                **What it finds:**
                - Price above all key EMAs (9, 21, 50, 200)
                - Making higher highs and higher lows
                - RSI showing bullish momentum (50-75)
                - Volume supporting the uptrend

                **Best for:** Trending markets, momentum trading
                **Risk:** Uptrends can reverse suddenly - use stop-losses
            """,
            "learning": "Trend following is one of the most profitable strategies. This scanner finds the strongest uptrends where all timeframes agree."
        },
        "Pullback to Support": {
            "description": """
                🎯 **Pullback to Support Scanner**

                Finds low-risk entries in strong stocks temporarily pulling back.

                **What it finds:**
                - Overall uptrend intact (above 200 EMA)
                - Price pulled back to support level
                - RSI oversold on pullback (30-45)
                - Volume decreasing (healthy pullback)

                **Best for:** Better risk-reward entries, swing trading
                **Risk:** Support can break - confirm before entering
            """,
            "learning": "Best trades often come on pullbacks, not at highs. This scanner finds these low-risk entry points."
        },
        "Volume Breakout": {
            "description": """
                🚀 **Volume Breakout Scanner**

                Catches stocks breaking out with institutional participation.

                **What it finds:**
                - Price breaking above resistance
                - Volume 2x+ above average (institutional buying)
                - MACD turning positive
                - Not overextended (RSI < 80)

                **Best for:** Explosive moves, catching new trends
                **Risk:** False breakouts happen - wait for confirmation
            """,
            "learning": "Volume confirms everything. Without volume, breakouts often fail. This scanner finds volume-confirmed moves."
        },
        "Market Leaders": {
            "description": """
                👑 **Market Leaders Scanner**

                Finds stocks showing exceptional relative strength.

                **What it finds:**
                - Near ~1-year highs (within ~15–20%)
                - Positive 3-month performance
                - Meaningful trend strength (ADX)
                - Above 50 EMA with decent volume

                **Best for:** Finding the strongest stocks, momentum trading
                **Risk:** Leaders can become laggards - monitor closely
            """,
            "learning": "Market leaders often continue leading. These stocks have the strongest fundamentals and technicals."
        },
        "Consolidation Breakout": {
            "description": """
                💥 **Consolidation Breakout Scanner**

                Spots tight ranges ready to explode with volume.

                **What it finds:**
                - Tight consolidation (< 8% range)
                - Volume contraction during consolidation
                - Breakout with volume expansion
                - Above consolidation high

                **Best for:** Catching explosive moves after compression
                **Risk:** Direction uncertain - wait for confirmed breakout
            """,
            "learning": "Tight consolidation = energy building. When it breaks with volume, explosive moves often follow."
        },
        "Price Action Breakout": {
            "description": """
                🚀 **Bullish Price Action Breakout (Top 5)**

                Shows only the **5 highest-confidence bullish** setups.

                **Must have:**
                - Not in a downtrend (no bearish engulfing)
                - Breakout / near 20-day high **or** bullish candle pattern
                - Volume confirmation preferred
                - Strength score ≥ 55

                **Best for:** Catching bullish breakouts with conviction
            """,
            "learning": "Fewer, higher-quality bullish breakouts beat a long list of weak signals."
        },
        "Volume Profile Analysis": {
            "description": """
                - Identifies high-volume price levels
                - Analyzes volume distribution
                - Detects institutional buying/selling
                - Includes volume-weighted price levels
            """,
            "learning": "Volume profile shows where institutions are buying/selling."
        },
        "Market Structure Analysis": {
            "description": """
                - Analyzes market structure (HH/HL vs LH/LL)
                - Identifies trend strength and direction
                - Includes multiple timeframe analysis
                - Considers market regime (trending vs ranging)
            """,
            "learning": "Market structure reveals the underlying trend and potential reversal points."
        },
        "Multi-Factor Analysis": {
            "description": """
                - Combines price action, volume, and market structure
                - Includes relative strength analysis
                - Considers sector rotation
                - Integrates market breadth indicators
            """,
            "learning": "Multi-factor approach increases probability by requiring multiple confirmations."
        },
        "Custom Price Movement": {
            "description": """
                - Customizable price movement analysis
                - Set your own duration and percentage targets
                - Track stocks moving up or down by your specified amount
                - Includes volume trend and volatility analysis
            """,
            "learning": "Custom filters let you find specific price movements based on your criteria."
        }
    }

    if strategy in strategy_descriptions:
        st.markdown("### Strategy Details")
        st.markdown(strategy_descriptions[strategy]["description"])

        if st.session_state.show_educational:
            st.info(f"💡 **Learning Point:** {strategy_descriptions[strategy]['learning']}")

    # ===== CUSTOM PRICE MOVEMENT INPUTS =====
    if strategy == "Custom Price Movement":
        col1, col2, col3 = st.columns(3)
        with col1:
            duration_days = st.number_input(
                "Duration (Days)", min_value=1, max_value=365, value=30,
                help="Number of days to look back"
            )
        with col2:
            target_percentage = st.number_input(
                "Target Percentage", min_value=0.1, max_value=1000.0,
                value=10.0, step=0.1, help="Target percentage change"
            )
        with col3:
            direction = st.selectbox("Direction", ["up", "down"], help="Price movement direction")

    # ===== SCREENING BUTTON =====
    if st.button(
        "🔍 Start Screening",
        use_container_width=True,
        type="primary",
        disabled=not tokens,
    ):
        if not tokens:
            st.warning("Select at least one stock list.")
        else:
            with st.spinner(
                f"🔄 Scanning {len(tokens)} stocks "
                f"({', '.join(selected_lists)}) in parallel… "
                "(reuses cached prices if you already scanned today)"
            ):
                # Keep day cache — do NOT clear between strategies (big speed win)
                if strategy in [
                    "Strong Uptrend Scanner", "Pullback to Support", "Volume Breakout",
                    "Market Leaders", "Consolidation Breakout"
                ]:
                    screened_stocks, scan_stats = analyze_all_tokens(
                        alice, tokens, strategy,
                        exchange=st.session_state.selected_exchange
                    )
                elif strategy == "Custom Price Movement":
                    screened_stocks = analyze_all_tokens_custom(
                        alice, tokens, duration_days, target_percentage, direction,
                        exchange=st.session_state.selected_exchange
                    )
                    scan_stats = None
                else:
                    screened_stocks = analyze_all_tokens_advanced(
                        alice, tokens, strategy,
                        exchange=st.session_state.selected_exchange
                    )
                    scan_stats = None

            # Apply tier limits (Price Action Breakout already capped at top 5 bullish)
            if strategy == "Price Action Breakout":
                result_limit = 5
            else:
                tier_limits = {"Free": 10, "Premium": 50, "Pro": None}
                result_limit = tier_limits[st.session_state.user_tier]

            total_found = len(screened_stocks)
            if result_limit and total_found > result_limit:
                if strategy != "Price Action Breakout":
                    st.warning(
                        f"⚠️ Found {total_found} stocks, but {st.session_state.user_tier} tier "
                        f"shows top {result_limit}. Upgrade for full access!"
                    )
                screened_stocks = screened_stocks[:result_limit]

            if screened_stocks:
                if strategy == "Price Action Breakout":
                    st.success(
                        f"✅ Top {len(screened_stocks)} **bullish** breakout picks "
                        f"(highest confidence)"
                    )
                else:
                    st.success(f"✅ Found {len(screened_stocks)} stocks matching **{strategy}**")

                # === EDUCATIONAL COMPARISON (Premium/Pro only) ===
                if st.session_state.show_educational and st.session_state.user_tier in ["Premium", "Pro"]:
                    with st.expander("📊 Educational Analysis - Why These Stocks Qualified", expanded=True):
                        comparison = compare_stocks_educational(screened_stocks)

                        st.markdown("### 🏆 Top Picks Ranked by Technical Strength")

                        for i, stock in enumerate(comparison[:min(10, len(comparison))], 1):
                            strength = stock['strength']
                            if strength >= 75:
                                badge_class, badge_text = "strength-high", "STRONG"
                            elif strength >= 50:
                                badge_class, badge_text = "strength-medium", "MODERATE"
                            else:
                                badge_class, badge_text = "strength-low", "WEAK"

                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.markdown(f"**{i}. {stock['name']}** - ₹{stock['price']}")
                            with col2:
                                st.markdown(
                                    f'<span class="strength-badge {badge_class}">'
                                    f'{strength}/100 - {badge_text}</span>',
                                    unsafe_allow_html=True
                                )
                            with col3:
                                chart_link = generate_tradingview_link(
                                    stock['name'], st.session_state.selected_exchange
                                )
                                st.markdown(chart_link, unsafe_allow_html=True)

                            st.markdown(f"📌 **Pattern:** {stock['pattern']}")

                            if stock['why_strong']:
                                st.markdown(f"✓ **Why Strong:** {', '.join(stock['why_strong'])}")

                            if stock.get('educational_note'):
                                st.markdown(
                                    f'<div class="edu-note">💡 <strong>Learning:</strong> '
                                    f'{stock["educational_note"]}</div>',
                                    unsafe_allow_html=True
                                )

                            st.markdown("---")

                # === DAILY LESSON (Premium/Pro only) ===
                if st.session_state.show_educational and st.session_state.user_tier in ["Premium", "Pro"]:
                    with st.expander("🎓 Today's Technical Analysis Lesson"):
                        lesson = generate_daily_lesson(screened_stocks[:5], strategy)

                        st.markdown(f"## {lesson['title']}")
                        st.markdown(f"*{lesson['date']}*")

                        st.markdown("### 📚 Strategy Explanation")
                        st.markdown(lesson['strategy_explanation'])

                        st.markdown("### 🎯 Key Learnings")
                        for learning in lesson['key_learnings']:
                            st.markdown(f"- ✓ {learning}")

                        if lesson['examples']:
                            st.markdown("### 📊 Real Examples from Today's Scan")
                            for example in lesson['examples']:
                                st.markdown(f"**{example['stock']}**: {example['why_qualified']}")
                                if example['what_to_watch']:
                                    st.markdown(f"  - 👀 **Watch:** {example['what_to_watch']}")

                        st.markdown("### ❓ Quiz Questions (Test Your Understanding)")
                        for q in lesson['quiz_questions']:
                            st.markdown(f"- {q['question']}")

                # === RESULTS TABLE ===
                st.markdown("### 📋 Screening Results")

                df_results = pd.DataFrame(screened_stocks)

                # Format numeric columns safely
                for col in ["Close", "Start_Price", "Percentage_Change", "Volatility"]:
                    if col in df_results.columns:
                        df_results[col] = pd.to_numeric(df_results[col], errors="coerce").round(2)

                if "Strength" in df_results.columns:
                    df_results["Strength"] = pd.to_numeric(
                        df_results["Strength"], errors="coerce"
                    ).fillna(0).astype(int)

                if "Volume" in df_results.columns:
                    df_results["Volume"] = pd.to_numeric(df_results["Volume"], errors="coerce")

                if "Patterns" in df_results.columns:
                    df_results["Patterns"] = df_results["Patterns"].apply(
                        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
                    )

                if "Volume_Nodes" in df_results.columns:
                    df_results["Volume_Nodes"] = df_results["Volume_Nodes"].apply(
                        lambda x: ", ".join(map(str, x[:3])) if isinstance(x, list) and x else "None"
                    )

                # Sort by Strength
                if "Strength" in df_results.columns:
                    df_results = df_results.sort_values(by="Strength", ascending=False)

                # Add TradingView links
                if "Name" in df_results.columns:
                    df_results["Name"] = df_results["Name"].apply(
                        lambda x: generate_tradingview_link(x, st.session_state.selected_exchange)
                    )

                # Remove Educational_Note column from main table
                if "Educational_Note" in df_results.columns:
                    df_results = df_results.drop(columns=["Educational_Note"])

                st.markdown(df_results.to_html(escape=False, index=False), unsafe_allow_html=True)

                # Download button
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=(
                        f"{strategy.replace(' ', '_')}_"
                        f"{st.session_state.selected_exchange}_"
                        f"{datetime.datetime.now().strftime('%Y%m%d')}.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True
                )

                if st.session_state.user_tier == "Free":
                    st.info("🎓 **Upgrade to Premium** to unlock educational features, daily lessons, and see more results!")

            else:
                st.warning(f"No stocks found matching **{strategy}** criteria. Try:")
                st.markdown("- A different strategy")
                st.markdown("- A different stock list")
                st.markdown("- A different exchange (NSE/BSE)")

                if scan_stats:
                    st.markdown(
                        f"**Scan summary:** {scan_stats['tokens']} symbols scanned, "
                        f"{scan_stats['with_data']} returned price data, "
                        f"{scan_stats.get('no_data', 0)} had no data, "
                        f"{scan_stats['matched']} matched criteria, "
                        f"{scan_stats['errors']} API errors."
                    )
                    if scan_stats["errors"] > 0 and scan_stats["errors"] == scan_stats["tokens"]:
                        st.error(
                            "All API calls failed — log out and log in again with a fresh authCode. "
                            "If the problem persists, try after 5:30 PM IST when historical data is available."
                        )
                    elif scan_stats["with_data"] == 0 and scan_stats.get("no_data", 0) > 0:
                        st.error(
                            "No historical data was returned. AliceBlue daily data is often "
                            "unavailable during market hours (9:15 AM–3:30 PM IST). "
                            "Try again after 5:30 PM IST or on weekends."
                        )
                    elif scan_stats["with_data"] > 0 and scan_stats["matched"] == 0:
                        st.info(
                            "Price data loaded successfully, but no stocks met the strategy filters. "
                            "Try a broader list like NIFTY 500 or ALL STOCKS."
                        )

                st.info(
                    "Note: AliceBlue historical data is limited on weekdays during market hours "
                    "(typically available 5:30 PM–8:00 AM). If other strategies also return empty, "
                    "try again after market close."
                )

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📚 Educational Stock Screener Platform</p>
        <p style='font-size: 12px;'>This tool is for educational purposes only. Not financial advice. Always do your own research.</p>
        <p style='font-size: 12px;'>Built with ❤️ for students learning technical analysis</p>
    </div>
""", unsafe_allow_html=True)
