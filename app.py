import streamlit as st
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from functools import partial
from alice_client import initialize_alice, save_credentials, load_credentials
from advanced_analysis import (
    analyze_all_tokens_advanced,
    analyze_all_tokens_custom
)
from stock_analysis import analyze_all_tokens  # NEW: Import enhanced strategies
from educational_scorer import (  # NEW: Import educational features
    compare_stocks_educational,
    generate_daily_lesson
)
from stock_lists import STOCK_LISTS
from utils import generate_tradingview_link
import base64

st.set_page_config(
    page_title="Learning Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

def load_image_base64(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

logo_base64 = load_image_base64("assets/harion.jpg")

st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:14px;">
        <img src="data:image/jpeg;base64,{logo_base64}" width="65">
        <div>
            <h1 style="margin-bottom:0;">Harion Research – Learning Lab</h1>
            <p style="margin-top:0; color:#666;">
                Learn Technical Analysis While Screening NSE & BSE Stocks
            </p>
        </div>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)



# Set theme configuration using teal (#7FE2D3)
st.markdown("""
    <style>
        .button-box {
            margin-bottom: 1rem;
        }

        /* NSE Selected - Aqua */
        .nse-btn button {
            background-color: #A7D6D6 !important;
            color: black !important;
        }
        .nse-btn button:hover {
            background-color: #94CACA !important;
        }

        /* BSE Selected - Peach */
        .bse-btn button {
            background-color: #F9D5C2 !important;
            color: black !important;
        }
        .bse-btn button:hover {
            background-color: #F2C0AC !important;
        }

        /* Inactive Button - Blue Gray */
        .default-btn button {
            background-color: #DDEBF1 !important;
            color: black !important;
        }
        .default-btn button:hover {
            background-color: #CFE0E8 !important;
        }
        
        /* Strength Score Badge */
        .strength-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 14px;
        }
        .strength-high {
            background-color: #28a745;
            color: white;
        }
        .strength-medium {
            background-color: #ffc107;
            color: black;
        }
        .strength-low {
            background-color: #dc3545;
            color: white;
        }
        
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

# Initialize session state
if 'selected_exchange' not in st.session_state:
    st.session_state.selected_exchange = 'NSE'
if 'show_educational' not in st.session_state:
    st.session_state.show_educational = True  # Toggle for educational features
if 'user_tier' not in st.session_state:
    st.session_state.user_tier = 'Free'  # Free, Premium, or Pro

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

# Header
st.markdown("""
    <div class="header">
        <h1>📈 Learning Lab </h1>
        <p>Learn Technical Analysis While Screening NSE & BSE Stocks</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Authentication")
    user_id, api_key = load_credentials()

    if not user_id or not api_key:
        st.markdown("Enter AliceBlue API Credentials")
        new_user_id = st.text_input("User ID", type="password")
        new_api_key = st.text_input("API Key", type="password")
        if st.button("Login", use_container_width=True):
            save_credentials(new_user_id, new_api_key)
            st.success("Credentials saved!")
            st.rerun()
    
    st.markdown("---")
    
    # NEW: User Tier Selection (replace with actual auth later)
    st.markdown("### Account Type")
    st.session_state.user_tier = st.selectbox(
        "Select Tier",
        ["Free", "Premium", "Pro"],
        help="Different tiers unlock different features"
    )
    
    # Show tier benefits
    tier_info = {
        "Free": "• 2 strategies\n• Top 10 results\n• Basic view",
        "Premium": "• All strategies\n• Top 50 results\n• Educational features\n• Daily lessons",
        "Pro": "• Unlimited results\n• All features\n• Priority support\n• API access"
    }
    st.info(tier_info[st.session_state.user_tier])
    
    st.markdown("---")
    
    # Educational toggle
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

# Exchange Toggle
st.markdown('<div class="exchange-toggle">', unsafe_allow_html=True)

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

st.markdown('</div>', unsafe_allow_html=True)

# Initialize Alice
try:
    alice = initialize_alice()
except Exception as e:
    st.error(f"Failed to initialize AliceBlue API: {e}")
    alice = None

# Strategy selection based on tier
col1, col2 = st.columns(2)

with col1:
    available_lists = get_stock_lists_for_exchange(st.session_state.selected_exchange)
    selected_list = st.selectbox(
        "Select Stock List",
        list(available_lists.keys()),
        help="Choose a list of stocks to analyze"
    )

with col2:
    # Define available strategies based on tier
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

# Strategy descriptions with educational content
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
            - Near 52-week highs
            - Outperforming the market
            - Strong trend (ADX > 25)
            - Consistent volume
            
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
            - Identifies strong breakouts with volume confirmation
            - Analyzes candlestick patterns and price action
            - Considers multiple timeframe confirmation
            - Includes volume profile analysis
        """,
        "learning": "Classic price action strategy focusing on pattern recognition."
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

# Display strategy details
if strategy in strategy_descriptions:
    st.markdown("### Strategy Details")
    st.markdown(strategy_descriptions[strategy]["description"])
    
    if st.session_state.show_educational:
        st.info(f"💡 **Learning Point:** {strategy_descriptions[strategy]['learning']}")

# Custom Price Movement inputs
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
        direction = st.selectbox(
            "Direction", ["up", "down"], help="Price movement direction"
        )

# Screening button
if st.button("🔍 Start Screening", use_container_width=True, type="primary"):
    tokens = available_lists.get(selected_list, [])
    if not tokens:
        st.warning(f"No stocks found for {selected_list}.")
    else:
        with st.spinner("🔄 Analyzing stocks... This may take a moment."):
            # Use appropriate analysis function based on strategy
            if strategy in ["Strong Uptrend Scanner", "Pullback to Support", "Volume Breakout", 
                           "Market Leaders", "Consolidation Breakout"]:
                # Use NEW enhanced strategies
                screened_stocks = analyze_all_tokens(
                    alice, tokens, strategy,
                    exchange=st.session_state.selected_exchange
                )
            elif strategy == "Custom Price Movement":
                screened_stocks = analyze_all_tokens_custom(
                    alice, tokens, duration_days, target_percentage, direction,
                    exchange=st.session_state.selected_exchange
                )
            else:
                # Use original advanced strategies
                screened_stocks = analyze_all_tokens_advanced(
                    alice, tokens, strategy,
                    exchange=st.session_state.selected_exchange
                )
        
        # Apply tier limits
        tier_limits = {
            "Free": 10,
            "Premium": 50,
            "Pro": None  # Unlimited
        }
        
        result_limit = tier_limits[st.session_state.user_tier]
        total_found = len(screened_stocks)
        
        if result_limit and total_found > result_limit:
            st.warning(f"⚠️ Found {total_found} stocks, but {st.session_state.user_tier} tier shows top {result_limit}. Upgrade for full access!")
            screened_stocks = screened_stocks[:result_limit]
        
        if screened_stocks:
            st.success(f"✅ Found {len(screened_stocks)} stocks matching **{strategy}**")
            
            # === EDUCATIONAL COMPARISON (Premium/Pro only) ===
            if st.session_state.show_educational and st.session_state.user_tier in ["Premium", "Pro"]:
                with st.expander("📊 Educational Analysis - Why These Stocks Qualified", expanded=True):
                    comparison = compare_stocks_educational(screened_stocks)
                    
                    st.markdown("### 🏆 Top Picks Ranked by Technical Strength")
                    
                    for i, stock in enumerate(comparison[:min(10, len(comparison))], 1):
                        # Create strength badge
                        strength = stock['strength']
                        if strength >= 75:
                            badge_class = "strength-high"
                            badge_text = "STRONG"
                        elif strength >= 50:
                            badge_class = "strength-medium"
                            badge_text = "MODERATE"
                        else:
                            badge_class = "strength-low"
                            badge_text = "WEAK"
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{i}. {stock['name']}** - ₹{stock['price']}")
                        with col2:
                            st.markdown(f'<span class="strength-badge {badge_class}">{strength}/100 - {badge_text}</span>', unsafe_allow_html=True)
                        
                        st.markdown(f"📌 **Pattern:** {stock['pattern']}")
                        
                        if stock['why_strong']:
                            st.markdown(f"✓ **Why Strong:** {', '.join(stock['why_strong'])}")
                        
                        if stock.get('educational_note'):
                            st.markdown(f'<div class="edu-note">💡 <strong>Learning:</strong> {stock["educational_note"]}</div>', unsafe_allow_html=True)
                        
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
            
            # Clean and format data
            df = pd.DataFrame(screened_stocks)
            
            # Format numeric columns
            if "Close" in df.columns:
                df["Close"] = df["Close"].astype(float).round(2)
            if "Strength" in df.columns:
                df["Strength"] = df["Strength"].astype(int)
            
            # Format strategy-specific columns
            if strategy == "Custom Price Movement":
                if "Start_Price" in df.columns:
                    df["Start_Price"] = df["Start_Price"].astype(float).round(2)
                if "Percentage_Change" in df.columns:
                    df["Percentage_Change"] = df["Percentage_Change"].astype(float).round(2)
                if "Volatility" in df.columns:
                    df["Volatility"] = df["Volatility"].astype(float).round(2)
            else:
                if "Volume" in df.columns:
                    df["Volume"] = df["Volume"].astype(float)
                if "Patterns" in df.columns:
                    df["Patterns"] = df["Patterns"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
                if "Volume_Nodes" in df.columns:
                    df["Volume_Nodes"] = df["Volume_Nodes"].apply(
                        lambda x: ", ".join(map(str, x[:3])) if isinstance(x, list) and x else "None"
                    )
            
            # Sort by strength
            df = df.sort_values(by="Strength", ascending=False)
            
            # Add TradingView links
            if "Name" in df.columns:
                df["Name"] = df["Name"].apply(
                    lambda x: generate_tradingview_link(x, st.session_state.selected_exchange)
                )
            
            # Remove Educational_Note column from main table (shown in expander above)
            if "Educational_Note" in df.columns:
                df = df.drop(columns=["Educational_Note"])
            
            # Display table
            st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            # === DOWNLOAD OPTION ===
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"{strategy.replace(' ', '_')}_{st.session_state.selected_exchange}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Upgrade prompt for free users
            if st.session_state.user_tier == "Free":
                st.info("🎓 **Upgrade to Premium** to unlock educational features, daily lessons, and see more results!")
        
        else:
            st.warning(f"No stocks found matching **{strategy}** criteria. Try:")
            st.markdown("- A different strategy")
            st.markdown("- A different stock list")
            st.markdown("- A different exchange (NSE/BSE)")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📚 Educational Stock Screener Platform</p>
        <p style='font-size: 12px;'>This tool is for educational purposes only. Not financial advice. Always do your own research.</p>
        <p style='font-size: 12px;'>Built with ❤️ for students learning technical analysis</p>
    </div>
""", unsafe_allow_html=True)
