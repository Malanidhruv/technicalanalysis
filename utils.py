import pandas as pd
import streamlit as st
import json


def generate_tradingview_link(stock_name, exchange='NSE'):
    """Generate a TradingView link for a given stock."""
    exchange_prefix = 'NSE' if exchange == 'NSE' else 'BSE'
    clean_name = str(stock_name).strip()
    return (
        f'<a href="https://in.tradingview.com/chart?symbol={exchange_prefix}%3A{clean_name}" '
        f'target="_blank">{clean_name}</a>'
    )


def print_stocks_up(stocks, exchange='NSE'):
    """Prints the stocks that gained in descending order with TradingView links."""
    for stock in stocks:
        stock['Change (%)'] = float(stock.get('Change (%)', 0))

    stocks_sorted = sorted(stocks, key=lambda x: -x['Change (%)'])

    print("\nStocks going up:")
    print(f"{'Name':<20} {'Token':<10} {'Close':<10} {'Change (%)':<10}")
    print('-' * 50)

    for stock in stocks_sorted:
        link = generate_tradingview_link(stock['Name'], exchange)
        print(
            f"{stock['Name']:<20} {stock.get('Token', 'N/A'):<10} "
            f"{stock['Close']:<10.2f} {stock['Change (%)']:<10.2f}  {link}"
        )

    print('-' * 50)


def print_stocks_down(stocks, exchange='NSE'):
    """Prints the stocks that are falling in ascending order with TradingView links."""
    for stock in stocks:
        stock['Change (%)'] = float(stock.get('Change (%)', 0))

    stocks_sorted = sorted(stocks, key=lambda x: x['Change (%)'])

    print("\nStocks going down:")
    print(f"{'Name':<20} {'Token':<10} {'Close':<10} {'Change (%)':<10}")
    print('-' * 50)

    for stock in stocks_sorted:
        link = generate_tradingview_link(stock['Name'], exchange)
        print(
            f"{stock['Name']:<20} {stock.get('Token', 'N/A'):<10} "
            f"{stock['Close']:<10.2f} {stock['Change (%)']:<10.2f}  {link}"
        )

    print('-' * 50)


def display_buy_candidates(signals, exchange='NSE'):
    """Displays the top 10 buy candidates in a Streamlit app with clickable links."""
    st.subheader("🚀 Top 10 Buy Candidates (Sorted by Strength)")

    if not signals:
        st.warning("No buy candidates found.")
        return

    for signal in signals:
        signal['Strength'] = float(signal.get('Strength', 0))
        signal['Distance_pct'] = float(signal.get('Distance_pct', 0))

    sorted_signals = sorted(signals, key=lambda x: (-x['Strength'], x['Distance_pct']))
    top_candidates = sorted_signals[:10]

    df = pd.DataFrame(top_candidates)
    df['Name'] = df['Name'].apply(lambda x: generate_tradingview_link(x, exchange))

    cols = [c for c in ['Name', 'Close', 'Support', 'Strength', 'Distance_pct', 'RSI', 'Trend'] if c in df.columns]
    df = df[cols]

    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)


def display_sell_candidates(signals, exchange='NSE'):
    """Displays the top 10 sell candidates in a Streamlit app with clickable links."""
    st.subheader("🔻 Top 10 Sell Candidates (Sorted by Strength)")

    if not signals:
        st.warning("No sell candidates found.")
        return

    for signal in signals:
        signal['Strength'] = float(signal.get('Strength', 0))
        signal['Distance_pct'] = float(signal.get('Distance_pct', 0))

    sorted_signals = sorted(signals, key=lambda x: (-x['Strength'], x['Distance_pct']))
    top_candidates = sorted_signals[:10]

    df = pd.DataFrame(top_candidates)
    df['Name'] = df['Name'].apply(lambda x: generate_tradingview_link(x, exchange))

    cols = [c for c in ['Name', 'Close', 'Resistance', 'Strength', 'Distance_pct', 'RSI', 'Trend'] if c in df.columns]
    df = df[cols]

    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
