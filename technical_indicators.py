"""
Technical Indicators Library
Professional-grade technical analysis indicators for educational stock screening
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


def calculate_ema(prices, period):
    """Calculate Exponential Moving Average."""
    return prices.ewm(span=period, adjust=False).mean()


def calculate_sma(prices, period):
    """Calculate Simple Moving Average."""
    return prices.rolling(window=period).mean()


def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures momentum and identifies overbought/oversold conditions.
    - RSI > 70: Overbought (potentially overvalued)
    - RSI < 30: Oversold (potentially undervalued)
    - RSI 40-60: Neutral zone
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Fill NaN with neutral value


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD shows trend direction and momentum:
    - MACD > Signal: Bullish momentum
    - MACD < Signal: Bearish momentum
    - MACD crossing above Signal: Buy signal
    - MACD crossing below Signal: Sell signal
    
    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.
    
    Shows volatility and potential reversal zones:
    - Price at upper band: Potentially overbought
    - Price at lower band: Potentially oversold
    - Bands squeezing: Low volatility, potential breakout coming
    - Bands expanding: High volatility
    
    Returns: (upper_band, middle_band, lower_band)
    """
    middle_band = calculate_sma(prices, period)
    std = prices.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR).
    
    ATR measures volatility (not direction):
    - High ATR: High volatility
    - Low ATR: Low volatility
    - Used for position sizing and stop-loss placement
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_adx(df, period=14):
    """
    Calculate Average Directional Index (ADX).
    
    ADX measures trend strength (not direction):
    - ADX > 25: Strong trend
    - ADX < 20: Weak trend or ranging market
    - ADX > 50: Very strong trend
    
    Higher ADX = stronger trend (regardless of direction)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate +DM and -DM
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate ATR
    atr = calculate_atr(df, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx.fillna(0)


def calculate_stochastic(df, period=14, smooth_k=3, smooth_d=3):
    """
    Calculate Stochastic Oscillator.
    
    Shows momentum and overbought/oversold conditions:
    - Stochastic > 80: Overbought
    - Stochastic < 20: Oversold
    - %K crossing above %D: Bullish signal
    - %K crossing below %D: Bearish signal
    
    Returns: (percent_k, percent_d)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate %K
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    percent_k = percent_k.rolling(window=smooth_k).mean()
    
    # Calculate %D (signal line)
    percent_d = percent_k.rolling(window=smooth_d).mean()
    
    return percent_k.fillna(50), percent_d.fillna(50)


def find_support_resistance(df, window_size=20, min_touches=2):
    """
    Find support and resistance levels.
    
    Support: Price level where buying pressure prevents further decline
    Resistance: Price level where selling pressure prevents further rise
    
    Returns: (support_levels, resistance_levels)
    """
    close_prices = df['close'].values
    
    # Adjust window size based on data length
    window = max(int(len(df) * 0.05), 5)
    window = min(window, window_size)
    
    # Find local minima (support)
    local_min_indices = argrelextrema(close_prices, np.less_equal, order=window)[0]
    
    # Find local maxima (resistance)
    local_max_indices = argrelextrema(close_prices, np.greater_equal, order=window)[0]
    
    # Filter for recent levels (last 6 months)
    recent_threshold = max(0, len(df) - 126)
    
    support_levels = []
    for idx in local_min_indices:
        if idx >= recent_threshold:
            support_levels.append({
                'price': close_prices[idx],
                'index': idx,
                'strength': 1
            })
    
    resistance_levels = []
    for idx in local_max_indices:
        if idx >= recent_threshold:
            resistance_levels.append({
                'price': close_prices[idx],
                'index': idx,
                'strength': 1
            })
    
    # Cluster nearby levels (within 2% of each other)
    support_levels = cluster_levels(support_levels)
    resistance_levels = cluster_levels(resistance_levels)
    
    return support_levels, resistance_levels


def cluster_levels(levels, threshold=0.02):
    """Cluster nearby price levels together."""
    if not levels:
        return []
    
    clustered = []
    sorted_levels = sorted(levels, key=lambda x: x['price'])
    
    current_cluster = [sorted_levels[0]]
    
    for level in sorted_levels[1:]:
        # Check if level is within threshold of current cluster
        cluster_avg = np.mean([l['price'] for l in current_cluster])
        
        if abs(level['price'] - cluster_avg) / cluster_avg <= threshold:
            current_cluster.append(level)
        else:
            # Finalize current cluster
            clustered.append({
                'price': np.mean([l['price'] for l in current_cluster]),
                'strength': len(current_cluster)
            })
            current_cluster = [level]
    
    # Add last cluster
    if current_cluster:
        clustered.append({
            'price': np.mean([l['price'] for l in current_cluster]),
            'strength': len(current_cluster)
        })
    
    return clustered


def calculate_relative_strength(stock_prices, index_prices):
    """
    Calculate relative strength vs index.
    
    RS > 1: Stock outperforming index
    RS < 1: Stock underperforming index
    
    Used to find market leaders and laggards.
    """
    # Normalize both to start at 100
    stock_normalized = (stock_prices / stock_prices.iloc[0]) * 100
    index_normalized = (index_prices / index_prices.iloc[0]) * 100
    
    relative_strength = stock_normalized / index_normalized
    
    return relative_strength


def detect_trend(df, ema_short=50, ema_long=200):
    """
    Detect overall trend using EMAs.
    
    Returns: 'Strong Uptrend', 'Uptrend', 'Sideways', 'Downtrend', 'Strong Downtrend'
    """
    if len(df) < ema_long:
        return 'Insufficient Data'
    
    current_price = df['close'].iloc[-1]
    ema_short_val = df['close'].ewm(span=ema_short, adjust=False).mean().iloc[-1]
    ema_long_val = df['close'].ewm(span=ema_long, adjust=False).mean().iloc[-1]
    
    # Check EMA alignment
    price_above_short = current_price > ema_short_val
    price_above_long = current_price > ema_long_val
    short_above_long = ema_short_val > ema_long_val
    
    # Check EMA slope
    ema_short_slope = (ema_short_val - df['close'].ewm(span=ema_short, adjust=False).mean().iloc[-10]) / ema_short_val * 100
    
    if price_above_short and price_above_long and short_above_long:
        if ema_short_slope > 2:
            return 'Strong Uptrend'
        return 'Uptrend'
    elif not price_above_short and not price_above_long and not short_above_long:
        if ema_short_slope < -2:
            return 'Strong Downtrend'
        return 'Downtrend'
    else:
        return 'Sideways'


def calculate_volume_analysis(df):
    """
    Analyze volume patterns.
    
    Returns volume metrics and signals.
    """
    avg_volume_20 = df['volume'].rolling(20).mean().iloc[-1]
    avg_volume_50 = df['volume'].rolling(50).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]
    
    volume_ratio_20 = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
    volume_ratio_50 = current_volume / avg_volume_50 if avg_volume_50 > 0 else 1
    
    # Volume trend
    recent_avg = df['volume'].iloc[-5:].mean()
    historical_avg = df['volume'].iloc[-20:].mean()
    volume_trend = 'Increasing' if recent_avg > historical_avg else 'Decreasing'
    
    return {
        'current': current_volume,
        'avg_20': avg_volume_20,
        'avg_50': avg_volume_50,
        'ratio_20': volume_ratio_20,
        'ratio_50': volume_ratio_50,
        'trend': volume_trend,
        'surge': volume_ratio_20 > 1.5  # Volume surge indicator
    }


def calculate_momentum_score(df):
    """
    Calculate overall momentum score (0-100).
    
    Combines multiple momentum indicators into a single score.
    """
    score = 0
    max_score = 0
    
    # RSI contribution (0-25 points)
    rsi = calculate_rsi(df['close']).iloc[-1]
    if 40 <= rsi <= 60:
        score += 15  # Neutral is good
    elif 60 < rsi <= 70:
        score += 25  # Bullish momentum
    elif 30 <= rsi < 40:
        score += 10  # Slightly oversold
    max_score += 25
    
    # MACD contribution (0-25 points)
    macd, signal, hist = calculate_macd(df['close'])
    if hist.iloc[-1] > 0:
        score += 15
        if hist.iloc[-1] > hist.iloc[-2]:  # Increasing histogram
            score += 10
    max_score += 25
    
    # Trend contribution (0-25 points)
    trend = detect_trend(df)
    if trend == 'Strong Uptrend':
        score += 25
    elif trend == 'Uptrend':
        score += 20
    elif trend == 'Sideways':
        score += 10
    max_score += 25
    
    # Volume contribution (0-25 points)
    vol_analysis = calculate_volume_analysis(df)
    if vol_analysis['surge']:
        score += 15
    if vol_analysis['trend'] == 'Increasing':
        score += 10
    max_score += 25
    
    return int((score / max_score) * 100)


def get_indicator_signals(df):
    """
    Get buy/sell/neutral signals from all indicators.
    
    Returns a dictionary with signal for each indicator.
    """
    signals = {}
    
    # RSI Signal
    rsi = calculate_rsi(df['close']).iloc[-1]
    if rsi < 30:
        signals['RSI'] = 'Oversold (Bullish)'
    elif rsi > 70:
        signals['RSI'] = 'Overbought (Bearish)'
    elif 40 <= rsi <= 60:
        signals['RSI'] = 'Neutral'
    else:
        signals['RSI'] = 'Bullish' if rsi > 50 else 'Bearish'
    
    # MACD Signal
    macd, signal_line, hist = calculate_macd(df['close'])
    if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]:
        signals['MACD'] = 'Strong Bullish'
    elif hist.iloc[-1] > 0:
        signals['MACD'] = 'Bullish'
    elif hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]:
        signals['MACD'] = 'Strong Bearish'
    else:
        signals['MACD'] = 'Bearish'
    
    # Trend Signal
    signals['Trend'] = detect_trend(df)
    
    # Volume Signal
    vol = calculate_volume_analysis(df)
    if vol['surge'] and vol['trend'] == 'Increasing':
        signals['Volume'] = 'Strong Surge'
    elif vol['surge']:
        signals['Volume'] = 'Surge'
    elif vol['trend'] == 'Increasing':
        signals['Volume'] = 'Increasing'
    else:
        signals['Volume'] = 'Decreasing'
    
    return signals
