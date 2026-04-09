import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler


def get_historical_data(alice, token, from_date, to_date, interval="D", exchange='NSE'):
    """Fetch historical data and return as a DataFrame."""
    # Normalize exchange name for get_instrument_by_token
    exchange_name = 'BSE' if exchange == 'BSE' else 'NSE'
    instrument = alice.get_instrument_by_token(exchange_name, token)
    df = alice.get_historical(instrument, from_date, to_date, interval, exchange_name)
    df = df.dropna()
    return instrument, df


def identify_candlestick_patterns(df):
    """Identify common candlestick patterns."""
    patterns = []

    if len(df) < 2:
        return patterns

    df = df.copy()
    df['body'] = df['close'] - df['open']
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body_size'] = abs(df['body'])
    df['total_size'] = df['high'] - df['low']

    # Avoid division by zero
    df['total_size'] = df['total_size'].replace(0, np.nan)

    # Doji
    if pd.notna(df['total_size'].iloc[-1]):
        if df['body_size'].iloc[-1] <= 0.1 * df['total_size'].iloc[-1]:
            patterns.append('Doji')

    # Hammer
    if pd.notna(df['total_size'].iloc[-1]) and df['body_size'].iloc[-1] > 0:
        hammer = (
            (df['lower_shadow'].iloc[-1] > 2 * df['body_size'].iloc[-1]) and
            (df['upper_shadow'].iloc[-1] < df['body_size'].iloc[-1])
        )
        if hammer:
            patterns.append('Hammer')

    # Bullish Engulfing
    if len(df) >= 2:
        prev_bearish = df['body'].iloc[-2] < 0
        curr_bullish = df['body'].iloc[-1] > 0
        curr_open_below_prev_close = df['open'].iloc[-1] < df['close'].iloc[-2]
        curr_close_above_prev_open = df['close'].iloc[-1] > df['open'].iloc[-2]

        if prev_bearish and curr_bullish and curr_open_below_prev_close and curr_close_above_prev_open:
            patterns.append('Bullish Engulfing')

    # Bearish Engulfing
    if len(df) >= 2:
        prev_bullish = df['body'].iloc[-2] > 0
        curr_bearish = df['body'].iloc[-1] < 0
        curr_open_above_prev_close = df['open'].iloc[-1] > df['close'].iloc[-2]
        curr_close_below_prev_open = df['close'].iloc[-1] < df['open'].iloc[-2]

        if prev_bullish and curr_bearish and curr_open_above_prev_close and curr_close_below_prev_open:
            patterns.append('Bearish Engulfing')

    # Morning Star
    if len(df) >= 3:
        ts = df['total_size']
        first_bearish = df['body'].iloc[-3] < 0 and abs(df['body'].iloc[-3]) > (ts.iloc[-3] or 1) * 0.3
        second_small = df['body_size'].iloc[-2] < (ts.iloc[-2] or 1) * 0.3
        third_bullish = df['body'].iloc[-1] > 0 and abs(df['body'].iloc[-1]) > (ts.iloc[-1] or 1) * 0.3

        if first_bearish and second_small and third_bullish:
            patterns.append('Morning Star')

    return patterns


def analyze_volume_profile(df):
    """Analyze volume profile and identify significant price levels."""
    if len(df) < 10:
        return pd.DataFrame(columns=['price_level', 'volume'])

    price_min = df['low'].min()
    price_max = df['high'].max()
    price_range = price_max - price_min

    if price_range == 0:
        return pd.DataFrame(columns=['price_level', 'volume'])

    num_bins = 50
    bin_size = price_range / num_bins

    volume_profile = pd.DataFrame({
        'price_level': np.arange(price_min, price_max, bin_size),
        'volume': 0.0
    })

    for i in range(len(df)):
        candle_low = df['low'].iloc[i]
        candle_high = df['high'].iloc[i]
        candle_volume = df['volume'].iloc[i]

        low_bin = int((candle_low - price_min) / bin_size)
        high_bin = int((candle_high - price_min) / bin_size)

        low_bin = max(0, min(low_bin, len(volume_profile) - 1))
        high_bin = max(0, min(high_bin, len(volume_profile) - 1))

        bins_touched = max(1, high_bin - low_bin + 1)
        volume_per_bin = candle_volume / bins_touched

        for bin_idx in range(low_bin, high_bin + 1):
            if 0 <= bin_idx < len(volume_profile):
                volume_profile.loc[bin_idx, 'volume'] += volume_per_bin

    if volume_profile['volume'].sum() > 0:
        mean_volume = volume_profile['volume'].mean()
        std_volume = volume_profile['volume'].std()

        if std_volume > 0:
            high_volume_nodes = volume_profile[volume_profile['volume'] > mean_volume + std_volume]
        else:
            high_volume_nodes = volume_profile.nlargest(5, 'volume')
    else:
        high_volume_nodes = pd.DataFrame(columns=['price_level', 'volume'])

    return high_volume_nodes


def analyze_market_structure(df):
    """Analyze market structure using higher highs and lower lows."""
    if len(df) < 20:
        return "Undefined"

    window = 5
    local_max_indices = argrelextrema(df['high'].values, np.greater_equal, order=window)[0]
    local_min_indices = argrelextrema(df['low'].values, np.less_equal, order=window)[0]

    if len(local_max_indices) < 2 or len(local_min_indices) < 2:
        return "Undefined"

    num_swings = min(3, len(local_max_indices), len(local_min_indices))
    recent_max_indices = local_max_indices[-num_swings:]
    recent_min_indices = local_min_indices[-num_swings:]

    recent_max = df['high'].iloc[recent_max_indices]
    recent_min = df['low'].iloc[recent_min_indices]

    if len(recent_max) >= 2 and len(recent_min) >= 2:
        higher_highs = recent_max.iloc[-1] > recent_max.iloc[-2]
        higher_lows = recent_min.iloc[-1] > recent_min.iloc[-2]

        if higher_highs and higher_lows:
            return "Uptrend"
        elif not higher_highs and not higher_lows:
            return "Downtrend"
        else:
            return "Sideways"

    return "Undefined"


def analyze_stock_advanced(alice, token, strategy, exchange='NSE'):
    """Analyze stock using advanced strategies."""
    try:
        instrument, df = get_historical_data(
            alice, token,
            datetime.now() - timedelta(days=365),
            datetime.now(),
            "D",
            exchange
        )

        if df is None or len(df) < 100:
            return None

        result = {
            'Name': instrument.symbol,
            'Close': df['close'].iloc[-1],
            'Volume': df['volume'].iloc[-1],
            'Patterns': [],
            'Market_Structure': '',
            'Volume_Nodes': [],
            'Strength': 0
        }

        patterns = identify_candlestick_patterns(df)
        result['Patterns'] = patterns

        result['Market_Structure'] = analyze_market_structure(df)

        volume_nodes = analyze_volume_profile(df)
        result['Volume_Nodes'] = volume_nodes['price_level'].tolist() if len(volume_nodes) > 0 else []

        if strategy == "Price Action Breakout":
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]

            strength = 0
            if len(patterns) > 0:
                strength += len(patterns) * 2
            if pd.notna(avg_volume) and avg_volume > 0 and current_volume > avg_volume * 1.5:
                strength += 3

            result['Strength'] = strength

        elif strategy == "Volume Profile Analysis":
            current_price = df['close'].iloc[-1]
            if len(volume_nodes) > 0:
                volume_nodes_copy = volume_nodes.copy()
                volume_nodes_copy['distance'] = abs(volume_nodes_copy['price_level'] - current_price) / current_price
                nearby_nodes = volume_nodes_copy[volume_nodes_copy['distance'] < 0.02]
                result['Strength'] = len(nearby_nodes) * 3
            else:
                result['Strength'] = 0

        elif strategy == "Market Structure Analysis":
            strength = 0
            if result['Market_Structure'] in ['Uptrend', 'Downtrend']:
                strength += 5
            if len(patterns) > 0:
                strength += len(patterns)
            result['Strength'] = strength

        elif strategy == "Multi-Factor Analysis":
            strength = 0
            strength += len(patterns) * 2
            strength += min(len(result['Volume_Nodes']), 5)
            strength += 5 if result['Market_Structure'] in ['Uptrend', 'Downtrend'] else 0
            result['Strength'] = strength

        return result if result['Strength'] > 0 else None

    except Exception as e:
        print(f"Error analyzing token {token}: {e}")
        return None


def analyze_all_tokens_advanced(alice, tokens, strategy, exchange='NSE'):
    """Analyze all tokens using advanced strategies in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_token = {
            executor.submit(analyze_stock_advanced, alice, token, strategy, exchange): token
            for token in tokens
        }
        for future in as_completed(future_to_token):
            token = future_to_token[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing token {token}: {e}")

    return results


def analyze_price_movement(df, duration_days, target_percentage, direction='up'):
    """
    Analyze price movement over a specified duration.

    Args:
        df: DataFrame with price data
        duration_days: Number of trading days to look back
        target_percentage: Target percentage change
        direction: 'up' or 'down'

    Returns:
        tuple: (percentage_change, met_criteria)
    """
    if len(df) < duration_days + 1:
        return 0, False

    # Get start price duration_days trading days ago
    start_idx = -(duration_days + 1)
    if abs(start_idx) > len(df):
        start_idx = -len(df)

    start_price = df['close'].iloc[start_idx]
    current_price = df['close'].iloc[-1]

    if start_price == 0:
        return 0, False

    percentage_change = ((current_price - start_price) / start_price) * 100

    if direction == 'up':
        met_criteria = percentage_change >= target_percentage
    else:
        met_criteria = percentage_change <= -target_percentage

    return percentage_change, met_criteria


def analyze_stock_custom(alice, token, duration_days, target_percentage, direction='up', exchange='NSE'):
    """
    Analyze stock based on custom price movement criteria.
    """
    try:
        lookback_days = max(duration_days * 2, 365)
        instrument, df = get_historical_data(
            alice, token,
            datetime.now() - timedelta(days=lookback_days),
            datetime.now(),
            "D",
            exchange
        )

        if df is None or len(df) < duration_days + 1:
            return None

        percentage_change, met_criteria = analyze_price_movement(
            df, duration_days, target_percentage, direction
        )

        if not met_criteria:
            return None

        recent_volume = df['volume'].iloc[-5:].mean()
        historical_volume = df['volume'].iloc[-20:].mean()
        volume_trend = (
            recent_volume > historical_volume
            if pd.notna(recent_volume) and pd.notna(historical_volume)
            else False
        )

        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100

        result = {
            'Name': instrument.symbol,
            'Close': df['close'].iloc[-1],
            'Start_Price': df['close'].iloc[-(duration_days + 1)],
            'Percentage_Change': round(percentage_change, 2),
            'Volume_Trend': 'Increasing' if volume_trend else 'Decreasing',
            'Volatility': round(volatility, 2),
            'Duration_Days': duration_days,
            'Direction': direction.capitalize(),
            'Strength': round(abs(percentage_change) / target_percentage, 2)
        }

        return result

    except Exception as e:
        print(f"Error analyzing token {token}: {e}")
        return None


def analyze_all_tokens_custom(alice, tokens, duration_days, target_percentage, direction='up', exchange='NSE'):
    """Analyze all tokens using custom criteria in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_token = {
            executor.submit(
                analyze_stock_custom,
                alice, token, duration_days, target_percentage, direction, exchange
            ): token for token in tokens
        }
        for future in as_completed(future_to_token):
            token = future_to_token[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing token {token}: {e}")

    return results
