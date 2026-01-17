import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler


def get_historical_data(alice, token, from_date, to_date, interval="D", exchange='NSE'):
    """Fetch historical data and return as a DataFrame."""
    exchange_name = 'BSE (1)' if exchange == 'BSE' else 'NSE'
    instrument = alice.get_instrument_by_token(exchange_name, token)
    historical_data = alice.get_historical(instrument, from_date, to_date, interval)
    df = pd.DataFrame(historical_data).dropna()
    return instrument, df


def identify_candlestick_patterns(df):
    """Identify common candlestick patterns."""
    patterns = []
    
    if len(df) < 2:
        return patterns
    
    # Calculate basic candle properties
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['body'] = df['close'] - df['open']
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body_size'] = abs(df['body'])
    df['total_size'] = df['high'] - df['low']
    
    # Avoid division by zero
    df['total_size'] = df['total_size'].replace(0, np.nan)
    
    # Doji - body is less than 10% of total candle size
    if pd.notna(df['total_size'].iloc[-1]):
        doji = (df['body_size'].iloc[-1] <= 0.1 * df['total_size'].iloc[-1])
        if doji:
            patterns.append('Doji')
    
    # Hammer - long lower shadow, small body, small upper shadow
    if pd.notna(df['total_size'].iloc[-1]) and df['body_size'].iloc[-1] > 0:
        hammer = (
            (df['lower_shadow'].iloc[-1] > 2 * df['body_size'].iloc[-1]) and
            (df['upper_shadow'].iloc[-1] < df['body_size'].iloc[-1])
        )
        if hammer:
            patterns.append('Hammer')
    
    # Bullish Engulfing - FIXED LOGIC
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
    
    # Morning Star - three candle pattern
    if len(df) >= 3:
        first_bearish = df['body'].iloc[-3] < 0 and abs(df['body'].iloc[-3]) > df['total_size'].iloc[-3] * 0.3
        second_small = df['body_size'].iloc[-2] < df['total_size'].iloc[-2] * 0.3
        third_bullish = df['body'].iloc[-1] > 0 and abs(df['body'].iloc[-1]) > df['total_size'].iloc[-1] * 0.3
        
        if first_bearish and second_small and third_bullish:
            patterns.append('Morning Star')
    
    return patterns


def analyze_volume_profile(df):
    """Analyze volume profile and identify significant price levels - IMPROVED LOGIC."""
    if len(df) < 10:
        return pd.DataFrame(columns=['price_level', 'volume'])
    
    # Create price bins
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_range = price_max - price_min
    
    if price_range == 0:
        return pd.DataFrame(columns=['price_level', 'volume'])
    
    num_bins = 50
    bin_size = price_range / num_bins
    
    # Initialize volume profile
    volume_profile = pd.DataFrame({
        'price_level': np.arange(price_min, price_max, bin_size),
        'volume': 0.0
    })
    
    # FIXED: Distribute volume across the entire candle range (high to low)
    for i in range(len(df)):
        candle_low = df['low'].iloc[i]
        candle_high = df['high'].iloc[i]
        candle_volume = df['volume'].iloc[i]
        
        # Find bins that this candle touches
        low_bin = int((candle_low - price_min) / bin_size)
        high_bin = int((candle_high - price_min) / bin_size)
        
        # Ensure bins are within range
        low_bin = max(0, min(low_bin, len(volume_profile) - 1))
        high_bin = max(0, min(high_bin, len(volume_profile) - 1))
        
        # Distribute volume evenly across touched bins
        bins_touched = max(1, high_bin - low_bin + 1)
        volume_per_bin = candle_volume / bins_touched
        
        for bin_idx in range(low_bin, high_bin + 1):
            if 0 <= bin_idx < len(volume_profile):
                volume_profile.loc[bin_idx, 'volume'] += volume_per_bin
    
    # Identify high volume nodes
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
    """Analyze market structure using higher highs and lower lows - FIXED LOGIC."""
    if len(df) < 20:  # Need minimum data for structure analysis
        return "Undefined"
    
    # Find local maxima and minima
    window = 5
    local_max_indices = argrelextrema(df['high'].values, np.greater_equal, order=window)[0]
    local_min_indices = argrelextrema(df['low'].values, np.less_equal, order=window)[0]
    
    # FIXED: Check if we have enough swing points
    if len(local_max_indices) < 2 or len(local_min_indices) < 2:
        return "Undefined"
    
    # Get last swing points (at least 2 of each)
    num_swings = min(3, len(local_max_indices), len(local_min_indices))
    recent_max_indices = local_max_indices[-num_swings:]
    recent_min_indices = local_min_indices[-num_swings:]
    
    recent_max = df['high'].iloc[recent_max_indices]
    recent_min = df['low'].iloc[recent_min_indices]
    
    # Determine trend by comparing swing points
    if len(recent_max) >= 2 and len(recent_min) >= 2:
        # Check if making higher highs
        higher_highs = recent_max.iloc[-1] > recent_max.iloc[-2]
        # Check if making higher lows
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
            alice, token, datetime.now() - timedelta(days=365), datetime.now(), "D", exchange
        )
        
        if len(df) < 100:
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
        
        # Analyze candlestick patterns
        patterns = identify_candlestick_patterns(df)
        result['Patterns'] = patterns
        
        # Analyze market structure
        result['Market_Structure'] = analyze_market_structure(df)
        
        # Analyze volume profile
        volume_nodes = analyze_volume_profile(df)
        result['Volume_Nodes'] = volume_nodes['price_level'].tolist() if len(volume_nodes) > 0 else []
        
        # Calculate overall strength based on strategy - IMPROVED LOGIC
        if strategy == "Price Action Breakout":
            # Strong breakouts with volume confirmation
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            strength = 0
            if len(patterns) > 0:
                strength += len(patterns) * 2
            if pd.notna(avg_volume) and avg_volume > 0 and current_volume > avg_volume * 1.5:
                strength += 3
            
            result['Strength'] = strength
                
        elif strategy == "Volume Profile Analysis":
            # High volume nodes near current price
            current_price = df['close'].iloc[-1]
            if len(volume_nodes) > 0:
                volume_nodes_copy = volume_nodes.copy()
                volume_nodes_copy['distance'] = abs(volume_nodes_copy['price_level'] - current_price) / current_price
                nearby_nodes = volume_nodes_copy[volume_nodes_copy['distance'] < 0.02]
                result['Strength'] = len(nearby_nodes) * 3
            else:
                result['Strength'] = 0
                
        elif strategy == "Market Structure Analysis":
            # Strong trend with confirmation
            strength = 0
            if result['Market_Structure'] in ['Uptrend', 'Downtrend']:
                strength += 5
            if len(patterns) > 0:
                strength += len(patterns)
            result['Strength'] = strength
                
        elif strategy == "Multi-Factor Analysis":
            # Combine all factors
            strength = 0
            strength += len(patterns) * 2  # Candlestick patterns
            strength += min(len(result['Volume_Nodes']), 5)  # Volume nodes (capped at 5)
            strength += 5 if result['Market_Structure'] in ['Uptrend', 'Downtrend'] else 0  # Market structure
            result['Strength'] = strength
        
        # Only return if we have meaningful signals
        return result if result['Strength'] > 0 else None
        
    except Exception as e:
        print(f"Error analyzing {token}: {e}")
        return None


def analyze_all_tokens_advanced(alice, tokens, strategy, exchange='NSE'):
    """Analyze all tokens using advanced strategies in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=50) as executor:
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
                print(f"Error processing {token}: {e}")
    
    return results


def analyze_price_movement(df, duration_days, target_percentage, direction='up'):
    """
    Analyze price movement over a specified duration - FIXED LOGIC.
    
    Args:
        df: DataFrame with price data
        duration_days: Number of days to look back
        target_percentage: Target percentage change
        direction: 'up' or 'down' for price movement direction
    
    Returns:
        tuple: (percentage_change, met_criteria)
    """
    if len(df) < duration_days + 1:  # Need at least duration_days + 1 rows
        return 0, False
    
    # FIXED: Use proper indexing to get price from duration_days ago
    # iloc[-duration_days-1] gets the price duration_days trading days ago
    # But we want exactly duration_days ago, so we use -duration_days
    start_idx = -(duration_days + 1)  # +1 because we want duration_days ago, not duration_days from end
    
    # Ensure we don't go out of bounds
    if abs(start_idx) > len(df):
        start_idx = -len(df)
    
    start_price = df['close'].iloc[start_idx]
    current_price = df['close'].iloc[-1]
    
    # Calculate percentage change
    percentage_change = ((current_price - start_price) / start_price) * 100
    
    # Check if criteria is met
    if direction == 'up':
        met_criteria = percentage_change >= target_percentage
    else:  # down
        met_criteria = percentage_change <= -target_percentage
    
    return percentage_change, met_criteria


def analyze_stock_custom(alice, token, duration_days, target_percentage, direction='up', exchange='NSE'):
    """
    Analyze stock based on custom price movement criteria.
    
    Args:
        alice: AliceBlue API instance
        token: Stock token
        duration_days: Number of days to look back
        target_percentage: Target percentage change
        direction: 'up' or 'down' for price movement direction
        exchange: 'NSE' or 'BSE'
    
    Returns:
        dict: Analysis results or None if criteria not met
    """
    try:
        # Get more historical data than needed to ensure we have enough
        lookback_days = max(duration_days * 2, 365)  # At least double the duration or 1 year
        instrument, df = get_historical_data(
            alice, token, 
            datetime.now() - timedelta(days=lookback_days), 
            datetime.now(), 
            "D", 
            exchange
        )
        
        if len(df) < duration_days + 1:
            return None
        
        # Calculate price movement
        percentage_change, met_criteria = analyze_price_movement(
            df, duration_days, target_percentage, direction
        )
        
        if not met_criteria:
            return None
        
        # Additional analysis for context
        recent_volume = df['volume'].iloc[-5:].mean()
        historical_volume = df['volume'].iloc[-20:].mean()
        volume_trend = recent_volume > historical_volume if pd.notna(recent_volume) and pd.notna(historical_volume) else False
        
        # Calculate volatility (annualized)
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        result = {
            'Name': instrument.symbol,
            'Close': df['close'].iloc[-1],
            'Start_Price': df['close'].iloc[-(duration_days + 1)],
            'Percentage_Change': round(percentage_change, 2),
            'Volume_Trend': 'Increasing' if volume_trend else 'Decreasing',
            'Volatility': round(volatility, 2),
            'Duration_Days': duration_days,
            'Direction': direction.capitalize(),
            'Strength': round(abs(percentage_change) / target_percentage, 2)  # Normalized strength
        }
        
        return result
        
    except Exception as e:
        print(f"Error analyzing {token}: {e}")
        return None


def analyze_all_tokens_custom(alice, tokens, duration_days, target_percentage, direction='up', exchange='NSE'):
    """Analyze all tokens using custom criteria in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_token = {
            executor.submit(
                analyze_stock_custom, 
                alice, 
                token, 
                duration_days, 
                target_percentage, 
                direction, 
                exchange
            ): token for token in tokens
        }
        for future in as_completed(future_to_token):
            token = future_to_token[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {token}: {e}")
    
    return results
