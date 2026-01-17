"""
Enhanced Stock Analysis with Multiple Professional Strategies
Educational screener for teaching technical analysis
"""

import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.signal import argrelextrema
from alice_client import get_cached_historical_data
from technical_indicators import (
    calculate_ema, calculate_rsi, calculate_macd, calculate_adx,
    find_support_resistance, calculate_volume_analysis, detect_trend,
    calculate_momentum_score, get_indicator_signals
)


# ============================================================================
# STRATEGY 1: STRONG UPTREND SCANNER (Your Core Strategy - Enhanced)
# ============================================================================

def analyze_strong_uptrend(df, instrument):
    """
    Identify stocks in strong uptrends with all EMAs aligned.
    
    EDUCATIONAL NOTES:
    - This strategy finds stocks with strong momentum
    - All timeframes (short, medium, long) are bullish
    - Best used in bull markets
    - Look for entry on minor pullbacks
    
    Criteria:
    - Price > 50 EMA > 200 EMA (bullish alignment)
    - Making higher highs and higher lows
    - RSI between 50-70 (momentum without overbought)
    - Volume above average
    """
    try:
        if len(df) < 200:
            return None
        
        # Calculate indicators
        df['9_EMA'] = calculate_ema(df['close'], 9)
        df['21_EMA'] = calculate_ema(df['close'], 21)
        df['50_EMA'] = calculate_ema(df['close'], 50)
        df['200_EMA'] = calculate_ema(df['close'], 200)
        df['RSI'] = calculate_rsi(df['close'])
        
        current_price = df['close'].iloc[-1]
        ema_9 = df['9_EMA'].iloc[-1]
        ema_21 = df['21_EMA'].iloc[-1]
        ema_50 = df['50_EMA'].iloc[-1]
        ema_200 = df['200_EMA'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        
        # Check EMA alignment (perfect order)
        ema_aligned = current_price > ema_9 > ema_21 > ema_50 > ema_200
        
        # Check for higher highs and higher lows
        recent_highs = df['high'].iloc[-20:].values
        recent_lows = df['low'].iloc[-20:].values
        
        higher_highs = recent_highs[-1] >= np.max(recent_highs[:-5])
        higher_lows = recent_lows[-1] >= np.min(recent_lows[-10:-5])
        
        # Volume confirmation
        vol_analysis = calculate_volume_analysis(df)
        volume_ok = vol_analysis['ratio_20'] > 0.8  # At least 80% of average
        
        # RSI in bullish zone
        rsi_ok = 50 <= rsi <= 75
        
        # Calculate strength score
        strength = 0
        if ema_aligned:
            strength += 40
        if higher_highs and higher_lows:
            strength += 30
        if rsi_ok:
            strength += 20
        if volume_ok:
            strength += 10
        
        if strength >= 70:  # Need at least 70/100 to qualify
            # Calculate distance from 200 EMA (trend strength)
            distance_from_200ema = ((current_price - ema_200) / ema_200) * 100
            
            return {
                'Name': instrument.symbol,
                'Close': round(current_price, 2),
                'Strength': strength,
                'RSI': round(rsi, 1),
                '50_EMA': round(ema_50, 2),
                '200_EMA': round(ema_200, 2),
                'Distance_200EMA': round(distance_from_200ema, 1),
                'Volume_Trend': vol_analysis['trend'],
                'Pattern': 'Strong Uptrend',
                'Educational_Note': f"All EMAs aligned bullishly. Price is {distance_from_200ema:.1f}% above 200 EMA showing strong trend."
            }
        
        return None
        
    except Exception as e:
        print(f"Error in uptrend analysis: {e}")
        return None


# ============================================================================
# STRATEGY 2: PULLBACK TO SUPPORT SCANNER
# ============================================================================

def analyze_pullback_entry(df, instrument):
    """
    Find stocks in uptrend pulling back to support levels.
    
    EDUCATIONAL NOTES:
    - This finds healthy pullbacks in strong trends
    - Best risk-reward entries (buy the dip in uptrend)
    - Wait for support to hold before entering
    - Lower risk than buying at highs
    
    Criteria:
    - Overall uptrend (price above 200 EMA)
    - Pulled back to support (EMA or previous low)
    - RSI showing oversold on pullback (< 40)
    - Volume decreasing on pullback (healthy)
    """
    try:
        if len(df) < 200:
            return None
        
        # Calculate indicators
        df['50_EMA'] = calculate_ema(df['close'], 50)
        df['200_EMA'] = calculate_ema(df['close'], 200)
        df['RSI'] = calculate_rsi(df['close'])
        
        current_price = df['close'].iloc[-1]
        ema_50 = df['50_EMA'].iloc[-1]
        ema_200 = df['200_EMA'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        
        # Must be in overall uptrend
        in_uptrend = current_price > ema_200 and ema_50 > ema_200
        
        if not in_uptrend:
            return None
        
        # Find support levels
        support_levels, _ = find_support_resistance(df)
        
        # Check if price is near support (within 3%)
        near_support = False
        closest_support = None
        
        for support in support_levels:
            distance = abs(current_price - support['price']) / current_price
            if distance <= 0.03:  # Within 3%
                near_support = True
                closest_support = support['price']
                break
        
        # Check if near 50 EMA support
        near_ema_support = abs(current_price - ema_50) / current_price <= 0.02
        
        # RSI showing pullback
        rsi_pullback = 30 <= rsi <= 45
        
        # Volume analysis
        vol_analysis = calculate_volume_analysis(df)
        volume_decreasing = vol_analysis['trend'] == 'Decreasing'
        
        # Recent price action (pullback from recent high)
        recent_high = df['high'].iloc[-10:].max()
        pullback_depth = ((recent_high - current_price) / recent_high) * 100
        healthy_pullback = 3 <= pullback_depth <= 15
        
        strength = 0
        if near_support or near_ema_support:
            strength += 35
        if rsi_pullback:
            strength += 30
        if volume_decreasing:
            strength += 20
        if healthy_pullback:
            strength += 15
        
        if strength >= 60:
            return {
                'Name': instrument.symbol,
                'Close': round(current_price, 2),
                'Strength': strength,
                'RSI': round(rsi, 1),
                'Support': round(closest_support if closest_support else ema_50, 2),
                'Pullback_Depth': round(pullback_depth, 1),
                'Volume_Trend': vol_analysis['trend'],
                'Pattern': 'Pullback to Support',
                'Educational_Note': f"Uptrend stock pulling back {pullback_depth:.1f}% to support. RSI oversold at {rsi:.0f}. Good risk-reward entry zone."
            }
        
        return None
        
    except Exception as e:
        print(f"Error in pullback analysis: {e}")
        return None


# ============================================================================
# STRATEGY 3: VOLUME BREAKOUT SCANNER
# ============================================================================

def analyze_volume_breakout(df, instrument):
    """
    Find stocks breaking out with strong volume.
    
    EDUCATIONAL NOTES:
    - Volume confirms breakout (not false breakout)
    - High volume = institutional participation
    - Best when breaking resistance or consolidation
    - Can signal start of new trend
    
    Criteria:
    - Price breaking above resistance
    - Volume 2x+ above average
    - MACD turning positive
    - Not already extended (RSI < 80)
    """
    try:
        if len(df) < 100:
            return None
        
        # Calculate indicators
        df['RSI'] = calculate_rsi(df['close'])
        macd, signal, hist = calculate_macd(df['close'])
        
        current_price = df['close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        
        # Find resistance levels
        _, resistance_levels = find_support_resistance(df)
        
        # Check if breaking resistance
        breaking_resistance = False
        broken_level = None
        
        for resistance in resistance_levels:
            # Price broke above resistance in last 5 days
            if current_price > resistance['price'] * 1.01:  # 1% above
                # Check if it was below this level recently
                recent_low = df['low'].iloc[-10:].min()
                if recent_low < resistance['price']:
                    breaking_resistance = True
                    broken_level = resistance['price']
                    break
        
        # Volume analysis
        vol_analysis = calculate_volume_analysis(df)
        volume_surge = vol_analysis['ratio_20'] >= 1.5  # 1.5x above average
        
        # MACD confirmation
        macd_bullish = hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]
        
        # Not overextended
        not_overextended = rsi < 80
        
        # Price action (strong green candle)
        last_candle_gain = ((df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]) * 100
        strong_candle = last_candle_gain > 1  # At least 1% gain
        
        strength = 0
        if breaking_resistance:
            strength += 35
        if volume_surge:
            strength += 30
        if macd_bullish:
            strength += 20
        if strong_candle:
            strength += 15
        
        if strength >= 65 and not_overextended:
            return {
                'Name': instrument.symbol,
                'Close': round(current_price, 2),
                'Strength': strength,
                'RSI': round(rsi, 1),
                'Resistance_Broken': round(broken_level, 2) if broken_level else 'N/A',
                'Volume_Ratio': round(vol_analysis['ratio_20'], 1),
                'Last_Candle_Gain': round(last_candle_gain, 1),
                'Pattern': 'Volume Breakout',
                'Educational_Note': f"Breaking resistance with {vol_analysis['ratio_20']:.1f}x volume. Strong institutional buying detected."
            }
        
        return None
        
    except Exception as e:
        print(f"Error in breakout analysis: {e}")
        return None


# ============================================================================
# STRATEGY 4: MARKET LEADERS SCANNER
# ============================================================================

def analyze_market_leaders(df, instrument):
    """
    Find stocks showing exceptional strength.
    
    EDUCATIONAL NOTES:
    - Market leaders often continue leading
    - These stocks have strong fundamentals + technicals
    - Good for momentum trading
    - Often the first to recover in corrections
    
    Criteria:
    - Near 52-week high
    - Strong relative performance
    - Consistent volume
    - Trend strength (ADX > 25)
    """
    try:
        if len(df) < 252:  # Need 1 year of data
            return None
        
        # Calculate indicators
        df['50_EMA'] = calculate_ema(df['close'], 50)
        df['RSI'] = calculate_rsi(df['close'])
        adx = calculate_adx(df)
        
        current_price = df['close'].iloc[-1]
        
        # 52-week high/low
        week_52_high = df['high'].iloc[-252:].max()
        week_52_low = df['low'].iloc[-252:].min()
        
        # Distance from 52-week high
        distance_from_high = ((week_52_high - current_price) / week_52_high) * 100
        near_52w_high = distance_from_high <= 10  # Within 10% of 52W high
        
        # Performance metrics
        price_3m_ago = df['close'].iloc[-63] if len(df) > 63 else df['close'].iloc[0]
        performance_3m = ((current_price - price_3m_ago) / price_3m_ago) * 100
        
        # Trend strength
        adx_val = adx.iloc[-1] if len(adx) > 0 else 0
        strong_trend = adx_val > 25
        
        # Volume consistency
        vol_analysis = calculate_volume_analysis(df)
        consistent_volume = vol_analysis['ratio_20'] > 0.7  # Not drying up
        
        # Above key EMA
        above_50ema = current_price > df['50_EMA'].iloc[-1]
        
        strength = 0
        if near_52w_high:
            strength += 30
        if performance_3m > 15:  # Outperforming by 15%+
            strength += 25
        elif performance_3m > 10:
            strength += 15
        if strong_trend:
            strength += 25
        if consistent_volume:
            strength += 10
        if above_50ema:
            strength += 10
        
        if strength >= 70:
            return {
                'Name': instrument.symbol,
                'Close': round(current_price, 2),
                'Strength': strength,
                '52W_High': round(week_52_high, 2),
                'Distance_52W': round(distance_from_high, 1),
                '3M_Performance': round(performance_3m, 1),
                'ADX': round(adx_val, 1),
                'Pattern': 'Market Leader',
                'Educational_Note': f"Strong performer: +{performance_3m:.1f}% in 3M. Only {distance_from_high:.1f}% from 52W high. Leading the market."
            }
        
        return None
        
    except Exception as e:
        print(f"Error in market leader analysis: {e}")
        return None


# ============================================================================
# STRATEGY 5: CONSOLIDATION BREAKOUT SCANNER
# ============================================================================

def analyze_consolidation_breakout(df, instrument):
    """
    Find stocks breaking out of tight consolidation.
    
    EDUCATIONAL NOTES:
    - Consolidation = market indecision
    - Breakout = decision made (new trend starts)
    - Tight consolidation = big move coming
    - Volume expansion confirms breakout
    
    Criteria:
    - Tight range for 10+ days
    - Low volatility during consolidation
    - Breakout with volume expansion
    - Above previous consolidation high
    """
    try:
        if len(df) < 50:
            return None
        
        current_price = df['close'].iloc[-1]
        
        # Check for consolidation (last 15 days)
        consolidation_period = df.iloc[-15:]
        
        # Calculate range
        high_in_period = consolidation_period['high'].max()
        low_in_period = consolidation_period['low'].min()
        range_pct = ((high_in_period - low_in_period) / low_in_period) * 100
        
        # Tight consolidation
        tight_range = range_pct < 8  # Less than 8% range
        
        # Volume contraction during consolidation
        vol_consolidation = consolidation_period['volume'].iloc[:-3].mean()
        vol_recent = df['volume'].iloc[-3:].mean()
        volume_expanding = vol_recent > vol_consolidation * 1.3
        
        # Price breaking above consolidation
        breaking_high = current_price > high_in_period * 0.995  # At or above high
        
        # Check previous trend before consolidation
        pre_consolidation = df.iloc[-30:-15]
        was_uptrending = pre_consolidation['close'].iloc[-1] > pre_consolidation['close'].iloc[0]
        
        # RSI check
        df['RSI'] = calculate_rsi(df['close'])
        rsi = df['RSI'].iloc[-1]
        rsi_ok = 45 <= rsi <= 75
        
        strength = 0
        if tight_range:
            strength += 30
        if volume_expanding:
            strength += 30
        if breaking_high:
            strength += 25
        if was_uptrending:
            strength += 10
        if rsi_ok:
            strength += 5
        
        if strength >= 70:
            return {
                'Name': instrument.symbol,
                'Close': round(current_price, 2),
                'Strength': strength,
                'Consolidation_Range': round(range_pct, 1),
                'Consolidation_High': round(high_in_period, 2),
                'Volume_Expansion': round((vol_recent / vol_consolidation), 1),
                'RSI': round(rsi, 1),
                'Pattern': 'Consolidation Breakout',
                'Educational_Note': f"Breaking {range_pct:.1f}% consolidation with {(vol_recent/vol_consolidation):.1f}x volume. Potential explosive move."
            }
        
        return None
        
    except Exception as e:
        print(f"Error in consolidation analysis: {e}")
        return None


# ============================================================================
# LEGACY STRATEGIES (Keep for backward compatibility)
# ============================================================================

def analyze_bullish(df, instrument):
    """Legacy bullish strategy - kept for compatibility."""
    # Your original code (keeping it as is)
    try:
        close_prices = df['close'].values
        window_size = max(int(len(df) * 0.05), 5)
        local_min = argrelextrema(close_prices, np.less_equal, order=window_size)[0]

        valid_supports = []
        for m in local_min:
            if m < len(df) - 126:
                continue
            support_price = close_prices[m]
            current_price = close_prices[-1]
            if 1.05 <= (current_price / support_price) <= 1.20:
                if df['volume'].iloc[-1] > df['volume'].iloc[m] * 0.8:
                    valid_supports.append({
                        'price': support_price,
                        'touches': 1
                    })

        if not valid_supports:
            return None

        strongest_support = max(valid_supports, key=lambda x: x['touches'])
        current_price = close_prices[-1]
        distance_pct = ((current_price - strongest_support['price']) / strongest_support['price']) * 100

        ema_crossover = df['50_EMA'].iloc[-1] > df['200_EMA'].iloc[-1]
        rsi_value = df['RSI'].iloc[-1]
        rsi_ok = 30 <= rsi_value <= 70

        if ema_crossover and rsi_ok:
            return {
                'Name': instrument.symbol,
                'Close': current_price,
                'Support': strongest_support['price'],
                'Strength': strongest_support['touches'],
                'Distance_pct': distance_pct,
                'RSI': rsi_value,
                'Trend': 'Bullish'
            }
        return None

    except Exception as e:
        print(f"Error in bullish analysis: {e}")
        return None


def analyze_bearish(df, instrument):
    """Legacy bearish strategy - kept for compatibility."""
    # Your original code (keeping it as is)
    try:
        close_prices = df['close'].values
        window_size = max(int(len(df) * 0.05), 5)
        local_max = argrelextrema(close_prices, np.greater_equal, order=window_size)[0]

        valid_resistances = []
        for m in local_max:
            if m < len(df) - 126:
                continue
            resistance_price = close_prices[m]
            current_price = close_prices[-1]
            if 0.80 <= (current_price / resistance_price) <= 0.95:
                if df['volume'].iloc[-1] > df['volume'].iloc[m] * 0.8:
                    valid_resistances.append({
                        'price': resistance_price,
                        'touches': 1
                    })

        if not valid_resistances:
            return None

        strongest_resistance = max(valid_resistances, key=lambda x: x['touches'])
        current_price = close_prices[-1]
        distance_pct = ((strongest_resistance['price'] - current_price) / current_price) * 100

        ema_crossover = df['50_EMA'].iloc[-1] < df['200_EMA'].iloc[-1]
        rsi_value = df['RSI'].iloc[-1]
        rsi_ok = 30 <= rsi_value <= 70

        if ema_crossover and rsi_ok:
            return {
                'Name': instrument.symbol,
                'Close': current_price,
                'Resistance': strongest_resistance['price'],
                'Strength': strongest_resistance['touches'],
                'Distance_pct': distance_pct,
                'RSI': rsi_value,
                'Trend': 'Bearish'
            }
        return None

    except Exception as e:
        print(f"Error in bearish analysis: {e}")
        return None


# ============================================================================
# MAIN ANALYSIS DISPATCHER
# ============================================================================

def analyze_stock(alice, token, strategy, exchange='NSE'):
    """Main analysis function with strategy routing."""
    try:
        # Get historical data
        instrument, df = get_cached_historical_data(
            alice, token, 
            datetime.now() - timedelta(days=365), 
            datetime.now(), 
            "D", 
            exchange
        )
        
        if len(df) < 100:
            return None

        # Calculate basic indicators for all strategies
        df['50_EMA'] = calculate_ema(df['close'], 50)
        df['200_EMA'] = calculate_ema(df['close'], 200)
        df['RSI'] = calculate_rsi(df['close'])

        # Route to appropriate strategy
        if strategy == "Strong Uptrend Scanner":
            return analyze_strong_uptrend(df, instrument)
        
        elif strategy == "Pullback to Support":
            return analyze_pullback_entry(df, instrument)
        
        elif strategy == "Volume Breakout":
            return analyze_volume_breakout(df, instrument)
        
        elif strategy == "Market Leaders":
            return analyze_market_leaders(df, instrument)
        
        elif strategy == "Consolidation Breakout":
            return analyze_consolidation_breakout(df, instrument)
        
        # Legacy strategies
        elif strategy == "EMA, RSI & Support Zone (Buy)":
            return analyze_bullish(df, instrument)
        
        elif strategy == "EMA, RSI & Resistance Zone (Sell)":
            return analyze_bearish(df, instrument)
        
        return None

    except Exception as e:
        print(f"Error analyzing {token}: {e}")
        return None


def analyze_stock_batch(alice, tokens, strategy, exchange='NSE', batch_size=50):
    """Analyze a batch of stocks in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_token = {
            executor.submit(analyze_stock, alice, token, strategy, exchange): token
            for token in tokens[:batch_size]
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


def analyze_all_tokens(alice, tokens, strategy, exchange='NSE'):
    """Analyze all tokens with optimized batch processing."""
    results = []
    batch_size = 50
    total_batches = (len(tokens) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(tokens))
        batch_tokens = tokens[start_idx:end_idx]
        
        batch_results = analyze_stock_batch(alice, batch_tokens, strategy, exchange, batch_size)
        results.extend(batch_results)
    
    return results
