"""
Educational Technical Strength Scoring System
Provides comprehensive analysis with learning points for students
"""

import pandas as pd
import numpy as np
from technical_indicators import (
    calculate_ema, calculate_rsi, calculate_macd, calculate_adx,
    calculate_bollinger_bands, calculate_stochastic,
    find_support_resistance, calculate_volume_analysis,
    detect_trend, get_indicator_signals, calculate_momentum_score
)


def calculate_technical_strength_score(df):
    """
    Calculate comprehensive technical strength score (0-100).
    
    This score combines multiple technical factors:
    - Trend Strength (30 points)
    - Momentum Indicators (30 points)
    - Volume Analysis (20 points)
    - Pattern Recognition (20 points)
    
    Returns: Dictionary with score and breakdown
    """
    
    score_breakdown = {
        'total_score': 0,
        'trend_score': 0,
        'momentum_score': 0,
        'volume_score': 0,
        'pattern_score': 0,
        'details': {}
    }
    
    try:
        if len(df) < 200:
            return score_breakdown
        
        # ===== TREND ANALYSIS (30 points) =====
        trend_score = 0
        
        # EMA alignment (15 points)
        df['9_EMA'] = calculate_ema(df['close'], 9)
        df['21_EMA'] = calculate_ema(df['close'], 21)
        df['50_EMA'] = calculate_ema(df['close'], 50)
        df['200_EMA'] = calculate_ema(df['close'], 200)
        
        current_price = df['close'].iloc[-1]
        
        if current_price > df['200_EMA'].iloc[-1]:
            trend_score += 8  # Above long-term trend
        if df['50_EMA'].iloc[-1] > df['200_EMA'].iloc[-1]:
            trend_score += 7  # Medium above long term
        
        # ADX - Trend strength (15 points)
        adx = calculate_adx(df)
        if len(adx) > 0:
            adx_val = adx.iloc[-1]
            if adx_val > 50:
                trend_score += 15  # Very strong trend
            elif adx_val > 35:
                trend_score += 12  # Strong trend
            elif adx_val > 25:
                trend_score += 8   # Moderate trend
            elif adx_val > 15:
                trend_score += 4   # Weak trend
            
            score_breakdown['details']['adx'] = round(adx_val, 1)
        
        score_breakdown['trend_score'] = min(trend_score, 30)
        
        # ===== MOMENTUM ANALYSIS (30 points) =====
        momentum_score = 0
        
        # RSI (10 points)
        rsi = calculate_rsi(df['close']).iloc[-1]
        if 40 <= rsi <= 60:
            momentum_score += 8   # Neutral - balanced
        elif 60 < rsi <= 70:
            momentum_score += 10  # Bullish momentum
        elif 30 <= rsi < 40:
            momentum_score += 6   # Slightly oversold
        elif rsi > 70:
            momentum_score += 4   # Overbought - be cautious
        
        score_breakdown['details']['rsi'] = round(rsi, 1)
        
        # MACD (10 points)
        macd, signal, hist = calculate_macd(df['close'])
        if hist.iloc[-1] > 0:
            momentum_score += 5
            if hist.iloc[-1] > hist.iloc[-2]:  # Increasing
                momentum_score += 5
        
        score_breakdown['details']['macd_histogram'] = round(hist.iloc[-1], 2)
        
        # Stochastic (10 points)
        k, d = calculate_stochastic(df)
        if 20 <= k.iloc[-1] <= 80:  # Not extreme
            if k.iloc[-1] > d.iloc[-1]:  # Bullish
                momentum_score += 10
            else:
                momentum_score += 5
        
        score_breakdown['details']['stochastic_k'] = round(k.iloc[-1], 1)
        
        score_breakdown['momentum_score'] = min(momentum_score, 30)
        
        # ===== VOLUME ANALYSIS (20 points) =====
        volume_score = 0
        
        vol_analysis = calculate_volume_analysis(df)
        
        # Volume surge (10 points)
        if vol_analysis['surge']:
            volume_score += 10
        elif vol_analysis['ratio_20'] > 1.2:
            volume_score += 7
        elif vol_analysis['ratio_20'] > 1.0:
            volume_score += 5
        
        # Volume trend (10 points)
        if vol_analysis['trend'] == 'Increasing':
            volume_score += 10
        else:
            volume_score += 3
        
        score_breakdown['details']['volume_ratio'] = round(vol_analysis['ratio_20'], 2)
        score_breakdown['details']['volume_trend'] = vol_analysis['trend']
        
        score_breakdown['volume_score'] = min(volume_score, 20)
        
        # ===== PATTERN ANALYSIS (20 points) =====
        pattern_score = 0
        
        # Higher highs and higher lows (10 points)
        recent_highs = df['high'].iloc[-20:]
        recent_lows = df['low'].iloc[-20:]
        
        if len(recent_highs) >= 10:
            higher_highs = recent_highs.iloc[-1] >= recent_highs.iloc[-10:].max() * 0.99
            higher_lows = recent_lows.iloc[-1] >= recent_lows.iloc[-10:].median()
            
            if higher_highs and higher_lows:
                pattern_score += 10
            elif higher_highs or higher_lows:
                pattern_score += 5
        
        # Support/Resistance proximity (10 points)
        supports, resistances = find_support_resistance(df)
        
        near_support = False
        near_resistance = False
        
        for support in supports:
            if abs(current_price - support['price']) / current_price < 0.02:
                near_support = True
                break
        
        for resistance in resistances:
            if abs(current_price - resistance['price']) / current_price < 0.02:
                near_resistance = True
                break
        
        if near_support:
            pattern_score += 7  # Near support is good for buying
        if near_resistance:
            pattern_score += 3  # Near resistance - caution
        
        if not near_support and not near_resistance:
            pattern_score += 5  # In clear zone
        
        score_breakdown['pattern_score'] = min(pattern_score, 20)
        
        # ===== TOTAL SCORE =====
        score_breakdown['total_score'] = (
            score_breakdown['trend_score'] +
            score_breakdown['momentum_score'] +
            score_breakdown['volume_score'] +
            score_breakdown['pattern_score']
        )
        
        return score_breakdown
        
    except Exception as e:
        print(f"Error calculating score: {e}")
        return score_breakdown


def get_educational_analysis(df, instrument_name):
    """
    Generate educational analysis report for a stock.
    
    Provides detailed breakdown with learning points.
    """
    
    analysis = {
        'stock_name': instrument_name,
        'current_price': 0,
        'technical_strength': 0,
        'trend': 'Unknown',
        'signals': {},
        'key_levels': {},
        'learning_points': [],
        'summary': ''
    }
    
    try:
        if len(df) < 100:
            analysis['summary'] = "Insufficient data for analysis"
            return analysis
        
        # Basic info
        analysis['current_price'] = round(df['close'].iloc[-1], 2)
        
        # Calculate strength score
        score_breakdown = calculate_technical_strength_score(df)
        analysis['technical_strength'] = score_breakdown['total_score']
        
        # Get trend
        analysis['trend'] = detect_trend(df)
        
        # Get all indicator signals
        analysis['signals'] = get_indicator_signals(df)
        
        # Calculate key levels
        df['50_EMA'] = calculate_ema(df['close'], 50)
        df['200_EMA'] = calculate_ema(df['close'], 200)
        
        supports, resistances = find_support_resistance(df)
        
        analysis['key_levels'] = {
            '50_EMA': round(df['50_EMA'].iloc[-1], 2),
            '200_EMA': round(df['200_EMA'].iloc[-1], 2),
            'support_levels': [round(s['price'], 2) for s in supports[:3]],
            'resistance_levels': [round(r['price'], 2) for r in resistances[:3]]
        }
        
        # Generate learning points based on analysis
        learning_points = []
        
        # Trend learning
        if analysis['trend'] in ['Strong Uptrend', 'Uptrend']:
            learning_points.append({
                'category': 'Trend',
                'point': f"Stock is in {analysis['trend']}. Price is above key moving averages, indicating bullish sentiment.",
                'what_it_means': "Uptrends suggest the stock is in demand. Higher highs and higher lows show buyers are in control."
            })
        elif analysis['trend'] in ['Downtrend', 'Strong Downtrend']:
            learning_points.append({
                'category': 'Trend',
                'point': f"Stock is in {analysis['trend']}. Exercise caution or wait for reversal signals.",
                'what_it_means': "Downtrends indicate selling pressure. It's risky to buy falling stocks without reversal confirmation."
            })
        
        # RSI learning
        rsi = score_breakdown['details'].get('rsi', 50)
        if rsi > 70:
            learning_points.append({
                'category': 'Momentum',
                'point': f"RSI is {rsi:.0f} (Overbought). Stock may be due for a pullback.",
                'what_it_means': "RSI above 70 suggests the stock has rallied strongly. In strong uptrends, it can stay overbought for extended periods."
            })
        elif rsi < 30:
            learning_points.append({
                'category': 'Momentum',
                'point': f"RSI is {rsi:.0f} (Oversold). Potential bounce opportunity if other factors align.",
                'what_it_means': "RSI below 30 suggests heavy selling. This can be a buying opportunity IF the overall trend is still up."
            })
        
        # Volume learning
        vol_trend = score_breakdown['details'].get('volume_trend', 'Unknown')
        vol_ratio = score_breakdown['details'].get('volume_ratio', 1.0)
        
        if vol_ratio > 1.5:
            learning_points.append({
                'category': 'Volume',
                'point': f"Volume surge detected ({vol_ratio:.1f}x average). Strong institutional interest.",
                'what_it_means': "High volume confirms price moves. Volume surge often precedes significant moves. Watch for direction."
            })
        
        # Support/Resistance learning
        if analysis['key_levels']['support_levels']:
            nearest_support = analysis['key_levels']['support_levels'][0]
            distance = abs(analysis['current_price'] - nearest_support) / analysis['current_price'] * 100
            
            if distance < 3:
                learning_points.append({
                    'category': 'Key Levels',
                    'point': f"Price near support at ₹{nearest_support}. Watch for bounce or breakdown.",
                    'what_it_means': "Support levels are where buyers historically stepped in. A bounce here would be bullish; breakdown would be bearish."
                })
        
        # Overall strength learning
        if analysis['technical_strength'] >= 75:
            learning_points.append({
                'category': 'Overall Assessment',
                'point': f"Strong technical setup (Score: {analysis['technical_strength']}/100). Multiple bullish factors aligned.",
                'what_it_means': "When multiple indicators agree, the probability of a successful trade increases. This is called 'confluence'."
            })
        elif analysis['technical_strength'] >= 50:
            learning_points.append({
                'category': 'Overall Assessment',
                'point': f"Moderate technical strength (Score: {analysis['technical_strength']}/100). Some bullish factors present.",
                'what_it_means': "Mixed signals suggest caution. Wait for more confirmation or look for better setups."
            })
        
        analysis['learning_points'] = learning_points
        
        # Generate summary
        if analysis['technical_strength'] >= 70:
            strength_text = "technically strong"
        elif analysis['technical_strength'] >= 50:
            strength_text = "showing moderate strength"
        else:
            strength_text = "technically weak"
        
        analysis['summary'] = (
            f"{instrument_name} is currently {strength_text} with a score of {analysis['technical_strength']}/100. "
            f"The stock is in a {analysis['trend']} with {analysis['signals']['RSI']} RSI. "
            f"Volume is {vol_trend.lower()}. "
        )
        
        return analysis
        
    except Exception as e:
        print(f"Error in educational analysis: {e}")
        analysis['summary'] = f"Error in analysis: {str(e)}"
        return analysis


def compare_stocks_educational(results_list):
    """
    Compare multiple stocks and rank them by technical strength.
    
    Helps students understand which setups are strongest.
    """
    
    if not results_list:
        return []
    
    comparison = []
    
    for result in results_list:
        stock_info = {
            'name': result.get('Name', 'Unknown'),
            'price': result.get('Close', 0),
            'strength': result.get('Strength', 0),
            'pattern': result.get('Pattern', 'N/A'),
            'why_strong': []
        }
        
        # Analyze why it's strong
        if result.get('RSI'):
            rsi = result['RSI']
            if 50 <= rsi <= 70:
                stock_info['why_strong'].append(f"Bullish RSI ({rsi:.0f})")
        
        if result.get('Volume_Trend') == 'Increasing':
            stock_info['why_strong'].append("Increasing volume")
        
        if result.get('Distance_200EMA'):
            dist = result['Distance_200EMA']
            if dist > 0:
                stock_info['why_strong'].append(f"{dist:.1f}% above 200 EMA")
        
        if result.get('Educational_Note'):
            stock_info['educational_note'] = result['Educational_Note']
        
        comparison.append(stock_info)
    
    # Sort by strength
    comparison.sort(key=lambda x: x['strength'], reverse=True)
    
    return comparison


def generate_daily_lesson(top_stocks, strategy_used):
    """
    Generate daily learning content based on screener results.
    
    Educational feature to teach students.
    """
    
    lesson = {
        'title': f'Daily Technical Analysis Lesson - {strategy_used}',
        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'strategy_explanation': '',
        'examples': [],
        'key_learnings': [],
        'quiz_questions': []
    }
    
    # Strategy explanations
    strategy_lessons = {
        'Strong Uptrend Scanner': {
            'explanation': """
            This strategy identifies stocks in strong, established uptrends. 
            
            Key Concepts:
            - EMA Alignment: When shorter EMAs are above longer ones, it shows strength at all timeframes
            - Higher Highs & Higher Lows: The signature of an uptrend
            - Volume Confirmation: Volume should support price increases
            
            When to Use: Bull markets, momentum trading, swing trading uptrends
            
            Risk Management: Uptrends can reverse. Always use stop-losses below recent support.
            """,
            'key_learnings': [
                "Uptrends are your friend - trade with the trend, not against it",
                "All timeframes should align for strongest signals",
                "Volume confirms the move - without volume, uptrends are weak",
                "Even strong stocks pull back - that's when you enter"
            ]
        },
        'Pullback to Support': {
            'explanation': """
            This strategy finds low-risk entries in strong stocks.
            
            Key Concepts:
            - Buying the Dip: Enter strong stocks when they temporarily pull back
            - Support Levels: Previous lows where buyers stepped in
            - Risk-Reward: Better entry = lower risk, higher potential reward
            
            When to Use: When strong stocks have minor corrections in uptrends
            
            Risk Management: Support can break. Set stop-loss just below support level.
            """,
            'key_learnings': [
                "Best trades often come on pullbacks, not at highs",
                "Support is psychological - where buyers previously found value",
                "Decreasing volume on pullback is healthy (shows no panic)",
                "Wait for support to hold before entering"
            ]
        },
        'Volume Breakout': {
            'explanation': """
            This strategy catches stocks breaking out with institutional participation.
            
            Key Concepts:
            - Breakout: Price moving above resistance
            - Volume Confirmation: High volume shows real buying, not false breakout
            - Institutional Money: Big players move big volume
            
            When to Use: Stock market leaders, new trends starting, strong momentum
            
            Risk Management: False breakouts happen. Confirm with volume and follow-through.
            """,
            'key_learnings': [
                "Volume confirms everything - without it, breakouts often fail",
                "Institutional buying precedes big moves",
                "First breakout often strongest - don't chase late",
                "Resistance becomes support after breakout"
            ]
        }
    }
    
    if strategy_used in strategy_lessons:
        lesson['strategy_explanation'] = strategy_lessons[strategy_used]['explanation']
        lesson['key_learnings'] = strategy_lessons[strategy_used]['key_learnings']
    
    # Add examples from top stocks
    for stock in top_stocks[:3]:
        example = {
            'stock': stock.get('Name', 'N/A'),
            'why_qualified': stock.get('Educational_Note', 'Strong technical setup'),
            'what_to_watch': ''
        }
        
        # Add specific watch points
        if stock.get('RSI'):
            if stock['RSI'] > 60:
                example['what_to_watch'] = 'Watch for RSI to cool down on any pullback'
            elif stock['RSI'] < 40:
                example['what_to_watch'] = 'Watch for RSI to turn up from oversold'
        
        lesson['examples'].append(example)
    
    # Generate quiz questions
    lesson['quiz_questions'] = [
        {
            'question': f"What does {strategy_used} strategy primarily look for?",
            'learning_goal': "Understand the core concept of each strategy"
        },
        {
            'question': "Why is volume important in technical analysis?",
            'learning_goal': "Understand volume's role in confirming price moves"
        },
        {
            'question': "What's the difference between support and resistance?",
            'learning_goal': "Master key level identification"
        }
    ]
    
    return lesson
