import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def find_swings(df: pd.DataFrame, window: int = 3) -> Tuple[List[Tuple], List[Tuple]]:
    """Detect swing highs/lows using local extrema with more lenient detection."""
    highs = []
    lows = []
    
    if len(df) < window * 2:
        return [], []
    
    # Use simpler local extrema detection - find peaks and troughs
    for i in range(window, len(df) - window):
        # Check if it's a local high
        if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
            highs.append((df.index[i], df['high'].iloc[i]))
        
        # Check if it's a local low
        if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
            lows.append((df.index[i], df['low'].iloc[i]))
    
    return highs, lows

def detect_trend(swings_high: List[Tuple], swings_low: List[Tuple], lookback: int = 2) -> str:
    """Detect uptrend (HH/HL), downtrend (LH/LL), or neutral."""
    # Require at least lookback swings
    if len(swings_high) < lookback or len(swings_low) < lookback:
        return "neutral"
    
    recent_highs = swings_high[-lookback:]
    recent_lows = swings_low[-lookback:]
    
    # Uptrend: Higher highs and higher lows
    hh = all(recent_highs[i][1] > recent_highs[i-1][1] for i in range(1, lookback))
    hl = all(recent_lows[i][1] > recent_lows[i-1][1] for i in range(1, lookback))
    if hh and hl:
        return "uptrend"
    
    # Downtrend: Lower highs and lower lows
    lh = all(recent_highs[i][1] < recent_highs[i-1][1] for i in range(1, lookback))
    ll = all(recent_lows[i][1] < recent_lows[i-1][1] for i in range(1, lookback))
    if lh and ll:
        return "downtrend"
    
    return "neutral"

def detect_trend_from_price_action(df: pd.DataFrame) -> str:
    """Fallback: Detect trend from recent price action and moving averages."""
    if len(df) < 20:
        return "neutral"
    
    # Calculate moving averages
    ma_10 = df['close'].rolling(window=10).mean().iloc[-1]
    ma_20 = df['close'].rolling(window=20).mean().iloc[-1]
    ma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else ma_20
    
    current_price = df['close'].iloc[-1]
    
    # Check price relative to MAs
    if current_price > ma_10 > ma_20 > ma_50:
        return "uptrend"
    elif current_price < ma_10 < ma_20 < ma_50:
        return "downtrend"
    
    # Simpler check: last 10 closes vs 20-period avg
    last_10_avg = df['close'].iloc[-10:].mean()
    ma_20_avg = ma_20
    
    if last_10_avg > ma_20_avg * 1.002:  # 0.2% above
        return "uptrend"
    elif last_10_avg < ma_20_avg * 0.998:  # 0.2% below
        return "downtrend"
    
    # Check recent higher/lower closes
    recent_closes = df['close'].iloc[-10:].values
    if len(recent_closes) >= 5:
        if recent_closes[-1] > recent_closes[0]:
            return "uptrend"
        elif recent_closes[-1] < recent_closes[0]:
            return "downtrend"
    
    return "neutral"

def get_recent_swings(df: pd.DataFrame, window: int = 3) -> Dict:
    highs, lows = find_swings(df, window)
    trend = detect_trend(highs, lows)
    
    # Fallback to price action trend detection if no clear swing trend
    if trend == "neutral" and len(df) > 20:
        trend = detect_trend_from_price_action(df)
    
    return {
        "last_swing_high": highs[-1] if highs else None,
        "last_swing_low": lows[-1] if lows else None,
        "prev_swing_high": highs[-2] if len(highs) > 1 else None,
        "prev_swing_low": lows[-2] if len(lows) > 1 else None,
        "trend": trend
    }



