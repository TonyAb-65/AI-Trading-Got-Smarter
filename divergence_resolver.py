"""
Divergence Resolver - Detects when divergences resolve or expire
Surgical addition - does not modify existing trading logic
"""

from database import get_session, DivergenceEvent
from datetime import datetime, timedelta

# Timeframe to minutes mapping
TIMEFRAME_MINUTES = {
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1H': 60,
    '4H': 240,
    '1D': 1440
}

def check_divergence_resolution(event, current_price, trend_context=None):
    """
    Check if a divergence event has resolved
    
    Resolution criteria:
    - Price moved 2%+ in predicted direction (bearish = down, bullish = up)
    - OR divergence disappeared from indicators (if trend_context provided)
    - OR max tracking window exceeded (20 candles)
    
    Args:
        event: DivergenceEvent object
        current_price: Current market price
        trend_context: Optional - current indicator trends
    
    Returns:
        (resolved, outcome) - (bool, str or None)
        outcome: 'success', 'failed', 'expired', or None
    """
    detection_price = event.detection_price
    price_change_pct = ((current_price - detection_price) / detection_price) * 100
    
    # Calculate time elapsed
    time_elapsed = datetime.utcnow() - event.detected_at
    timeframe_minutes = TIMEFRAME_MINUTES.get(event.timeframe, 60)
    candles_elapsed = time_elapsed.total_seconds() / 60 / timeframe_minutes
    
    # Check max tracking window (20 candles)
    if candles_elapsed > 20:
        return True, 'expired'
    
    # Check price movement threshold (2%)
    if event.divergence_type == 'bearish':
        # Bearish divergence expects price to fall
        if price_change_pct <= -2.0:
            return True, 'success'
        elif price_change_pct >= 3.0:
            # Price went up too much - divergence failed
            return True, 'failed'
    
    elif event.divergence_type == 'bullish':
        # Bullish divergence expects price to rise
        if price_change_pct >= 2.0:
            return True, 'success'
        elif price_change_pct <= -3.0:
            # Price went down too much - divergence failed
            return True, 'failed'
    
    # Check if divergence disappeared from indicators
    if trend_context and event.indicator in trend_context:
        current_divergence = trend_context[event.indicator].get('divergence', 'none')
        if current_divergence == 'none':
            # Divergence cleared - check if it succeeded
            if event.divergence_type == 'bearish' and price_change_pct < 0:
                return True, 'success'
            elif event.divergence_type == 'bullish' and price_change_pct > 0:
                return True, 'success'
            else:
                return True, 'failed'
    
    return False, None

def resolve_active_divergences(symbol, current_price, trend_context=None):
    """
    Check all active divergences for a symbol and resolve if needed
    
    Args:
        symbol: Trading pair
        current_price: Current market price
        trend_context: Optional - current indicator trends
    
    Returns:
        Number of divergences resolved
    """
    try:
        session = get_session()
        
        # Get all active divergences for this symbol
        active_events = session.query(DivergenceEvent).filter(
            DivergenceEvent.symbol == symbol,
            DivergenceEvent.status == 'active'
        ).all()
        
        resolved_count = 0
        
        for event in active_events:
            resolved, outcome = check_divergence_resolution(event, current_price, trend_context)
            
            if resolved:
                # Calculate resolution metrics
                time_elapsed = datetime.utcnow() - event.detected_at
                timeframe_minutes = TIMEFRAME_MINUTES.get(event.timeframe, 60)
                candles_elapsed = int(time_elapsed.total_seconds() / 60 / timeframe_minutes)
                
                # Update event
                event.resolved_at = datetime.utcnow()
                event.resolution_price = current_price
                event.resolution_candles = candles_elapsed
                event.resolution_outcome = outcome
                event.status = 'resolved'
                
                resolved_count += 1
        
        session.commit()
        session.close()
        
        return resolved_count
        
    except Exception as e:
        print(f"Error resolving divergences: {e}")
        try:
            session.close()
        except:
            pass
        return 0
