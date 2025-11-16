"""
Divergence Logger - Tracks when divergences are detected
Surgical addition - does not modify existing trading logic
"""

from database import get_session, DivergenceEvent
from datetime import datetime

def log_divergence(symbol, timeframe, indicator, divergence_type, current_price):
    """
    Log a new divergence detection event
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USD')
        timeframe: Chart timeframe (e.g., '1H')
        indicator: Indicator name (e.g., 'OBV', 'RSI')
        divergence_type: 'bullish' or 'bearish'
        current_price: Current market price
    
    Returns:
        DivergenceEvent ID or None if error
    """
    if divergence_type not in ['bullish', 'bearish']:
        return None
    
    try:
        session = get_session()
        
        # Check if active divergence already exists for this combo
        existing = session.query(DivergenceEvent).filter(
            DivergenceEvent.symbol == symbol,
            DivergenceEvent.timeframe == timeframe,
            DivergenceEvent.indicator == indicator,
            DivergenceEvent.divergence_type == divergence_type,
            DivergenceEvent.status == 'active'
        ).first()
        
        # If already logged and active, don't duplicate
        if existing:
            session.close()
            return existing.id
        
        # Create new divergence event
        event = DivergenceEvent(
            symbol=symbol,
            timeframe=timeframe,
            indicator=indicator,
            divergence_type=divergence_type,
            detection_price=current_price,
            detected_at=datetime.utcnow(),
            status='active'
        )
        
        session.add(event)
        session.commit()
        event_id = event.id
        session.close()
        
        return event_id
        
    except Exception as e:
        print(f"Error logging divergence: {e}")
        try:
            session.close()
        except:
            pass
        return None

def log_divergences_from_context(symbol, timeframe, trend_context, current_price):
    """
    Extract and log all divergences from trend_context
    
    Args:
        symbol: Trading pair
        timeframe: Chart timeframe
        trend_context: Dictionary with indicator trends
        current_price: Current market price
    
    Returns:
        List of logged event IDs
    """
    logged_events = []
    
    print(f"🔍 Checking divergences for {symbol} {timeframe} (price: {current_price})")
    
    # List of indicators that track divergence
    indicators_with_divergence = ['RSI', 'Stochastic', 'MFI', 'OBV']
    
    for indicator in indicators_with_divergence:
        if indicator in trend_context:
            ctx = trend_context[indicator]
            divergence = ctx.get('divergence', 'none')
            
            print(f"  {indicator}: divergence={divergence}")
            
            if divergence in ['bullish', 'bearish']:
                print(f"  ✅ Logging {divergence} divergence for {indicator}")
                event_id = log_divergence(
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator=indicator,
                    divergence_type=divergence,
                    current_price=current_price
                )
                if event_id:
                    print(f"  📊 Logged divergence event ID: {event_id}")
                    logged_events.append(event_id)
                else:
                    print(f"  ⚠️ Failed to log divergence")
    
    if logged_events:
        print(f"✅ Total divergences logged: {len(logged_events)}")
    else:
        print(f"  No divergences detected")
    
    return logged_events
