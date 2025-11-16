"""
Divergence Analytics - Calculates timing statistics from resolved divergences
Surgical addition - does not modify existing trading logic
"""

from database import get_session, DivergenceEvent, DivergenceStats
from datetime import datetime
import statistics

# Timeframe to minutes mapping
TIMEFRAME_MINUTES = {
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1H': 60,
    '4H': 240,
    '1D': 1440
}

def classify_divergence_speed(avg_candles):
    """
    Classify divergence speed based on candle count
    
    Speed classes (timeframe-relative):
    - FAST: <2 candles (too quick to profit)
    - ACTIONABLE: 2-6 candles (scalp window exists)
    - SLOW: >6 candles (takes too long)
    
    Args:
        avg_candles: Average resolution time in candles
    
    Returns:
        str: 'fast', 'actionable', or 'slow'
    """
    if avg_candles < 2:
        return 'fast'
    elif avg_candles <= 6:
        return 'actionable'
    else:
        return 'slow'

def calculate_divergence_stats(indicator, timeframe, divergence_type):
    """
    Calculate statistics for a specific divergence pattern
    
    Args:
        indicator: Indicator name (e.g., 'OBV')
        timeframe: Chart timeframe (e.g., '1H')
        divergence_type: 'bullish' or 'bearish'
    
    Returns:
        dict with stats or None if insufficient data
    """
    try:
        session = get_session()
        
        # Get all resolved events for this pattern
        events = session.query(DivergenceEvent).filter(
            DivergenceEvent.indicator == indicator,
            DivergenceEvent.timeframe == timeframe,
            DivergenceEvent.divergence_type == divergence_type,
            DivergenceEvent.status == 'resolved',
            DivergenceEvent.resolution_candles.isnot(None)
        ).all()
        
        session.close()
        
        # Need at least 5 samples for meaningful stats
        if len(events) < 5:
            return None
        
        # Extract resolution candles
        candles = [e.resolution_candles for e in events if e.resolution_candles > 0]
        
        if len(candles) < 5:
            return None
        
        # Calculate statistics
        avg_candles = statistics.mean(candles)
        median_candles = statistics.median(candles)
        
        # Calculate 90th percentile manually
        sorted_candles = sorted(candles)
        p90_index = int(len(sorted_candles) * 0.9)
        p90_candles = sorted_candles[min(p90_index, len(sorted_candles)-1)]
        
        # Convert to minutes
        timeframe_minutes = TIMEFRAME_MINUTES.get(timeframe, 60)
        avg_minutes = avg_candles * timeframe_minutes
        
        # Calculate success rate
        successes = len([e for e in events if e.resolution_outcome == 'success'])
        success_rate = (successes / len(events)) * 100
        
        # Classify speed
        speed_class = classify_divergence_speed(avg_candles)
        
        return {
            'avg_candles': round(avg_candles, 1),
            'avg_minutes': round(avg_minutes, 1),
            'median_candles': round(median_candles, 1),
            'p90_candles': round(p90_candles, 1),
            'speed_class': speed_class,
            'success_rate': round(success_rate, 1),
            'sample_size': len(events)
        }
        
    except Exception as e:
        print(f"Error calculating divergence stats: {e}")
        return None

def update_all_divergence_stats():
    """
    Update statistics for all divergence patterns
    Background job - runs nightly or on-demand
    
    Returns:
        Number of stat records updated
    """
    indicators = ['OBV', 'RSI', 'Stochastic', 'MFI']
    timeframes = ['5m', '15m', '30m', '1H', '4H', '1D']
    divergence_types = ['bullish', 'bearish']
    
    updated_count = 0
    
    try:
        session = get_session()
        
        for indicator in indicators:
            for timeframe in timeframes:
                for div_type in divergence_types:
                    
                    stats = calculate_divergence_stats(indicator, timeframe, div_type)
                    
                    if stats:
                        # Check if record exists
                        existing = session.query(DivergenceStats).filter(
                            DivergenceStats.indicator == indicator,
                            DivergenceStats.timeframe == timeframe,
                            DivergenceStats.divergence_type == div_type
                        ).first()
                        
                        if existing:
                            # Update existing
                            existing.avg_resolution_candles = stats['avg_candles']
                            existing.avg_resolution_minutes = stats['avg_minutes']
                            existing.median_resolution_candles = stats['median_candles']
                            existing.p90_resolution_candles = stats['p90_candles']
                            existing.speed_class = stats['speed_class']
                            existing.success_rate = stats['success_rate']
                            existing.sample_size = stats['sample_size']
                            existing.last_updated = datetime.utcnow()
                        else:
                            # Create new record
                            new_stat = DivergenceStats(
                                indicator=indicator,
                                timeframe=timeframe,
                                divergence_type=div_type,
                                avg_resolution_candles=stats['avg_candles'],
                                avg_resolution_minutes=stats['avg_minutes'],
                                median_resolution_candles=stats['median_candles'],
                                p90_resolution_candles=stats['p90_candles'],
                                speed_class=stats['speed_class'],
                                success_rate=stats['success_rate'],
                                sample_size=stats['sample_size']
                            )
                            session.add(new_stat)
                        
                        updated_count += 1
        
        session.commit()
        session.close()
        
        print(f"✅ Updated {updated_count} divergence stat records")
        return updated_count
        
    except Exception as e:
        print(f"Error updating divergence stats: {e}")
        try:
            session.close()
        except:
            pass
        return 0

def get_divergence_timing_info(indicator, timeframe, divergence_type):
    """
    Get timing intelligence for a specific divergence pattern
    Used for displaying in UI
    
    Args:
        indicator: Indicator name
        timeframe: Chart timeframe
        divergence_type: 'bullish' or 'bearish'
    
    Returns:
        dict with timing info or None if no data
    """
    try:
        session = get_session()
        
        stat = session.query(DivergenceStats).filter(
            DivergenceStats.indicator == indicator,
            DivergenceStats.timeframe == timeframe,
            DivergenceStats.divergence_type == divergence_type
        ).first()
        
        session.close()
        
        if not stat or stat.sample_size < 5:
            return None
        
        return {
            'avg_candles': stat.avg_resolution_candles,
            'avg_hours': stat.avg_resolution_minutes / 60,
            'median_candles': stat.median_resolution_candles,
            'speed_class': stat.speed_class,
            'success_rate': stat.success_rate,
            'sample_size': stat.sample_size,
            'recommendation': get_speed_recommendation(stat.speed_class, divergence_type)
        }
        
    except Exception as e:
        print(f"Error getting divergence timing info: {e}")
        return None

def get_speed_recommendation(speed_class, divergence_type):
    """
    Get trading recommendation based on divergence speed
    
    Args:
        speed_class: 'fast', 'actionable', or 'slow'
        divergence_type: 'bullish' or 'bearish'
    
    Returns:
        str: Trading recommendation
    """
    if speed_class == 'fast':
        return "⚠️ FAST divergence - High risk! Consider waiting for reversal"
    elif speed_class == 'actionable':
        return "✅ ACTIONABLE - Quick scalp possible with tight stops"
    elif speed_class == 'slow':
        return "⏱️ SLOW divergence - Longer hold needed, harder to time exit"
    else:
        return "📊 Insufficient data for timing recommendation"
