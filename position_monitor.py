from datetime import datetime, timedelta
import time
from database import get_session, ActivePosition, MarketData
from api_integrations import get_market_data_unified, get_current_price
from technical_indicators import TechnicalIndicators, calculate_support_resistance
from whale_tracker import WhaleTracker
from divergence_resolver import resolve_active_divergences
from telegram_notifier import send_telegram_alert
import json
import numpy as np

# Cache to track last momentum direction per position (for Telegram notification filtering)
# Only notify when direction actually changes (bullish <-> bearish)
_last_momentum_direction = {}

class PositionMonitor:
    def __init__(self, ml_engine=None):
        self.check_interval_minutes = 15
        self.ml_engine = ml_engine  # Inject MLEngine for profile similarity calculation
        
        # Key indicators for quick Tier 0 check
        self.quick_check_indicators = ['RSI', 'MACD', 'Stochastic_%K', 'ADX', 'MFI']
        
    def check_active_positions(self):
        session = get_session()
        results = []
        
        try:
            active_positions = session.query(ActivePosition).filter(
                ActivePosition.is_active == True
            ).all()
            
            for position in active_positions:
                result = self._analyze_position(position, session)
                results.append(result)
                
                position.last_check_time = datetime.utcnow()
                position.current_recommendation = result['recommendation']
                
                if result['current_price']:
                    position.current_price = result['current_price']
                
                session.commit()
            
            return results
            
        except Exception as e:
            print(f"Error checking positions: {e}")
            session.rollback()
            return []
        finally:
            session.close()
    
    def _analyze_position(self, position, session):
        try:
            current_price = get_current_price(position.symbol, position.market_type)
            
            if not current_price:
                return {
                    'symbol': position.symbol,
                    'status': 'error',
                    'message': 'Unable to fetch current price',
                    'recommendation': 'HOLD',
                    'current_price': None,
                    'monitoring_alerts': []
                }
            
            timeframe = getattr(position, 'timeframe', '1H') or '1H'
            df = get_market_data_unified(position.symbol, position.market_type, timeframe, 100)
            
            if df is None or len(df) < 20:
                return {
                    'symbol': position.symbol,
                    'status': 'error',
                    'message': 'Unable to fetch market data',
                    'recommendation': 'HOLD',
                    'current_price': current_price,
                    'monitoring_alerts': []
                }
            
            tech_indicators = TechnicalIndicators(df)
            indicators_df = tech_indicators.calculate_all_indicators()
            indicators = tech_indicators.get_latest_indicators()
            signals = tech_indicators.get_trend_signals()
            
            # Calculate support/resistance and trend context FIRST (use enriched indicators_df, same as Market Analysis)
            support_levels, resistance_levels = calculate_support_resistance(indicators_df)
            trend_context = tech_indicators.get_trend_context(position.symbol, position.market_type)
            
            # Enrich indicators with S/R and trend context (same as market analysis)
            indicators['support_levels'] = support_levels
            indicators['resistance_levels'] = resistance_levels
            indicators['trend_context'] = trend_context
            
            monitoring_alerts = self._check_tight_monitoring(
                position, current_price, indicators, signals
            )
            
            # EARLY WARNING: Check momentum timing BEFORE profile deviation
            # Momentum shifts faster than profile patterns - gives earlier heads-up
            momentum_alerts = self._check_momentum_timing(
                position, tech_indicators, timeframe
            )
            monitoring_alerts.extend(momentum_alerts)
            
            # Add profile comparison alerts (deeper analysis)
            # NOW uses enriched indicators with S/R and trend_context
            profile_alerts = self._check_profile_deviation(
                position, indicators
            )
            monitoring_alerts.extend(profile_alerts)
            
            # Check and resolve active divergences (timing intelligence)
            try:
                resolve_active_divergences(position.symbol, current_price, trend_context)
            except Exception as e:
                print(f"Divergence resolution check failed: {e}")
            
            whale_tracker = WhaleTracker(tech_indicators.df)
            whale_movements = whale_tracker.detect_whale_movements()
            smart_money = whale_tracker.detect_smart_money()
            
            entry_price = position.entry_price
            pnl_percentage = ((current_price - entry_price) / entry_price) * 100
            
            if position.trade_type == 'SHORT':
                pnl_percentage = -pnl_percentage
            
            recommendation = self._generate_recommendation(
                position, current_price, pnl_percentage, signals, 
                support_levels, resistance_levels, whale_movements, smart_money,
                monitoring_alerts
            )
            
            market_data = MarketData(
                symbol=position.symbol,
                market_type=position.market_type,
                timestamp=datetime.utcnow(),
                open_price=float(df.iloc[-1]['open']),
                high_price=float(df.iloc[-1]['high']),
                low_price=float(df.iloc[-1]['low']),
                close_price=float(df.iloc[-1]['close']),
                volume=float(df.iloc[-1]['volume']),
                indicators=indicators
            )
            session.add(market_data)
            
            current_obv_slope = indicators.get('obv_slope', 0)
            position.last_obv_slope = current_obv_slope
            position.monitoring_alerts = monitoring_alerts
            
            # Send Telegram alerts ONLY when momentum DIRECTION changes (bullish <-> bearish)
            # This prevents notifications for small indicator fluctuations
            global _last_momentum_direction
            
            # Get current momentum direction from timing analysis
            current_momentum = tech_indicators.get_momentum_timing(60)  # 1H timeframe
            current_direction = current_momentum.get('momentum_direction', 'neutral') if current_momentum else 'neutral'
            
            # Create position key for tracking
            position_key = f"{position.id}_{position.symbol}"
            last_direction = _last_momentum_direction.get(position_key, None)
            
            # Determine if direction actually changed (bullish <-> bearish)
            direction_changed = False
            if last_direction is not None:
                # Only trigger on actual direction flip (not neutral transitions)
                if (last_direction == 'bullish' and current_direction == 'bearish') or \
                   (last_direction == 'bearish' and current_direction == 'bullish'):
                    direction_changed = True
                    print(f"📢 Momentum direction changed for {position.symbol}: {last_direction} → {current_direction}")
            
            # Update cached direction
            _last_momentum_direction[position_key] = current_direction
            
            # Only send Telegram notification if direction actually changed
            if direction_changed:
                momentum_alert_types = [
                    'MOMENTUM_SHIFT',      # 3+ signals aligned against position
                    'MOMENTUM_WARNING',    # 2 signals or early opposition
                    'MOMENTUM_DIVERGENCE', # OBV divergence (smart money conflict)
                    'MOMENTUM_ADX_CONFLICT' # ADX direction conflict
                ]
                for severity_level in ['EARLY_WARNING', 'LOW']:
                    momentum_alerts_filtered = [a for a in monitoring_alerts 
                                       if a.get('severity') == severity_level 
                                       and a.get('type') in momentum_alert_types]
                    if momentum_alerts_filtered:
                        try:
                            alert = momentum_alerts_filtered[0]
                            send_telegram_alert(
                                symbol=position.symbol,
                                position_type=position.trade_type,
                                entry_price=entry_price,
                                current_price=current_price,
                                pnl_percentage=pnl_percentage,
                                severity=severity_level,
                                alert_message=f"DIRECTION CHANGED: {last_direction} → {current_direction}",
                                recommendation=alert.get('recommendation', recommendation['action']),
                                reason=recommendation['reason']
                            )
                            break  # Only send one notification per direction change
                        except Exception as e:
                            print(f"Telegram notification error ({severity_level}): {e}")
            
            return {
                'symbol': position.symbol,
                'trade_type': position.trade_type,
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl_percentage': round(pnl_percentage, 2),
                'recommendation': recommendation['action'],
                'reason': recommendation['reason'],
                'signals': signals,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'whale_activity': len(whale_movements) > 0,
                'smart_money_signal': smart_money[0]['type'] if smart_money else None,
                'monitoring_alerts': monitoring_alerts,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error analyzing position {position.symbol}: {e}")
            return {
                'symbol': position.symbol,
                'status': 'error',
                'message': str(e),
                'recommendation': 'HOLD',
                'current_price': None,
                'monitoring_alerts': []
            }
    
    def _check_tight_monitoring(self, position, current_price, indicators, signals):
        alerts = []
        
        if not position.stop_loss:
            return alerts
        
        entry_price = position.entry_price
        stop_loss = position.stop_loss
        
        if position.trade_type == 'SHORT':
            distance_to_sl = stop_loss - entry_price
            
            if distance_to_sl <= 0:
                return alerts
            
            current_distance = current_price - entry_price
            threshold_price = entry_price + (distance_to_sl * 0.6)
            
            if current_price >= threshold_price:
                pct_to_sl = (current_distance / distance_to_sl) * 100
                alerts.append({
                    'type': 'DANGER_ZONE',
                    'severity': 'HIGH',
                    'message': f'⚠️ Price at ${current_price:,.2f} - {pct_to_sl:.0f}% to stop loss (${stop_loss:,.2f})',
                    'recommendation': 'CONSIDER EXIT - Approaching stop loss'
                })
            
            obv_slope = indicators.get('obv_slope', 0)
            if obv_slope > 0 and position.last_obv_slope and position.last_obv_slope < 0:
                alerts.append({
                    'type': 'OBV_FLIP',
                    'severity': 'HIGH',
                    'message': f'🔄 OBV flipped BULLISH (slope: +{obv_slope:,.0f}) - Smart money buying',
                    'recommendation': 'CONSIDER EXIT - Momentum against SHORT position'
                })
            elif obv_slope > 0:
                alerts.append({
                    'type': 'OBV_WARNING',
                    'severity': 'MEDIUM',
                    'message': f'📈 OBV rising (+{obv_slope:,.0f}) - Smart money accumulating',
                    'recommendation': 'Monitor closely - Trend against SHORT'
                })
        
        elif position.trade_type == 'LONG':
            distance_to_sl = entry_price - stop_loss
            
            if distance_to_sl <= 0:
                return alerts
            
            current_distance = entry_price - current_price
            threshold_price = entry_price - (distance_to_sl * 0.6)
            
            if current_price <= threshold_price:
                pct_to_sl = (current_distance / distance_to_sl) * 100
                alerts.append({
                    'type': 'DANGER_ZONE',
                    'severity': 'HIGH',
                    'message': f'⚠️ Price at ${current_price:,.2f} - {pct_to_sl:.0f}% to stop loss (${stop_loss:,.2f})',
                    'recommendation': 'CONSIDER EXIT - Approaching stop loss'
                })
            
            obv_slope = indicators.get('obv_slope', 0)
            if obv_slope < 0 and position.last_obv_slope and position.last_obv_slope > 0:
                alerts.append({
                    'type': 'OBV_FLIP',
                    'severity': 'HIGH',
                    'message': f'🔄 OBV flipped BEARISH (slope: {obv_slope:,.0f}) - Smart money selling',
                    'recommendation': 'CONSIDER EXIT - Momentum against LONG position'
                })
            elif obv_slope < 0:
                alerts.append({
                    'type': 'OBV_WARNING',
                    'severity': 'MEDIUM',
                    'message': f'📉 OBV falling ({obv_slope:,.0f}) - Smart money distributing',
                    'recommendation': 'Monitor closely - Trend against LONG'
                })
        
        return alerts
    
    def _check_momentum_timing(self, position, tech_indicators, timeframe):
        """
        EARLY WARNING SYSTEM - Checks momentum timing before profile deviation.
        Momentum timing changes FASTER than profile patterns, giving early heads-up.
        
        Uses 5-indicator momentum analysis (RSI multi-TF, KDJ, MACD, ADX, OBV)
        to detect when momentum is shifting against the position direction.
        """
        alerts = []
        
        try:
            # Convert timeframe string to minutes
            timeframe_map = {'5m': 5, '15m': 15, '30m': 30, '1H': 60, '4H': 240, '1D': 1440}
            tf_minutes = timeframe_map.get(timeframe, 60)
            
            # Get current momentum timing analysis
            momentum = tech_indicators.get_momentum_timing(tf_minutes)
            
            if not momentum or momentum.get('advisory') == 'Insufficient data for timing analysis':
                return alerts
            
            momentum_dir = momentum.get('momentum_direction', 'neutral')
            signals_aligned = momentum.get('signals_aligned', 0)
            est_candles = momentum.get('estimated_candles', 0)
            est_hours = momentum.get('estimated_hours', 0)
            
            # Format time display
            if est_hours >= 24:
                time_display = f"~{est_hours/24:.1f} days"
            elif est_hours >= 1:
                time_display = f"~{est_hours:.0f}h"
            else:
                time_display = f"~{est_hours*60:.0f}m"
            
            # Check for momentum conflict with position direction
            # Strong conflict: momentum_dir is opposite to position (3+ signals)
            # Weak conflict: early signs of opposition even if momentum_dir is neutral (2 signals)
            
            strong_conflict = False
            weak_conflict = False
            conflict_message = ""
            
            # Get RSI and KDJ alignment from momentum details
            rsi_alignment = momentum.get('rsi_alignment', 'neutral')
            kdj_dynamics = momentum.get('kdj_dynamics', 'neutral')
            
            # Get confirmation details for ADX and OBV divergence checks
            details = momentum.get('details', {})
            adx_confirms = details.get('adx_confirms', False)
            obv_confirms = details.get('obv_confirms', False)
            divergence_warning = details.get('divergence_warning')
            obv_flow = momentum.get('obv_flow', 'neutral')
            
            if position.trade_type == 'LONG':
                # Check for bearish signals against LONG position
                if momentum_dir == 'bearish':
                    strong_conflict = True
                    conflict_message = f"⚡ EARLY WARNING: Bearish momentum detected ({signals_aligned}/5 signals)"
                elif signals_aligned >= 2 and (
                    'bearish' in rsi_alignment or 
                    'bearish' in kdj_dynamics or
                    'reversal_down' in kdj_dynamics
                ):
                    weak_conflict = True
                    conflict_message = f"📊 Momentum weakening: Early bearish signals ({signals_aligned}/5)"
                    
            elif position.trade_type == 'SHORT':
                # Check for bullish signals against SHORT position
                if momentum_dir == 'bullish':
                    strong_conflict = True
                    conflict_message = f"⚡ EARLY WARNING: Bullish momentum detected ({signals_aligned}/5 signals)"
                elif signals_aligned >= 2 and (
                    'bullish' in rsi_alignment or 
                    'bullish' in kdj_dynamics or
                    'reversal_up' in kdj_dynamics
                ):
                    weak_conflict = True
                    conflict_message = f"📊 Momentum weakening: Early bullish signals ({signals_aligned}/5)"
            
            # ========== OBV DIVERGENCE WARNING ==========
            # Smart money divergence is critical - surface even if no direction conflict
            if divergence_warning:
                obv_warning_msg = ""
                if position.trade_type == 'LONG' and divergence_warning == 'bearish_divergence':
                    obv_warning_msg = f"🔴 OBV DIVERGENCE: Smart money selling while price rising"
                elif position.trade_type == 'SHORT' and divergence_warning == 'bullish_divergence':
                    obv_warning_msg = f"🟢 OBV DIVERGENCE: Smart money buying while price falling"
                
                if obv_warning_msg:
                    # OBV divergence = immediate LOW warning, regardless of other signals
                    alerts.append({
                        'type': 'MOMENTUM_DIVERGENCE',
                        'severity': 'LOW',
                        'message': obv_warning_msg,
                        'recommendation': 'Smart money flow conflicts with position - watch closely',
                        'obv_flow': obv_flow,
                        'divergence_type': divergence_warning
                    })
                    print(f"🔄 OBV Divergence Alert: {position.symbol} ({position.trade_type}) - {divergence_warning}")
            
            # ========== ADX DIRECTION CONFLICT WARNING ==========
            # ADX > 25 but DI direction opposes position = potential trend conflict
            adx_value = details.get('ADX') if details else None
            di_plus = details.get('di_plus', 0)
            di_minus = details.get('di_minus', 0)
            
            if adx_value and adx_value > 25:
                adx_direction_conflict = False
                if position.trade_type == 'LONG' and di_minus > di_plus:
                    adx_direction_conflict = True
                    adx_conflict_msg = f"⚠️ ADX Conflict: Strong trend ({adx_value:.0f}) but DI- > DI+ (bearish direction)"
                elif position.trade_type == 'SHORT' and di_plus > di_minus:
                    adx_direction_conflict = True
                    adx_conflict_msg = f"⚠️ ADX Conflict: Strong trend ({adx_value:.0f}) but DI+ > DI- (bullish direction)"
                
                if adx_direction_conflict:
                    alerts.append({
                        'type': 'MOMENTUM_ADX_CONFLICT',
                        'severity': 'LOW',
                        'message': adx_conflict_msg,
                        'recommendation': 'Trend direction conflicts with position',
                        'adx_value': adx_value,
                        'di_plus': di_plus,
                        'di_minus': di_minus
                    })
                    print(f"⚠️ ADX Direction Conflict: {position.symbol} ({position.trade_type}) - ADX={adx_value:.0f}, DI+={di_plus:.1f}, DI-={di_minus:.1f}")
            
            # ========== MAIN MOMENTUM CONFLICT ALERTS ==========
            if strong_conflict and signals_aligned >= 3:
                # Strong momentum conflict (3+ signals aligned against position)
                alerts.append({
                    'type': 'MOMENTUM_SHIFT',
                    'severity': 'EARLY_WARNING',
                    'message': f"{conflict_message} - may persist {est_candles:.0f} candles ({time_display})",
                    'recommendation': 'Monitor closely - momentum shifting against position',
                    'momentum_direction': momentum_dir,
                    'signals_aligned': signals_aligned,
                    'estimated_candles': est_candles,
                    'estimated_hours': est_hours
                })
                print(f"⚡ Momentum Early Warning: {position.symbol} ({position.trade_type}) - {momentum_dir} momentum ({signals_aligned}/5)")
                
            elif weak_conflict or (strong_conflict and signals_aligned == 2):
                # Weak momentum conflict (2 signals or early opposition detected)
                direction_text = momentum_dir if momentum_dir != 'neutral' else ('bearish' if position.trade_type == 'LONG' else 'bullish')
                alerts.append({
                    'type': 'MOMENTUM_WARNING',
                    'severity': 'LOW',
                    'message': conflict_message,
                    'recommendation': 'Watch for further momentum changes',
                    'momentum_direction': direction_text,
                    'signals_aligned': signals_aligned
                })
                print(f"📊 Momentum Warning: {position.symbol} ({position.trade_type}) - early {direction_text} signals ({signals_aligned}/5)")
            
            return alerts
            
        except Exception as e:
            print(f"Momentum timing check error: {e}")
            return alerts
    
    def _check_profile_deviation(self, position, current_indicators):
        """
        Three-Tier Profile-Based Risk Management:
        Tier 0: Quick difference check (exit early if no change detected)
        Tier 1: Deep similarity using M1's proven normalization pipeline
        Tier 2: Full M1 re-analysis if major deviation detected (similarity < 55%)
        """
        alerts = []
        
        # Check if we have entry snapshot
        if not position.indicators_snapshot:
            return alerts
        
        try:
            entry_indicators = position.indicators_snapshot
            
            # ========== TIER 0: Quick Difference Check ==========
            # Exit early if key indicators haven't changed significantly (>15%)
            no_significant_change = True
            for indicator in self.quick_check_indicators:
                entry_val = entry_indicators.get(indicator)
                current_val = current_indicators.get(indicator)
                
                if entry_val is not None and current_val is not None and entry_val != 0:
                    pct_change = abs((current_val - entry_val) / entry_val) * 100
                    if pct_change > 15:  # 15% change threshold
                        no_significant_change = False
                        break
            
            # Early exit: No significant changes detected
            if no_significant_change:
                return alerts  # Position stable - no deeper analysis needed
            
            # ========== TIER 1: Deep Profile Similarity (M1's Proven Method) ==========
            if not self.ml_engine or not self.ml_engine.scaler_fitted:
                # ML Engine not available - skip Tier 1 & 2
                print("⚠️  ML Engine not available for profile similarity - using basic monitoring only")
                return alerts
            
            # Use M1's proven normalization pipeline
            try:
                # Prepare and normalize entry profile
                entry_features = self.ml_engine.prepare_features(entry_indicators, trade_type=None)
                entry_features = entry_features.flatten()[1:]  # Skip trade_type feature
                entry_normalized = self.ml_engine.scaler.transform(entry_features.reshape(1, -1)).flatten()
                
                # Prepare and normalize current profile
                current_features = self.ml_engine.prepare_features(current_indicators, trade_type=None)
                current_features = current_features.flatten()[1:]  # Skip trade_type feature
                current_normalized = self.ml_engine.scaler.transform(current_features.reshape(1, -1)).flatten()
                
                # Calculate cosine similarity between normalized profiles
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(
                    entry_normalized.reshape(1, -1),
                    current_normalized.reshape(1, -1)
                )[0][0]
                
                similarity_pct = similarity * 100
                print(f"📊 Profile Similarity (Entry vs Current): {similarity_pct:.1f}%")
                
                # Tier 1 Decision Thresholds
                if similarity_pct >= 75:
                    # High similarity - position stable
                    return alerts
                
                elif similarity_pct >= 55:
                    # Medium similarity - pattern weakening
                    alerts.append({
                        'type': 'PATTERN_WEAKENING',
                        'severity': 'MEDIUM',
                        'message': f"⚠️ Entry pattern weakening (similarity: {similarity_pct:.1f}%)",
                        'recommendation': 'Monitor closely - Profile diverging from entry'
                    })
                    return alerts  # Don't proceed to Tier 2 yet
                
                # ========== TIER 2: Full M1 Re-Analysis (Major Deviation) ==========
                # Similarity < 55% - Major deviation detected
                print(f"🔍 Major profile deviation ({similarity_pct:.1f}%) - Running full M1 re-analysis...")
                
                # Run full M1 prediction on current market conditions
                m1_result = self.ml_engine.predict(current_indicators)
                
                if m1_result:
                    m1_signal = m1_result.get('signal', 'HOLD')
                    m1_confidence = m1_result.get('confidence', 0)
                    
                    # Check if M1 recommendation reversed (opposite of entry direction)
                    if position.trade_type == 'LONG' and m1_signal == 'SHORT':
                        alerts.append({
                            'type': 'PATTERN_REVERSED',
                            'severity': 'HIGH',
                            'message': f"🔴 PATTERN REVERSED - M1 now recommends SHORT ({m1_confidence:.1f}% confidence)",
                            'recommendation': 'EXIT NOW - Market conditions completely reversed'
                        })
                    elif position.trade_type == 'SHORT' and m1_signal == 'LONG':
                        alerts.append({
                            'type': 'PATTERN_REVERSED',
                            'severity': 'HIGH',
                            'message': f"🔴 PATTERN REVERSED - M1 now recommends LONG ({m1_confidence:.1f}% confidence)",
                            'recommendation': 'EXIT NOW - Market conditions completely reversed'
                        })
                    else:
                        # M1 still agrees with direction, but profile has deviated significantly
                        alerts.append({
                            'type': 'PATTERN_WEAKENING',
                            'severity': 'MEDIUM',
                            'message': f"⚠️ Profile deviated ({similarity_pct:.1f}%) but M1 still {m1_signal} ({m1_confidence:.1f}%)",
                            'recommendation': 'Consider tightening stops - Conditions changed but direction holds'
                        })
                
            except Exception as e:
                print(f"Error in Tier 1/2 analysis: {e}")
                import traceback
                traceback.print_exc()
            
            return alerts
            
        except Exception as e:
            print(f"Profile comparison error: {e}")
            return alerts
    
    def _generate_recommendation(self, position, current_price, pnl_percentage, 
                                 signals, support_levels, resistance_levels,
                                 whale_movements, smart_money, monitoring_alerts=None):
        
        reasons = []
        
        # PRIORITY 1: Check HIGH severity monitoring alerts (approaching stop loss, OBV flip)
        if monitoring_alerts:
            high_severity_alerts = [alert for alert in monitoring_alerts if alert.get('severity') == 'HIGH']
            if high_severity_alerts:
                alert_messages = [alert.get('message', '') for alert in high_severity_alerts]
                return {
                    'action': 'EXIT',
                    'reason': f"⚠️ RISK ALERT: {'; '.join(alert_messages)}"
                }
        
        # PRIORITY 2: Check if stop loss actually hit
        if position.stop_loss:
            if position.trade_type == 'LONG' and current_price <= position.stop_loss:
                return {
                    'action': 'EXIT',
                    'reason': 'Stop loss hit'
                }
            elif position.trade_type == 'SHORT' and current_price >= position.stop_loss:
                return {
                    'action': 'EXIT',
                    'reason': 'Stop loss hit'
                }
        
        if position.take_profit:
            if position.trade_type == 'LONG' and current_price >= position.take_profit:
                return {
                    'action': 'EXIT',
                    'reason': 'Take profit target reached'
                }
            elif position.trade_type == 'SHORT' and current_price <= position.take_profit:
                return {
                    'action': 'EXIT',
                    'reason': 'Take profit target reached'
                }
        
        bearish_count = 0
        bullish_count = 0
        
        for indicator, signal in signals.items():
            if signal in ['bearish', 'overbought', 'strong_downtrend']:
                bearish_count += 1
            elif signal in ['bullish', 'oversold', 'strong_uptrend']:
                bullish_count += 1
        
        if position.trade_type == 'LONG':
            if bearish_count >= 4:
                reasons.append(f"Bearish signals detected ({bearish_count} indicators)")
                return {
                    'action': 'EXIT',
                    'reason': ', '.join(reasons)
                }
            
            if resistance_levels and current_price >= resistance_levels[0] * 0.98:
                reasons.append(f"Approaching resistance at {resistance_levels[0]}")
            
            if pnl_percentage > 5:
                reasons.append(f"Profit target reached: +{pnl_percentage:.2f}%")
                return {
                    'action': 'EXIT',
                    'reason': ', '.join(reasons)
                }
        
        elif position.trade_type == 'SHORT':
            if bullish_count >= 4:
                reasons.append(f"Bullish signals detected ({bullish_count} indicators)")
                return {
                    'action': 'EXIT',
                    'reason': ', '.join(reasons)
                }
            
            if support_levels and current_price <= support_levels[0] * 1.02:
                reasons.append(f"Approaching support at {support_levels[0]}")
            
            if pnl_percentage > 5:
                reasons.append(f"Profit target reached: +{pnl_percentage:.2f}%")
                return {
                    'action': 'EXIT',
                    'reason': ', '.join(reasons)
                }
        
        if smart_money:
            latest_signal = smart_money[0]
            if position.trade_type == 'LONG' and latest_signal['type'] == 'distribution':
                reasons.append("Smart money distribution detected")
                return {
                    'action': 'EXIT',
                    'reason': ', '.join(reasons)
                }
            elif position.trade_type == 'SHORT' and latest_signal['type'] == 'accumulation':
                reasons.append("Smart money accumulation detected")
                return {
                    'action': 'EXIT',
                    'reason': ', '.join(reasons)
                }
        
        if pnl_percentage < -3:
            reasons.append(f"Position in loss: {pnl_percentage:.2f}%")
            
            if position.trade_type == 'LONG' and bearish_count > bullish_count:
                return {
                    'action': 'EXIT',
                    'reason': 'Cutting losses - trend against position'
                }
            elif position.trade_type == 'SHORT' and bullish_count > bearish_count:
                return {
                    'action': 'EXIT',
                    'reason': 'Cutting losses - trend against position'
                }
        
        # HOLD - Position stable (no exit conditions triggered)
        # Three-tier profile monitoring provides advanced pattern analysis
        return {
            'action': 'HOLD',
            'reason': 'Position looks healthy - monitoring continues'
        }
    
    def add_position(self, symbol, market_type, trade_type, entry_price, 
                     quantity=None, stop_loss=None, take_profit=None, timeframe='1H', indicators=None, m2_entry_quality=None):
        session = get_session()
        
        try:
            position = ActivePosition(
                symbol=symbol,
                market_type=market_type,
                trade_type=trade_type,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timeframe=timeframe,
                entry_time=datetime.utcnow(),
                current_price=entry_price,
                is_active=True,
                indicators_snapshot=indicators,
                m2_entry_quality=m2_entry_quality
            )
            
            session.add(position)
            session.commit()
            
            return {
                'success': True,
                'message': f'Position added for {symbol}',
                'position_id': position.id
            }
            
        except Exception as e:
            session.rollback()
            return {
                'success': False,
                'message': f'Error adding position: {str(e)}'
            }
        finally:
            session.close()
    
    def close_position(self, position_id, exit_price, outcome, exit_type=None, notes=None):
        session = get_session()
        
        try:
            print(f"\n🔒 CLOSING POSITION - ID: {position_id}")
            
            position = session.query(ActivePosition).filter(
                ActivePosition.id == position_id,
                ActivePosition.is_active == True
            ).first()
            
            if not position:
                print(f"❌ No active position found with ID {position_id}")
                return {
                    'success': False,
                    'message': f'No active position found with ID {position_id}'
                }
            
            print(f"✅ Found position: {position.symbol} ({position.trade_type} @ ${position.entry_price})")
            print(f"   Position ID: {position.id}, Is Active: {position.is_active}")
            
            position.is_active = False
            
            from database import Trade
            
            # Extract consolidation score from indicators for M2 learning
            consolidation_score = None
            if position.indicators_snapshot and isinstance(position.indicators_snapshot, dict):
                consolidation_data = position.indicators_snapshot.get('consolidation', {})
                if consolidation_data:
                    consolidation_score = consolidation_data.get('consolidation_score')
            
            trade = Trade(
                symbol=position.symbol,
                market_type=position.market_type,
                trade_type=position.trade_type,
                entry_price=position.entry_price,
                exit_price=exit_price,
                entry_time=position.entry_time,
                exit_time=datetime.utcnow(),
                quantity=position.quantity,
                outcome=outcome,
                exit_type=exit_type,
                notes=notes,
                indicators_at_entry=position.indicators_snapshot,
                m2_entry_quality=position.m2_entry_quality
            )
            
            # Set consolidation_score if Trade model has the column
            if consolidation_score is not None:
                try:
                    trade.consolidation_score = consolidation_score
                    print(f"📊 Consolidation score at entry: {consolidation_score}/100")
                except AttributeError:
                    pass  # Column not yet added to model
            
            # Calculate P&L percentage (works even without quantity)
            if position.trade_type == 'LONG':
                price_diff = exit_price - position.entry_price
            else:  # SHORT
                price_diff = position.entry_price - exit_price
            
            trade.profit_loss_percentage = (price_diff / position.entry_price) * 100
            
            # Calculate absolute P&L
            if position.quantity and position.quantity > 0:
                # User provided quantity - calculate real dollar P&L
                trade.profit_loss = price_diff * position.quantity
                trade.notes = notes
            else:
                # No quantity - use NORMALIZED P&L for fair comparison across different assets
                # Assume hypothetical $10,000 position size for consistent analytics
                hypothetical_position_size = 10000
                hypothetical_quantity = hypothetical_position_size / position.entry_price
                trade.profit_loss = price_diff * hypothetical_quantity
                
                # Add note that this is normalized for transparency (preserve any existing notes)
                note_prefix = "[Normalized P&L: $10k position] "
                if notes:
                    trade.notes = note_prefix + notes
                else:
                    trade.notes = note_prefix.strip()
            
            session.add(trade)
            session.commit()
            
            print(f"✅ Position {position.id} closed successfully: {position.symbol}")
            print(f"   Trade created with ID: {trade.id}")
            
            # ========== PREDICTION VS OUTCOME LOGGING (Console Only) ==========
            # Log M2 prediction accuracy to console - no database writes to avoid affecting trade closure
            m2_score = position.m2_entry_quality
            if m2_score is not None:
                m2_predicted_good = m2_score >= 50  # M2 >= 50% = predicted good entry
                actual_win = outcome.lower() == 'win'
                # M2 is CORRECT if: (predicted good AND won) OR (predicted poor AND lost)
                m2_correct = (m2_predicted_good and actual_win) or (not m2_predicted_good and not actual_win)
                
                print(f"📊 PREDICTION VS OUTCOME:")
                print(f"   M2 Entry Quality: {m2_score:.1f}% ({'Good' if m2_predicted_good else 'Poor'} entry predicted)")
                print(f"   Actual Outcome: {outcome.upper()}")
                print(f"   M2 Prediction: {'✅ CORRECT' if m2_correct else '❌ INCORRECT'}")
            # ========== END PREDICTION VS OUTCOME LOGGING ==========
            
            print(f"   Learning from trade...")
            
            from ml_engine import MLTradingEngine
            ml_engine = MLTradingEngine()
            ml_engine.learn_from_trade(trade.id)
            
            print(f"✅ ML learning complete for trade {trade.id}\n")
            
            return {
                'success': True,
                'message': f'Position closed for {position.symbol}',
                'trade_id': trade.id
            }
            
        except Exception as e:
            session.rollback()
            return {
                'success': False,
                'message': f'Error closing position: {str(e)}'
            }
        finally:
            session.close()
    
    def update_entry_price(self, symbol, new_entry_price, old_entry_price=None):
        """Update the entry price of an active position"""
        session = get_session()
        
        try:
            query = session.query(ActivePosition).filter(
                ActivePosition.symbol == symbol,
                ActivePosition.is_active == True
            )
            
            if old_entry_price is not None:
                query = query.filter(ActivePosition.entry_price == old_entry_price)
            
            position = query.first()
            
            if not position:
                return {
                    'success': False,
                    'message': f'No active position found for {symbol}'
                }
            
            old_entry = position.entry_price
            position.entry_price = new_entry_price
            
            session.commit()
            
            return {
                'success': True,
                'message': f'Entry price updated from ${old_entry:,.2f} to ${new_entry_price:,.2f}',
                'old_price': old_entry,
                'new_price': new_entry_price
            }
            
        except Exception as e:
            session.rollback()
            return {
                'success': False,
                'message': f'Error updating entry price: {str(e)}'
            }
        finally:
            session.close()
