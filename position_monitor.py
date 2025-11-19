from datetime import datetime, timedelta
import time
from database import get_session, ActivePosition, MarketData
from api_integrations import get_market_data_unified, get_current_price
from technical_indicators import TechnicalIndicators, calculate_support_resistance
from whale_tracker import WhaleTracker
from divergence_resolver import resolve_active_divergences
import json
import numpy as np

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
            tech_indicators.calculate_all_indicators()
            indicators = tech_indicators.get_latest_indicators()
            signals = tech_indicators.get_trend_signals()
            
            # Calculate support/resistance and trend context FIRST
            support_levels, resistance_levels = calculate_support_resistance(df)
            trend_context = tech_indicators.get_trend_context(position.symbol, position.market_type)
            
            # Enrich indicators with S/R and trend context (same as market analysis)
            indicators['support_levels'] = support_levels
            indicators['resistance_levels'] = resistance_levels
            indicators['trend_context'] = trend_context
            
            monitoring_alerts = self._check_tight_monitoring(
                position, current_price, indicators, signals
            )
            
            # Add profile comparison alerts (early warning system)
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
