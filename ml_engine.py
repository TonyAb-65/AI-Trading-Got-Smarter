import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import os
import base64
from io import BytesIO
from datetime import datetime
from database import get_session, Trade, ModelPerformance, IndicatorPerformance, MLModel

class MLTradingEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        self.feature_columns = []
        self.m2_model = None  # Meta-labeling model for entry quality
        
        # Cache profiles to avoid re-fitting scaler on every predict()
        self.cached_profiles = None
        self.profiles_trade_count = 0
        
        self.indicator_weights = {
            'RSI': 2.0,
            'MACD': 2.0,
            'ADX': 2.0,
            'Stoch_K': 1.0,
            'MFI': 1.0,
            'CCI': 1.0,
            'SMA': 1.0
        }
        
        self._load_indicator_weights()
        self._load_m2_model()  # Load M2 model if exists
    
    def calculate_safe_tp(self, entry_price, atr_tp, signal, support_levels, resistance_levels):
        """
        Calculate safe Take Profit that respects support/resistance levels.
        Places TP BEFORE key levels to avoid bounce-reversal scenarios.
        
        Args:
            entry_price: Entry price
            atr_tp: ATR-based TP target (momentum-based)
            signal: 'LONG' or 'SHORT'
            support_levels: List of support prices below current price
            resistance_levels: List of resistance prices above current price
        
        Returns:
            Adjusted TP price that stops before S/R levels
        """
        # Safety buffer: stop 0.3% before S/R level to ensure fill
        buffer_pct = 0.003
        
        if signal == 'LONG':
            # For LONG: Check if any resistance between entry and ATR target
            if resistance_levels:
                for resistance in sorted(resistance_levels):
                    # If resistance is between entry and target
                    if entry_price < resistance < atr_tp:
                        # Place TP just before resistance
                        safe_tp = resistance * (1 - buffer_pct)
                        print(f"📊 TP adjusted: {atr_tp:.2f} → {safe_tp:.2f} (stops before resistance at {resistance:.2f})")
                        return safe_tp
            # No resistance in the way, use ATR target
            return atr_tp
        
        elif signal == 'SHORT':
            # For SHORT: Check if any support between ATR target and entry
            if support_levels:
                for support in sorted(support_levels, reverse=True):
                    # If support is between target and entry
                    if atr_tp < support < entry_price:
                        # Place TP just before support
                        safe_tp = support * (1 + buffer_pct)
                        print(f"📊 TP adjusted: {atr_tp:.2f} → {safe_tp:.2f} (stops before support at {support:.2f})")
                        return safe_tp
            # No support in the way, use ATR target
            return atr_tp
        
        return atr_tp
    
    def get_adaptive_ml_weight(self):
        """
        Calculate adaptive ML weight based on trade count.
        As system learns from more trades, ML's learned patterns should dominate over hardcoded rules.
        
        Returns:
            float: ML weight (0.0 to 1.0)
                0-10 trades: 0.2 (20% ML, 80% rules - bootstrap phase)
                10-30 trades: 0.5 (50% ML, 50% rules - learning phase)
                30+ trades: 0.8 (80% ML, 20% rules - mature phase)
        """
        session = get_session()
        try:
            trade_count = session.query(Trade).filter(
                Trade.exit_price.isnot(None),
                Trade.outcome.isnot(None),
                Trade.indicators_at_entry.isnot(None)
            ).count()
            
            if trade_count < 10:
                ml_weight = 0.2
                phase = "Bootstrap"
            elif trade_count < 30:
                ml_weight = 0.5
                phase = "Learning"
            else:
                ml_weight = 0.8
                phase = "Mature"
            
            print(f"🧠 Adaptive Learning: {trade_count} trades → {ml_weight*100:.0f}% ML, {(1-ml_weight)*100:.0f}% Rules ({phase} phase)")
            return ml_weight
            
        except Exception as e:
            print(f"Error calculating adaptive weight: {e}")
            return 0.5
        finally:
            session.close()
    
    def prepare_features(self, indicators, trade_type=None):
        """
        Prepare features for ML models - ORIGINAL VERSION (no volatility features)
        """
        features = []
        feature_names = [
            'RSI', 'MACD', 'MACD_hist', 'Stoch_K', 'Stoch_D',
            'MFI', 'CCI', 'ADX', 'DI_plus', 'DI_minus',
            'current_price', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'volume', 'Volume_SMA'
        ]
        
        # Add trade direction as first feature (1.0 for LONG, -1.0 for SHORT, 0.0 for unknown)
        if trade_type == 'LONG':
            features.append(1.0)
        elif trade_type == 'SHORT':
            features.append(-1.0)
        else:
            features.append(0.0)
        
        for feat in feature_names:
            val = indicators.get(feat)
            if val is not None and not pd.isna(val):
                features.append(float(val))
            else:
                features.append(0.0)
        
        if indicators.get('current_price') and indicators.get('SMA_20'):
            features.append((indicators['current_price'] - indicators['SMA_20']) / indicators['SMA_20'])
        else:
            features.append(0.0)
        
        if indicators.get('current_price') and indicators.get('SMA_50'):
            features.append((indicators['current_price'] - indicators['SMA_50']) / indicators['SMA_50'])
        else:
            features.append(0.0)
        
        if indicators.get('MACD') and indicators.get('MACD_signal'):
            features.append(indicators['MACD'] - indicators['MACD_signal'])
        else:
            features.append(0.0)
        
        if indicators.get('volume') and indicators.get('Volume_SMA') and indicators['Volume_SMA'] > 0:
            features.append(indicators['volume'] / indicators['Volume_SMA'])
        else:
            features.append(1.0)
        
        # Add trend context features (duration, slope, divergence)
        trend_context = indicators.get('trend_context', {})
        
        # RSI trend features
        rsi_ctx = trend_context.get('RSI', {})
        features.append(float(rsi_ctx.get('duration_candles', 0)))
        features.append(float(rsi_ctx.get('slope', 0.0)))
        divergence = rsi_ctx.get('divergence', 'none')
        features.append(1.0 if divergence == 'bullish' else -1.0 if divergence == 'bearish' else 0.0)
        
        # Stochastic trend features
        stoch_ctx = trend_context.get('Stochastic', {})
        features.append(float(stoch_ctx.get('duration_candles', 0)))
        features.append(float(stoch_ctx.get('slope', 0.0)))
        divergence = stoch_ctx.get('divergence', 'none')
        features.append(1.0 if divergence == 'bullish' else -1.0 if divergence == 'bearish' else 0.0)
        
        # MFI trend features
        mfi_ctx = trend_context.get('MFI', {})
        features.append(float(mfi_ctx.get('duration_candles', 0)))
        features.append(float(mfi_ctx.get('slope', 0.0)))
        divergence = mfi_ctx.get('divergence', 'none')
        features.append(1.0 if divergence == 'bullish' else -1.0 if divergence == 'bearish' else 0.0)
        
        # OBV trend features (volume-based, no duration tracking)
        obv_ctx = trend_context.get('OBV', {})
        features.append(float(obv_ctx.get('slope', 0.0)))
        divergence = obv_ctx.get('divergence', 'none')
        features.append(1.0 if divergence == 'bullish' else -1.0 if divergence == 'bearish' else 0.0)
        
        # Support/Resistance features (NEW - helps ML understand price levels)
        current_price = indicators.get('current_price', 0)
        support_levels = indicators.get('support_levels', [])
        resistance_levels = indicators.get('resistance_levels', [])
        
        # Distance to nearest support/resistance (as % of price)
        nearest_support = support_levels[0] if support_levels else current_price * 0.98
        nearest_resistance = resistance_levels[0] if resistance_levels else current_price * 1.02
        
        support_distance_pct = ((current_price - nearest_support) / current_price * 100) if current_price > 0 else 0
        resistance_distance_pct = ((nearest_resistance - current_price) / current_price * 100) if current_price > 0 else 0
        
        features.append(float(support_distance_pct))
        features.append(float(resistance_distance_pct))
        
        # At support/resistance zone (1 if within 1% of level, else 0)
        at_support = 1.0 if support_distance_pct < 1.0 else 0.0
        at_resistance = 1.0 if resistance_distance_pct < 1.0 else 0.0
        
        features.append(at_support)
        features.append(at_resistance)
        
        # Number of S/R levels (strength indicator)
        support_strength = float(len(support_levels))
        resistance_strength = float(len(resistance_levels))
        
        features.append(support_strength)
        features.append(resistance_strength)
        
        # Divergence Timing Intelligence Features (NEW - helps M2 assess entry timing)
        from divergence_analytics import get_divergence_timing_info
        from divergence_resolver import TIMEFRAME_MINUTES
        
        # Check for any active divergence in indicators
        has_divergence = 0.0
        timing_candles_elapsed = 0.0
        timing_avg_resolution = 0.0
        timing_speed_encoded = 0.0  # -1=fast, 0=actionable, 1=slow
        timing_success_rate = 0.0
        
        timeframe = indicators.get('timeframe', '1H')
        detected_at = indicators.get('divergence_detected_at')  # Timestamp when divergence was detected
        
        # Check each indicator for divergences
        for ind_name in ['RSI', 'MFI', 'Stochastic', 'OBV']:
            ind_ctx = trend_context.get(ind_name, {})
            div = ind_ctx.get('divergence', 'none')
            
            if div != 'none':
                has_divergence = 1.0
                
                # Calculate candles elapsed if we have detection time
                if detected_at:
                    try:
                        from datetime import datetime
                        if isinstance(detected_at, str):
                            detected_time = datetime.fromisoformat(detected_at.replace('Z', '+00:00'))
                        else:
                            detected_time = detected_at
                        
                        time_elapsed = datetime.utcnow() - detected_time
                        timeframe_minutes = TIMEFRAME_MINUTES.get(timeframe, 60)
                        timing_candles_elapsed = time_elapsed.total_seconds() / 60 / timeframe_minutes
                    except:
                        timing_candles_elapsed = 0.0
                
                # Get historical timing intelligence for this divergence pattern
                timing_info = get_divergence_timing_info(ind_name, timeframe, div)
                if timing_info:
                    timing_avg_resolution = timing_info['avg_candles']
                    timing_success_rate = timing_info['success_rate'] / 100.0  # Normalize to 0-1
                    
                    # Encode speed class
                    if timing_info['speed_class'] == 'fast':
                        timing_speed_encoded = -1.0
                    elif timing_info['speed_class'] == 'actionable':
                        timing_speed_encoded = 0.0
                    elif timing_info['speed_class'] == 'slow':
                        timing_speed_encoded = 1.0
                
                break  # Use first divergence found
        
        features.append(has_divergence)
        features.append(float(timing_candles_elapsed))
        features.append(float(timing_avg_resolution))
        features.append(timing_speed_encoded)
        features.append(timing_success_rate)
        
        # ========== Momentum Timing Features (NEW - helps M2 assess entry timing) ==========
        # Multi-timeframe RSI values for momentum analysis
        rsi_6 = indicators.get('RSI_6')
        rsi_12 = indicators.get('RSI_12')
        rsi_24 = indicators.get('RSI_24')
        stoch_j = indicators.get('Stoch_J')
        
        features.append(float(rsi_6) if rsi_6 is not None else 50.0)
        features.append(float(rsi_12) if rsi_12 is not None else 50.0)
        features.append(float(rsi_24) if rsi_24 is not None else 50.0)
        features.append(float(stoch_j) if stoch_j is not None else 50.0)
        
        # RSI alignment feature: RSI_6 > RSI_12 > RSI_24 = bullish accelerating
        rsi_6_val = rsi_6 if rsi_6 is not None else 50.0
        rsi_12_val = rsi_12 if rsi_12 is not None else 50.0
        rsi_24_val = rsi_24 if rsi_24 is not None else 50.0
        
        if rsi_6_val > rsi_12_val > rsi_24_val:
            rsi_alignment_encoded = 1.0  # Bullish accelerating
        elif rsi_6_val < rsi_12_val < rsi_24_val:
            rsi_alignment_encoded = -1.0  # Bearish accelerating
        elif rsi_6_val < rsi_12_val and rsi_6_val > 50:
            rsi_alignment_encoded = 0.5  # Bullish weakening
        elif rsi_6_val > rsi_12_val and rsi_6_val < 50:
            rsi_alignment_encoded = -0.5  # Bearish weakening
        else:
            rsi_alignment_encoded = 0.0  # Neutral
        
        features.append(rsi_alignment_encoded)
        
        # KDJ dynamics feature: J > K > D = bullish momentum
        stoch_k = indicators.get('Stoch_K', 50.0)
        stoch_d = indicators.get('Stoch_D', 50.0)
        stoch_j_val = stoch_j if stoch_j is not None else 50.0
        
        if stoch_j_val > 100:
            kdj_dynamics_encoded = -0.8  # J peaked - reversal down likely
        elif stoch_j_val < 0:
            kdj_dynamics_encoded = 0.8  # J bottomed - reversal up likely
        elif stoch_j_val > stoch_k > stoch_d:
            kdj_dynamics_encoded = 1.0  # Bullish aligned
        elif stoch_j_val < stoch_k < stoch_d:
            kdj_dynamics_encoded = -1.0  # Bearish aligned
        else:
            kdj_dynamics_encoded = 0.0  # Neutral
        
        features.append(kdj_dynamics_encoded)
        
        # Momentum timing from pre-calculated analysis (if available)
        momentum_timing = indicators.get('momentum_timing', {})
        
        # Momentum direction: bullish=1, bearish=-1, reversal_imminent=0.5/-0.5, mixed=0
        momentum_dir = momentum_timing.get('momentum_direction', 'neutral')
        if momentum_dir == 'bullish':
            momentum_dir_encoded = 1.0
        elif momentum_dir == 'bearish':
            momentum_dir_encoded = -1.0
        elif momentum_dir == 'reversal_imminent':
            momentum_dir_encoded = 0.0  # Reversal = good entry point
        else:
            momentum_dir_encoded = 0.0
        
        features.append(momentum_dir_encoded)
        
        # Estimated candles before reversal (normalized: divide by 10 to keep scale reasonable)
        est_candles = momentum_timing.get('estimated_candles', 0)
        features.append(float(est_candles) / 10.0)
        
        # Timing confidence (0-1 scale)
        timing_confidence = momentum_timing.get('timing_confidence', 0.5)
        features.append(float(timing_confidence))
        
        self.feature_columns = ['trade_type'] + feature_names + [
            'price_vs_sma20', 'price_vs_sma50', 'macd_divergence', 'volume_ratio',
            'rsi_duration', 'rsi_slope', 'rsi_divergence',
            'stoch_duration', 'stoch_slope', 'stoch_divergence',
            'mfi_duration', 'mfi_slope', 'mfi_divergence',
            'obv_slope', 'obv_divergence',
            'support_distance_pct', 'resistance_distance_pct',
            'at_support_zone', 'at_resistance_zone',
            'support_strength', 'resistance_strength',
            'div_has_active', 'div_candles_elapsed', 'div_avg_resolution',
            'div_speed_class', 'div_success_rate',
            'mt_rsi_6', 'mt_rsi_12', 'mt_rsi_24', 'mt_stoch_j',
            'mt_rsi_alignment', 'mt_kdj_dynamics', 'mt_momentum_dir',
            'mt_est_candles', 'mt_timing_confidence'
        ]
        
        return np.array(features).reshape(1, -1)
    
    def classify_volatility_regime(self, indicators):
        """
        Classify current market volatility regime based on multiple volatility metrics.
        
        Returns:
            tuple: (regime_label, regime_score)
                regime_label: 'LOW', 'MEDIUM', 'HIGH', or 'EXTREME'
                regime_score: Numerical score 0-100
        """
        atr_percentile = indicators.get('ATR_percentile', 50.0)
        bb_width_pct = indicators.get('BB_width_pct', 0.0)
        variance_14 = indicators.get('variance_14', 0.0)
        wick_ratio = indicators.get('wick_to_body_ratio', 1.0)
        
        # Primary classifier: ATR Percentile (most reliable)
        # Secondary factors: BB width, variance, wicks
        
        # Calculate composite volatility score (0-100)
        score = atr_percentile * 0.5  # 50% weight on ATR percentile
        
        # BB width contribution (normalize to 0-100 scale, 10% price range = 100)
        bb_contribution = min(bb_width_pct / 10.0 * 100, 100) * 0.2  # 20% weight
        score += bb_contribution
        
        # Variance contribution (normalize to 0-100 scale)
        variance_contribution = min(variance_14 * 10, 100) * 0.2  # 20% weight
        score += variance_contribution
        
        # Wick ratio contribution (high wicks = panic/volatility)
        # Normal wick ratio is 1-2, extreme is >5
        wick_contribution = min((wick_ratio - 1.0) / 4.0 * 100, 100) * 0.1  # 10% weight
        score += wick_contribution
        
        # Classify into regime based on score
        if score < 30:
            regime = 'LOW'
        elif score < 55:
            regime = 'MEDIUM'
        elif score < 75:
            regime = 'HIGH'
        else:
            regime = 'EXTREME'
        
        return regime, min(score, 100.0)
    
    def build_trade_profiles(self):
        """
        Build average indicator profiles from closed trades with feature normalization:
        - Winning LONG profile
        - Winning SHORT profile
        - Losing LONG profile
        - Losing SHORT profile
        """
        session = get_session()
        
        try:
            trades = session.query(Trade).filter(
                Trade.exit_price.isnot(None),
                Trade.outcome.isnot(None),
                Trade.indicators_at_entry.isnot(None)
            ).all()
            
            if len(trades) < 5:
                print(f"Not enough trades to build profiles. Need at least 5, have {len(trades)}")
                return None
            
            # STEP 1: Collect ALL features from ALL trades to fit scaler
            all_features = []
            trade_features_map = {}
            
            for trade in trades:
                try:
                    features = self.prepare_features(trade.indicators_at_entry, trade_type=None)
                    features_flat = features.flatten()[1:]  # Skip trade_type feature
                    all_features.append(features_flat)
                    trade_features_map[trade.id] = features_flat
                except Exception as feature_error:
                    print(f"⚠️  Skipping trade {trade.id} - error extracting features: {feature_error}")
                    continue
            
            if len(all_features) < 5:
                print(f"Not enough valid feature sets. Need at least 5, have {len(all_features)}")
                return None
            
            # STEP 2: Fit scaler on ALL features (ensures consistent normalization)
            all_features_array = np.array(all_features)
            self.scaler.fit(all_features_array)
            self.scaler_fitted = True
            print(f"✅ Scaler fitted on {len(all_features)} trade feature sets")
            
            # STEP 3: Separate trades into 4 categories
            win_long_trades = [t for t in trades if t.outcome.upper() == 'WIN' and t.trade_type == 'LONG']
            win_short_trades = [t for t in trades if t.outcome.upper() == 'WIN' and t.trade_type == 'SHORT']
            loss_long_trades = [t for t in trades if t.outcome.upper() == 'LOSS' and t.trade_type == 'LONG']
            loss_short_trades = [t for t in trades if t.outcome.upper() == 'LOSS' and t.trade_type == 'SHORT']
            
            print(f"📊 Building normalized profiles from {len(trades)} trades:")
            print(f"   Winning LONG: {len(win_long_trades)}, Winning SHORT: {len(win_short_trades)}")
            print(f"   Losing LONG: {len(loss_long_trades)}, Losing SHORT: {len(loss_short_trades)}")
            
            # STEP 4: Build normalized average profile for each category
            profiles = {}
            
            for category, category_trades in [
                ('win_long', win_long_trades),
                ('win_short', win_short_trades),
                ('loss_long', loss_long_trades),
                ('loss_short', loss_short_trades)
            ]:
                if len(category_trades) > 0:
                    # Get normalized features for this category
                    normalized_features_list = []
                    for trade in category_trades:
                        if trade.id in trade_features_map:
                            # Transform using fitted scaler
                            features_normalized = self.scaler.transform(trade_features_map[trade.id].reshape(1, -1))
                            normalized_features_list.append(features_normalized.flatten())
                    
                    # Calculate average normalized profile
                    if len(normalized_features_list) > 0:
                        profiles[category] = np.mean(normalized_features_list, axis=0)
                        print(f"   ✅ {category}: {len(normalized_features_list)} normalized trades")
                    else:
                        profiles[category] = None
                        print(f"   ⚠️  {category}: No valid feature data")
                else:
                    profiles[category] = None
                    print(f"   ⚠️  {category}: No trades in category")
            
            return profiles
            
        except Exception as e:
            print(f"❌ Error building profiles: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            session.close()
    
    def calculate_similarity(self, current_indicators, profile):
        """Calculate cosine similarity between normalized current indicators and a profile"""
        if profile is None:
            return 0.0
        
        try:
            # Prepare current features (without trade_type)
            current_features = self.prepare_features(current_indicators, trade_type=None)
            current_features = current_features.flatten()[1:]  # Skip trade_type feature
            
            # Normalize current features using fitted scaler
            if self.scaler_fitted:
                current_features = self.scaler.transform(current_features.reshape(1, -1)).flatten()
            else:
                print("⚠️  Scaler not fitted - using raw features (may be inaccurate)")
            
            # Reshape for cosine_similarity
            current_features = current_features.reshape(1, -1)
            profile = profile.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(current_features, profile)[0][0]
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def train_meta_model(self, min_trades=10):
        """
        Train M2 meta-labeling model to assess entry quality.
        Learns which predicted trades are worth taking based on full indicator profile.
        
        Returns:
            True if training successful, False otherwise
        """
        session = get_session()
        
        try:
            # Get closed trades with full indicator data
            trades = session.query(Trade).filter(
                Trade.exit_price.isnot(None),
                Trade.outcome.isnot(None),
                Trade.indicators_at_entry.isnot(None)
            ).all()
            
            if len(trades) < min_trades:
                print(f"⚠️  Not enough trades for M2 training. Need {min_trades}, have {len(trades)}")
                self.m2_model = None
                return False
            
            # Prepare training data
            X_train = []
            y_train = []
            
            for trade in trades:
                try:
                    # Get features at entry (all 43 indicators)
                    features = self.prepare_features(trade.indicators_at_entry, trade_type=trade.trade_type)
                    
                    # Label: 1 if trade won, 0 if lost
                    label = 1 if trade.outcome.upper() == 'WIN' else 0
                    
                    X_train.append(features.flatten())
                    y_train.append(label)
                    
                except Exception as e:
                    print(f"⚠️  Skipping trade {trade.id} in M2 training: {e}")
                    continue
            
            if len(X_train) < min_trades:
                print(f"⚠️  Not enough valid trades for M2. Need {min_trades}, have {len(X_train)}")
                self.m2_model = None
                return False
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Train XGBoost classifier for entry quality
            print(f"🔧 Training M2 meta-model on {len(X_train)} trades...")
            
            self.m2_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            
            self.m2_model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = self.m2_model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)
            
            win_count = sum(y_train)
            loss_count = len(y_train) - win_count
            
            print(f"✅ M2 meta-model trained successfully!")
            print(f"   Training accuracy: {accuracy:.1%}")
            print(f"   Win examples: {win_count}, Loss examples: {loss_count}")
            
            # Save M2 model to database for persistence
            self._save_m2_model()
            
            return True
            
        except Exception as e:
            print(f"❌ Error training M2 meta-model: {e}")
            import traceback
            traceback.print_exc()
            self.m2_model = None
            return False
        finally:
            session.close()
    
    def get_m2_advisory_reason(self, indicators: dict, direction: str, momentum_timing: dict = None) -> str:
        """
        Generate specific reason why M2 is warning about entry timing.
        Now includes momentum timing analysis from multi-timeframe RSI and KDJ.
        Uses timeframe-aware timing (candles + actual hours).
        """
        reasons = []
        
        # ========== Momentum Timing Analysis (Primary Factor for Entry Timing) ==========
        if momentum_timing and momentum_timing.get('momentum_direction'):
            est_candles = momentum_timing.get('estimated_candles', 0)
            est_hours = momentum_timing.get('estimated_hours', 0)
            tf_label = momentum_timing.get('timeframe_label', '1H')
            momentum_dir = momentum_timing.get('momentum_direction')
            
            # Format time display
            if est_hours >= 24:
                time_display = f"~{est_hours/24:.1f} days"
            elif est_hours >= 1:
                time_display = f"~{est_hours:.0f}h"
            else:
                time_display = f"~{est_hours*60:.0f}m"
            
            # Check if momentum conflicts with trade direction
            if direction == 'SHORT' and momentum_dir in ['bullish', 'mixed']:
                if est_candles >= 2:
                    reasons.append(f"HOLD: Bullish momentum persists ~{est_candles:.0f} {tf_label} candles ({time_display})")
                    details = momentum_timing.get('details', {})
                    if details:
                        rsi_6 = details.get('RSI_6', 0)
                        stoch_j = details.get('Stoch_J', 0)
                        if rsi_6 > 60:
                            reasons.append(f"RSI_6={rsi_6:.0f} still rising")
                        if stoch_j > 50 and stoch_j < 100:
                            reasons.append(f"J={stoch_j:.0f} not peaked")
            
            elif direction == 'LONG' and momentum_dir in ['bearish', 'mixed']:
                if est_candles >= 2:
                    reasons.append(f"HOLD: Bearish momentum persists ~{est_candles:.0f} {tf_label} candles ({time_display})")
                    details = momentum_timing.get('details', {})
                    if details:
                        rsi_6 = details.get('RSI_6', 0)
                        stoch_j = details.get('Stoch_J', 0)
                        if rsi_6 < 40:
                            reasons.append(f"RSI_6={rsi_6:.0f} still falling")
                        if stoch_j < 50 and stoch_j > 0:
                            reasons.append(f"J={stoch_j:.0f} not bottomed")
            
            # Good entry timing - reversal imminent
            elif momentum_dir == 'reversal_imminent':
                reasons.append(f"ENTER: Reversal imminent within 1-2 {tf_label} candles")
            
            # Momentum aligns with direction - good entry
            elif (direction == 'LONG' and momentum_dir == 'bullish') or \
                 (direction == 'SHORT' and momentum_dir == 'bearish'):
                # Momentum supports the trade direction - this is good
                pass  # Don't add warning, let other factors speak
        
        # Check divergence status
        has_divergence = indicators.get('has_divergence', False)
        candles_elapsed = indicators.get('candles_elapsed', 0)
        
        if not has_divergence:
            reasons.append("No divergence detected")
        elif candles_elapsed > 10:
            reasons.append(f"Late entry ({candles_elapsed} candles after divergence)")
        
        # Check trend strength
        adx = indicators.get('ADX', 0)
        if adx and adx < 20:
            reasons.append("Weak trend (ADX < 20) - wait for confirmation")
        elif adx and adx > 25:
            # Check trend direction
            di_plus = indicators.get('DI_plus', indicators.get('DI+', 0))
            di_minus = indicators.get('DI_minus', indicators.get('DI-', 0))
            
            if di_plus and di_minus:
                if direction == 'LONG' and di_minus > di_plus:
                    reasons.append("Trend still falling (DI- > DI+)")
                elif direction == 'SHORT' and di_plus > di_minus:
                    reasons.append("Trend still rising (DI+ > DI-)")
        
        # Check MACD direction
        macd = indicators.get('MACD')
        macd_signal = indicators.get('MACD_signal')
        if macd is not None and macd_signal is not None:
            if direction == 'SHORT' and macd > macd_signal:
                reasons.append("MACD still bullish - wait for crossover")
            elif direction == 'LONG' and macd < macd_signal:
                reasons.append("MACD still bearish - wait for crossover")
        
        # Check for clear reversal signal via OBV
        obv_signal = indicators.get('obv_signal', 'neutral')
        if isinstance(obv_signal, str):
            if direction == 'LONG' and 'falling' in obv_signal.lower():
                reasons.append("OBV falling - smart money still selling")
            elif direction == 'SHORT' and 'rising' in obv_signal.lower():
                reasons.append("OBV rising - smart money still buying")
        
        if reasons:
            return " - ".join(reasons)
        else:
            return "Entry timing may be suboptimal - consider waiting"
    
    def assess_entry_quality(self, indicators, m1_confidence, predicted_direction):
        """
        M2 meta-model: Assess whether this trade is worth taking.
        
        Args:
            indicators: Current market indicators (all 43)
            m1_confidence: M1 pattern matching confidence (0-1)
            predicted_direction: 'LONG' or 'SHORT' from M1
        
        Returns:
            entry_quality: Score from 0.0 to 1.0, or None if M2 not trained
                - 0.0-0.3: Poor entry (likely late or unfavorable conditions)
                - 0.3-0.6: Moderate entry quality
                - 0.6-1.0: Good entry quality
                - None: M2 not available yet (need 10+ trades)
        """
        # If M2 not trained, return None (no filtering)
        if self.m2_model is None:
            print("⚠️  M2 model not available - need 10+ trades to train")
            return None
        
        try:
            # Prepare features with predicted direction
            features = self.prepare_features(indicators, trade_type=predicted_direction)
            features = features.reshape(1, -1)
            
            # Get probability of winning trade
            win_probability = self.m2_model.predict_proba(features)[0][1]
            
            return float(win_probability)
            
        except Exception as e:
            print(f"⚠️  Error in M2 assessment: {e}")
            return 1.0  # Fallback to neutral
    
    def train_models(self, min_trades=5):
        """
        Profile-based learning system:
        Rebuilds winning/losing profiles and calculates indicator accuracy
        """
        session = get_session()
        
        try:
            trades = session.query(Trade).filter(
                Trade.exit_price.isnot(None),
                Trade.outcome.isnot(None),
                Trade.indicators_at_entry.isnot(None)
            ).all()
            
            if len(trades) < min_trades:
                print(f"Not enough trades for profile building. Need {min_trades}, have {len(trades)}")
                return False
            
            # Build profiles
            profiles = self.build_trade_profiles()
            
            if profiles is None:
                print("Failed to build profiles")
                return False
            
            # Calculate overall accuracy based on profile matching
            # For each trade, predict using profiles and check if it matches actual outcome
            correct_predictions = 0
            total_predictions = 0
            
            for trade in trades:
                if not trade.indicators_at_entry:
                    continue
                
                # Get similarity scores
                sim_win_long = self.calculate_similarity(trade.indicators_at_entry, profiles.get('win_long'))
                sim_win_short = self.calculate_similarity(trade.indicators_at_entry, profiles.get('win_short'))
                sim_loss_long = self.calculate_similarity(trade.indicators_at_entry, profiles.get('loss_long'))
                sim_loss_short = self.calculate_similarity(trade.indicators_at_entry, profiles.get('loss_short'))
                
                # Predict direction based on similarity
                long_score = sim_win_long + sim_loss_short
                short_score = sim_win_short + sim_loss_long
                
                predicted_direction = 'LONG' if long_score > short_score else 'SHORT'
                
                # Check if prediction matches actual trade (case-insensitive for outcome)
                was_correct = (predicted_direction == trade.trade_type and trade.outcome.upper() == 'WIN') or \
                             (predicted_direction != trade.trade_type and trade.outcome.upper() == 'LOSS')
                
                if was_correct:
                    correct_predictions += 1
                total_predictions += 1
            
            overall_accuracy = (correct_predictions / total_predictions) if total_predictions > 0 else 0.0
            
            # Deactivate all old models (including legacy RandomForest/XGBoost)
            session.query(ModelPerformance).update({'is_active': False})
            session.commit()
            
            # Save overall performance metrics
            self._save_performance_metrics('ProfileMatching', {
                'accuracy': overall_accuracy,
                'precision': overall_accuracy,
                'recall': overall_accuracy,
                'f1': overall_accuracy
            }, len(trades))
            
            print(f"✅ Profile-based learning complete!")
            print(f"   Analyzed {len(trades)} trades")
            print(f"   Overall accuracy: {overall_accuracy:.1%}")
            print(f"   Win LONG examples: {len([t for t in trades if t.outcome.upper()=='WIN' and t.trade_type=='LONG'])}")
            print(f"   Win SHORT examples: {len([t for t in trades if t.outcome.upper()=='WIN' and t.trade_type=='SHORT'])}")
            print(f"   Loss LONG examples: {len([t for t in trades if t.outcome.upper()=='LOSS' and t.trade_type=='LONG'])}")
            print(f"   Loss SHORT examples: {len([t for t in trades if t.outcome.upper()=='LOSS' and t.trade_type=='SHORT'])}")
            
            # Train M2 meta-labeling model (entry quality filter)
            print(f"\n🔧 Training M2 meta-labeling model...")
            self.train_meta_model(min_trades=10)
            
            return True
            
        except Exception as e:
            print(f"Error in profile-based learning: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            session.close()
    
    def predict(self, indicators):
        rule_based_result = self._rule_based_prediction(indicators)
        
        try:
            # ========== NEW: Volatility Regime Classification ==========
            current_regime, regime_score = self.classify_volatility_regime(indicators)
            print(f"🌡️  Current Volatility Regime: {current_regime} (score: {regime_score:.1f}/100)")
            
            # Build profiles from historical trades (ORIGINAL LOGIC - NO CHANGES)
            from database import get_session, Trade
            session = get_session()
            try:
                current_trade_count = session.query(Trade).filter(
                    Trade.exit_price.isnot(None),
                    Trade.outcome.isnot(None),
                    Trade.indicators_at_entry.isnot(None)
                ).count()
            finally:
                session.close()
            
            if self.cached_profiles is None or current_trade_count != self.profiles_trade_count:
                self.cached_profiles = self.build_trade_profiles()
                self.profiles_trade_count = current_trade_count
            
            profiles = self.cached_profiles
            
            if profiles is None:
                print("⚠️  Not enough trades to build profiles - using rule-based prediction only")
                return rule_based_result
            
            # Calculate similarity to each profile (ORIGINAL LOGIC)
            sim_win_long = self.calculate_similarity(indicators, profiles.get('win_long'))
            sim_win_short = self.calculate_similarity(indicators, profiles.get('win_short'))
            sim_loss_long = self.calculate_similarity(indicators, profiles.get('loss_long'))
            sim_loss_short = self.calculate_similarity(indicators, profiles.get('loss_short'))
            
            print(f"📊 Pattern Similarity Scores:")
            print(f"   Win LONG: {sim_win_long:.3f}, Win SHORT: {sim_win_short:.3f}")
            print(f"   Loss LONG: {sim_loss_long:.3f}, Loss SHORT: {sim_loss_short:.3f}")
            
            # SURGICAL FIX: Add volatility warning ONLY (no logic changes)
            regime_warning = None
            if current_regime in ['HIGH', 'EXTREME']:
                regime_warning = f"⚠️ {current_regime} volatility detected - consider waiting for calmer market conditions for optimal entry"
                print(f"   {regime_warning}")
            
            # Determine ML recommendation based on WIN patterns ONLY
            # LOSS patterns are used as FILTER only (to block, not flip)
            # This is the CONSERVATIVE approach - no "flip to opposite" logic
            
            long_score = sim_win_long   # Only Win LONG favors LONG
            short_score = sim_win_short  # Only Win SHORT favors SHORT
            
            # Ensure scores are non-negative (cosine similarity can be negative)
            # Shift both scores to positive range by adding the minimum
            min_score = min(long_score, short_score)
            if min_score < 0:
                long_score = long_score - min_score  # Shift to positive
                short_score = short_score - min_score
            
            # Normalize ML scores to 0-1 range
            total_score = long_score + short_score
            if total_score > 0:
                ml_long_probability = long_score / total_score
                ml_short_probability = short_score / total_score
            else:
                ml_long_probability = 0.5
                ml_short_probability = 0.5
            
            # Get rule-based score
            rule_score = rule_based_result['bullish_score'] - rule_based_result['bearish_score']
            rule_normalized = (rule_score + 20) / 40
            rule_normalized = max(0, min(1, rule_normalized))
            
            # ADAPTIVE WEIGHTING: Combine ML pattern matching with rule-based signals
            # Weight adjusts based on trade count (more trades = trust ML more)
            ml_weight = self.get_adaptive_ml_weight()
            rule_weight = 1 - ml_weight
            
            long_final_prob = (ml_long_probability * ml_weight) + (rule_normalized * rule_weight)
            short_final_prob = (ml_short_probability * ml_weight) + ((1 - rule_normalized) * rule_weight)
            
            current_price = indicators.get('current_price', 0)
            atr = indicators.get('ATR', current_price * 0.02)
            min_distance = max(atr, current_price * 0.002)
            
            reasons = rule_based_result.get('reasons', [])
            
            # Calculate margin between directions
            prob_difference = abs(long_final_prob - short_final_prob)
            
            # ========== PATTERN MATCHING LOGIC (CONSERVATIVE APPROACH) ==========
            # Step 1: Check if fresh analysis matches ANY WIN pattern
            # Step 2: If matches WIN pattern → recommend that direction
            # Step 3: If matches LOSS pattern → HOLD (block, don't flip)
            # Step 4: If no match to any pattern → HOLD
            
            WIN_SIMILARITY_THRESHOLD = 0.30    # Need at least 30% similarity to a win pattern
            LOSS_SIMILARITY_THRESHOLD = 0.40   # 40% similarity to loss pattern = block
            
            loss_pattern_warning = None
            force_hold = False
            
            # Get the best win pattern match
            best_win_match = max(sim_win_long, sim_win_short)
            best_win_direction = 'LONG' if sim_win_long >= sim_win_short else 'SHORT'
            
            print(f"   Best WIN match: {best_win_direction} at {best_win_match*100:.1f}%")
            
            # CHECK 1: Does it match ANY win pattern?
            no_win_match = False
            if best_win_match < WIN_SIMILARITY_THRESHOLD:
                force_hold = True
                no_win_match = True
                reasons.append(f"🚫 HOLD: No strong WIN pattern match (best: {best_win_direction} {best_win_match*100:.1f}% < {WIN_SIMILARITY_THRESHOLD*100:.0f}% threshold)")
                print(f"   🚫 No WIN pattern match - forcing HOLD")
            
            # CHECK 2: If about to recommend LONG, check LOSS LONG pattern
            if not force_hold and long_final_prob > short_final_prob:
                if sim_loss_long > LOSS_SIMILARITY_THRESHOLD:
                    loss_pattern_warning = f"⚠️ LOSS PATTERN: {sim_loss_long*100:.1f}% similar to losing LONG trades"
                    print(f"   {loss_pattern_warning}")
                    force_hold = True
                    reasons.append(f"🚫 HOLD: Matches LONG LOSS pattern ({sim_loss_long*100:.1f}%) - avoiding this setup")
            
            # CHECK 3: If about to recommend SHORT, check LOSS SHORT pattern
            if not force_hold and short_final_prob > long_final_prob:
                if sim_loss_short > LOSS_SIMILARITY_THRESHOLD:
                    loss_pattern_warning = f"⚠️ LOSS PATTERN: {sim_loss_short*100:.1f}% similar to losing SHORT trades"
                    print(f"   {loss_pattern_warning}")
                    force_hold = True
                    reasons.append(f"🚫 HOLD: Matches SHORT LOSS pattern ({sim_loss_short*100:.1f}%) - avoiding this setup")
            
            # ========== END PATTERN MATCHING LOGIC ==========
            
            # Choose direction with higher probability
            # Recommend if: (1) >60% absolute, OR (2) wins by >8% margin AND >52%
            # BUT respect force_hold if loss pattern detected
            if not force_hold and long_final_prob > short_final_prob and (long_final_prob > 0.6 or (prob_difference > 0.08 and long_final_prob > 0.52)):
                signal = 'LONG'
                entry_price = current_price
                stop_loss = current_price - (2 * min_distance)
                
                # Calculate ATR-based TP, then adjust for support/resistance
                atr_tp = current_price + (3 * min_distance)
                support_levels = indicators.get('support_levels', [])
                resistance_levels = indicators.get('resistance_levels', [])
                take_profit = self.calculate_safe_tp(entry_price, atr_tp, 'LONG', support_levels, resistance_levels)
                
                recommendation = f"Strong LONG signal. Enter at {entry_price:.2f}"
                reasons.append(f"WIN Pattern Similarity: LONG {sim_win_long*100:.1f}%, SHORT {sim_win_short*100:.1f}%")
                reasons.append(f"Final ({ml_weight*100:.0f}% ML + {rule_weight*100:.0f}% Rules): LONG {long_final_prob*100:.1f}%, SHORT {short_final_prob*100:.1f}%")
                
                # NEW: Add volatility regime info
                reasons.append(f"Volatility Regime: {current_regime} (score: {regime_score:.1f}/100)")
                if regime_warning:
                    reasons.append(f"⚠️ {regime_warning}")
                
                final_probability = long_final_prob
                ml_win_probability = ml_long_probability
                
                # M2 Entry Quality Assessment (Advisory Only - Does Not Block)
                entry_quality = self.assess_entry_quality(indicators, long_final_prob, 'LONG')
                
                if entry_quality is not None:
                    print(f"🎯 M2 Entry Quality: {entry_quality:.1%}")
                    
                    # ALWAYS check momentum conflict regardless of M2 score
                    momentum_timing = indicators.get('momentum_timing', {})
                    momentum_dir = momentum_timing.get('momentum_direction', 'neutral')
                    est_candles = momentum_timing.get('estimated_candles', 0)
                    est_hours = momentum_timing.get('estimated_hours', 0)
                    signals_aligned = momentum_timing.get('signals_aligned', 0)
                    
                    # Check for momentum conflict: LONG trade but bearish momentum
                    momentum_conflict = momentum_dir == 'bearish' and est_candles >= 2
                    
                    # M2 provides advisory warnings but does NOT block the signal
                    if entry_quality < 0.5:
                        reasons.append(f"⚠️ M2 Entry Quality: {entry_quality*100:.1f}% (below 50% threshold)")
                        advisory_reason = self.get_m2_advisory_reason(indicators, 'LONG', momentum_timing)
                        reasons.append(f"⚠️ M2 Advisory: {advisory_reason}")
                    elif momentum_conflict:
                        # HIGH M2 but momentum conflicts - CRITICAL WARNING
                        time_display = f"~{est_hours:.0f}h" if est_hours >= 1 else f"~{est_hours*60:.0f}m"
                        reasons.append(f"✅ M2 Entry Quality: {entry_quality*100:.1f}%")
                        reasons.append(f"⚠️ MOMENTUM CONFLICT: Bearish momentum ({signals_aligned}/5) persists {est_candles:.0f} candles ({time_display}) - price may drop before reversal")
                    else:
                        reasons.append(f"✅ M2 Entry Quality: {entry_quality*100:.1f}% (good entry timing)")
                else:
                    # M2 not available - proceed without advisory
                    reasons.append("ℹ️ M2 quality filter not yet available (need 10+ trades to train)")
                
            elif not force_hold and short_final_prob > long_final_prob and (short_final_prob > 0.6 or (prob_difference > 0.08 and short_final_prob > 0.52)):
                signal = 'SHORT'
                entry_price = current_price
                stop_loss = current_price + (2 * min_distance)
                
                # Calculate ATR-based TP, then adjust for support/resistance
                atr_tp = current_price - (3 * min_distance)
                support_levels = indicators.get('support_levels', [])
                resistance_levels = indicators.get('resistance_levels', [])
                take_profit = self.calculate_safe_tp(entry_price, atr_tp, 'SHORT', support_levels, resistance_levels)
                
                recommendation = f"Strong SHORT signal. Enter at {entry_price:.2f}"
                reasons.append(f"WIN Pattern Similarity: LONG {sim_win_long*100:.1f}%, SHORT {sim_win_short*100:.1f}%")
                reasons.append(f"Final ({ml_weight*100:.0f}% ML + {rule_weight*100:.0f}% Rules): LONG {long_final_prob*100:.1f}%, SHORT {short_final_prob*100:.1f}%")
                
                # NEW: Add volatility regime info
                reasons.append(f"Volatility Regime: {current_regime} (score: {regime_score:.1f}/100)")
                if regime_warning:
                    reasons.append(f"⚠️ {regime_warning}")
                
                final_probability = short_final_prob
                ml_win_probability = ml_short_probability
                
                # M2 Entry Quality Assessment (Advisory Only - Does Not Block)
                entry_quality = self.assess_entry_quality(indicators, short_final_prob, 'SHORT')
                
                if entry_quality is not None:
                    print(f"🎯 M2 Entry Quality: {entry_quality:.1%}")
                    
                    # ALWAYS check momentum conflict regardless of M2 score
                    momentum_timing = indicators.get('momentum_timing', {})
                    momentum_dir = momentum_timing.get('momentum_direction', 'neutral')
                    est_candles = momentum_timing.get('estimated_candles', 0)
                    est_hours = momentum_timing.get('estimated_hours', 0)
                    signals_aligned = momentum_timing.get('signals_aligned', 0)
                    
                    # Check for momentum conflict: SHORT trade but bullish momentum
                    momentum_conflict = momentum_dir == 'bullish' and est_candles >= 2
                    
                    # M2 provides advisory warnings but does NOT block the signal
                    if entry_quality < 0.5:
                        reasons.append(f"⚠️ M2 Entry Quality: {entry_quality*100:.1f}% (below 50% threshold)")
                        advisory_reason = self.get_m2_advisory_reason(indicators, 'SHORT', momentum_timing)
                        reasons.append(f"⚠️ M2 Advisory: {advisory_reason}")
                    elif momentum_conflict:
                        # HIGH M2 but momentum conflicts - CRITICAL WARNING
                        time_display = f"~{est_hours:.0f}h" if est_hours >= 1 else f"~{est_hours*60:.0f}m"
                        reasons.append(f"✅ M2 Entry Quality: {entry_quality*100:.1f}%")
                        reasons.append(f"⚠️ MOMENTUM CONFLICT: Bullish momentum ({signals_aligned}/5) persists {est_candles:.0f} candles ({time_display}) - price may rise before reversal")
                    else:
                        reasons.append(f"✅ M2 Entry Quality: {entry_quality*100:.1f}% (good entry timing)")
                else:
                    # M2 not available - proceed without advisory
                    reasons.append("ℹ️ M2 quality filter not yet available (need 10+ trades to train)")
                
            else:
                signal = 'HOLD'
                entry_price = None
                stop_loss = None
                take_profit = None
                entry_quality = None  # No M2 assessment for HOLD signals
                
                # Different message based on WHY we're holding
                if force_hold and no_win_match:
                    # HOLD because no WIN pattern match
                    recommendation = "HOLD - No matching win pattern. Wait for better setup."
                    reasons.append(f"WIN Pattern Similarity: LONG {sim_win_long*100:.1f}%, SHORT {sim_win_short*100:.1f}%")
                elif force_hold and loss_pattern_warning:
                    # HOLD because LOSS pattern detected
                    recommendation = "HOLD - Loss pattern detected. Wait for better setup."
                    reasons.append(f"WIN Pattern Similarity: LONG {sim_win_long*100:.1f}%, SHORT {sim_win_short*100:.1f}%")
                    reasons.append(loss_pattern_warning)
                else:
                    # HOLD because no clear winner between LONG/SHORT
                    recommendation = "No clear signal. Wait for better opportunity."
                    reasons.append(f"WIN Pattern Similarity: LONG {sim_win_long*100:.1f}%, SHORT {sim_win_short*100:.1f}%")
                    reasons.append(f"Final ({ml_weight*100:.0f}% ML + {rule_weight*100:.0f}% Rules): LONG {long_final_prob*100:.1f}%, SHORT {short_final_prob*100:.1f}% - No clear winner")
                
                final_probability = max(long_final_prob, short_final_prob)
                ml_win_probability = max(ml_long_probability, ml_short_probability)
            
            return {
                'signal': signal,
                'confidence': round(max(final_probability, 1 - final_probability) * 100, 2),
                'win_probability': round(final_probability * 100, 2),
                'recommendation': recommendation,
                'entry_price': round(entry_price, 2) if entry_price else None,
                'stop_loss': round(stop_loss, 2) if stop_loss else None,
                'take_profit': round(take_profit, 2) if take_profit else None,
                'ml_probability': round(ml_win_probability * 100, 2),
                'rule_probability': round(rule_normalized * 100, 2),
                'm2_entry_quality': round(entry_quality * 100, 2) if entry_quality is not None else None,
                'method': 'profile_matching',
                'reasons': reasons
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'recommendation': f'Error making prediction: {str(e)}',
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None
            }
    
    def _rule_based_prediction(self, indicators):
        """
        Rule-based prediction using technical indicators.
        Always makes a prediction even without ML models.
        Learning happens AFTER trades are executed.
        """
        current_price = indicators.get('current_price', 0)
        atr = indicators.get('ATR', current_price * 0.02)
        min_distance = max(atr, current_price * 0.002)
        
        bullish_signals = 0
        bearish_signals = 0
        reasons = []
        
        rsi_weight = self.indicator_weights.get('RSI', 2.0)
        
        # Get trend context for duration-aware signals
        trend_context = indicators.get('trend_context', {})
        rsi_ctx = trend_context.get('RSI', {})
        
        rsi = indicators.get('RSI')
        if rsi:
            # Let ML learn temporal patterns - rules only provide basic signals
            if rsi < 30:
                bullish_signals += 2 * rsi_weight
                reasons.append(f"RSI oversold ({rsi:.1f}) - strong buy signal (weight: {rsi_weight:.1f}x + 0.0 bonus)")
                
            elif rsi < 40:
                bullish_signals += 1 * rsi_weight
                reasons.append(f"RSI below 40 ({rsi:.1f}) - buy signal (weight: {rsi_weight:.1f}x)")
                
            elif rsi > 70:
                bearish_signals += 2 * rsi_weight
                reasons.append(f"RSI overbought ({rsi:.1f}) - strong sell signal (weight: {rsi_weight:.1f}x + 0.0 bonus)")
                
            elif rsi > 60:
                bearish_signals += 1 * rsi_weight
                reasons.append(f"RSI above 60 ({rsi:.1f}) - sell signal (weight: {rsi_weight:.1f}x)")
        
        macd_weight = self.indicator_weights.get('MACD', 2.0)
        
        macd = indicators.get('MACD')
        macd_signal = indicators.get('MACD_signal')
        if macd and macd_signal:
            if macd > macd_signal:
                bullish_signals += 2 * macd_weight
                reasons.append(f"MACD bullish crossover (weight: {macd_weight:.1f}x)")
            else:
                bearish_signals += 2 * macd_weight
                reasons.append(f"MACD bearish crossover (weight: {macd_weight:.1f}x)")
        
        stoch_weight = self.indicator_weights.get('Stoch_K', 1.0)
        stoch_ctx = trend_context.get('Stochastic', {})
        
        stoch_k = indicators.get('Stoch_K')
        if stoch_k:
            # Let ML learn temporal patterns - rules only provide basic signals
            if stoch_k < 20:
                bullish_signals += 1 * stoch_weight
                reasons.append(f"Stochastic oversold ({stoch_k:.1f}) (weight: {stoch_weight:.1f}x + 0.0 bonus)")
                
            elif stoch_k > 80:
                bearish_signals += 1 * stoch_weight
                reasons.append(f"Stochastic overbought ({stoch_k:.1f}) (weight: {stoch_weight:.1f}x + 0.0 bonus)")
        
        adx_weight = self.indicator_weights.get('ADX', 2.0)
        
        adx = indicators.get('ADX')
        di_plus = indicators.get('DI_plus', 0)
        di_minus = indicators.get('DI_minus', 0)
        if adx and adx > 25:
            if di_plus > di_minus:
                bullish_signals += 2 * adx_weight
                reasons.append(f"Strong uptrend (ADX {adx:.1f}, +DI > -DI, weight: {adx_weight:.1f}x)")
            else:
                bearish_signals += 2 * adx_weight
                reasons.append(f"Strong downtrend (ADX {adx:.1f}, -DI > +DI, weight: {adx_weight:.1f}x)")
        
        sma_weight = self.indicator_weights.get('SMA', 1.0)
        
        price = indicators.get('current_price', 0)
        sma_20 = indicators.get('SMA_20')
        sma_50 = indicators.get('SMA_50')
        if price and sma_20:
            if price > sma_20:
                bullish_signals += 1 * sma_weight
                reasons.append(f"Price above SMA 20 (weight: {sma_weight:.1f}x)")
            else:
                bearish_signals += 1 * sma_weight
                reasons.append(f"Price below SMA 20 (weight: {sma_weight:.1f}x)")
        
        if price and sma_50:
            if price > sma_50:
                bullish_signals += 1 * sma_weight
                reasons.append(f"Price above SMA 50 (weight: {sma_weight:.1f}x)")
            else:
                bearish_signals += 1 * sma_weight
                reasons.append(f"Price below SMA 50 (weight: {sma_weight:.1f}x)")
        
        mfi_weight = self.indicator_weights.get('MFI', 1.0)
        mfi_ctx = trend_context.get('MFI', {})
        
        mfi = indicators.get('MFI')
        if mfi:
            # Let ML learn temporal patterns - rules only provide basic signals
            if mfi < 20:
                bullish_signals += 1 * mfi_weight
                reasons.append(f"MFI oversold ({mfi:.1f}) (weight: {mfi_weight:.1f}x + 0.0 bonus)")
                
            elif mfi > 80:
                bearish_signals += 1 * mfi_weight
                reasons.append(f"MFI overbought ({mfi:.1f}) (weight: {mfi_weight:.1f}x + 0.0 bonus)")
        
        # OBV Divergence Analysis (Volume-based)
        obv_ctx = trend_context.get('OBV', {})
        obv_divergence = obv_ctx.get('divergence', 'none')
        obv_slope = obv_ctx.get('slope', 0.0)
        
        if obv_divergence == 'bullish':
            # Price falling but volume accumulating = smart money buying
            divergence_signal = 2.5
            bullish_signals += divergence_signal
            slope_info = "accumulation" if obv_slope > 0 else ""
            reason = f"Bullish OBV divergence detected - smart money accumulating"
            if slope_info:
                reason += f" [{slope_info}]"
            reasons.append(reason + f" (+{divergence_signal:.1f} signal)")
            
        elif obv_divergence == 'bearish':
            # Price rising but volume distributing = smart money selling
            divergence_signal = 2.5
            bearish_signals += divergence_signal
            slope_info = "distribution" if obv_slope < 0 else ""
            reason = f"Bearish OBV divergence detected - smart money distributing"
            if slope_info:
                reason += f" [{slope_info}]"
            reasons.append(reason + f" (+{divergence_signal:.1f} signal)")
        
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            total_signals = 1
        
        if bullish_signals > bearish_signals * 1.5:
            signal = 'LONG'
            confidence = min(95, (bullish_signals / total_signals) * 100)
            entry_price = current_price
            stop_loss = current_price - (2 * min_distance)
            
            # Calculate ATR-based TP, then adjust for support/resistance
            atr_tp = current_price + (3 * min_distance)
            support_levels = indicators.get('support_levels', [])
            resistance_levels = indicators.get('resistance_levels', [])
            take_profit = self.calculate_safe_tp(entry_price, atr_tp, 'LONG', support_levels, resistance_levels)
            
            recommendation = f"Rule-based LONG signal (no ML training yet)"
        elif bearish_signals > bullish_signals * 1.5:
            signal = 'SHORT'
            confidence = min(95, (bearish_signals / total_signals) * 100)
            entry_price = current_price
            stop_loss = current_price + (2 * min_distance)
            
            # Calculate ATR-based TP, then adjust for support/resistance
            atr_tp = current_price - (3 * min_distance)
            support_levels = indicators.get('support_levels', [])
            resistance_levels = indicators.get('resistance_levels', [])
            take_profit = self.calculate_safe_tp(entry_price, atr_tp, 'SHORT', support_levels, resistance_levels)
            
            recommendation = f"Rule-based SHORT signal (no ML training yet)"
        else:
            signal = 'HOLD'
            confidence = abs(bullish_signals - bearish_signals) / total_signals * 100
            entry_price = None
            stop_loss = None
            take_profit = None
            recommendation = "Mixed signals - wait for clearer setup"
            reasons.append(f"Bullish: {bullish_signals}, Bearish: {bearish_signals} - too close")
        
        return {
            'signal': signal,
            'confidence': round(confidence, 1),
            'recommendation': recommendation,
            'entry_price': round(entry_price, 2) if entry_price else None,
            'stop_loss': round(stop_loss, 2) if stop_loss else None,
            'take_profit': round(take_profit, 2) if take_profit else None,
            'reasons': reasons,
            'method': 'rule_based',
            'bullish_score': bullish_signals,
            'bearish_score': bearish_signals
        }
    
    def _save_performance_metrics(self, model_name, metrics, total_trades):
        session = get_session()
        
        try:
            session.query(ModelPerformance).filter(
                ModelPerformance.model_name == model_name
            ).update({'is_active': False})
            
            perf = ModelPerformance(
                model_name=model_name,
                version='1.0',
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1'],
                total_trades=total_trades,
                is_active=True
            )
            
            session.add(perf)
            session.commit()
            
        except Exception as e:
            print(f"Error saving performance metrics: {e}")
            session.rollback()
        finally:
            session.close()
    
    def learn_from_trade(self, trade_id):
        """
        Learn from a single completed trade immediately.
        
        DUAL LEARNING APPROACH:
        1. INDIVIDUAL LEARNING (Every Trade):
           - Updates indicator weights from EVERY trade (wins AND losses)
           - Lightweight, immediate feedback
           
        2. BULK RETRAINING (Every 10 Trades):
           - At 10 trades: Train models on all 10 trades
           - At 20 trades: Train models on all 20 trades
           - At 30 trades: Train models on all 30 trades
           - Continues every 10 trades (40, 50, 60...)
           - Always uses ALL historical trades (wins + losses)
        
        This ensures the system learns from both wins AND losses continuously.
        """
        session = get_session()
        
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            
            if not trade or not trade.exit_price or not trade.outcome:
                return False
            
            print(f"\n📚 Learning from trade #{trade_id}:")
            print(f"   Direction: {trade.trade_type}, Outcome: {trade.outcome}")
            print(f"   P&L: ${trade.profit_loss:.2f}")
            
            # STEP 1: Incremental weight update (runs every trade)
            if trade.indicators_at_entry:
                self._learn_from_single_trade(trade)
            
            # STEP 2: Check if models need retraining
            total_trades = session.query(Trade).filter(
                Trade.exit_price.isnot(None),
                Trade.outcome.isnot(None)
            ).count()
            
            print(f"   Total completed trades: {total_trades}")
            
            # RETRAINING SCHEDULE:
            # - First training at 10 trades (learns from all 10)
            # - Then retrain every 10 trades: 20, 30, 40, 50...
            # - Each retrain uses ALL historical trades (wins AND losses)
            # - Prevents duplicate retraining using persistent milestone tracking
            
            last_training_count = self._get_last_training_count()
            current_milestone = (total_trades // 10) * 10  # Round down to nearest 10
            last_milestone = (last_training_count // 10) * 10
            
            # Retrain if we've hit a new 10-trade milestone
            should_retrain = (total_trades >= 10 and current_milestone > last_milestone)
            
            # TRIGGER BATCH ANALYSIS IF:
            # Major milestone (every 100 trades) for comprehensive diagnostics
            should_batch_analyze = (total_trades % 100 == 0 and total_trades >= 100)
            
            if should_retrain:
                print(f"\n🎯 Milestone: {current_milestone} trades completed!")
                print(f"   Retraining models on ALL {total_trades} trades (wins AND losses)...")
                
                # Count wins vs losses for transparency
                wins = session.query(Trade).filter(
                    Trade.exit_price.isnot(None),
                    Trade.outcome == 'win'
                ).count()
                losses = session.query(Trade).filter(
                    Trade.exit_price.isnot(None),
                    Trade.outcome == 'loss'
                ).count()
                
                print(f"   Training data: {wins} wins, {losses} losses")
                
                success = self.train_models(min_trades=10)
                if success:
                    self._save_last_training_count(total_trades)
                    print(f"✅ Models retrained successfully!")
                    print(f"   ML now understands patterns from {total_trades} trades of experience!")
                else:
                    print(f"⚠️  Retraining failed - will retry at next milestone")
            elif should_batch_analyze:
                print(f"\n🎯 Major milestone: {total_trades} trades!")
                print("   Triggering comprehensive batch analysis...")
                self.batch_analysis()
            else:
                next_milestone = ((total_trades // 10) + 1) * 10
                print(f"   Next retraining at {next_milestone} trades ({next_milestone - total_trades} more needed)")
            
            return True
            
        except Exception as e:
            print(f"Error learning from trade: {e}")
            return False
        finally:
            session.close()
    
    def _learn_from_single_trade(self, trade):
        """
        Immediately update indicator weights based on this single trade.
        This makes the system learn from EVERY trade, not just batches.
        """
        is_win = trade.outcome == 'win'
        indicators = trade.indicators_at_entry
        
        print(f"   🧠 Updating weights based on this trade...")
        
        learning_rate = 0.05
        
        for ind_name in ['RSI', 'MACD', 'ADX', 'Stoch_K', 'MFI', 'SMA', 'CCI']:
            if ind_name in indicators and indicators[ind_name] is not None:
                current_weight = self.indicator_weights.get(ind_name, 1.0)
                
                if is_win:
                    new_weight = min(3.0, current_weight * (1 + learning_rate))
                    print(f"   ⬆️ {ind_name}: {current_weight:.2f} → {new_weight:.2f} (win)")
                else:
                    new_weight = max(0.5, current_weight * (1 - learning_rate))
                    print(f"   ⬇️ {ind_name}: {current_weight:.2f} → {new_weight:.2f} (loss)")
                
                self.indicator_weights[ind_name] = new_weight
        
        price = indicators.get('current_price', 0)
        sma_20 = indicators.get('SMA_20')
        sma_50 = indicators.get('SMA_50')
        if (sma_20 or sma_50) and 'SMA' not in indicators:
            current_weight = self.indicator_weights.get('SMA', 1.0)
            if is_win:
                new_weight = min(3.0, current_weight * (1 + learning_rate))
                print(f"   ⬆️ SMA: {current_weight:.2f} → {new_weight:.2f} (win)")
            else:
                new_weight = max(0.5, current_weight * (1 - learning_rate))
                print(f"   ⬇️ SMA: {current_weight:.2f} → {new_weight:.2f} (loss)")
            self.indicator_weights['SMA'] = new_weight
        
        self._save_indicator_weights()
        print(f"   💾 Updated weights saved - System is now smarter!")
        
        # Track per-indicator performance
        self._track_indicator_performance(trade)
    
    def _track_indicator_performance(self, trade):
        """
        Track which specific indicators were correct/wrong for this trade.
        Updates the IndicatorPerformance table in the database.
        """
        session = get_session()
        
        try:
            is_win = trade.outcome == 'win'
            trade_direction = trade.trade_type
            indicators = trade.indicators_at_entry
            
            if not indicators:
                return
            
            # Analyze each indicator's signal
            indicator_signals = self._evaluate_indicator_signals(indicators, trade_direction)
            
            # Update database for each indicator
            for ind_name, signal in indicator_signals.items():
                # Track ALL indicators (including neutral ones)
                # Neutral indicators count as "aligned with trade direction"
                
                # Get or create indicator performance record
                ind_perf = session.query(IndicatorPerformance).filter(
                    IndicatorPerformance.indicator_name == ind_name
                ).first()
                
                if not ind_perf:
                    ind_perf = IndicatorPerformance(
                        indicator_name=ind_name,
                        correct_count=0,
                        wrong_count=0,
                        total_signals=0,
                        accuracy_rate=0.0,
                        weight_multiplier=self.indicator_weights.get(ind_name, 1.0)
                    )
                    session.add(ind_perf)
                
                # Update counts - properly account for trade direction and signal
                ind_perf.total_signals += 1
                
                # Determine if indicator was correct based on trade direction, signal, and outcome
                # Neutral signals count as "correct" since they didn't oppose the trade
                is_long = trade_direction == 'LONG'
                indicator_was_correct = False
                
                if signal == 'neutral':
                    # Neutral indicators are always counted as correct (didn't oppose the trade)
                    indicator_was_correct = True
                elif is_long:
                    # LONG trade: bullish signal + win = correct, bearish signal + loss = correct
                    is_bullish_signal = signal == 'bullish'
                    if (is_bullish_signal and is_win) or (not is_bullish_signal and not is_win):
                        indicator_was_correct = True
                else:
                    # SHORT trade: bearish signal + win = correct, bullish signal + loss = correct
                    is_bullish_signal = signal == 'bullish'
                    if (not is_bullish_signal and is_win) or (is_bullish_signal and not is_win):
                        indicator_was_correct = True
                
                if indicator_was_correct:
                    ind_perf.correct_count += 1
                else:
                    ind_perf.wrong_count += 1
                
                # Recalculate accuracy
                ind_perf.accuracy_rate = (ind_perf.correct_count / ind_perf.total_signals * 100) if ind_perf.total_signals > 0 else 0.0
                ind_perf.weight_multiplier = self.indicator_weights.get(ind_name, 1.0)
                ind_perf.last_updated = datetime.utcnow()
            
            session.commit()
            
        except Exception as e:
            print(f"   Error tracking indicator performance: {e}")
            session.rollback()
        finally:
            session.close()
    
    def _evaluate_indicator_signals(self, indicators, trade_direction):
        """
        Evaluate which signal each indicator gave (bullish/bearish/neutral).
        Returns dict of {indicator_name: 'bullish'|'bearish'|'neutral'}
        """
        signals = {}
        
        # RSI
        rsi = indicators.get('RSI')
        if rsi is not None:
            if rsi < 45:
                signals['RSI'] = 'bullish'  # Oversold or leaning bullish
            elif rsi > 55:
                signals['RSI'] = 'bearish'  # Overbought or leaning bearish
            else:
                signals['RSI'] = 'neutral'
        
        # MACD
        macd = indicators.get('MACD')
        macd_signal = indicators.get('MACD_signal')
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                signals['MACD'] = 'bullish'
            elif macd < macd_signal:
                signals['MACD'] = 'bearish'
            else:
                signals['MACD'] = 'neutral'
        
        # Stochastic
        stoch_k = indicators.get('Stoch_K')
        if stoch_k is not None:
            if stoch_k < 45:
                signals['Stochastic'] = 'bullish'  # Oversold or leaning bullish
            elif stoch_k > 55:
                signals['Stochastic'] = 'bearish'  # Overbought or leaning bearish
            else:
                signals['Stochastic'] = 'neutral'
        
        # ADX (trend strength, not direction - use DI)
        adx = indicators.get('ADX')
        di_plus = indicators.get('DI_plus')
        di_minus = indicators.get('DI_minus')
        if di_plus is not None and di_minus is not None and adx is not None:
            if adx < 25:
                signals['ADX'] = 'neutral'  # Weak trend
            elif di_plus > di_minus:
                signals['ADX'] = 'bullish'
            else:
                signals['ADX'] = 'bearish'
        
        # MFI
        mfi = indicators.get('MFI')
        if mfi is not None:
            if mfi < 45:
                signals['MFI'] = 'bullish'  # Oversold or leaning bullish
            elif mfi > 55:
                signals['MFI'] = 'bearish'  # Overbought or leaning bearish
            else:
                signals['MFI'] = 'neutral'
        
        # CCI
        cci = indicators.get('CCI')
        if cci is not None:
            if cci < 0:
                signals['CCI'] = 'bullish'  # Below zero - bearish momentum weakening
            elif cci > 0:
                signals['CCI'] = 'bearish'  # Above zero - bullish momentum weakening  
            else:
                signals['CCI'] = 'neutral'
        
        # OBV (volume flow)
        obv = indicators.get('OBV')
        if obv is not None:
            # Simplified: positive OBV suggests accumulation (bullish)
            if obv > 0:
                signals['OBV'] = 'bullish'
            elif obv < 0:
                signals['OBV'] = 'bearish'
            else:
                signals['OBV'] = 'neutral'
        
        return signals
    
    def batch_analysis(self):
        """
        Comprehensive analysis every 30 trades.
        Analyzes wins vs losses, indicator performance, and improves predictions.
        """
        session = get_session()
        
        try:
            trades = session.query(Trade).filter(
                Trade.exit_price.isnot(None),
                Trade.outcome.isnot(None)
            ).all()
            
            if len(trades) < 30:
                print(f"Not enough trades for batch analysis. Need 30, have {len(trades)}")
                return
            
            total_trades = len(trades)
            wins = [t for t in trades if t.outcome == 'win']
            losses = [t for t in trades if t.outcome == 'loss']
            
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"📊 BATCH ANALYSIS - {total_trades} Trades")
            print(f"{'='*60}")
            print(f"\n✅ Wins: {len(wins)} ({len(wins)/total_trades*100:.1f}%)")
            print(f"❌ Losses: {len(losses)} ({len(losses)/total_trades*100:.1f}%)")
            print(f"🎯 Overall Win Rate: {win_rate:.1f}%")
            
            total_pnl = sum(t.profit_loss for t in trades if t.profit_loss)
            avg_win = sum(t.profit_loss for t in wins if t.profit_loss) / len(wins) if wins else 0
            avg_loss = sum(t.profit_loss for t in losses if t.profit_loss) / len(losses) if losses else 0
            
            print(f"\n💰 P&L Analysis:")
            print(f"   Total P&L: ${total_pnl:.2f}")
            print(f"   Avg Win: ${avg_win:.2f}")
            print(f"   Avg Loss: ${avg_loss:.2f}")
            print(f"   Risk/Reward: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "   Risk/Reward: N/A")
            
            indicator_performance = self._analyze_indicator_performance(wins, losses)
            
            print(f"\n📈 Indicator Performance Analysis:")
            for indicator, stats in indicator_performance.items():
                if stats['count'] > 5:
                    print(f"   {indicator}: {stats['win_rate']:.1f}% win rate ({stats['wins']}/{stats['count']} trades)")
            
            print(f"\n🔍 What Led to Wins:")
            self._analyze_winning_patterns(wins)
            
            print(f"\n⚠️  What Led to Losses:")
            self._analyze_losing_patterns(losses)
            
            self._update_indicator_weights(indicator_performance)
            
            print(f"\n🤖 Retraining ML models with all {total_trades} trades...")
            success = self.train_models(min_trades=30)
            
            if success:
                print(f"✅ Models retrained successfully!")
                print(f"   System is now smarter based on {total_trades} trades of experience!")
            
            print(f"\n{'='*60}\n")
            
        except Exception as e:
            print(f"Error in batch analysis: {e}")
            import traceback
            traceback.print_exc()
        finally:
            session.close()
    
    def _analyze_indicator_performance(self, wins, losses):
        """Analyze which indicators performed best in winning trades"""
        performance = {}
        
        all_trades = wins + losses
        
        for trade in all_trades:
            if not trade.indicators_at_entry:
                continue
            
            indicators = trade.indicators_at_entry
            is_win = trade.outcome == 'win'
            
            for ind_name in ['RSI', 'MACD', 'ADX', 'MFI', 'CCI', 'Stoch_K', 'SMA']:
                if ind_name not in performance:
                    performance[ind_name] = {'wins': 0, 'count': 0, 'win_rate': 0}
                
                if ind_name in indicators and indicators[ind_name] is not None:
                    performance[ind_name]['count'] += 1
                    if is_win:
                        performance[ind_name]['wins'] += 1
                elif ind_name == 'SMA' and (indicators.get('SMA_20') or indicators.get('SMA_50')):
                    performance['SMA']['count'] += 1
                    if is_win:
                        performance['SMA']['wins'] += 1
        
        for ind_name in performance:
            if performance[ind_name]['count'] > 0:
                performance[ind_name]['win_rate'] = (performance[ind_name]['wins'] / performance[ind_name]['count']) * 100
        
        return performance
    
    def _analyze_winning_patterns(self, wins):
        """Identify common patterns in winning trades"""
        if not wins:
            return
        
        rsi_oversold_wins = 0
        rsi_overbought_wins = 0
        macd_bullish_wins = 0
        strong_trend_wins = 0
        
        for trade in wins:
            if not trade.indicators_at_entry:
                continue
            
            ind = trade.indicators_at_entry
            
            if ind.get('RSI'):
                if ind['RSI'] < 30 and trade.trade_type == 'LONG':
                    rsi_oversold_wins += 1
                elif ind['RSI'] > 70 and trade.trade_type == 'SHORT':
                    rsi_overbought_wins += 1
            
            if ind.get('MACD') and ind.get('MACD_signal'):
                if ind['MACD'] > ind['MACD_signal'] and trade.trade_type == 'LONG':
                    macd_bullish_wins += 1
            
            if ind.get('ADX') and ind['ADX'] > 25:
                strong_trend_wins += 1
        
        total = len(wins)
        print(f"   - RSI oversold + LONG: {rsi_oversold_wins}/{total} ({rsi_oversold_wins/total*100:.1f}%)")
        print(f"   - RSI overbought + SHORT: {rsi_overbought_wins}/{total} ({rsi_overbought_wins/total*100:.1f}%)")
        print(f"   - MACD bullish + LONG: {macd_bullish_wins}/{total} ({macd_bullish_wins/total*100:.1f}%)")
        print(f"   - Strong trend (ADX>25): {strong_trend_wins}/{total} ({strong_trend_wins/total*100:.1f}%)")
    
    def _analyze_losing_patterns(self, losses):
        """Identify what was missed in losing trades"""
        if not losses:
            return
        
        weak_trend_losses = 0
        mixed_signals_losses = 0
        
        for trade in losses:
            if not trade.indicators_at_entry:
                continue
            
            ind = trade.indicators_at_entry
            
            if ind.get('ADX') and ind['ADX'] < 20:
                weak_trend_losses += 1
            
            rsi = ind.get('RSI')
            macd = ind.get('MACD')
            macd_sig = ind.get('MACD_signal')
            
            if rsi and macd and macd_sig:
                rsi_signal = 'bullish' if rsi < 50 else 'bearish'
                macd_signal = 'bullish' if macd > macd_sig else 'bearish'
                
                if rsi_signal != macd_signal:
                    mixed_signals_losses += 1
        
        total = len(losses)
        print(f"   - Weak trend (ADX<20): {weak_trend_losses}/{total} ({weak_trend_losses/total*100:.1f}%)")
        print(f"   - Mixed RSI/MACD signals: {mixed_signals_losses}/{total} ({mixed_signals_losses/total*100:.1f}%)")
        print(f"   💡 Lesson: Avoid trading when trend is weak or indicators conflict")
    
    def _update_indicator_weights(self, indicator_performance):
        """
        Update indicator weights based on performance.
        Better performing indicators get higher weights in future predictions.
        """
        print(f"\n🔄 Updating indicator weights based on performance...")
        
        for indicator, stats in indicator_performance.items():
            if stats['count'] < 5:
                continue
            
            win_rate = stats['win_rate']
            
            if win_rate > 70:
                self.indicator_weights[indicator] = min(3.0, self.indicator_weights.get(indicator, 1.0) * 1.2)
                print(f"   ⬆️ {indicator}: Increased weight to {self.indicator_weights[indicator]:.2f} (win rate: {win_rate:.1f}%)")
            elif win_rate < 40:
                self.indicator_weights[indicator] = max(0.5, self.indicator_weights.get(indicator, 1.0) * 0.8)
                print(f"   ⬇️ {indicator}: Decreased weight to {self.indicator_weights[indicator]:.2f} (win rate: {win_rate:.1f}%)")
            else:
                print(f"   ➡️ {indicator}: Weight unchanged at {self.indicator_weights.get(indicator, 1.0):.2f} (win rate: {win_rate:.1f}%)")
        
        self._save_indicator_weights()
    
    def _save_indicator_weights(self):
        """Save learned indicator weights to disk"""
        try:
            import json
            weights_file = f'{self.model_dir}/indicator_weights.json'
            with open(weights_file, 'w') as f:
                json.dump(self.indicator_weights, f)
            print(f"   💾 Indicator weights saved")
        except Exception as e:
            print(f"   Error saving weights: {e}")
    
    def _load_indicator_weights(self):
        """Load learned indicator weights from disk"""
        try:
            import json
            weights_file = f'{self.model_dir}/indicator_weights.json'
            if os.path.exists(weights_file):
                with open(weights_file, 'r') as f:
                    loaded_weights = json.load(f)
                    self.indicator_weights.update(loaded_weights)
                print(f"📂 Loaded learned indicator weights from previous sessions")
        except Exception as e:
            pass
    
    def _load_m2_model(self):
        """
        Load M2 meta-labeling model from database.
        This ensures M2 survives across app restarts.
        """
        try:
            session = get_session()
            model_record = session.query(MLModel).filter(
                MLModel.model_name == 'M2_MetaLabeling'
            ).first()
            session.close()
            
            if model_record:
                # Deserialize model from base64
                model_bytes = base64.b64decode(model_record.model_data)
                buffer = BytesIO(model_bytes)
                self.m2_model = joblib.load(buffer)
                print(f"📂 Loaded M2 meta-labeling model from database (version {model_record.version})")
            else:
                print(f"ℹ️  No saved M2 model found - will train when 10+ trades available")
        except Exception as e:
            print(f"⚠️  Could not load M2 model: {e}")
            self.m2_model = None
    
    def _save_m2_model(self):
        """
        Save M2 meta-labeling model to database.
        This allows M2 to persist across app restarts.
        """
        if self.m2_model is None:
            return
        
        try:
            session = get_session()
            
            # Serialize model to bytes
            buffer = BytesIO()
            joblib.dump(self.m2_model, buffer)
            buffer.seek(0)
            model_bytes = buffer.read()
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            
            # Check if model already exists
            existing = session.query(MLModel).filter(
                MLModel.model_name == 'M2_MetaLabeling'
            ).first()
            
            if existing:
                # Update existing model
                existing.model_data = model_b64
                existing.updated_at = datetime.utcnow()
                existing.version += 1
                print(f"💾 Updated M2 model in database (version {existing.version})")
            else:
                # Create new model record
                new_model = MLModel(
                    model_name='M2_MetaLabeling',
                    model_data=model_b64,
                    version=1
                )
                session.add(new_model)
                print(f"💾 Saved M2 model to database (version 1)")
            
            session.commit()
            session.close()
            
        except Exception as e:
            print(f"⚠️  Could not save M2 model: {e}")
            if session:
                session.rollback()
                session.close()
    
    def _get_last_training_count(self):
        """
        Get the trade count at last training time.
        Returns 0 if never trained or file doesn't exist (thread-safe default).
        """
        try:
            import json
            metadata_file = f'{self.model_dir}/training_metadata.json'
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('last_training_count', 0)
        except Exception as e:
            pass
        return 0
    
    def _save_last_training_count(self, count):
        """
        Save the trade count after training.
        This enables tracking of NEW trades since last training.
        """
        try:
            import json
            metadata = {'last_training_count': count}
            metadata_file = f'{self.model_dir}/training_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            print(f"   💾 Training metadata saved (last count: {count})")
        except Exception as e:
            print(f"   Warning: Could not save training metadata: {e}")
    
    def backfill_indicator_performance(self):
        """
        Backfill indicator performance table for all historical trades.
        Called during manual retrain to populate graphs from existing trades.
        
        This fixes the issue where manual retrain only trains ML models
        but doesn't populate the IndicatorPerformance table needed for graphs.
        """
        session = get_session()
        
        try:
            print(f"\n📊 Backfilling indicator performance data...")
            
            # Clear existing indicator performance data to avoid duplicates
            session.query(IndicatorPerformance).delete()
            session.commit()
            
            # Get all completed trades with indicators
            trades = session.query(Trade).filter(
                Trade.exit_price.isnot(None),
                Trade.outcome.isnot(None),
                Trade.indicators_at_entry.isnot(None)
            ).all()
            
            print(f"   Processing {len(trades)} trades with indicator data...")
            
            # Track performance for each trade
            for trade in trades:
                self._track_indicator_performance(trade)
            
            session.commit()
            print(f"✅ Backfilled indicator performance for {len(trades)} trades!")
            return len(trades)
            
        except Exception as e:
            print(f"❌ Error backfilling indicator performance: {e}")
            session.rollback()
            return 0
        finally:
            session.close()
