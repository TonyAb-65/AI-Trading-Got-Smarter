import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import base64
from io import BytesIO
from datetime import datetime
from database import get_session, Trade, ModelPerformance, IndicatorPerformance, MLModel

class MLTradingEngine:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
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
    
    def prepare_features(self, indicators):
        features = []
        feature_names = [
            'RSI', 'MACD', 'MACD_hist', 'Stoch_K', 'Stoch_D',
            'MFI', 'CCI', 'ADX', 'DI_plus', 'DI_minus',
            'current_price', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'volume', 'Volume_SMA'
        ]
        
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
        
        self.feature_columns = feature_names + [
            'price_vs_sma20', 'price_vs_sma50', 'macd_divergence', 'volume_ratio',
            'rsi_duration', 'rsi_slope', 'rsi_divergence',
            'stoch_duration', 'stoch_slope', 'stoch_divergence',
            'mfi_duration', 'mfi_slope', 'mfi_divergence',
            'obv_slope', 'obv_divergence',
            'support_distance_pct', 'resistance_distance_pct',
            'at_support_zone', 'at_resistance_zone',
            'support_strength', 'resistance_strength'
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train_models(self, min_trades=10):
        session = get_session()
        
        try:
            trades = session.query(Trade).filter(
                Trade.exit_price.isnot(None),
                Trade.outcome.isnot(None)
            ).all()
            
            if len(trades) < min_trades:
                print(f"Not enough trades for training. Need {min_trades}, have {len(trades)}")
                return False
            
            X = []
            y = []
            
            for trade in trades:
                if trade.indicators_at_entry:
                    features = self.prepare_features(trade.indicators_at_entry)
                    X.append(features.flatten())
                    y.append(1 if trade.outcome == 'win' else 0)
            
            if len(X) < min_trades:
                print("Not enough valid feature data")
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            X_scaled = self.scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
            
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            self.rf_model.fit(X_train, y_train)
            
            self.xgb_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            self.xgb_model.fit(X_train, y_train)
            
            rf_pred = self.rf_model.predict(X_test)
            xgb_pred = self.xgb_model.predict(X_test)
            
            rf_metrics = {
                'accuracy': accuracy_score(y_test, rf_pred),
                'precision': precision_score(y_test, rf_pred, zero_division=0),
                'recall': recall_score(y_test, rf_pred, zero_division=0),
                'f1': f1_score(y_test, rf_pred, zero_division=0)
            }
            
            xgb_metrics = {
                'accuracy': accuracy_score(y_test, xgb_pred),
                'precision': precision_score(y_test, xgb_pred, zero_division=0),
                'recall': recall_score(y_test, xgb_pred, zero_division=0),
                'f1': f1_score(y_test, xgb_pred, zero_division=0)
            }
            
            self._save_models()
            self._save_performance_metrics('RandomForest', rf_metrics, len(trades))
            self._save_performance_metrics('XGBoost', xgb_metrics, len(trades))
            
            print(f"Models trained successfully on {len(trades)} trades")
            print(f"RF Accuracy: {rf_metrics['accuracy']:.2%}, XGB Accuracy: {xgb_metrics['accuracy']:.2%}")
            
            return True
            
        except Exception as e:
            print(f"Error training models: {e}")
            return False
        finally:
            session.close()
    
    def predict(self, indicators):
        models_available = False
        if self.rf_model is None or self.xgb_model is None:
            models_available = self._load_models()
        else:
            models_available = True
        
        rule_based_result = self._rule_based_prediction(indicators)
        
        if not models_available:
            return rule_based_result
        
        try:
            features = self.prepare_features(indicators)
            
            # Safety check: Ensure scaler is fitted before using
            if not hasattr(self.scaler, 'n_features_in_'):
                print("⚠️  Scaler not fitted - using rule-based prediction")
                return rule_based_result
            
            # Feature dimension compatibility check
            expected_features = self.scaler.n_features_in_
            actual_features = features.shape[1]
            
            if expected_features != actual_features:
                print(f"⚠️  Model dimension mismatch: Expected {expected_features} features, got {actual_features}")
                print(f"   Models were trained with old feature set - invalidating and using rule-based prediction")
                print(f"   System will retrain automatically when enough new trades accumulate (10+ trades needed)")
                
                # Invalidate old models and reinitialize scaler for future training
                self.rf_model = None
                self.xgb_model = None
                self.scaler = StandardScaler()  # Fresh scaler for retraining with new features
                
                # Delete outdated models from database
                try:
                    session = get_session()
                    deleted_count = session.query(MLModel).delete()
                    session.commit()
                    session.close()
                    if deleted_count > 0:
                        print(f"   ✅ Deleted {deleted_count} outdated models from database")
                except Exception as e:
                    print(f"   ⚠️  Error deleting outdated models: {e}")
                    if 'session' in locals():
                        session.rollback()
                        session.close()
                
                return rule_based_result
            
            features_scaled = self.scaler.transform(features)
            
            rf_proba = self.rf_model.predict_proba(features_scaled)[0]
            xgb_proba = self.xgb_model.predict_proba(features_scaled)[0]
            
            ensemble_proba = (rf_proba + xgb_proba) / 2
            ml_win_probability = ensemble_proba[1]
            
            rule_score = rule_based_result['bullish_score'] - rule_based_result['bearish_score']
            rule_normalized = (rule_score + 20) / 40
            rule_normalized = max(0, min(1, rule_normalized))
            
            final_probability = (ml_win_probability * 0.7) + (rule_normalized * 0.3)
            
            current_price = indicators.get('current_price', 0)
            atr = indicators.get('ATR', current_price * 0.02)
            min_distance = max(atr, current_price * 0.002)
            
            reasons = rule_based_result.get('reasons', [])
            
            if final_probability > 0.6:
                signal = 'LONG'
                entry_price = current_price
                stop_loss = current_price - (2 * min_distance)
                take_profit = current_price + (3 * min_distance)
                recommendation = f"Strong LONG signal. Enter at {entry_price:.2f}"
                reasons.append(f"ML models: {ml_win_probability*100:.1f}%, Weighted rules: {rule_normalized*100:.1f}%, Final: {final_probability*100:.1f}%")
            elif final_probability < 0.4:
                signal = 'SHORT'
                entry_price = current_price
                stop_loss = current_price + (2 * min_distance)
                take_profit = current_price - (3 * min_distance)
                recommendation = f"Strong SHORT signal. Enter at {entry_price:.2f}"
                reasons.append(f"ML models: {(1-ml_win_probability)*100:.1f}%, Weighted rules: {(1-rule_normalized)*100:.1f}%, Final: {(1-final_probability)*100:.1f}%")
            else:
                signal = 'HOLD'
                entry_price = None
                stop_loss = None
                take_profit = None
                recommendation = "No clear signal. Wait for better opportunity."
                reasons.append(f"Hybrid confidence below threshold (final: {final_probability*100:.1f}%)")
            
            return {
                'signal': signal,
                'confidence': round(max(final_probability, 1 - final_probability) * 100, 2),
                'win_probability': round(final_probability * 100, 2),
                'recommendation': recommendation,
                'entry_price': round(entry_price, 2) if entry_price else None,
                'stop_loss': round(stop_loss, 2) if stop_loss else None,
                'take_profit': round(take_profit, 2) if take_profit else None,
                'rf_confidence': round(rf_proba[1] * 100, 2),
                'xgb_confidence': round(xgb_proba[1] * 100, 2),
                'ml_probability': round(ml_win_probability * 100, 2),
                'rule_probability': round(rule_normalized * 100, 2),
                'method': 'hybrid',
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
    
    def _save_models(self):
        try:
            session = get_session()
            
            def serialize_model(model):
                buffer = BytesIO()
                joblib.dump(model, buffer)
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode('utf-8')
            
            rf_data = serialize_model(self.rf_model)
            xgb_data = serialize_model(self.xgb_model)
            scaler_data = serialize_model(self.scaler)
            
            for model_name, model_data in [
                ('rf_model', rf_data),
                ('xgb_model', xgb_data),
                ('scaler', scaler_data)
            ]:
                existing = session.query(MLModel).filter(MLModel.model_name == model_name).first()
                if existing:
                    existing.model_data = model_data
                    existing.updated_at = datetime.utcnow()
                    existing.version += 1
                else:
                    new_model = MLModel(
                        model_name=model_name,
                        model_data=model_data
                    )
                    session.add(new_model)
            
            session.commit()
            session.close()
            print("✅ Models saved to database successfully")
        except Exception as e:
            print(f"❌ Error saving models to database: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
    
    def _load_models(self):
        try:
            session = get_session()
            
            def deserialize_model(model_data):
                decoded = base64.b64decode(model_data.encode('utf-8'))
                buffer = BytesIO(decoded)
                return joblib.load(buffer)
            
            rf_record = session.query(MLModel).filter(MLModel.model_name == 'rf_model').first()
            xgb_record = session.query(MLModel).filter(MLModel.model_name == 'xgb_model').first()
            scaler_record = session.query(MLModel).filter(MLModel.model_name == 'scaler').first()
            
            session.close()
            
            if rf_record and xgb_record and scaler_record:
                self.rf_model = deserialize_model(rf_record.model_data)
                self.xgb_model = deserialize_model(xgb_record.model_data)
                self.scaler = deserialize_model(scaler_record.model_data)
                print(f"✅ Models loaded from database (version: {rf_record.version})")
                return True
            else:
                print("⚠️  No models found in database")
                return False
        except Exception as e:
            print(f"❌ Error loading models from database: {e}")
            if 'session' in locals():
                session.close()
            return False
    
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
            duration = rsi_ctx.get('duration_candles', 0)
            slope = rsi_ctx.get('slope', 0.0)
            divergence = rsi_ctx.get('divergence', 'none')
            
            if rsi < 30:
                base_signal = 2 * rsi_weight
                # Duration bonus: longer oversold = stronger reversal signal
                duration_bonus = min(2.0, duration / 10) if duration > 10 else 0
                # Momentum bonus: rising while oversold = bullish
                momentum_bonus = 1.0 if slope > 0.5 else 0
                # Divergence bonus
                divergence_bonus = 1.5 if divergence == 'bullish' else 0
                
                total_signal = base_signal + duration_bonus + momentum_bonus + divergence_bonus
                bullish_signals += total_signal
                
                context_info = []
                if duration > 10:
                    context_info.append(f"{duration} candles")
                if slope > 0.5:
                    context_info.append("rising momentum")
                if divergence == 'bullish':
                    context_info.append("bullish divergence")
                
                reason = f"RSI oversold ({rsi:.1f})"
                if context_info:
                    reason += f" [{', '.join(context_info)}]"
                reason += f" - strong buy signal (weight: {rsi_weight:.1f}x + {total_signal - base_signal:.1f} bonus)"
                reasons.append(reason)
                
            elif rsi < 40:
                bullish_signals += 1 * rsi_weight
                reasons.append(f"RSI below 40 ({rsi:.1f}) - buy signal (weight: {rsi_weight:.1f}x)")
                
            elif rsi > 70:
                base_signal = 2 * rsi_weight
                # Duration bonus: longer overbought = stronger reversal signal
                duration_bonus = min(2.0, duration / 10) if duration > 10 else 0
                # Momentum bonus: falling while overbought = bearish
                momentum_bonus = 1.0 if slope < -0.5 else 0
                # Divergence bonus
                divergence_bonus = 1.5 if divergence == 'bearish' else 0
                
                total_signal = base_signal + duration_bonus + momentum_bonus + divergence_bonus
                bearish_signals += total_signal
                
                context_info = []
                if duration > 10:
                    context_info.append(f"{duration} candles")
                if slope < -0.5:
                    context_info.append("falling momentum")
                if divergence == 'bearish':
                    context_info.append("bearish divergence")
                
                reason = f"RSI overbought ({rsi:.1f})"
                if context_info:
                    reason += f" [{', '.join(context_info)}]"
                reason += f" - strong sell signal (weight: {rsi_weight:.1f}x + {total_signal - base_signal:.1f} bonus)"
                reasons.append(reason)
                
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
            duration = stoch_ctx.get('duration_candles', 0)
            slope = stoch_ctx.get('slope', 0.0)
            divergence = stoch_ctx.get('divergence', 'none')
            
            if stoch_k < 20:
                base_signal = 1 * stoch_weight
                duration_bonus = min(1.0, duration / 15) if duration > 10 else 0
                momentum_bonus = 0.5 if slope > 0.5 else 0
                divergence_bonus = 1.0 if divergence == 'bullish' else 0
                
                total_signal = base_signal + duration_bonus + momentum_bonus + divergence_bonus
                bullish_signals += total_signal
                
                context_info = []
                if duration > 10:
                    context_info.append(f"{duration} candles")
                if slope > 0.5:
                    context_info.append("rising")
                if divergence == 'bullish':
                    context_info.append("bullish div")
                
                reason = f"Stochastic oversold ({stoch_k:.1f})"
                if context_info:
                    reason += f" [{', '.join(context_info)}]"
                reason += f" (weight: {stoch_weight:.1f}x + {total_signal - base_signal:.1f} bonus)"
                reasons.append(reason)
                
            elif stoch_k > 80:
                base_signal = 1 * stoch_weight
                duration_bonus = min(1.0, duration / 15) if duration > 10 else 0
                momentum_bonus = 0.5 if slope < -0.5 else 0
                divergence_bonus = 1.0 if divergence == 'bearish' else 0
                
                total_signal = base_signal + duration_bonus + momentum_bonus + divergence_bonus
                bearish_signals += total_signal
                
                context_info = []
                if duration > 10:
                    context_info.append(f"{duration} candles")
                if slope < -0.5:
                    context_info.append("falling")
                if divergence == 'bearish':
                    context_info.append("bearish div")
                
                reason = f"Stochastic overbought ({stoch_k:.1f})"
                if context_info:
                    reason += f" [{', '.join(context_info)}]"
                reason += f" (weight: {stoch_weight:.1f}x + {total_signal - base_signal:.1f} bonus)"
                reasons.append(reason)
        
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
            duration = mfi_ctx.get('duration_candles', 0)
            slope = mfi_ctx.get('slope', 0.0)
            divergence = mfi_ctx.get('divergence', 'none')
            
            if mfi < 20:
                base_signal = 1 * mfi_weight
                duration_bonus = min(1.0, duration / 15) if duration > 10 else 0
                momentum_bonus = 0.5 if slope > 0.5 else 0
                divergence_bonus = 1.0 if divergence == 'bullish' else 0
                
                total_signal = base_signal + duration_bonus + momentum_bonus + divergence_bonus
                bullish_signals += total_signal
                
                context_info = []
                if duration > 10:
                    context_info.append(f"{duration} candles")
                if slope > 0.5:
                    context_info.append("rising")
                if divergence == 'bullish':
                    context_info.append("bullish div")
                
                reason = f"MFI oversold ({mfi:.1f})"
                if context_info:
                    reason += f" [{', '.join(context_info)}]"
                reason += f" (weight: {mfi_weight:.1f}x + {total_signal - base_signal:.1f} bonus)"
                reasons.append(reason)
                
            elif mfi > 80:
                base_signal = 1 * mfi_weight
                duration_bonus = min(1.0, duration / 15) if duration > 10 else 0
                momentum_bonus = 0.5 if slope < -0.5 else 0
                divergence_bonus = 1.0 if divergence == 'bearish' else 0
                
                total_signal = base_signal + duration_bonus + momentum_bonus + divergence_bonus
                bearish_signals += total_signal
                
                context_info = []
                if duration > 10:
                    context_info.append(f"{duration} candles")
                if slope < -0.5:
                    context_info.append("falling")
                if divergence == 'bearish':
                    context_info.append("bearish div")
                
                reason = f"MFI overbought ({mfi:.1f})"
                if context_info:
                    reason += f" [{', '.join(context_info)}]"
                reason += f" (weight: {mfi_weight:.1f}x + {total_signal - base_signal:.1f} bonus)"
                reasons.append(reason)
        
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
            take_profit = current_price + (3 * min_distance)
            recommendation = f"Rule-based LONG signal (no ML training yet)"
        elif bearish_signals > bullish_signals * 1.5:
            signal = 'SHORT'
            confidence = min(95, (bearish_signals / total_signals) * 100)
            entry_price = current_price
            stop_loss = current_price + (2 * min_distance)
            take_profit = current_price - (3 * min_distance)
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
            print(f"   Direction: {trade.direction}, Outcome: {trade.outcome}")
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
                if ind['RSI'] < 30 and trade.direction == 'LONG':
                    rsi_oversold_wins += 1
                elif ind['RSI'] > 70 and trade.direction == 'SHORT':
                    rsi_overbought_wins += 1
            
            if ind.get('MACD') and ind.get('MACD_signal'):
                if ind['MACD'] > ind['MACD_signal'] and trade.direction == 'LONG':
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
