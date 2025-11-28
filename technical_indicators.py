import pandas as pd
import numpy as np
import pandas_ta as ta

class TechnicalIndicators:
    def __init__(self, df):
        self.df = df.copy()
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
    def calculate_all_indicators(self):
        df = self.df.copy()
        
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # ========== NEW: Multi-Timeframe RSI for Momentum Timing ==========
        df['RSI_6'] = ta.rsi(df['close'], length=6)    # Fast RSI (short-term momentum)
        df['RSI_12'] = ta.rsi(df['close'], length=12)  # Medium RSI
        df['RSI_24'] = ta.rsi(df['close'], length=24)  # Slow RSI (longer-term momentum)
        
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']
            df['MACD_hist'] = macd['MACDh_12_26_9']
        
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        if stoch is not None:
            df['Stoch_K'] = stoch['STOCHk_14_3_3']
            df['Stoch_D'] = stoch['STOCHd_14_3_3']
            # ========== NEW: KDJ J Line for Momentum Timing ==========
            # J = 3*K - 2*D (leads K and D, more sensitive to reversals)
            df['Stoch_J'] = 3 * df['Stoch_K'] - 2 * df['Stoch_D']
        
        df['OBV'] = ta.obv(df['close'], df['volume'])
        
        df['MFI'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        
        df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['ADX'] = adx['ADX_14']
            df['DI_plus'] = adx['DMP_14']
            df['DI_minus'] = adx['DMN_14']
        
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['SMA_200'] = ta.sma(df['close'], length=200)
        
        df['EMA_12'] = ta.ema(df['close'], length=12)
        df['EMA_26'] = ta.ema(df['close'], length=26)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None and not bbands.empty:
            bbu_col = [col for col in bbands.columns if col.startswith('BBU')]
            bbm_col = [col for col in bbands.columns if col.startswith('BBM')]
            bbl_col = [col for col in bbands.columns if col.startswith('BBL')]
            
            if bbu_col and bbm_col and bbl_col:
                df['BB_upper'] = bbands[bbu_col[0]]
                df['BB_middle'] = bbands[bbm_col[0]]
                df['BB_lower'] = bbands[bbl_col[0]]
        
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        df['Volume_SMA'] = ta.sma(df['volume'], length=20)
        
        # ========== NEW: Volatility Features for Regime Classification ==========
        # ATR Percentile (relative to last 200 candles)
        if len(df) >= 200:
            df['ATR_percentile'] = df['ATR'].rolling(window=200, min_periods=50).apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50.0
            )
        else:
            df['ATR_percentile'] = df['ATR'].expanding(min_periods=20).apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50.0
            )
        
        # Bollinger Band Width Percentage (normalized volatility)
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns and 'BB_middle' in df.columns:
            df['BB_width_pct'] = ((df['BB_upper'] - df['BB_lower']) / df['BB_middle']) * 100
        
        # Rolling Variance (14 and 50 period price variance)
        df['variance_14'] = df['close'].pct_change().rolling(window=14).var() * 100
        df['variance_50'] = df['close'].pct_change().rolling(window=50).var() * 100
        
        # Wick-to-Body Ratio (measures panic/indecision)
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['total_wick'] = df['upper_wick'] + df['lower_wick']
        df['wick_to_body_ratio'] = df['total_wick'] / (df['body_size'] + 0.0001)  # Avoid division by zero
        
        # ATR as percentage of price (normalized volatility)
        df['ATR_pct_price'] = (df['ATR'] / df['close']) * 100
        
        self.df = df
        return df
    
    def get_latest_indicators(self):
        if len(self.df) == 0:
            return {}
        
        latest = self.df.iloc[-1]
        indicators = {
            'RSI': latest.get('RSI'),
            # NEW: Multi-timeframe RSI for momentum timing
            'RSI_6': latest.get('RSI_6'),
            'RSI_12': latest.get('RSI_12'),
            'RSI_24': latest.get('RSI_24'),
            'MACD': latest.get('MACD'),
            'MACD_signal': latest.get('MACD_signal'),
            'MACD_hist': latest.get('MACD_hist'),
            'Stoch_K': latest.get('Stoch_K'),
            'Stoch_D': latest.get('Stoch_D'),
            # NEW: KDJ J line for momentum timing
            'Stoch_J': latest.get('Stoch_J'),
            'OBV': latest.get('OBV'),
            'MFI': latest.get('MFI'),
            'CCI': latest.get('CCI'),
            'ADX': latest.get('ADX'),
            'DI_plus': latest.get('DI_plus'),
            'DI_minus': latest.get('DI_minus'),
            'SMA_20': latest.get('SMA_20'),
            'SMA_50': latest.get('SMA_50'),
            'SMA_200': latest.get('SMA_200'),
            'EMA_12': latest.get('EMA_12'),
            'EMA_26': latest.get('EMA_26'),
            'EMA_50': latest.get('EMA_50'),
            'BB_upper': latest.get('BB_upper'),
            'BB_middle': latest.get('BB_middle'),
            'BB_lower': latest.get('BB_lower'),
            'ATR': latest.get('ATR'),
            'Volume_SMA': latest.get('Volume_SMA'),
            'current_price': latest.get('close'),
            'volume': latest.get('volume'),
            # NEW: Volatility features
            'ATR_percentile': latest.get('ATR_percentile'),
            'BB_width_pct': latest.get('BB_width_pct'),
            'variance_14': latest.get('variance_14'),
            'variance_50': latest.get('variance_50'),
            'wick_to_body_ratio': latest.get('wick_to_body_ratio'),
            'ATR_pct_price': latest.get('ATR_pct_price')
        }
        
        return {k: float(v) if pd.notna(v) else None for k, v in indicators.items()}
    
    def get_trend_signals(self):
        indicators = self.get_latest_indicators()
        signals = {}
        
        if indicators.get('RSI') is not None:
            rsi = indicators['RSI']
            if rsi < 30:
                signals['RSI'] = 'oversold'
            elif rsi > 70:
                signals['RSI'] = 'overbought'
            else:
                signals['RSI'] = 'neutral'
        
        if indicators.get('MACD') and indicators.get('MACD_signal'):
            if indicators['MACD'] > indicators['MACD_signal']:
                signals['MACD'] = 'bullish'
            else:
                signals['MACD'] = 'bearish'
        
        if indicators.get('Stoch_K') is not None:
            stoch_k = indicators['Stoch_K']
            if stoch_k < 20:
                signals['Stochastic'] = 'oversold'
            elif stoch_k > 80:
                signals['Stochastic'] = 'overbought'
            else:
                signals['Stochastic'] = 'neutral'
        
        if indicators.get('ADX') is not None:
            adx = indicators['ADX']
            if adx > 25:
                if indicators.get('DI_plus', 0) > indicators.get('DI_minus', 0):
                    signals['ADX'] = 'strong_uptrend'
                else:
                    signals['ADX'] = 'strong_downtrend'
            else:
                signals['ADX'] = 'weak_trend'
        
        current_price = indicators.get('current_price')
        if current_price and indicators.get('BB_upper') and indicators.get('BB_lower'):
            if current_price > indicators['BB_upper']:
                signals['Bollinger'] = 'overbought'
            elif current_price < indicators['BB_lower']:
                signals['Bollinger'] = 'oversold'
            else:
                signals['Bollinger'] = 'neutral'
        
        if current_price:
            ma_signals = []
            if indicators.get('SMA_20'):
                ma_signals.append('bullish' if current_price > indicators['SMA_20'] else 'bearish')
            if indicators.get('SMA_50'):
                ma_signals.append('bullish' if current_price > indicators['SMA_50'] else 'bearish')
            if indicators.get('EMA_12'):
                ma_signals.append('bullish' if current_price > indicators['EMA_12'] else 'bearish')
            
            if ma_signals:
                bullish_count = ma_signals.count('bullish')
                signals['Moving_Averages'] = 'bullish' if bullish_count > len(ma_signals) / 2 else 'bearish'
        
        return signals
    
    def get_momentum_timing(self, timeframe_minutes=60):
        """
        Analyze multi-timeframe RSI and KDJ to estimate momentum persistence.
        Returns timing analysis showing how long current momentum is likely to continue.
        
        Args:
            timeframe_minutes: The selected chart timeframe in minutes (60=1H, 240=4H, etc.)
        
        Uses:
        - RSI_6, RSI_12, RSI_24: Multi-timeframe momentum alignment
        - Stoch_K, Stoch_D, Stoch_J: KDJ momentum dynamics
        
        Returns dict with:
        - momentum_direction: 'bullish', 'bearish', or 'neutral'
        - estimated_candles: estimated candles before reversal likely
        - estimated_hours: estimated hours before reversal (timeframe-aware)
        - timeframe_label: human-readable timeframe (e.g., "1H", "4H")
        - rsi_alignment: how RSI timeframes are aligned
        - kdj_dynamics: J/K/D relationship analysis
        - timing_confidence: confidence in timing estimate
        - advisory: human-readable timing advisory
        """
        indicators = self.get_latest_indicators()
        
        # Convert timeframe to label for display
        timeframe_labels = {5: '5m', 15: '15m', 30: '30m', 60: '1H', 240: '4H', 1440: '1D'}
        timeframe_label = timeframe_labels.get(timeframe_minutes, f'{timeframe_minutes}m')
        
        # Get multi-timeframe RSI values
        rsi_6 = indicators.get('RSI_6')
        rsi_12 = indicators.get('RSI_12')
        rsi_24 = indicators.get('RSI_24')
        rsi_14 = indicators.get('RSI')
        
        # Get KDJ values
        stoch_k = indicators.get('Stoch_K')
        stoch_d = indicators.get('Stoch_D')
        stoch_j = indicators.get('Stoch_J')
        
        # Initialize result
        result = {
            'momentum_direction': 'neutral',
            'estimated_candles': 0,
            'estimated_hours': 0,
            'timeframe_minutes': timeframe_minutes,
            'timeframe_label': timeframe_label,
            'rsi_alignment': 'neutral',
            'kdj_dynamics': 'neutral',
            'timing_confidence': 0.0,
            'advisory': '',
            'details': {}
        }
        
        # Check if we have all required data
        if None in [rsi_6, rsi_12, rsi_24, stoch_k, stoch_d, stoch_j]:
            result['advisory'] = 'Insufficient data for timing analysis'
            return result
        
        # Store raw values for display
        result['details'] = {
            'RSI_6': round(rsi_6, 2),
            'RSI_12': round(rsi_12, 2),
            'RSI_24': round(rsi_24, 2),
            'Stoch_K': round(stoch_k, 2),
            'Stoch_D': round(stoch_d, 2),
            'Stoch_J': round(stoch_j, 2)
        }
        
        # ========== RSI Multi-Timeframe Analysis ==========
        # When RSI_6 > RSI_12 > RSI_24: Momentum BUILDING (bullish accelerating)
        # When RSI_6 < RSI_12 < RSI_24: Momentum FADING (bearish accelerating)
        # When aligned but RSI_6 < RSI_12: Momentum starting to weaken
        
        rsi_bullish_building = rsi_6 > rsi_12 > rsi_24
        rsi_bearish_building = rsi_6 < rsi_12 < rsi_24
        rsi_bullish_fading = rsi_6 < rsi_12 and rsi_12 >= rsi_24 and rsi_6 > 50
        rsi_bearish_fading = rsi_6 > rsi_12 and rsi_12 <= rsi_24 and rsi_6 < 50
        
        # Determine RSI alignment
        if rsi_bullish_building:
            result['rsi_alignment'] = 'bullish_accelerating'
            rsi_momentum = 'bullish'
            rsi_candles = 4 + (rsi_6 - rsi_24) / 10  # More spread = more momentum
        elif rsi_bearish_building:
            result['rsi_alignment'] = 'bearish_accelerating'
            rsi_momentum = 'bearish'
            rsi_candles = 4 + (rsi_24 - rsi_6) / 10
        elif rsi_bullish_fading:
            result['rsi_alignment'] = 'bullish_weakening'
            rsi_momentum = 'bullish_weak'
            rsi_candles = 2  # Reversal likely soon
        elif rsi_bearish_fading:
            result['rsi_alignment'] = 'bearish_weakening'
            rsi_momentum = 'bearish_weak'
            rsi_candles = 2
        else:
            result['rsi_alignment'] = 'neutral'
            rsi_momentum = 'neutral'
            rsi_candles = 1
        
        # Overbought/Oversold extremes reduce timing
        if rsi_6 > 80:
            rsi_candles = max(1, rsi_candles - 2)  # Near extreme, reversal soon
        elif rsi_6 < 20:
            rsi_candles = max(1, rsi_candles - 2)
        
        # ========== KDJ Dynamics Analysis ==========
        # J is the leading indicator (most sensitive)
        # K is medium speed
        # D is the slowest (lagging)
        #
        # J > K > D: Bullish momentum building
        # J < K < D: Bearish momentum building
        # J peaked (>100 or starting to fall while K/D still rising): Reversal starting
        # J bottomed (<0 or starting to rise while K/D still falling): Reversal starting
        
        kdj_bullish = stoch_j > stoch_k > stoch_d
        kdj_bearish = stoch_j < stoch_k < stoch_d
        kdj_j_peaked = stoch_j > 100 or (stoch_j < stoch_k and stoch_k > stoch_d)
        kdj_j_bottomed = stoch_j < 0 or (stoch_j > stoch_k and stoch_k < stoch_d)
        
        if kdj_bullish and not kdj_j_peaked:
            result['kdj_dynamics'] = 'bullish_aligned'
            kdj_momentum = 'bullish'
            kdj_candles = 3 + (stoch_j - stoch_d) / 20
        elif kdj_bearish and not kdj_j_bottomed:
            result['kdj_dynamics'] = 'bearish_aligned'
            kdj_momentum = 'bearish'
            kdj_candles = 3 + (stoch_d - stoch_j) / 20
        elif kdj_j_peaked:
            result['kdj_dynamics'] = 'j_peaked_reversal_likely'
            kdj_momentum = 'reversal_down'
            kdj_candles = 1
        elif kdj_j_bottomed:
            result['kdj_dynamics'] = 'j_bottomed_reversal_likely'
            kdj_momentum = 'reversal_up'
            kdj_candles = 1
        else:
            result['kdj_dynamics'] = 'neutral'
            kdj_momentum = 'neutral'
            kdj_candles = 2
        
        # Extreme J values reduce timing
        if stoch_j > 100:
            kdj_candles = max(1, kdj_candles - 1)
        elif stoch_j < 0:
            kdj_candles = max(1, kdj_candles - 1)
        
        # ========== Combine RSI and KDJ for Overall Timing ==========
        estimated_candles = (rsi_candles + kdj_candles) / 2
        
        # Determine overall momentum direction
        if rsi_momentum in ['bullish', 'bullish_weak'] and kdj_momentum in ['bullish', 'reversal_up']:
            result['momentum_direction'] = 'bullish'
            result['timing_confidence'] = 0.8 if rsi_momentum == 'bullish' and kdj_momentum == 'bullish' else 0.5
        elif rsi_momentum in ['bearish', 'bearish_weak'] and kdj_momentum in ['bearish', 'reversal_down']:
            result['momentum_direction'] = 'bearish'
            result['timing_confidence'] = 0.8 if rsi_momentum == 'bearish' and kdj_momentum == 'bearish' else 0.5
        elif kdj_momentum in ['reversal_up', 'reversal_down']:
            result['momentum_direction'] = 'reversal_imminent'
            result['timing_confidence'] = 0.7
            estimated_candles = 1
        else:
            result['momentum_direction'] = 'mixed'
            result['timing_confidence'] = 0.4
        
        result['estimated_candles'] = round(max(1, estimated_candles), 1)
        
        # ========== Calculate Actual Time from Candles ==========
        estimated_hours = (result['estimated_candles'] * timeframe_minutes) / 60
        result['estimated_hours'] = round(estimated_hours, 1)
        
        # Format time for display (hours or days)
        if estimated_hours >= 24:
            time_display = f"~{estimated_hours/24:.1f} days"
        elif estimated_hours >= 1:
            time_display = f"~{estimated_hours:.0f} hours"
        else:
            time_display = f"~{estimated_hours*60:.0f} mins"
        
        candle_display = f"{result['estimated_candles']:.0f} {timeframe_label} candles"
        
        # ========== Generate Advisory ==========
        if result['momentum_direction'] == 'bullish':
            if estimated_candles >= 3:
                result['advisory'] = f"Bullish momentum likely persists {candle_display} ({time_display}) - RSI: {result['rsi_alignment']}, KDJ: {result['kdj_dynamics']}"
            else:
                result['advisory'] = f"Bullish momentum weakening, ~{candle_display} ({time_display}) before potential reversal"
        elif result['momentum_direction'] == 'bearish':
            if estimated_candles >= 3:
                result['advisory'] = f"Bearish momentum likely persists {candle_display} ({time_display}) - RSI: {result['rsi_alignment']}, KDJ: {result['kdj_dynamics']}"
            else:
                result['advisory'] = f"Bearish momentum weakening, ~{candle_display} ({time_display}) before potential reversal"
        elif result['momentum_direction'] == 'reversal_imminent':
            result['advisory'] = f"Reversal signals detected! J-line {'peaked' if kdj_j_peaked else 'bottomed'} - expect direction change within 1-2 {timeframe_label} candles ({time_display})"
        else:
            result['advisory'] = f"Mixed signals - momentum unclear, wait for alignment (RSI: {result['rsi_alignment']}, KDJ: {result['kdj_dynamics']})"
        
        return result
    
    def detect_candlestick_patterns(self):
        """Detect common candlestick patterns in the most recent candles"""
        if len(self.df) < 3:
            return {}
        
        patterns = {}
        
        # Get last 3 candles
        df = self.df.tail(3).reset_index(drop=True)
        
        # Current candle (most recent)
        curr = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else None
        prev2 = df.iloc[-3] if len(df) >= 3 else None
        
        # Calculate candle properties
        curr_body = abs(curr['close'] - curr['open'])
        curr_range = curr['high'] - curr['low']
        curr_upper_shadow = curr['high'] - max(curr['open'], curr['close'])
        curr_lower_shadow = min(curr['open'], curr['close']) - curr['low']
        
        # HAMMER (Bullish reversal)
        # Small body at top, long lower shadow (2-3x body), small upper shadow
        if curr_range > 0:
            if (curr_lower_shadow >= 2 * curr_body and 
                curr_upper_shadow <= 0.1 * curr_range and
                curr_body >= 0.1 * curr_range):
                patterns['Hammer'] = 'bullish'
        
        # SHOOTING STAR (Bearish reversal)
        # Small body at bottom, long upper shadow (2-3x body), small lower shadow
        if curr_range > 0:
            if (curr_upper_shadow >= 2 * curr_body and
                curr_lower_shadow <= 0.1 * curr_range and
                curr_body >= 0.1 * curr_range):
                patterns['Shooting_Star'] = 'bearish'
        
        # DOJI (Indecision)
        # Very small body (open ≈ close)
        if curr_range > 0 and curr_body <= 0.05 * curr_range:
            patterns['Doji'] = 'neutral'
        
        # BULLISH ENGULFING (Bullish reversal)
        if prev is not None:
            prev_body = abs(prev['close'] - prev['open'])
            if (prev['close'] < prev['open'] and  # Previous red
                curr['close'] > curr['open'] and  # Current green
                curr['open'] < prev['close'] and  # Opens below prev close
                curr['close'] > prev['open']):    # Closes above prev open
                patterns['Bullish_Engulfing'] = 'bullish'
        
        # BEARISH ENGULFING (Bearish reversal)
        if prev is not None:
            if (prev['close'] > prev['open'] and  # Previous green
                curr['close'] < curr['open'] and  # Current red
                curr['open'] > prev['close'] and  # Opens above prev close
                curr['close'] < prev['open']):    # Closes below prev open
                patterns['Bearish_Engulfing'] = 'bearish'
        
        # MORNING STAR (Bullish reversal - 3 candle pattern)
        if prev is not None and prev2 is not None:
            prev2_body = abs(prev2['close'] - prev2['open'])
            prev_body = abs(prev['close'] - prev['open'])
            
            if (prev2['close'] < prev2['open'] and  # First: large red
                prev_body < prev2_body * 0.3 and     # Second: small body (doji-like)
                curr['close'] > curr['open'] and     # Third: green
                curr['close'] > (prev2['open'] + prev2['close']) / 2):  # Closes above midpoint
                patterns['Morning_Star'] = 'bullish'
        
        # EVENING STAR (Bearish reversal - 3 candle pattern)
        if prev is not None and prev2 is not None:
            prev2_body = abs(prev2['close'] - prev2['open'])
            prev_body = abs(prev['close'] - prev['open'])
            
            if (prev2['close'] > prev2['open'] and  # First: large green
                prev_body < prev2_body * 0.3 and     # Second: small body (doji-like)
                curr['close'] < curr['open'] and     # Third: red
                curr['close'] < (prev2['open'] + prev2['close']) / 2):  # Closes below midpoint
                patterns['Evening_Star'] = 'bearish'
        
        return patterns
    
    def get_pattern_signals(self):
        """Get trading signals from candlestick patterns"""
        patterns = self.detect_candlestick_patterns()
        
        if not patterns:
            return 'neutral'
        
        # Count bullish vs bearish signals
        bullish_count = sum(1 for signal in patterns.values() if signal == 'bullish')
        bearish_count = sum(1 for signal in patterns.values() if signal == 'bearish')
        
        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'
    
    def get_indicator_history(self, symbol, market_type, lookback=50):
        """
        Query historical indicator data from MarketData table.
        Returns list of indicator snapshots (most recent last).
        """
        from database import get_session, MarketData
        
        session = get_session()
        try:
            # Query last N market data records for this symbol
            records = session.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.market_type == market_type,
                MarketData.indicators.isnot(None)
            ).order_by(MarketData.timestamp.desc()).limit(lookback).all()
            
            # Reverse to get chronological order (oldest first)
            records = list(reversed(records))
            
            # Extract indicator values from JSON
            history = []
            for record in records:
                if record.indicators:
                    snapshot = {
                        'timestamp': record.timestamp,
                        'close_price': record.close_price,
                        **record.indicators  # Unpack all indicators
                    }
                    history.append(snapshot)
            
            return history
            
        except Exception as e:
            print(f"Error getting indicator history: {e}")
            return []
        finally:
            session.close()
    
    def _calculate_duration(self, history, indicator_name, threshold_low, threshold_high):
        """
        Calculate how many consecutive candles an indicator has been in a zone.
        Returns: (duration_candles, zone_type)
        zone_type: 'oversold', 'overbought', 'neutral'
        """
        if not history or len(history) == 0:
            return 0, 'neutral'
        
        duration = 0
        zone_type = 'neutral'
        
        # Check current state
        current_value = history[-1].get(indicator_name)
        if current_value is None:
            return 0, 'neutral'
        
        if current_value < threshold_low:
            zone_type = 'oversold'
        elif current_value > threshold_high:
            zone_type = 'overbought'
        else:
            return 0, 'neutral'
        
        # Count backwards while in same zone
        for i in range(len(history) - 1, -1, -1):
            value = history[i].get(indicator_name)
            if value is None:
                break
            
            if zone_type == 'oversold' and value < threshold_low:
                duration += 1
            elif zone_type == 'overbought' and value > threshold_high:
                duration += 1
            else:
                break
        
        return duration, zone_type
    
    def _calculate_slope(self, history, indicator_name, window=10):
        """
        Calculate slope/momentum of an indicator using linear regression.
        Positive slope = rising, negative slope = falling.
        """
        if not history or len(history) < window:
            return 0.0
        
        # Get last N values
        recent = history[-window:]
        values = [h.get(indicator_name) for h in recent if h.get(indicator_name) is not None]
        
        if len(values) < 3:
            return 0.0
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values, dtype=float)
        
        # Calculate slope using least squares
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) ** 2)
        
        return float(slope)
    
    def _detect_divergence(self, history, indicator_name, window=10):
        """
        Detect divergence between price and indicator.
        Bullish divergence: Price falling but indicator rising
        Bearish divergence: Price rising but indicator falling
        """
        if not history or len(history) < window:
            return 'none'
        
        # Get price and indicator slopes
        recent = history[-window:]
        prices = [h.get('close_price') for h in recent if h.get('close_price') is not None]
        indicator_values = [h.get(indicator_name) for h in recent if h.get(indicator_name) is not None]
        
        if len(prices) < 3 or len(indicator_values) < 3:
            return 'none'
        
        # Calculate slopes
        price_x = np.arange(len(prices))
        price_y = np.array(prices, dtype=float)
        price_slope = (len(price_x) * np.sum(price_x * price_y) - np.sum(price_x) * np.sum(price_y)) / \
                     (len(price_x) * np.sum(price_x * price_x) - np.sum(price_x) ** 2)
        
        ind_x = np.arange(len(indicator_values))
        ind_y = np.array(indicator_values, dtype=float)
        ind_slope = (len(ind_x) * np.sum(ind_x * ind_y) - np.sum(ind_x) * np.sum(ind_y)) / \
                   (len(ind_x) * np.sum(ind_x * ind_x) - np.sum(ind_x) ** 2)
        
        # Detect divergence (slopes have opposite signs)
        if price_slope < -0.01 and ind_slope > 0.01:
            return 'bullish'  # Price falling, indicator rising
        elif price_slope > 0.01 and ind_slope < -0.01:
            return 'bearish'  # Price rising, indicator falling
        else:
            return 'none'
    
    def get_trend_context(self, symbol, market_type):
        """
        Get comprehensive trend context for key indicators.
        Returns dict with duration, slope, and divergence for RSI, Stochastic, MFI, and OBV.
        """
        history = self.get_indicator_history(symbol, market_type, lookback=50)
        
        if not history:
            return {}
        
        context = {}
        
        # RSI Analysis
        rsi_duration, rsi_zone = self._calculate_duration(history, 'RSI', 30, 70)
        rsi_slope = self._calculate_slope(history, 'RSI', window=10)
        rsi_divergence = self._detect_divergence(history, 'RSI', window=10)
        
        context['RSI'] = {
            'zone': rsi_zone,
            'duration_candles': rsi_duration,
            'slope': rsi_slope,
            'divergence': rsi_divergence
        }
        
        # Stochastic Analysis
        stoch_duration, stoch_zone = self._calculate_duration(history, 'Stoch_K', 20, 80)
        stoch_slope = self._calculate_slope(history, 'Stoch_K', window=10)
        stoch_divergence = self._detect_divergence(history, 'Stoch_K', window=10)
        
        context['Stochastic'] = {
            'zone': stoch_zone,
            'duration_candles': stoch_duration,
            'slope': stoch_slope,
            'divergence': stoch_divergence
        }
        
        # MFI Analysis
        mfi_duration, mfi_zone = self._calculate_duration(history, 'MFI', 20, 80)
        mfi_slope = self._calculate_slope(history, 'MFI', window=10)
        mfi_divergence = self._detect_divergence(history, 'MFI', window=10)
        
        context['MFI'] = {
            'zone': mfi_zone,
            'duration_candles': mfi_duration,
            'slope': mfi_slope,
            'divergence': mfi_divergence
        }
        
        # OBV Analysis (Volume-based)
        # OBV doesn't have zones like RSI, only slope and divergence
        obv_slope = self._calculate_slope(history, 'OBV', window=10)
        obv_divergence = self._detect_divergence(history, 'OBV', window=10)
        
        context['OBV'] = {
            'slope': obv_slope,
            'divergence': obv_divergence
        }
        
        return context

def calculate_support_resistance(df, lookback=20):
    if len(df) < lookback:
        return [], []
    
    recent_df = df.tail(lookback * 3)
    
    highs = recent_df['high'].values
    lows = recent_df['low'].values
    
    resistance_levels = []
    support_levels = []
    
    for i in range(lookback, len(highs) - lookback):
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            resistance_levels.append(highs[i])
        
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            support_levels.append(lows[i])
    
    resistance_levels = sorted(list(set([round(r, 2) for r in resistance_levels])), reverse=True)[:3]
    support_levels = sorted(list(set([round(s, 2) for s in support_levels])), reverse=True)[:3]
    
    current_price = df.iloc[-1]['close']
    resistance_levels = [r for r in resistance_levels if r > current_price]
    support_levels = [s for s in support_levels if s < current_price]
    
    pivot_high = df.tail(10)['high'].max()
    pivot_low = df.tail(10)['low'].min()
    pivot_close = df.iloc[-1]['close']
    
    pivot = (pivot_high + pivot_low + pivot_close) / 3
    r1 = 2 * pivot - pivot_low
    s1 = 2 * pivot - pivot_high
    r2 = pivot + (pivot_high - pivot_low)
    s2 = pivot - (pivot_high - pivot_low)
    
    for level in [r1, r2]:
        if level > current_price and level not in resistance_levels:
            resistance_levels.append(round(level, 2))
    
    for level in [s1, s2]:
        if level < current_price and level not in support_levels:
            support_levels.append(round(level, 2))
    
    resistance_levels = sorted(resistance_levels)[:3]
    support_levels = sorted(support_levels, reverse=True)[:3]
    
    return support_levels, resistance_levels
