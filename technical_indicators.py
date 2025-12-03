import pandas as pd
import numpy as np
import pandas_ta as ta

class TechnicalIndicators:
    def __init__(self, df):
        self.df = df.copy()
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        self.data_quality_issues = []
        self._validate_data()
        
    def _validate_data(self):
        """
        Validate OHLCV data quality before indicator calculations.
        Checks for: duplicates, missing volume, outliers, minimum data.
        Does NOT modify data - only logs issues for transparency.
        """
        self.data_quality_issues = []
        df = self.df
        
        # 1. Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicates = df['timestamp'].duplicated().sum()
            if duplicates > 0:
                self.data_quality_issues.append(f"⚠️ {duplicates} duplicate timestamps found")
                # Remove duplicates, keep last (most recent data)
                self.df = df.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
                df = self.df
        
        # 2. Check for missing/zero volume
        if 'volume' in df.columns:
            zero_volume = (df['volume'] == 0).sum() + df['volume'].isna().sum()
            if zero_volume > len(df) * 0.1:  # More than 10% missing volume
                self.data_quality_issues.append(f"⚠️ {zero_volume}/{len(df)} candles have zero/missing volume")
        
        # 3. Check for price outliers (>50% single-candle moves)
        if 'close' in df.columns and len(df) > 1:
            pct_changes = df['close'].pct_change().abs()
            extreme_moves = (pct_changes > 0.5).sum()  # 50% moves
            if extreme_moves > 0:
                self.data_quality_issues.append(f"⚠️ {extreme_moves} extreme price moves (>50%) detected")
        
        # 4. Check OHLC consistency (high >= low, high >= open/close, low <= open/close)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df['high'] < df['low']) | 
                (df['high'] < df['open']) | 
                (df['high'] < df['close']) |
                (df['low'] > df['open']) | 
                (df['low'] > df['close'])
            ).sum()
            if invalid_ohlc > 0:
                self.data_quality_issues.append(f"⚠️ {invalid_ohlc} candles have invalid OHLC (high<low or similar)")
        
        # 5. Minimum data check
        if len(df) < 50:
            self.data_quality_issues.append(f"⚠️ Only {len(df)} candles - some indicators may be unreliable (need 50+)")
        
        # Log issues if any found
        if self.data_quality_issues:
            print(f"📊 Data Quality Check: {len(self.data_quality_issues)} issue(s) detected")
            for issue in self.data_quality_issues:
                print(f"   {issue}")
        
    def get_data_quality_summary(self):
        """Return data quality issues for display in UI if needed."""
        return self.data_quality_issues
        
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
        Analyze 5 momentum indicators to estimate momentum persistence.
        Returns timing analysis showing how long current momentum is likely to continue.
        
        Args:
            timeframe_minutes: The selected chart timeframe in minutes (60=1H, 240=4H, etc.)
        
        Uses (5 Pro-Level Indicators):
        - RSI_6, RSI_12, RSI_24: Multi-timeframe momentum alignment
        - Stoch_K, Stoch_D, Stoch_J: KDJ momentum dynamics
        - MACD, MACD_signal, MACD_hist: Trend confirmation
        - ADX: Trend strength measurement
        - OBV slope: Smart money accumulation/distribution
        
        Returns dict with:
        - momentum_direction: 'bullish', 'bearish', or 'neutral'
        - estimated_candles: estimated candles before reversal likely
        - estimated_hours: estimated hours before reversal (timeframe-aware)
        - timeframe_label: human-readable timeframe (e.g., "1H", "4H")
        - rsi_alignment: how RSI timeframes are aligned
        - kdj_dynamics: J/K/D relationship analysis
        - macd_trend: MACD confirmation status
        - adx_strength: trend strength status
        - obv_flow: smart money direction
        - timing_confidence: confidence in timing estimate
        - signals_aligned: count of aligned signals out of 5
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
        
        # Get MACD values (NEW)
        macd = indicators.get('MACD')
        macd_signal = indicators.get('MACD_signal')
        macd_hist = indicators.get('MACD_hist')
        
        # Get ADX value (NEW)
        adx = indicators.get('ADX')
        
        # Initialize result
        result = {
            'momentum_direction': 'neutral',
            'estimated_candles': 0,
            'estimated_hours': 0,
            'timeframe_minutes': timeframe_minutes,
            'timeframe_label': timeframe_label,
            'rsi_alignment': 'neutral',
            'kdj_dynamics': 'neutral',
            'macd_trend': 'neutral',
            'adx_strength': 'weak',
            'obv_flow': 'neutral',
            'timing_confidence': 0.0,
            'signals_aligned': 0,
            'advisory': '',
            'details': {}
        }
        
        # Check if we have minimum required data (RSI + KDJ are essential)
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
            'Stoch_J': round(stoch_j, 2),
            'MACD': round(macd, 4) if macd else None,
            'MACD_signal': round(macd_signal, 4) if macd_signal else None,
            'MACD_hist': round(macd_hist, 4) if macd_hist else None,
            'ADX': round(adx, 2) if adx else None
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
        
        # ========== MACD Trend Confirmation (NEW) ==========
        # MACD > Signal = bullish, histogram growing = momentum building
        # MACD < Signal = bearish, histogram shrinking = momentum fading
        macd_momentum = 'neutral'
        macd_candles = 2  # Default neutral contribution
        
        if macd is not None and macd_signal is not None and macd_hist is not None:
            if macd > macd_signal:
                if macd_hist > 0:
                    result['macd_trend'] = 'bullish_confirmed'
                    macd_momentum = 'bullish'
                    macd_candles = 3 + abs(macd_hist) * 10  # Stronger histogram = more momentum
                else:
                    result['macd_trend'] = 'bullish_weakening'
                    macd_momentum = 'bullish_weak'
                    macd_candles = 2
            elif macd < macd_signal:
                if macd_hist < 0:
                    result['macd_trend'] = 'bearish_confirmed'
                    macd_momentum = 'bearish'
                    macd_candles = 3 + abs(macd_hist) * 10
                else:
                    result['macd_trend'] = 'bearish_weakening'
                    macd_momentum = 'bearish_weak'
                    macd_candles = 2
            else:
                result['macd_trend'] = 'neutral'
                macd_momentum = 'neutral'
                macd_candles = 1  # Crossover imminent
            
            # Cap MACD candles contribution
            macd_candles = min(macd_candles, 6)
        
        # ========== ADX Trend Strength (NEW) ==========
        # ADX > 25 = trend has strength, rising = momentum increasing
        # ADX < 20 = weak/ranging market, timing less reliable
        adx_multiplier = 1.0  # Default - no adjustment
        
        if adx is not None:
            if adx > 40:
                result['adx_strength'] = 'very_strong'
                adx_multiplier = 1.3  # Strong trend extends timing
            elif adx > 25:
                result['adx_strength'] = 'strong'
                adx_multiplier = 1.15  # Moderate extension
            elif adx > 20:
                result['adx_strength'] = 'moderate'
                adx_multiplier = 1.0  # No change
            else:
                result['adx_strength'] = 'weak'
                adx_multiplier = 0.7  # Weak trend = less reliable timing
        
        # ========== OBV Smart Money Flow (NEW) ==========
        # Calculate OBV slope using LINEAR REGRESSION (same method as display for consistency)
        obv_momentum = 'neutral'
        obv_slope = 0.0
        
        if len(self.df) >= 10:
            try:
                obv_recent = self.df['OBV'].tail(10).dropna()
                price_recent = self.df['close'].tail(10).dropna()
                
                if len(obv_recent) >= 5 and len(price_recent) >= 5:
                    # Use LINEAR REGRESSION for OBV slope (consistent with display)
                    x_obv = np.arange(len(obv_recent))
                    y_obv = obv_recent.values.astype(float)
                    n_obv = len(x_obv)
                    obv_slope = (n_obv * np.sum(x_obv * y_obv) - np.sum(x_obv) * np.sum(y_obv)) / \
                               (n_obv * np.sum(x_obv * x_obv) - np.sum(x_obv) ** 2)
                    
                    # Use LINEAR REGRESSION for price slope too
                    x_price = np.arange(len(price_recent))
                    y_price = price_recent.values.astype(float)
                    n_price = len(x_price)
                    price_slope = (n_price * np.sum(x_price * y_price) - np.sum(x_price) * np.sum(y_price)) / \
                                 (n_price * np.sum(x_price * x_price) - np.sum(x_price) ** 2)
                    
                    # Store slope for debugging
                    result['details']['obv_slope_raw'] = float(obv_slope)
                    
                    # OBV rising = accumulation (smart money buying)
                    # OBV falling = distribution (smart money selling)
                    # Use threshold to avoid noise (slope must be meaningful)
                    obv_threshold = abs(obv_slope) * 0.1  # 10% of slope magnitude as threshold
                    
                    if obv_slope > obv_threshold:
                        result['obv_flow'] = 'accumulation'
                        obv_momentum = 'bullish'
                        # Check for bullish divergence (price down but OBV up)
                        if price_slope < 0:
                            result['obv_flow'] = 'bullish_divergence'
                    elif obv_slope < -obv_threshold:
                        result['obv_flow'] = 'distribution'
                        obv_momentum = 'bearish'
                        # Check for bearish divergence (price up but OBV down)
                        if price_slope > 0:
                            result['obv_flow'] = 'bearish_divergence'
                    else:
                        result['obv_flow'] = 'neutral'
                        obv_momentum = 'neutral'
            except Exception as e:
                print(f"OBV analysis error: {e}")
                pass  # OBV analysis failed, continue with neutral
        
        # ========== Combine Indicators: Core (RSI/KDJ/MACD) + Confirmers (ADX/OBV) ==========
        # Core indicators determine direction, confirmers validate or warn of divergence
        
        base_candles = (rsi_candles + kdj_candles + macd_candles) / 3
        
        # Determine CORE momentum direction from RSI, KDJ, MACD (3 core indicators)
        core_bullish = 0
        core_bearish = 0
        
        # RSI signal (core)
        if rsi_momentum in ['bullish', 'bullish_weak']:
            core_bullish += 1
        elif rsi_momentum in ['bearish', 'bearish_weak']:
            core_bearish += 1
        
        # KDJ signal (core)
        if kdj_momentum in ['bullish', 'reversal_up']:
            core_bullish += 1
        elif kdj_momentum in ['bearish', 'reversal_down']:
            core_bearish += 1
        
        # MACD signal (core)
        if macd_momentum in ['bullish', 'bullish_weak']:
            core_bullish += 1
        elif macd_momentum in ['bearish', 'bearish_weak']:
            core_bearish += 1
        
        # Determine core direction (need 2/3 agreement)
        if core_bullish >= 2:
            core_direction = 'bullish'
        elif core_bearish >= 2:
            core_direction = 'bearish'
        else:
            core_direction = 'mixed'
        
        result['details']['core_direction'] = core_direction
        result['details']['core_bullish'] = core_bullish
        result['details']['core_bearish'] = core_bearish
        
        # ========== ADX/OBV Confirmation Check ==========
        # These confirm or warn about potential divergence
        adx_confirms = False
        obv_confirms = False
        divergence_warning = None
        
        # Get DI values for direction confirmation
        di_plus = indicators.get('DI_plus', 0)
        di_minus = indicators.get('DI_minus', 0)
        
        # ADX confirmation: Strong trend AND direction must match core direction
        # ADX > 25 = strong trend, but DI+ vs DI- determines direction
        if adx is not None and adx > 25:
            if core_direction == 'bullish' and di_plus > di_minus:
                adx_confirms = True  # Strong bullish trend confirmed by DI+
                estimated_candles = base_candles * adx_multiplier
            elif core_direction == 'bearish' and di_minus > di_plus:
                adx_confirms = True  # Strong bearish trend confirmed by DI-
                estimated_candles = base_candles * adx_multiplier
            else:
                # ADX strong but direction conflicts - treat as divergence
                adx_confirms = False
                estimated_candles = base_candles * 0.7  # Reduce timing due to conflict
        else:
            # Weak ADX - trend may not persist
            estimated_candles = base_candles * 0.8
        
        # OBV confirmation: Smart money aligned with core direction
        if core_direction == 'bullish':
            if obv_momentum == 'bullish':
                obv_confirms = True
            elif obv_momentum == 'bearish':
                # OBV DIVERGENCE: Core says bullish but smart money selling
                divergence_warning = 'bearish_divergence'
                result['obv_flow'] = 'smart_money_selling'
                estimated_candles = max(1, estimated_candles * 0.5)  # Sharply reduce timing
        elif core_direction == 'bearish':
            if obv_momentum == 'bearish':
                obv_confirms = True
            elif obv_momentum == 'bullish':
                # OBV DIVERGENCE: Core says bearish but smart money buying
                divergence_warning = 'bullish_divergence'
                result['obv_flow'] = 'smart_money_buying'
                estimated_candles = max(1, estimated_candles * 0.5)  # Sharply reduce timing
        
        # Store confirmation status
        result['details']['adx_confirms'] = adx_confirms
        result['details']['obv_confirms'] = obv_confirms
        result['details']['divergence_warning'] = divergence_warning
        result['details']['di_plus'] = round(di_plus, 1) if di_plus else None
        result['details']['di_minus'] = round(di_minus, 1) if di_minus else None
        result['details']['obv_momentum'] = obv_momentum
        
        # Calculate overall signal alignment
        # Core: 3 possible, Confirmers: 2 possible
        total_aligned = max(core_bullish, core_bearish)
        if adx_confirms:
            total_aligned += 1
        if obv_confirms:
            total_aligned += 1
        
        result['signals_aligned'] = total_aligned
        
        # ========== Determine Momentum Direction ==========
        # Core direction is primary, but divergence warning can override confidence
        
        # Check for divergence FIRST - this is critical
        if divergence_warning:
            # Divergence detected - momentum may reverse soon
            result['momentum_direction'] = core_direction  # Still show core direction
            result['timing_confidence'] = 0.3  # Very low confidence due to divergence
            estimated_candles = max(1, estimated_candles * 0.4)  # Sharply reduce timing
        elif total_aligned >= 4:
            # Strong alignment: 3 core + ADX + OBV all confirm
            result['momentum_direction'] = core_direction
            result['timing_confidence'] = 0.95  # Highest confidence
        elif total_aligned == 3 and adx_confirms:
            # Good alignment with strong trend
            result['momentum_direction'] = core_direction
            result['timing_confidence'] = 0.85
        elif total_aligned == 3:
            # Core aligned but confirmers mixed
            result['momentum_direction'] = core_direction
            result['timing_confidence'] = 0.7
        elif kdj_momentum in ['reversal_up', 'reversal_down']:
            # KDJ signaling imminent reversal
            result['momentum_direction'] = 'reversal_imminent'
            result['timing_confidence'] = 0.6
            estimated_candles = 1
        elif core_direction != 'mixed':
            # Core has direction but weak confirmation
            result['momentum_direction'] = core_direction
            result['timing_confidence'] = 0.5
            estimated_candles = max(1, estimated_candles * 0.7)
        else:
            # No clear direction
            result['momentum_direction'] = 'mixed'
            result['timing_confidence'] = 0.3
            estimated_candles = max(1, estimated_candles * 0.5)
        
        # ========== STEP 2: HISTORICAL PATTERN TIMING (NEW) ==========
        # Use historical data to refine timing based on how long similar patterns lasted
        # Direction was determined by fresh analysis (Step 1), timing uses historical patterns
        
        historical_timing = self._get_historical_zone_duration()
        result['details']['historical_timing'] = historical_timing
        
        if historical_timing.get('has_data'):
            # Get current zone durations
            rsi_zone_duration = historical_timing.get('rsi_duration', 0)
            kdj_zone_duration = historical_timing.get('kdj_duration', 0)
            rsi_zone = historical_timing.get('rsi_zone', 'neutral')
            kdj_zone = historical_timing.get('kdj_zone', 'neutral')
            
            # Calculate remaining candles based on historical averages
            # Average overbought/oversold duration is typically 3-8 candles
            avg_zone_duration = 5  # Default historical average
            
            if rsi_zone in ['overbought', 'oversold']:
                # If already in zone for X candles, remaining = avg - current
                rsi_remaining = max(1, avg_zone_duration - rsi_zone_duration)
                # Weight historical timing into estimate (50% historical, 50% spread-based)
                estimated_candles = (estimated_candles + rsi_remaining) / 2
                result['details']['rsi_zone_info'] = f"{rsi_zone} for {rsi_zone_duration} candles, ~{rsi_remaining} remaining"
            
            if kdj_zone in ['overbought', 'oversold']:
                kdj_remaining = max(1, avg_zone_duration - kdj_zone_duration)
                # KDJ tends to reverse faster than RSI
                estimated_candles = min(estimated_candles, kdj_remaining + 1)
                result['details']['kdj_zone_info'] = f"{kdj_zone} for {kdj_zone_duration} candles, ~{kdj_remaining} remaining"
            
            # If both in extreme zones for extended period, reversal is imminent
            if rsi_zone_duration >= 6 and kdj_zone_duration >= 4:
                estimated_candles = max(1, min(estimated_candles, 2))
                result['details']['extended_extreme'] = True
        
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
        
        # Build signal summary for advisory
        signal_summary = f"{result['signals_aligned']}/5 signals"
        confirm_status = []
        if adx_confirms:
            confirm_status.append("ADX")
        if obv_confirms:
            confirm_status.append("OBV")
        confirm_str = f" [{'+'.join(confirm_status)} confirm]" if confirm_status else ""
        
        # ========== Generate Enhanced Advisory with Divergence Warnings ==========
        
        # DIVERGENCE WARNING TAKES PRIORITY
        if divergence_warning:
            if divergence_warning == 'bearish_divergence':
                result['advisory'] = f"DIVERGENCE WARNING: Core {core_direction.upper()} but OBV shows SMART MONEY SELLING - reversal likely within {candle_display} ({time_display})"
            elif divergence_warning == 'bullish_divergence':
                result['advisory'] = f"DIVERGENCE WARNING: Core {core_direction.upper()} but OBV shows SMART MONEY BUYING - reversal likely within {candle_display} ({time_display})"
        elif result['momentum_direction'] == 'bullish':
            if result['signals_aligned'] >= 4:
                result['advisory'] = f"STRONG Bullish: {signal_summary}{confirm_str} - momentum persists {candle_display} ({time_display})"
            elif result['signals_aligned'] == 3:
                result['advisory'] = f"Bullish: {signal_summary}{confirm_str} - momentum likely {candle_display} ({time_display})"
            else:
                missing = []
                if not adx_confirms:
                    missing.append("ADX weak")
                if not obv_confirms:
                    missing.append("OBV unconfirmed")
                missing_str = f" [{', '.join(missing)}]" if missing else ""
                result['advisory'] = f"Weak Bullish: {signal_summary}{missing_str} - timing uncertain ~{candle_display} ({time_display})"
        elif result['momentum_direction'] == 'bearish':
            if result['signals_aligned'] >= 4:
                result['advisory'] = f"STRONG Bearish: {signal_summary}{confirm_str} - momentum persists {candle_display} ({time_display})"
            elif result['signals_aligned'] == 3:
                result['advisory'] = f"Bearish: {signal_summary}{confirm_str} - momentum likely {candle_display} ({time_display})"
            else:
                missing = []
                if not adx_confirms:
                    missing.append("ADX weak")
                if not obv_confirms:
                    missing.append("OBV unconfirmed")
                missing_str = f" [{', '.join(missing)}]" if missing else ""
                result['advisory'] = f"Weak Bearish: {signal_summary}{missing_str} - timing uncertain ~{candle_display} ({time_display})"
        elif result['momentum_direction'] == 'reversal_imminent':
            result['advisory'] = f"REVERSAL: J-line {'peaked' if kdj_j_peaked else 'bottomed'} - expect direction change within 1-2 {timeframe_label} candles"
        else:
            # Mixed signals - core indicators disagree
            result['advisory'] = f"MIXED: Core split ({core_bullish}B/{core_bearish}B) - no clear momentum, wait for alignment"
        
        # Add indicator breakdown to details
        result['details']['signal_breakdown'] = {
            'RSI': rsi_momentum,
            'KDJ': kdj_momentum,
            'MACD': macd_momentum,
            'OBV': obv_momentum,
            'ADX': result['adx_strength'],
            'adx_confirms': adx_confirms,
            'obv_confirms': obv_confirms,
            'divergence_warning': divergence_warning
        }
        
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
    
    def _get_historical_zone_duration(self):
        """
        Calculate how many consecutive candles RSI and KDJ have been in extreme zones.
        Uses the DataFrame directly for historical analysis.
        
        Returns dict with:
        - rsi_duration: candles RSI has been overbought/oversold
        - rsi_zone: 'overbought', 'oversold', or 'neutral'
        - kdj_duration: candles KDJ J has been in extreme zone
        - kdj_zone: 'overbought', 'oversold', or 'neutral'
        - has_data: whether sufficient data exists
        """
        result = {
            'has_data': False,
            'rsi_duration': 0,
            'rsi_zone': 'neutral',
            'kdj_duration': 0,
            'kdj_zone': 'neutral'
        }
        
        if len(self.df) < 5:
            return result
        
        try:
            # Get RSI values from DataFrame (use RSI_14 or RSI column)
            rsi_col = 'RSI' if 'RSI' in self.df.columns else 'RSI_14' if 'RSI_14' in self.df.columns else None
            stoch_j_col = 'STOCHk_14_3_3' if 'STOCHk_14_3_3' in self.df.columns else 'Stoch_J' if 'Stoch_J' in self.df.columns else None
            
            # Calculate RSI zone duration
            if rsi_col and rsi_col in self.df.columns:
                rsi_values = self.df[rsi_col].dropna().values
                if len(rsi_values) >= 3:
                    current_rsi = rsi_values[-1]
                    
                    # Determine current zone
                    if current_rsi > 70:
                        result['rsi_zone'] = 'overbought'
                        threshold = 70
                        # Count backwards while in overbought
                        duration = 0
                        for i in range(len(rsi_values) - 1, -1, -1):
                            if rsi_values[i] > threshold:
                                duration += 1
                            else:
                                break
                        result['rsi_duration'] = duration
                        result['has_data'] = True
                        
                    elif current_rsi < 30:
                        result['rsi_zone'] = 'oversold'
                        threshold = 30
                        # Count backwards while in oversold
                        duration = 0
                        for i in range(len(rsi_values) - 1, -1, -1):
                            if rsi_values[i] < threshold:
                                duration += 1
                            else:
                                break
                        result['rsi_duration'] = duration
                        result['has_data'] = True
            
            # Calculate KDJ J-line zone duration
            if stoch_j_col and stoch_j_col in self.df.columns:
                j_values = self.df[stoch_j_col].dropna().values
                if len(j_values) >= 3:
                    current_j = j_values[-1]
                    
                    if current_j > 80:
                        result['kdj_zone'] = 'overbought'
                        threshold = 80
                        duration = 0
                        for i in range(len(j_values) - 1, -1, -1):
                            if j_values[i] > threshold:
                                duration += 1
                            else:
                                break
                        result['kdj_duration'] = duration
                        result['has_data'] = True
                        
                    elif current_j < 20:
                        result['kdj_zone'] = 'oversold'
                        threshold = 20
                        duration = 0
                        for i in range(len(j_values) - 1, -1, -1):
                            if j_values[i] < threshold:
                                duration += 1
                            else:
                                break
                        result['kdj_duration'] = duration
                        result['has_data'] = True
            
            return result
            
        except Exception as e:
            print(f"Historical zone duration error: {e}")
            return result
    
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
