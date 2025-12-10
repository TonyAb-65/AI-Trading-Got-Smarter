import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import pytz

from database import init_db, get_session, Trade, ActivePosition, ModelPerformance
from api_integrations import get_market_data_unified, get_current_price, OKXClient
from technical_indicators import TechnicalIndicators, calculate_support_resistance, analyze_consolidation_state
from whale_tracker import WhaleTracker
from ml_engine import MLTradingEngine
from position_monitor import PositionMonitor
from scheduler import get_scheduler
from divergence_logger import log_divergences_from_context
from divergence_analytics import get_divergence_timing_info

st.set_page_config(
    page_title="AI Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_database():
    init_db()

@st.cache_resource
def get_ml_engine():
    """Get shared ML Engine instance for profile similarity calculations"""
    ml_engine = MLTradingEngine()
    
    # Build profiles immediately to ensure scaler is fitted for position monitoring
    try:
        profiles = ml_engine.build_trade_profiles()
        if profiles:
            print(f"✅ ML Engine initialized with {len([p for p in profiles.values() if p is not None])} profiles")
        else:
            print("ℹ️  ML Engine created but profiles not yet available (need 5+ trades)")
    except Exception as e:
        print(f"⚠️  Error building profiles during initialization: {e}")
    
    return ml_engine

@st.cache_resource
def start_background_scheduler():
    ml_engine = get_ml_engine()  # Get shared ML Engine first
    scheduler = get_scheduler(ml_engine)  # Pass it to scheduler
    scheduler.start()
    return scheduler

initialize_database()
start_background_scheduler()

CRYPTO_PAIRS = [
    'BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD', 'ADA/USD',
    'DOGE/USD', 'MATIC/USD', 'DOT/USD', 'AVAX/USD', 'LINK/USD'
]

FOREX_PAIRS = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD',
    'USD/CAD', 'NZD/USD'
]

METALS = [
    'XAU/USD', 'XAG/USD', 'XPT/USD', 'XPD/USD'
]

# Riyadh Timezone (GMT+3)
RIYADH_TZ = pytz.timezone('Asia/Riyadh')

def format_price(price):
    """
    Smart price formatting based on value:
    - Prices < $1: 5 decimals (e.g., $0.55495)
    - Prices $1-$10: 4 decimals (e.g., $5.5549)
    - Prices >= $10: 2 decimals (e.g., $45,234.50)
    """
    if price is None:
        return "N/A"
    
    if price < 1:
        return f"${price:,.5f}"
    elif price < 10:
        return f"${price:,.4f}"
    else:
        return f"${price:,.2f}"

def check_global_alerts():
    """Check all active positions for HIGH severity alerts"""
    try:
        ml_engine = get_ml_engine()
        monitor = PositionMonitor(ml_engine=ml_engine)
        results = monitor.check_active_positions()
        
        high_alerts = []
        for result in results:
            if result.get('status') == 'success' and result.get('monitoring_alerts'):
                for alert in result['monitoring_alerts']:
                    if alert.get('severity') == 'HIGH':
                        high_alerts.append({
                            'symbol': result['symbol'],
                            'message': alert['message'],
                            'recommendation': result['recommendation']
                        })
        
        return high_alerts
    except Exception as e:
        return []

def convert_to_riyadh_time(utc_dt):
    """Convert UTC datetime to Riyadh time (GMT+3)"""
    if utc_dt is None:
        return None
    
    # If datetime is naive (no timezone), assume it's UTC
    if utc_dt.tzinfo is None:
        utc_dt = pytz.utc.localize(utc_dt)
    
    # Convert to Riyadh time
    riyadh_dt = utc_dt.astimezone(RIYADH_TZ)
    return riyadh_dt

def check_api_keys():
    twelve_data_key = os.getenv('TWELVE_DATA_API_KEY')
    okx_key = os.getenv('OKX_API_KEY')
    
    if not twelve_data_key:
        st.error("⚠️ TWELVE_DATA_API_KEY is required for the platform to work!")
        with st.expander("🔑 How to add your Twelve Data API key"):
            st.write("""
            **Twelve Data provides ALL market data (Crypto, Forex, Metals)**
            
            1. Sign up at https://twelvedata.com/ (FREE account)
            2. Copy your API key from the dashboard
            3. In Replit, click the 🔒 Secrets tab (lock icon on left sidebar)
            4. Click "New Secret"
            5. Key: `TWELVE_DATA_API_KEY`
            6. Value: [paste your API key]
            7. Refresh this page
            
            **Free tier includes:**
            - 800 API calls per day
            - All crypto pairs (BTC, ETH, etc.)
            - All forex pairs (EUR/USD, GBP/USD, etc.)
            - Precious metals (Gold, Silver, etc.)
            """)
        return False
    
    st.success("✅ Twelve Data API key configured - All markets available!")
    if not okx_key:
        st.info("💡 Optional: Add OKX_API_KEY for orderbook data and whale tracking on crypto")
    
    return True

def convert_to_heikin_ashi(df):
    """Convert regular OHLC data to Heikin-Ashi candles for clearer trend visualization"""
    ha_df = df.copy()
    
    # Calculate Heikin-Ashi values
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = df['open'].copy()
    
    # First HA Open = (first Open + first Close) / 2
    ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    
    # Calculate HA Open (average of previous HA Open and HA Close)
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    
    ha_high = df[['high', 'open', 'close']].max(axis=1)
    ha_low = df[['low', 'open', 'close']].min(axis=1)
    
    # Apply HA High/Low using HA Open and HA Close
    for i in range(len(df)):
        ha_high.iloc[i] = max(df['high'].iloc[i], ha_open.iloc[i], ha_close.iloc[i])
        ha_low.iloc[i] = min(df['low'].iloc[i], ha_open.iloc[i], ha_close.iloc[i])
    
    ha_df['open'] = ha_open
    ha_df['high'] = ha_high
    ha_df['low'] = ha_low
    ha_df['close'] = ha_close
    
    return ha_df

def plot_candlestick_chart(df, indicators_df, symbol, support_levels, resistance_levels):
    # Convert to Heikin-Ashi candles for clearer trend visualization
    ha_df = convert_to_heikin_ashi(df)
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.15],
        subplot_titles=(f'{symbol} Price Action (Heikin-Ashi)', 'RSI', 'MACD', 'Volume')
    )
    
    fig.add_trace(
        go.Candlestick(
            x=ha_df['timestamp'],
            open=ha_df['open'],
            high=ha_df['high'],
            low=ha_df['low'],
            close=ha_df['close'],
            name='HA Price'
        ),
        row=1, col=1
    )
    
    if 'SMA_20' in indicators_df.columns:
        fig.add_trace(
            go.Scatter(x=indicators_df['timestamp'], y=indicators_df['SMA_20'],
                      name='SMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    if 'SMA_50' in indicators_df.columns:
        fig.add_trace(
            go.Scatter(x=indicators_df['timestamp'], y=indicators_df['SMA_50'],
                      name='SMA 50', line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    if 'EMA_12' in indicators_df.columns:
        fig.add_trace(
            go.Scatter(x=indicators_df['timestamp'], y=indicators_df['EMA_12'],
                      name='EMA 12', line=dict(color='green', width=1, dash='dash')),
            row=1, col=1
        )
    
    if 'BB_upper' in indicators_df.columns:
        fig.add_trace(
            go.Scatter(x=indicators_df['timestamp'], y=indicators_df['BB_upper'],
                      name='BB Upper', line=dict(color='gray', width=1, dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=indicators_df['timestamp'], y=indicators_df['BB_lower'],
                      name='BB Lower', line=dict(color='gray', width=1, dash='dot'),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
    
    for level in support_levels:
        fig.add_hline(y=level, line_dash="dash", line_color="green", 
                     annotation_text=f"S: {level}", row=1, col=1)
    
    for level in resistance_levels:
        fig.add_hline(y=level, line_dash="dash", line_color="red",
                     annotation_text=f"R: {level}", row=1, col=1)
    
    if 'RSI' in indicators_df.columns:
        fig.add_trace(
            go.Scatter(x=indicators_df['timestamp'], y=indicators_df['RSI'],
                      name='RSI', line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    if 'MACD' in indicators_df.columns:
        fig.add_trace(
            go.Scatter(x=indicators_df['timestamp'], y=indicators_df['MACD'],
                      name='MACD', line=dict(color='blue', width=2)),
            row=3, col=1
        )
    if 'MACD_signal' in indicators_df.columns:
        fig.add_trace(
            go.Scatter(x=indicators_df['timestamp'], y=indicators_df['MACD_signal'],
                      name='Signal', line=dict(color='orange', width=2)),
            row=3, col=1
        )
    if 'MACD_hist' in indicators_df.columns:
        colors = ['green' if val >= 0 else 'red' for val in indicators_df['MACD_hist']]
        fig.add_trace(
            go.Bar(x=indicators_df['timestamp'], y=indicators_df['MACD_hist'],
                  name='Histogram', marker_color=colors),
            row=3, col=1
        )
    
    fig.add_trace(
        go.Bar(x=df['timestamp'], y=df['volume'], name='Volume',
              marker_color='lightblue'),
        row=4, col=1
    )
    
    fig.update_layout(
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

st.title("📈 AI-Powered Trading Analysis Platform")
st.markdown("**Real-time Market Analysis | ML Predictions | Position Monitoring**")

check_api_keys()

menu = st.sidebar.selectbox(
    "Navigation",
    ["Market Analysis", "Trading Signals", "Position Tracker", "Risk Calculator", "Performance Analytics", "Model Training"]
)

# GLOBAL ALERT BANNER - Shows HIGH severity warnings on all tabs
global_alerts = check_global_alerts()
if global_alerts:
    for alert in global_alerts:
        st.error(f"🚨 **CRITICAL ALERT: {alert['symbol']}** - {alert['message']}")
        if menu != "Position Tracker":
            st.info("👉 Go to **Position Tracker** tab to view details and take action")
    st.divider()

if menu == "Market Analysis":
    st.header("📊 Market Analysis Dashboard")
    
    # ML Learning Explanation
    with st.expander("🧠 **How ML Learning Improves Predictions**", expanded=False):
        st.write("""
        **The AI continuously learns from your trading results:**
        
        1. **Individual Learning** (After Every Trade):
           - When you close a trade, the system immediately updates indicator weights
           - Winning trades → indicators get stronger influence
           - Losing trades → indicators get weaker influence
           
        2. **Bulk Retraining** (Every 10 Trades):
           - At 10, 20, 30, 40... trades, ML models retrain on ALL your historical data
           - Models learn patterns from both wins AND losses
           - Training uses 80% data for learning, 20% for validation
           
        3. **Applied to Market Analysis**:
           - Trained models analyze current market conditions
           - ML prediction combines with rule-based technical indicators
           - Higher model confidence = stronger buy/sell signals
           - System shows "ML Confidence: 65%" to indicate prediction strength
           
        **Example:** After 30 trades, if RSI consistently led to wins but MACD led to losses,
        the system will trust RSI more and reduce MACD's influence in future predictions.
        
        This is why accuracy improves over time - the AI learns YOUR trading style!
        """)
    
    col1, col2, col3 = st.columns(3)
    
    # Initialize from stored values (for persistence across tab switches)
    default_market_type = st.session_state.get('last_market_type', 'crypto')
    default_symbol = st.session_state.get('last_symbol', 'BTC/USD')
    default_timeframe = st.session_state.get('last_timeframe', '1H')
    
    with col1:
        market_type = st.selectbox(
            "Market Type", 
            ["crypto", "forex", "metals", "custom"],
            index=["crypto", "forex", "metals", "custom"].index(default_market_type) if default_market_type in ["crypto", "forex", "metals", "custom"] else 0,
            key="market_analysis_market_type"
        )
    
    with col2:
        if market_type == "crypto":
            symbol_index = CRYPTO_PAIRS.index(default_symbol) if default_symbol in CRYPTO_PAIRS else 0
            symbol = st.selectbox("Select Pair", CRYPTO_PAIRS, index=symbol_index, key="market_analysis_symbol")
        elif market_type == "forex":
            symbol_index = FOREX_PAIRS.index(default_symbol) if default_symbol in FOREX_PAIRS else 0
            symbol = st.selectbox("Select Pair", FOREX_PAIRS, index=symbol_index, key="market_analysis_symbol")
        elif market_type == "metals":
            symbol_index = METALS.index(default_symbol) if default_symbol in METALS else 0
            symbol = st.selectbox("Select Metal", METALS, index=symbol_index, key="market_analysis_symbol")
        else:  # custom
            symbol = st.text_input("🔍 Enter Symbol (e.g., AAPL/USD, TSLA/USD, LTC/USD)", value=default_symbol, placeholder="BTC/USD", key="market_analysis_symbol").upper()
    
    with col3:
        timeframe_index = ["5m", "15m", "30m", "1H", "4H", "1D"].index(default_timeframe) if default_timeframe in ["5m", "15m", "30m", "1H", "4H", "1D"] else 3
        timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "1H", "4H", "1D"], index=timeframe_index, key="market_analysis_timeframe")
    
    if st.button("Analyze Market", type="primary"):
        if not symbol or symbol.strip() == "":
            st.error("❌ Please enter a trading pair symbol")
            st.stop()
        
        # Clear old analysis if parameters changed (only when explicitly analyzing)
        current_params = f"{symbol}_{market_type}_{timeframe}"
        if 'analysis_params' in st.session_state and st.session_state['analysis_params'] != current_params:
            for key in ['analysis_data', 'analysis_params', 'last_prediction', 'last_symbol', 'last_market_type', 'last_timeframe', 'last_indicators']:
                if key in st.session_state:
                    del st.session_state[key]
        
        # Map 'custom' to 'forex' for API compatibility (Twelve Data treats most symbols as forex pairs)
        api_market_type = "forex" if market_type == "custom" else market_type
        
        with st.spinner("Fetching market data..."):
            try:
                df = get_market_data_unified(symbol, api_market_type, timeframe, 100)
                
                if df is None:
                    st.error(f"❌ Failed to fetch data for {symbol} on {timeframe} timeframe. API may have returned an error.")
                    st.info("💡 Try a different timeframe or check if your API key is valid")
                    st.stop()
                    
                if len(df) == 0:
                    st.error(f"❌ No data available for {symbol}")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                st.stop()
            
            if df is not None and len(df) > 0:
                tech = TechnicalIndicators(df)
                indicators_df = tech.calculate_all_indicators()
                latest_indicators = tech.get_latest_indicators()
                signals = tech.get_trend_signals()
                
                # NEW: Get momentum timing analysis from multi-timeframe RSI and KDJ
                # Convert timeframe string to minutes for timing calculations
                timeframe_to_minutes = {'5m': 5, '15m': 15, '30m': 30, '1H': 60, '4H': 240, '1D': 1440}
                timeframe_minutes = timeframe_to_minutes.get(timeframe, 60)
                momentum_timing = tech.get_momentum_timing(timeframe_minutes=timeframe_minutes)
                latest_indicators['momentum_timing'] = momentum_timing
                
                # Get historical trend context for duration/slope/divergence analysis
                trend_context = tech.get_trend_context(symbol, api_market_type)
                latest_indicators['trend_context'] = trend_context
                
                support_levels, resistance_levels = calculate_support_resistance(indicators_df)
                
                # Add S/R levels to indicators for ML predictions
                latest_indicators['support_levels'] = support_levels
                latest_indicators['resistance_levels'] = resistance_levels
                
                # NEW: Consolidation detection (advisory only - does NOT affect M1)
                consolidation = analyze_consolidation_state(indicators_df, latest_indicators, support_levels, resistance_levels)
                latest_indicators['consolidation'] = consolidation
                
                # Get ML prediction
                ml_engine = MLTradingEngine()
                prediction = ml_engine.predict(latest_indicators)
                
                # Log divergences for timing intelligence
                try:
                    if 'trend_context' in latest_indicators:
                        current_price = latest_indicators.get('close', 0)
                        log_divergences_from_context(symbol, timeframe, latest_indicators['trend_context'], current_price)
                except Exception as e:
                    print(f"Divergence logging failed: {e}")
                
                # Get candlestick patterns
                patterns = tech.detect_candlestick_patterns()
                
                # Get whale data for crypto
                whale_data = None
                if market_type == "crypto":
                    try:
                        okx_client = OKXClient(api_key=os.getenv('OKX_API_KEY'))
                        orderbook = okx_client.get_orderbook(symbol)
                        whale_tracker = WhaleTracker(indicators_df, orderbook)
                        whale_data = {
                            'movements': whale_tracker.detect_whale_movements(),
                            'smart_money': whale_tracker.detect_smart_money(),
                            'volume_profile': whale_tracker.get_volume_profile()
                        }
                    except Exception as e:
                        print(f"Whale analysis failed: {e}")
                
                # Store ALL analysis data in session state for persistence
                st.session_state['analysis_data'] = {
                    'df': df,
                    'indicators_df': indicators_df,
                    'latest_indicators': latest_indicators,
                    'signals': signals,
                    'support_levels': support_levels,
                    'resistance_levels': resistance_levels,
                    'prediction': prediction,
                    'patterns': patterns,
                    'whale_data': whale_data,
                    'trend_context': trend_context,
                    'momentum_timing': momentum_timing,  # Multi-timeframe timing analysis
                    'consolidation': consolidation  # NEW: Consolidation advisory
                }
                st.session_state['analysis_params'] = current_params
                st.session_state['last_prediction'] = prediction
                st.session_state['last_symbol'] = symbol
                st.session_state['last_market_type'] = api_market_type
                st.session_state['last_timeframe'] = timeframe
                st.session_state['last_indicators'] = latest_indicators
                
                # Reset trade direction to match AI recommendation
                ai_signal = prediction.get('signal', 'LONG')
                new_direction = 'SHORT' if ai_signal == 'SHORT' else 'LONG'
                st.session_state['manual_direction'] = new_direction
                st.session_state['manual_direction_signal'] = new_direction
                
                # Pre-fill form fields with predicted values
                if prediction.get('entry_price'):
                    st.session_state['manual_entry_price'] = float(prediction['entry_price'])
                    st.session_state['manual_sl'] = float(prediction.get('stop_loss', 0.0))
                    st.session_state['manual_tp'] = float(prediction.get('take_profit', 0.0))
                else:
                    # Clear stale values for HOLD predictions
                    for key in ['manual_entry_price', 'manual_sl', 'manual_tp']:
                        if key in st.session_state:
                            del st.session_state[key]
                
                st.success(f"✅ Analysis complete for {symbol}")
    
    # Display stored analysis (persists across tab switches)
    if 'analysis_data' in st.session_state:
        data = st.session_state['analysis_data']
        symbol = st.session_state.get('last_symbol', '')
        market_type = st.session_state.get('last_market_type', 'crypto')
        timeframe = st.session_state.get('last_timeframe', '1H')
        
        df = data['df']
        indicators_df = data['indicators_df']
        latest_indicators = data['latest_indicators']
        signals = data['signals']
        support_levels = data['support_levels']
        resistance_levels = data['resistance_levels']
        prediction = data['prediction']
        patterns = data['patterns']
        whale_data = data['whale_data']
        trend_context = data['trend_context']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        current_price = latest_indicators.get('current_price', 0)
        
        with col1:
            st.metric("Current Price", format_price(current_price))
        with col2:
            rsi = latest_indicators.get('RSI')
            st.metric("RSI", f"{rsi:.1f}" if rsi else "N/A")
        with col3:
            adx = latest_indicators.get('ADX')
            st.metric("ADX", f"{adx:.1f}" if adx else "N/A")
        with col4:
            mfi = latest_indicators.get('MFI')
            st.metric("MFI", f"{mfi:.1f}" if mfi else "N/A")
        with col5:
            obv = latest_indicators.get('OBV')
            if obv is not None:
                obv_formatted = f"{obv:,.0f}" if abs(obv) > 1000 else f"{obv:.2f}"
                st.metric("OBV", obv_formatted)
            else:
                st.metric("OBV", "N/A")
        
        st.plotly_chart(
            plot_candlestick_chart(df, indicators_df, symbol, support_levels, resistance_levels),
            use_container_width=True
        )
        
        # Display ATR and volatility regime info
        atr_val = latest_indicators.get('ATR', 0)
        atr_pct = latest_indicators.get('ATR_pct_price', 0)
        atr_percentile = latest_indicators.get('ATR_percentile', 50)
        bb_width = latest_indicators.get('BB_width_pct', 0)
        current_price = latest_indicators.get('current_price', 0)
        
        # ONE CLEAN ROW: ATR | ATR Percentile | BB Width | Patterns | S/R
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("ATR", f"${atr_val:,.2f}" if current_price > 100 else f"${atr_val:.4f}")
            st.caption(f"{atr_pct:.2f}% of price")
        
        with col2:
            st.metric("ATR Percentile", f"{atr_percentile:.1f}%")
            if atr_percentile >= 75:
                st.caption("🔴 Extreme")
            elif atr_percentile >= 55:
                st.caption("🟡 High")
            elif atr_percentile >= 30:
                st.caption("🟢 Medium")
            else:
                st.caption("🟢 Low")
        
        with col3:
            st.metric("BB Width", f"{bb_width:.2f}%")
            st.caption("Bollinger Band")
        
        with col4:
            st.subheader("Patterns")
            if patterns:
                for pattern_name, signal in patterns.items():
                    if signal == 'bullish':
                        st.write(f"🟢 {pattern_name.replace('_', ' ')}")
                    elif signal == 'bearish':
                        st.write(f"🔴 {pattern_name.replace('_', ' ')}")
            else:
                st.info("None detected")
        
        with col5:
            st.subheader("S/R Levels")
            st.write("**Resistance:**")
            for r in resistance_levels[:2]:
                st.write(f"🔴 {format_price(r)}")
            st.write("**Support:**")
            for s in support_levels[:2]:
                st.write(f"🟢 {format_price(s)}")
        
        st.divider()
        
        # Technical Signals and Smart Money in full-width sections below
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Technical Signals")
            for indicator, signal in signals.items():
                color = "🟢" if signal in ['bullish', 'oversold', 'strong_uptrend'] else "🔴" if signal in ['bearish', 'overbought', 'strong_downtrend'] else "🟡"
                
                if indicator == 'ADX':
                    adx_val = latest_indicators.get('ADX', 0)
                    di_plus = latest_indicators.get('DI_plus', 0)
                    di_minus = latest_indicators.get('DI_minus', 0)
                    st.write(f"{color} **{indicator}**: {signal}")
                    st.caption(f"   ADX: {adx_val:.1f} | +DI: {di_plus:.1f} | -DI: {di_minus:.1f}")
                else:
                    st.write(f"{color} **{indicator}**: {signal}")
        
        with col2:
            st.subheader("🧠 Smart Money (OBV)")
            obv_ctx = trend_context.get('OBV', {})
            obv_slope = obv_ctx.get('slope', 0.0)
            obv_divergence = obv_ctx.get('divergence', 'none')
            
            if obv_divergence == 'bullish':
                st.write("🟢 **Bullish Divergence** - Accumulating")
                st.caption(f"Slope: {obv_slope:+.2f}")
                timing_info = get_divergence_timing_info('OBV', timeframe, 'bullish')
                if timing_info:
                    st.caption(f"⏱️ Resolves in {timing_info['avg_candles']:.1f} candles | {timing_info['success_rate']:.0f}% success")
            elif obv_divergence == 'bearish':
                st.write("🔴 **Bearish Divergence** - Distributing")
                st.caption(f"Slope: {obv_slope:+.2f}")
                timing_info = get_divergence_timing_info('OBV', timeframe, 'bearish')
                if timing_info:
                    st.caption(f"⏱️ Resolves in {timing_info['avg_candles']:.1f} candles | {timing_info['success_rate']:.0f}% success")
            else:
                slope_direction = "Rising" if obv_slope > 0.5 else "Falling" if obv_slope < -0.5 else "Flat"
                slope_color = "🟢" if obv_slope > 0.5 else "🔴" if obv_slope < -0.5 else "🟡"
                st.write(f"{slope_color} **{slope_direction}**")
                st.caption(f"Slope: {obv_slope:+.2f}")
        
        if market_type == "crypto" and whale_data:
            st.subheader("🐋 Whale & Smart Money Analysis")
            whale_movements = whale_data['movements']
            smart_money = whale_data['smart_money']
            volume_profile = whale_data['volume_profile']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if whale_movements:
                    st.write("**Recent Whale Activity:**")
                    for movement in whale_movements[:5]:
                        st.write(f"{movement['transaction_type']} - Impact Score: {movement['impact_score']:.1f}")
                else:
                    st.info("No significant whale activity detected")
            
            with col2:
                if smart_money:
                    st.write("**Smart Money Signals:**")
                    for signal in smart_money:
                        st.write(f"• {signal['description']} (Confidence: {signal['confidence']})")
                else:
                    st.info("No smart money signals detected")
            
            if volume_profile:
                st.write(f"**Volume Profile:** Current: {volume_profile['current_volume']:,.0f} | Avg: {volume_profile['average_volume']:,.0f} | {volume_profile['volume_vs_avg']:.0f}% of average")
        
        # NEW: Momentum Timing Analysis Section
        momentum_timing = data.get('momentum_timing', {})
        if momentum_timing and momentum_timing.get('advisory'):
            st.divider()
            st.subheader("⏱️ Momentum Timing Analysis")
            
            details = momentum_timing.get('details', {})
            momentum_dir = momentum_timing.get('momentum_direction', 'neutral')
            est_candles = momentum_timing.get('estimated_candles', 0)
            est_hours = momentum_timing.get('estimated_hours', 0)
            tf_label = momentum_timing.get('timeframe_label', '1H')
            confidence = momentum_timing.get('timing_confidence', 0)
            
            # Format time display
            if est_hours >= 24:
                time_display = f"~{est_hours/24:.1f} days"
            elif est_hours >= 1:
                time_display = f"~{est_hours:.0f} hours"
            else:
                time_display = f"~{est_hours*60:.0f} mins"
            
            # Display momentum direction with color and actual time
            if momentum_dir == 'bullish':
                st.success(f"📈 **Bullish Momentum** - Likely persists ~{est_candles:.0f} {tf_label} candles ({time_display})")
            elif momentum_dir == 'bearish':
                st.error(f"📉 **Bearish Momentum** - Likely persists ~{est_candles:.0f} {tf_label} candles ({time_display})")
            elif momentum_dir == 'reversal_imminent':
                st.warning(f"🔄 **Reversal Imminent** - Direction change expected within 1-2 {tf_label} candles ({time_display})")
            else:
                st.info(f"↔️ **Mixed/Neutral** - Wait for clearer signal")
            
            # Display multi-timeframe details in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Multi-Timeframe RSI:**")
                rsi_6 = details.get('RSI_6', 0)
                rsi_12 = details.get('RSI_12', 0)
                rsi_24 = details.get('RSI_24', 0)
                
                # Color based on values
                rsi6_color = "🟢" if rsi_6 > 60 else "🔴" if rsi_6 < 40 else "🟡"
                rsi12_color = "🟢" if rsi_12 > 60 else "🔴" if rsi_12 < 40 else "🟡"
                rsi24_color = "🟢" if rsi_24 > 60 else "🔴" if rsi_24 < 40 else "🟡"
                
                st.write(f"{rsi6_color} RSI₆: {rsi_6:.1f} (fast)")
                st.write(f"{rsi12_color} RSI₁₂: {rsi_12:.1f} (medium)")
                st.write(f"{rsi24_color} RSI₂₄: {rsi_24:.1f} (slow)")
                
                rsi_alignment = momentum_timing.get('rsi_alignment', 'neutral')
                st.caption(f"Alignment: {rsi_alignment.replace('_', ' ')}")
            
            with col2:
                st.write("**KDJ Dynamics:**")
                stoch_j = details.get('Stoch_J', 0)
                stoch_k = details.get('Stoch_K', 0)
                stoch_d = details.get('Stoch_D', 0)
                
                # J line extremes
                j_color = "🔴" if stoch_j > 100 else "🟢" if stoch_j < 0 else "🟡"
                k_color = "🔴" if stoch_k > 80 else "🟢" if stoch_k < 20 else "🟡"
                d_color = "🔴" if stoch_d > 80 else "🟢" if stoch_d < 20 else "🟡"
                
                st.write(f"{j_color} J: {stoch_j:.1f} (leading)")
                st.write(f"{k_color} K: {stoch_k:.1f} (medium)")
                st.write(f"{d_color} D: {stoch_d:.1f} (lagging)")
                
                kdj_dynamics = momentum_timing.get('kdj_dynamics', 'neutral')
                st.caption(f"Dynamics: {kdj_dynamics.replace('_', ' ')}")
            
            # Advisory message
            st.info(f"💡 **Timing Advisory:** {momentum_timing.get('advisory', 'No timing data')}")
            
            # Price Target Display (Step 3: Where momentum is heading)
            price_target = momentum_timing.get('price_target', {})
            if price_target and price_target.get('target_price'):
                st.divider()
                target_price = price_target.get('target_price', 0)
                current_price = price_target.get('current_price', 0)
                move_pct = price_target.get('move_percentage', 0)
                atr_target = price_target.get('atr_target', 0)
                sr_constrained = price_target.get('sr_constrained', False)
                constraint_level = price_target.get('constraint_level')
                constraint_type = price_target.get('constraint_type', '')
                
                # Format the target with direction indicator
                if momentum_dir == 'bullish':
                    target_icon = "🎯📈"
                    target_color = "success"
                elif momentum_dir == 'bearish':
                    target_icon = "🎯📉"
                    target_color = "error"
                else:
                    target_icon = "🎯"
                    target_color = "info"
                
                # Build the message
                target_msg = f"{target_icon} **Predicted Price Target**: {format_price(target_price)} ({move_pct:+.2f}%)"
                
                if sr_constrained and constraint_level:
                    target_msg += f"\n\n⚠️ Constrained by {constraint_type} at {format_price(constraint_level)}"
                    target_msg += f"\n(Original ATR target was {format_price(atr_target)})"
                
                if target_color == "success":
                    st.success(target_msg)
                elif target_color == "error":
                    st.error(target_msg)
                else:
                    st.info(target_msg)
                
                # Show calculation details in expander
                with st.expander("📊 Price Target Calculation Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Current Price:** {format_price(current_price)}")
                        st.write(f"**ATR-Based Target:** {format_price(atr_target)}")
                        st.write(f"**Estimated Candles:** {est_candles:.0f}")
                    with col2:
                        st.write(f"**Final Target:** {format_price(target_price)}")
                        st.write(f"**Expected Move:** {move_pct:+.2f}%")
                        if sr_constrained:
                            st.write(f"**Blocked by:** {constraint_type} @ {format_price(constraint_level)}")
                        else:
                            st.write("**S/R Check:** Path clear")
        
        # NEW: Consolidation Advisory Section (does NOT affect M1 prediction)
        consolidation = data.get('consolidation', {})
        if consolidation and consolidation.get('consolidation_score', 0) > 30:
            st.divider()
            st.subheader("📊 Range/Consolidation Analysis")
            
            score = consolidation.get('consolidation_score', 0)
            is_consolidating = consolidation.get('is_consolidating', False)
            advisory = consolidation.get('advisory', '')
            reasons = consolidation.get('reasons', [])
            breakout_up = consolidation.get('breakout_up')
            breakout_down = consolidation.get('breakout_down')
            
            # Score display with color coding
            if score >= 70:
                st.error(f"⚠️ **STRONG CONSOLIDATION** (Score: {score}/100)")
            elif score >= 50:
                st.warning(f"⚠️ **CONSOLIDATION DETECTED** (Score: {score}/100)")
            else:
                st.info(f"📊 **Mild Range Conditions** (Score: {score}/100)")
            
            # Advisory message
            if advisory:
                st.write(f"💡 **Advisory:** {advisory}")
            
            # Contributing factors
            if reasons:
                with st.expander("📋 Contributing Factors"):
                    for reason in reasons:
                        st.write(f"• {reason}")
            
            # Breakout levels to watch
            if breakout_up or breakout_down:
                col1, col2 = st.columns(2)
                with col1:
                    if breakout_up:
                        st.success(f"📈 **Bullish Breakout Above:** {format_price(breakout_up)}")
                with col2:
                    if breakout_down:
                        st.error(f"📉 **Bearish Breakout Below:** {format_price(breakout_down)}")
        
        st.divider()
        st.subheader("🤖 AI Trading Recommendation")
        
        # Defensive check for prediction data
        if prediction is None or not isinstance(prediction, dict):
            st.error("⚠️ Prediction data unavailable. Please re-analyze the market.")
        elif 'signal' not in prediction or 'confidence' not in prediction:
            st.warning("⚠️ Incomplete prediction data. Please re-analyze the market.")
        else:
            if prediction['signal'] == 'LONG':
                st.success(f"📈 **LONG** - Confidence: {prediction['confidence']:.1f}%")
            elif prediction['signal'] == 'SHORT':
                st.error(f"📉 **SHORT** - Confidence: {prediction['confidence']:.1f}%")
            else:
                st.warning(f"⏸️ **HOLD** - Confidence: {prediction['confidence']:.1f}%")
        
        if prediction and prediction.get('entry_price') is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entry Price", format_price(prediction['entry_price']))
            with col2:
                st.metric("Stop Loss", format_price(prediction.get('stop_loss', 0)))
            with col3:
                st.metric("Take Profit", format_price(prediction.get('take_profit', 0)))
        elif prediction:
            st.info(prediction.get('recommendation', 'Insufficient data for prediction. Models not trained yet.'))
        
        if prediction and prediction.get('reasons'):
            st.write("**Why this recommendation?**")
            for reason in prediction['reasons']:
                st.write(f"• {reason}")
                    
    
    if 'last_prediction' in st.session_state and st.session_state['last_prediction']:
        prediction = st.session_state['last_prediction']
        symbol = st.session_state.get('last_symbol', '')
        market_type = st.session_state.get('last_market_type', 'crypto')
        timeframe = st.session_state.get('last_timeframe', '1H')
        indicators = st.session_state.get('last_indicators', None)
        
        # Allow manual position tracking even when AI says HOLD
        st.divider()
        st.write("**💡 Want to track a position?**")
        
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            # Default to AI recommendation direction
            ai_signal = prediction.get('signal', 'LONG')
            default_index = 1 if ai_signal == 'SHORT' else 0
            
            manual_direction = st.selectbox(
                "Trade Direction",
                ["LONG", "SHORT"],
                key="manual_direction",
                index=default_index
            )
        
        with col2:
            manual_entry = st.number_input(
                "Entry Price",
                min_value=0.0,
                value=float(prediction.get('entry_price', 0.0)) if prediction.get('entry_price') else 0.0,
                step=0.01,
                key="manual_entry_price"
            )
        
        with col3:
            trade_quantity = st.number_input(
                "Quantity (optional)", 
                min_value=0.0, 
                step=0.01,
                key="market_analysis_qty",
                help="Enter trade size"
            )
        
        col1, col2 = st.columns([2, 2])
        
        with col1:
            manual_sl = st.number_input(
                "Stop Loss",
                min_value=0.0,
                value=float(prediction.get('stop_loss', 0.0)) if prediction.get('stop_loss') else 0.0,
                step=0.01,
                key="manual_sl"
            )
        
        with col2:
            manual_tp = st.number_input(
                "Take Profit (optional)",
                min_value=0.0,
                value=float(prediction.get('take_profit', 0.0)) if prediction.get('take_profit') else 0.0,
                step=0.01,
                key="manual_tp"
            )
        
        if prediction['signal'] == 'HOLD':
            st.warning("⚠️ AI recommends HOLD - You're entering a manual position")
        
        if st.button("📊 Track This Position", type="primary", key="take_trade_market"):
            if manual_entry <= 0:
                st.error("❌ Please enter a valid entry price")
            else:
                ml_engine = get_ml_engine()
                monitor = PositionMonitor(ml_engine=ml_engine)
                result = monitor.add_position(
                    symbol,
                    market_type,
                    manual_direction,
                    manual_entry,
                    quantity=trade_quantity if trade_quantity > 0 else None,
                    stop_loss=manual_sl if manual_sl > 0 else None,
                    take_profit=manual_tp if manual_tp > 0 else None,
                    timeframe=timeframe,
                    indicators=indicators,
                    m2_entry_quality=prediction.get('m2_entry_quality')
                )
                
                if result['success']:
                    st.success(f"✅ {result['message']}")
                    st.success(f"🎯 Position added: {symbol} {manual_direction} @ {format_price(manual_entry)}")
                    st.info("📍 Go to 'Position Tracker' to view and manage this position")
                    # Keep analysis data persistent - don't delete session state
                else:
                    st.error(f"❌ {result['message']}")

elif menu == "Trading Signals":
    st.header("🎯 Quick Signal Lookup")
    st.info("💡 Tip: Use 'Market Analysis' for full chart analysis + AI recommendation. This is for quick signal lookups only.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        market_type = st.selectbox("Market Type", ["crypto", "forex", "metals", "custom"], key="signal_market")
    
    with col2:
        if market_type == "crypto":
            symbol = st.selectbox("Select Pair", CRYPTO_PAIRS, key="signal_symbol")
        elif market_type == "forex":
            symbol = st.selectbox("Select Pair", FOREX_PAIRS, key="signal_symbol")
        elif market_type == "metals":
            symbol = st.selectbox("Select Metal", METALS, key="signal_symbol")
        else:  # custom
            symbol = st.text_input("🔍 Enter Symbol (e.g., AAPL/USD, TSLA/USD, LTC/USD)", placeholder="BTC/USD", key="signal_symbol_custom").upper()
    
    if st.button("Get AI Recommendation", type="primary"):
        if not symbol or symbol.strip() == "":
            st.error("❌ Please enter a trading pair symbol")
            st.stop()
        
        # Map 'custom' to 'forex' for API compatibility (Twelve Data treats most symbols as forex pairs)
        api_market_type = "forex" if market_type == "custom" else market_type
        
        with st.spinner("Analyzing with AI models..."):
            df = get_market_data_unified(symbol, api_market_type, '1H', 100)
            
            if df is not None and len(df) > 0:
                tech = TechnicalIndicators(df)
                indicators_df = tech.calculate_all_indicators()
                indicators = tech.get_latest_indicators()
                
                # Get historical trend context for duration/slope/divergence analysis
                trend_context = tech.get_trend_context(symbol, api_market_type)
                indicators['trend_context'] = trend_context
                
                # Calculate and add S/R levels for ML predictions
                support_levels, resistance_levels = calculate_support_resistance(indicators_df)
                indicators['support_levels'] = support_levels
                indicators['resistance_levels'] = resistance_levels
                
                ml_engine = MLTradingEngine()
                prediction = ml_engine.predict(indicators)
                
                st.session_state['last_signal_prediction'] = prediction
                st.session_state['last_signal_symbol'] = symbol
                st.session_state['last_signal_market_type'] = api_market_type  # Store the API-compatible market type
                st.session_state['last_signal_indicators'] = indicators
                
                # Reset trade direction to match AI recommendation
                ai_signal = prediction.get('signal', 'LONG')
                new_direction = 'SHORT' if ai_signal == 'SHORT' else 'LONG'
                st.session_state['manual_direction_signal'] = new_direction
                
                # Pre-fill form fields with predicted values
                if prediction.get('entry_price'):
                    st.session_state['manual_entry_price_signal'] = float(prediction['entry_price'])
                    st.session_state['manual_sl_signal'] = float(prediction.get('stop_loss', 0.0))
                    st.session_state['manual_tp_signal'] = float(prediction.get('take_profit', 0.0))
                else:
                    # Clear stale values for HOLD predictions
                    if 'manual_entry_price_signal' in st.session_state:
                        del st.session_state['manual_entry_price_signal']
                    if 'manual_sl_signal' in st.session_state:
                        del st.session_state['manual_sl_signal']
                    if 'manual_tp_signal' in st.session_state:
                        del st.session_state['manual_tp_signal']
                
                st.subheader("AI Recommendation")
                
                signal_color = "🟢" if prediction['signal'] == 'LONG' else "🔴" if prediction['signal'] == 'SHORT' else "🟡"
                st.markdown(f"## {signal_color} {prediction['signal']}")
                st.markdown(f"**Confidence:** {prediction['confidence']}%")
                st.info(prediction['recommendation'])
                
                if prediction['entry_price']:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Entry Price", format_price(prediction['entry_price']))
                    with col2:
                        st.metric("Stop Loss", format_price(prediction['stop_loss']))
                    with col3:
                        st.metric("Take Profit", format_price(prediction['take_profit']))
                
                with st.expander("Model Details"):
                    st.write(f"Random Forest Confidence: {prediction.get('rf_confidence', 0)}%")
                    st.write(f"XGBoost Confidence: {prediction.get('xgb_confidence', 0)}%")
                    st.write(f"Win Probability: {prediction.get('win_probability', 0)}%")
    
    if 'last_signal_prediction' in st.session_state and st.session_state['last_signal_prediction']:
        prediction = st.session_state['last_signal_prediction']
        symbol = st.session_state.get('last_signal_symbol', '')
        market_type = st.session_state.get('last_signal_market_type', 'crypto')
        signal_indicators = st.session_state.get('last_signal_indicators', None)
        
        # Allow manual position tracking even when AI says HOLD
        st.divider()
        st.write("**💡 Want to track a position?**")
        
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            # Default to AI recommendation direction
            ai_signal = prediction.get('signal', 'LONG')
            default_index = 1 if ai_signal == 'SHORT' else 0
            
            manual_direction_signal = st.selectbox(
                "Trade Direction",
                ["LONG", "SHORT"],
                key="manual_direction_signal",
                index=default_index
            )
        
        with col2:
            manual_entry_signal = st.number_input(
                "Entry Price",
                min_value=0.0,
                value=float(prediction.get('entry_price', 0.0)) if prediction.get('entry_price') else 0.0,
                step=0.01,
                key="manual_entry_price_signal"
            )
        
        with col3:
            trade_quantity_signal = st.number_input(
                "Quantity (optional)", 
                min_value=0.0, 
                step=0.01,
                key="signal_qty",
                help="Enter trade size"
            )
        
        col1, col2 = st.columns([2, 2])
        
        with col1:
            manual_sl_signal = st.number_input(
                "Stop Loss",
                min_value=0.0,
                value=float(prediction.get('stop_loss', 0.0)) if prediction.get('stop_loss') else 0.0,
                step=0.01,
                key="manual_sl_signal"
            )
        
        with col2:
            manual_tp_signal = st.number_input(
                "Take Profit (optional)",
                min_value=0.0,
                value=float(prediction.get('take_profit', 0.0)) if prediction.get('take_profit') else 0.0,
                step=0.01,
                key="manual_tp_signal"
            )
        
        if prediction['signal'] == 'HOLD':
            st.warning("⚠️ AI recommends HOLD - You're entering a manual position")
        
        if st.button("📊 Track This Position", type="primary", key="take_trade_signal"):
            if manual_entry_signal <= 0:
                st.error("❌ Please enter a valid entry price")
            else:
                ml_engine = get_ml_engine()
                monitor = PositionMonitor(ml_engine=ml_engine)
                result = monitor.add_position(
                    symbol,
                    market_type,
                    manual_direction_signal,
                    manual_entry_signal,
                    quantity=trade_quantity_signal if trade_quantity_signal > 0 else None,
                    stop_loss=manual_sl_signal if manual_sl_signal > 0 else None,
                    take_profit=manual_tp_signal if manual_tp_signal > 0 else None,
                    indicators=signal_indicators,
                    m2_entry_quality=prediction.get('m2_entry_quality')
                )
                
                if result['success']:
                    st.success(f"✅ {result['message']}")
                    st.success(f"🎯 Position added: {symbol} {manual_direction_signal} @ {format_price(manual_entry_signal)}")
                    st.info("📍 Go to 'Position Tracker' to view and manage this position")
                    # Keep signal data persistent - don't delete session state
                else:
                    st.error(f"❌ {result['message']}")

elif menu == "Position Tracker":
    st.header("📍 Active Position Monitor")
    
    tab1, tab2, tab3 = st.tabs(["Active Positions", "Add Position", "Close Position"])
    
    with tab1:
        st.subheader("Your Active Positions")
        
        # AUTO-CHECK: Automatically check positions when tab loads
        if 'auto_checked_positions' not in st.session_state:
            st.session_state['auto_checked_positions'] = False
        
        if not st.session_state['auto_checked_positions']:
            ml_engine = get_ml_engine()
            monitor = PositionMonitor(ml_engine=ml_engine)
            with st.spinner("Auto-checking positions..."):
                monitor.check_active_positions()
                st.session_state['auto_checked_positions'] = True
        
        if st.button("🔄 Check All Positions"):
            st.session_state['auto_checked_positions'] = False  # Reset to allow fresh check
            ml_engine = get_ml_engine()
            monitor = PositionMonitor(ml_engine=ml_engine)
            with st.spinner("Checking positions..."):
                results = monitor.check_active_positions()
                
                if results:
                    for result in results:
                        if result['status'] == 'success':
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.write(f"**{result['symbol']}**")
                                st.write(f"Type: {result['trade_type']}")
                            
                            with col2:
                                st.metric("Entry", format_price(result['entry_price']))
                                st.metric("Current", format_price(result['current_price']))
                            
                            with col3:
                                pnl_color = "normal" if result['pnl_percentage'] >= 0 else "inverse"
                                st.metric("P&L", f"{result['pnl_percentage']:+.2f}%", delta_color=pnl_color)
                            
                            with col4:
                                rec_color = "🟢" if result['recommendation'] == 'HOLD' else "🔴"
                                st.write(f"{rec_color} **{result['recommendation']}**")
                                st.write(result['reason'])
                            
                            st.divider()
                else:
                    st.info("No active positions")
        
        session = get_session()
        positions = session.query(ActivePosition).filter(ActivePosition.is_active == True).all()
        session.close()
        
        if positions:
            for pos in positions:
                with st.expander(f"**{pos.symbol}** ({pos.trade_type}) - Entry: {format_price(pos.entry_price)}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Entry Price", format_price(pos.entry_price))
                        st.metric("Stop Loss", format_price(pos.stop_loss) if pos.stop_loss else "Not set")
                    with col2:
                        st.metric("Current Price", format_price(pos.current_price) if pos.current_price else "N/A")
                        st.metric("Take Profit", format_price(pos.take_profit) if pos.take_profit else "Not set")
                    with col3:
                        riyadh_time = convert_to_riyadh_time(pos.entry_time)
                        st.write(f"**Entry Time:** {riyadh_time.strftime('%Y-%m-%d %H:%M')} (Riyadh)")
                        st.write(f"**Quantity:** {pos.quantity}" if pos.quantity else "**Quantity:** Not set")
                    
                    # Show tight monitoring alerts
                    if pos.monitoring_alerts and isinstance(pos.monitoring_alerts, list) and len(pos.monitoring_alerts) > 0:
                        st.divider()
                        st.write("### 🚨 **Tight Monitoring Alerts**")
                        
                        for alert in pos.monitoring_alerts:
                            severity = alert.get('severity', 'MEDIUM')
                            alert_type = alert.get('type', '')
                            message = alert.get('message', '')
                            recommendation = alert.get('recommendation', '')
                            
                            if severity == 'HIGH':
                                st.error(f"{message}")
                                st.warning(f"**💡 Recommendation:** {recommendation}")
                            elif severity == 'MEDIUM':
                                st.warning(f"{message}")
                                st.info(f"**💡 Recommendation:** {recommendation}")
                    
                    # Show automatic monitoring recommendation
                    if pos.current_recommendation:
                        st.divider()
                        rec_color = "🟢" if pos.current_recommendation == 'HOLD' else "🔴"
                        st.write(f"### {rec_color} Auto-Monitor: **{pos.current_recommendation}**")
                        
                        if pos.last_check_time:
                            last_check_riyadh = convert_to_riyadh_time(pos.last_check_time)
                            st.caption(f"Last checked: {last_check_riyadh.strftime('%Y-%m-%d %H:%M:%S')} (Riyadh)")
                        
                        st.info("💡 Position is being monitored automatically every 5 minutes")
                    
                    st.divider()
                    st.write("**✏️ Adjust Entry Price**")
                    st.caption("Update entry price if it differs from your actual live trading platform entry")
                    
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        new_entry = st.number_input(
                            "New Entry Price",
                            min_value=0.0,
                            value=float(pos.entry_price),
                            step=0.01,
                            key=f"edit_entry_{pos.id}"
                        )
                    
                    with col_b:
                        st.write("")
                        st.write("")
                        if st.button("Update Entry", key=f"update_btn_{pos.id}"):
                            if new_entry != pos.entry_price:
                                ml_engine = get_ml_engine()
                                monitor = PositionMonitor(ml_engine=ml_engine)
                                result = monitor.update_entry_price(pos.symbol, new_entry, old_entry_price=pos.entry_price)
                                
                                if result['success']:
                                    st.success(f"✅ {result['message']}")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result['message']}")
                            else:
                                st.info("No change - entry price is the same")
    
    with tab2:
        st.subheader("Add New Position")
        
        col1, col2 = st.columns(2)
        
        with col1:
            market_type = st.selectbox("Market Type", ["crypto", "forex", "metals", "custom"], key="add_market")
            if market_type == "crypto":
                symbol = st.selectbox("Pair", CRYPTO_PAIRS, key="add_symbol")
            elif market_type == "forex":
                symbol = st.selectbox("Pair", FOREX_PAIRS, key="add_symbol")
            elif market_type == "metals":
                symbol = st.selectbox("Metal", METALS, key="add_symbol")
            else:  # custom
                symbol = st.text_input("🔍 Enter Symbol (e.g., AAPL/USD, TSLA/USD)", placeholder="BTC/USD", key="add_symbol_custom").upper()
            
            trade_type = st.selectbox("Trade Type", ["LONG", "SHORT"])
        
        with col2:
            entry_price = st.number_input("Entry Price", min_value=0.01, value=1000.0, step=0.01, help="Enter your entry price")
            quantity = st.number_input("Quantity (optional)", min_value=0.0, value=0.0, step=0.01)
            stop_loss = st.number_input("Stop Loss (optional)", min_value=0.0, value=0.0, step=0.01)
            take_profit = st.number_input("Take Profit (optional)", min_value=0.0, value=0.0, step=0.01)
        
        if st.button("Add Position"):
            if not symbol or symbol.strip() == "":
                st.error("❌ Please enter a trading pair symbol")
                st.stop()
            
            # Map 'custom' to 'forex' for API compatibility
            api_market_type = "forex" if market_type == "custom" else market_type
            
            # Capture indicators at entry for ML learning
            try:
                df = get_market_data_unified(symbol, api_market_type, '1H', 100)
                if df is not None and len(df) >= 20:
                    tech = TechnicalIndicators(df)
                    tech.calculate_all_indicators()
                    indicators_snapshot = tech.get_latest_indicators()
                else:
                    indicators_snapshot = None
            except:
                indicators_snapshot = None
            
            ml_engine = get_ml_engine()
            monitor = PositionMonitor(ml_engine=ml_engine)
            result = monitor.add_position(
                symbol, api_market_type, trade_type, entry_price,
                quantity if quantity > 0 else None,
                stop_loss if stop_loss > 0 else None,
                take_profit if take_profit > 0 else None,
                timeframe='1H',
                indicators=indicators_snapshot
            )
            
            if result['success']:
                st.success(result['message'])
                st.rerun()
            else:
                st.error(result['message'])
    
    with tab3:
        st.subheader("Close Position")
        
        session = get_session()
        active_positions = session.query(ActivePosition).filter(ActivePosition.is_active == True).all()
        session.close()
        
        if active_positions:
            position_symbols = [f"{p.symbol} ({p.trade_type} @ {format_price(p.entry_price)})" for p in active_positions]
            selected = st.selectbox("Select Position", position_symbols, key="close_position_selector")
            
            selected_symbol = selected.split(" (")[0]
            selected_trade_type = selected.split("(")[1].split(" @ ")[0]
            selected_entry_str = selected.split(" @ ")[1].rstrip(")")
            
            selected_pos = next((p for p in active_positions 
                               if p.symbol == selected_symbol 
                               and p.trade_type == selected_trade_type
                               and format_price(p.entry_price) == selected_entry_str), None)
            
            if selected_pos:
                st.info(f"📊 Position: **{selected_pos.symbol}** | Entry: **{format_price(selected_pos.entry_price)}** | Current: **{format_price(selected_pos.current_price)}**" if selected_pos.current_price else f"📊 Position: **{selected_pos.symbol}** | Entry: **{format_price(selected_pos.entry_price)}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Exit Price**")
                if selected_pos and selected_pos.current_price:
                    st.caption(f"💡 Current market price: {format_price(selected_pos.current_price)}")
                    default_exit = float(selected_pos.current_price)
                else:
                    st.caption("💡 Enter your actual exit price from live platform")
                    default_exit = float(selected_pos.entry_price) if selected_pos else 0.0
                
                exit_price = st.number_input(
                    "Adjust exit price if needed", 
                    min_value=0.0, 
                    value=default_exit,
                    step=0.01, 
                    key=f"exit_price_{selected_pos.id}",
                    label_visibility="collapsed"
                )
                outcome = st.selectbox("Outcome", ["win", "loss"], key=f"outcome_{selected_pos.id}")
            
            with col2:
                exit_type = st.selectbox(
                    "Exit Type", 
                    ["Manual Exit", "TO Achieved (Stop Loss)", "TO Achieved (Take Profit)"],
                    help="Select how you exited: manually or if stop loss/take profit was hit",
                    key=f"exit_type_{selected_pos.id}"
                )
                
            notes = st.text_area(
                "Exit Notes (Optional)", 
                placeholder="Why did you exit? What did you learn from this trade?",
                help="Record your reasoning and observations for future learning",
                key=f"notes_{selected_pos.id}"
            )
            
            if st.button("Close Position", type="primary", key=f"close_btn_{selected_pos.id}"):
                ml_engine = get_ml_engine()
                monitor = PositionMonitor(ml_engine=ml_engine)
                result = monitor.close_position(
                    selected_pos.id, 
                    exit_price, 
                    outcome,
                    exit_type=exit_type,
                    notes=notes if notes else None
                )
                
                if result['success']:
                    st.success(f"✅ {result['message']}")
                    st.info("📚 System is learning from this trade to improve future predictions!")
                    st.rerun()
                else:
                    st.error(result['message'])
        else:
            st.info("No active positions to close")

elif menu == "Risk Calculator":
    st.header("🎯 Position Size Calculator")
    st.markdown("**Calculate the right position size based on your capital and risk tolerance**")
    
    calc_mode = st.radio(
        "Calculation Mode",
        ["📊 Risk-Based (Calculate Position)", "💰 Investment-Based (Enter Amount)"],
        horizontal=True,
        help="Risk-Based: Enter risk % to calculate position. Investment-Based: Enter invested amount directly."
    )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Capital & Settings")
        total_capital = st.number_input(
            "Total Trading Capital ($)", 
            min_value=10.0, 
            value=10000.0,
            step=100.0,
            help="Your total available trading capital"
        )
        
        leverage_options = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 125, 200]
        leverage = st.selectbox(
            "Leverage (1:X)",
            options=leverage_options,
            index=leverage_options.index(20),
            help="Select your broker leverage. Spot trading = 1x."
        )
        
        buying_power = total_capital * leverage
        st.info(f"💪 Buying Power: ${buying_power:,.2f} (Capital × {leverage}x)")
        
        if calc_mode == "📊 Risk-Based (Calculate Position)":
            st.subheader("Risk Settings")
            risk_percentage = st.slider(
                "Risk Per Trade (%)", 
                min_value=0.5, 
                max_value=5.0,
                value=1.0,
                step=0.25,
                help="Industry standard: 1-2%. Aggressive: 2-3%. Very risky: 3%+"
            )
            
            if risk_percentage <= 2.0:
                st.success(f"✅ {risk_percentage}% is a safe risk level")
            elif risk_percentage <= 3.0:
                st.warning(f"⚠️ {risk_percentage}% is aggressive - be careful!")
            else:
                st.error(f"❌ {risk_percentage}% is very risky - you could lose your capital quickly!")
            
            invested_amount = None
        else:
            st.subheader("Investment Amount")
            invested_amount = st.number_input(
                "Amount Invested in This Trade ($)",
                min_value=1.0,
                max_value=buying_power,
                value=min(100.0, buying_power),
                step=10.0,
                help="The actual amount you're investing from your capital"
            )
            
            margin_used_pct = (invested_amount / total_capital) * 100
            if margin_used_pct <= 10:
                st.success(f"✅ Using {margin_used_pct:.1f}% of capital - well diversified")
            elif margin_used_pct <= 30:
                st.info(f"ℹ️ Using {margin_used_pct:.1f}% of capital")
            else:
                st.warning(f"⚠️ Using {margin_used_pct:.1f}% of capital - consider diversifying")
            
            risk_percentage = None
    
    with col2:
        st.subheader("Trade Details")
        
        trade_direction = st.selectbox(
            "Trade Direction",
            ["LONG", "SHORT"],
            help="LONG = Buy low, sell high. SHORT = Sell high, buy back low."
        )
        
        entry_price = st.number_input(
            "Entry Price ($)", 
            min_value=0.01, 
            value=65000.0,
            step=0.01,
            help="Your planned entry price for this trade"
        )
        
        if trade_direction == "LONG":
            stop_loss_price = st.number_input(
                "Stop Loss Price ($)", 
                min_value=0.01,
                value=64000.0,
                step=0.01,
                help="Stop loss BELOW entry (cut losses if price drops)"
            )
            
            take_profit_price = st.number_input(
                "Take Profit Price ($)", 
                min_value=0.01,
                value=68000.0,
                step=0.01,
                help="Take profit ABOVE entry (lock profits when price rises)"
            )
        else:  # SHORT
            stop_loss_price = st.number_input(
                "Stop Loss Price ($)", 
                min_value=0.01,
                value=66000.0,
                step=0.01,
                help="Stop loss ABOVE entry (cut losses if price rises)"
            )
            
            take_profit_price = st.number_input(
                "Take Profit Price ($)", 
                min_value=0.01,
                value=62000.0,
                step=0.01,
                help="Take profit BELOW entry (lock profits when price drops)"
            )
    
    st.divider()
    
    distance_to_stop = abs(entry_price - stop_loss_price)
    
    # Calculate based on mode
    if calc_mode == "📊 Risk-Based (Calculate Position)":
        # Risk-Based Mode: Calculate position from risk %
        risk_amount = total_capital * (risk_percentage / 100)
        position_size_units = risk_amount / distance_to_stop if distance_to_stop > 0 else 0
        position_value = position_size_units * entry_price
        potential_loss = risk_amount
        actual_risk_pct = risk_percentage
    else:
        # Investment-Based Mode: Calculate from invested amount
        position_value = invested_amount * leverage  # Invested margin × leverage = position value
        position_size_units = position_value / entry_price
        potential_loss = position_size_units * distance_to_stop
        actual_risk_pct = (potential_loss / total_capital) * 100
        risk_amount = potential_loss
    
    # Validate stop loss and take profit placement
    validation_errors = []
    
    if trade_direction == "LONG":
        if stop_loss_price >= entry_price:
            validation_errors.append("❌ **LONG trades**: Stop loss must be BELOW entry price")
        if take_profit_price <= entry_price:
            validation_errors.append("❌ **LONG trades**: Take profit must be ABOVE entry price")
    else:  # SHORT
        if stop_loss_price <= entry_price:
            validation_errors.append("❌ **SHORT trades**: Stop loss must be ABOVE entry price")
        if take_profit_price >= entry_price:
            validation_errors.append("❌ **SHORT trades**: Take profit must be BELOW entry price")
    
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        st.info(f"""
        **{trade_direction} Trade Setup:**
        - Entry: {format_price(entry_price)}
        - Stop Loss should be: {'BELOW' if trade_direction == 'LONG' else 'ABOVE'} entry
        - Take Profit should be: {'ABOVE' if trade_direction == 'LONG' else 'BELOW'} entry
        """)
    
    if distance_to_stop > 0 and position_size_units > 0:
        
        # Calculate profit based on trade direction
        if trade_direction == "LONG":
            potential_profit = (take_profit_price - entry_price) * position_size_units
        else:  # SHORT
            potential_profit = (entry_price - take_profit_price) * position_size_units
        
        risk_reward_ratio = potential_profit / potential_loss if potential_loss > 0 and potential_profit > 0 else 0
        
        st.subheader("📊 Calculated Position Size")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Risk Amount", 
                format_price(risk_amount),
                help="Maximum you'll lose if stop loss is hit"
            )
        
        with col2:
            st.metric(
                "Position Size", 
                f"{position_size_units:,.4f} units",
                help="Number of units/coins to buy"
            )
        
        with col3:
            st.metric(
                "Investment", 
                format_price(position_value),
                help="Total amount to invest"
            )
        
        with col4:
            rr_color = "normal" if risk_reward_ratio >= 2 else "inverse"
            st.metric(
                "Risk/Reward", 
                f"1:{risk_reward_ratio:.1f}",
                delta_color=rr_color,
                help="Risk vs potential reward ratio"
            )
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Profit & Loss Scenarios")
            st.write(f"**If Stop Loss is Hit ({format_price(stop_loss_price)}):**")
            st.error(f"❌ Loss: {format_price(potential_loss)} ({actual_risk_pct:.2f}% of capital)")
            
            st.write(f"**If Take Profit is Hit ({format_price(take_profit_price)}):**")
            if potential_profit > 0:
                profit_percentage = (potential_profit / total_capital) * 100
                st.success(f"✅ Profit: {format_price(potential_profit)} ({profit_percentage:.2f}% of capital)")
            else:
                loss_percentage = abs((potential_profit / total_capital) * 100)
                st.error(f"❌ LOSS: {format_price(abs(potential_profit))} ({loss_percentage:.2f}% of capital)")
                st.warning(f"⚠️ This is a LOSS because for {trade_direction} trades, take profit must be {'above' if trade_direction == 'LONG' else 'below'} entry price!")
        
        with col2:
            st.subheader("🎯 Trade Quality Assessment")
            
            if risk_reward_ratio >= 3:
                st.success("✅ **Excellent** risk/reward ratio (3:1 or better)")
            elif risk_reward_ratio >= 2:
                st.success("✅ **Good** risk/reward ratio (2:1 to 3:1)")
            elif risk_reward_ratio >= 1.5:
                st.warning("⚠️ **Acceptable** risk/reward ratio (1.5:1 to 2:1)")
            else:
                st.error("❌ **Poor** risk/reward ratio (less than 1.5:1)\nConsider skipping this trade!")
            
            margin_required = position_value / leverage
            
            if position_value > buying_power:
                st.error(f"❌ **Warning**: Position ({format_price(position_value)}) exceeds buying power ({format_price(buying_power)})!")
                st.write("**Solutions:**")
                st.write(f"• Move stop loss closer to entry (currently {format_price(distance_to_stop)} away)")
                st.write(f"• Reduce risk/investment (currently {actual_risk_pct:.1f}% at risk)")
                st.write(f"• Increase leverage (currently {leverage}x)")
            elif margin_required > total_capital:
                st.error(f"❌ **Warning**: Margin required ({format_price(margin_required)}) exceeds your capital!")
                st.write("**Solutions:**")
                st.write(f"• Move stop loss closer to entry")
                st.write(f"• Reduce risk percentage")
            elif margin_required > total_capital * 0.5:
                st.warning(f"⚠️ This trade uses {(margin_required/total_capital)*100:.1f}% of your capital as margin")
                st.write("Consider diversifying across multiple trades")
            else:
                st.success(f"✅ Margin: {format_price(margin_required)} ({(margin_required/total_capital)*100:.1f}% of capital)")
        
        st.divider()
        
        st.subheader("📝 Summary")
        margin_required = position_value / leverage
        st.markdown(f"""
        **To execute this {trade_direction} trade at {leverage}x leverage:**
        1. {'Buy' if trade_direction == 'LONG' else 'Sell short'} **{position_size_units:,.4f} units** at **{format_price(entry_price)}**
        2. Position value: **{format_price(position_value)}** | Margin required: **{format_price(margin_required)}**
        3. Set stop loss at **{format_price(stop_loss_price)}** (risk: {format_price(potential_loss)})
        4. Set take profit at **{format_price(take_profit_price)}** (potential: {format_price(potential_profit)})
        5. Risk/Reward: **1:{risk_reward_ratio:.1f}**
        
        **If this looks good, you can add it to Position Tracker manually or use AI recommendations!**
        """)

elif menu == "Performance Analytics":
    st.header("📊 Performance Analytics")
    
    session = get_session()
    
    trades = session.query(Trade).filter(Trade.exit_price.isnot(None)).order_by(Trade.entry_time.desc()).all()
    
    if trades:
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.outcome == 'win'])
        losing_trades = len([t for t in trades if t.outcome == 'loss'])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Winning Trades", winning_trades)
        with col3:
            st.metric("Losing Trades", losing_trades)
        with col4:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Calculate detailed P&L analytics
        wins = [t for t in trades if t.outcome == 'win' and t.profit_loss is not None]
        losses = [t for t in trades if t.outcome == 'loss' and t.profit_loss is not None]
        
        total_pnl = sum(t.profit_loss for t in trades if t.profit_loss is not None)
        avg_win = sum(t.profit_loss for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.profit_loss for t in losses) / len(losses) if losses else 0
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Prominent P&L Analytics Section
        st.divider()
        st.subheader("💰 Profit & Loss Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pnl_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric("Total P&L", f"${total_pnl:.2f}", delta_color=pnl_color)
        
        with col2:
            st.metric("Avg Win", f"${avg_win:.2f}", delta_color="normal" if avg_win > 0 else "off")
        
        with col3:
            st.metric("Avg Loss", f"${avg_loss:.2f}", delta_color="inverse" if avg_loss < 0 else "off")
        
        with col4:
            rr_color = "normal" if risk_reward_ratio >= 2 else "inverse"
            st.metric("Risk/Reward", f"1:{risk_reward_ratio:.2f}", delta_color=rr_color)
        
        # Performance Quality Assessment
        if risk_reward_ratio >= 2:
            st.success("✅ **Excellent Risk/Reward** - Your wins are significantly larger than your losses!")
        elif risk_reward_ratio >= 1.5:
            st.info("📊 **Good Risk/Reward** - You're managing risk well, keep it up!")
        elif risk_reward_ratio >= 1:
            st.warning("⚠️ **Acceptable Risk/Reward** - Consider taking larger profits or cutting losses faster.")
        else:
            st.error("❌ **Poor Risk/Reward** - Your losses are larger than wins. Review your strategy!")
        
        # Win Rate vs R:R Analysis
        expected_value = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)
        
        st.write(f"**Expected Value per Trade:** ${expected_value:.2f}")
        
        if expected_value > 0:
            st.success(f"✅ Your trading strategy is profitable! Over time, you expect to make ${expected_value:.2f} per trade.")
        else:
            st.error(f"❌ Your strategy is losing money. Expected loss: ${expected_value:.2f} per trade. Adjust your approach!")
        
        st.divider()
        st.subheader("📋 Trade History")
        
        # Filter controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            all_symbols = sorted(list(set([t.symbol for t in trades])))
            filter_symbol = st.selectbox("Filter by Symbol", ["All"] + all_symbols, key="filter_symbol")
        
        with col2:
            filter_type = st.selectbox("Trade Type", ["All", "LONG", "SHORT"], key="filter_type")
        
        with col3:
            filter_outcome = st.selectbox("Outcome", ["All", "WIN", "LOSS"], key="filter_outcome")
        
        with col4:
            st.write("")
            st.write("")
            if st.button("🔄 Reset Filters"):
                st.rerun()
        
        # Apply filters
        filtered_trades = trades
        if filter_symbol != "All":
            filtered_trades = [t for t in filtered_trades if t.symbol == filter_symbol]
        if filter_type != "All":
            filtered_trades = [t for t in filtered_trades if t.trade_type == filter_type]
        if filter_outcome != "All":
            filtered_trades = [t for t in filtered_trades if t.outcome == filter_outcome.lower()]
        
        # Export to CSV button
        if len(trades) > 0:
            import io
            import json
            
            csv_buffer = io.StringIO()
            csv_buffer.write("ID,Symbol,Trade Type,Entry Price,Exit Price,Entry Time,Exit Time,Outcome,P&L,Duration (hours),Indicators At Entry\n")
            
            for t in trades:
                entry_time = t.entry_time.strftime("%Y-%m-%d %H:%M:%S") if t.entry_time else "N/A"
                exit_time = t.exit_time.strftime("%Y-%m-%d %H:%M:%S") if t.exit_time else "N/A"
                
                duration_hours = 0
                if t.entry_time and t.exit_time:
                    duration_hours = (t.exit_time - t.entry_time).total_seconds() / 3600
                
                indicators_json = json.dumps(t.indicators_at_entry) if t.indicators_at_entry else "{}"
                
                csv_buffer.write(f"{t.id},{t.symbol},{t.trade_type},{t.entry_price},{t.exit_price},"
                               f"{entry_time},{exit_time},{t.outcome.upper() if t.outcome else 'N/A'},"
                               f"{t.profit_loss if t.profit_loss else 0},{duration_hours:.2f},"
                               f'"{indicators_json}"\n')
            
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download All Trades (CSV)",
                data=csv_data,
                file_name=f"trading_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
        
        st.caption(f"Showing {len(filtered_trades)} of {len(trades)} trades")
        
        # Show/Hide toggle for trade history table
        show_trade_history = st.checkbox("📋 Show Trade History Details", value=False, key="show_trade_history")
        
        if show_trade_history:
            for trade in filtered_trades:
                exit_time_riyadh = convert_to_riyadh_time(trade.exit_time) if trade.exit_time else None
                exit_time_str = exit_time_riyadh.strftime('%Y-%m-%d %H:%M') if exit_time_riyadh else 'N/A'
                
                with st.expander(f"{trade.symbol} ({trade.trade_type}) - {trade.outcome.upper()} - {exit_time_str}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Entry Price", format_price(trade.entry_price))
                    with col2:
                        st.metric("Exit Price", format_price(trade.exit_price))
                    with col3:
                        pnl_color = "normal" if trade.profit_loss_percentage and trade.profit_loss_percentage >= 0 else "inverse"
                        st.metric("P&L", f"{trade.profit_loss_percentage:.2f}%" if trade.profit_loss_percentage else "N/A", delta_color=pnl_color)
                    with col4:
                        outcome_emoji = "✅" if trade.outcome == "win" else "❌"
                        st.metric("Outcome", f"{outcome_emoji} {trade.outcome.upper()}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if trade.exit_type:
                            exit_emoji = "🎯" if "TO Achieved" in trade.exit_type else "👤"
                            st.write(f"**Exit Type:** {exit_emoji} {trade.exit_type}")
                        else:
                            st.write("**Exit Type:** Not recorded")
                        
                        entry_time_riyadh = convert_to_riyadh_time(trade.entry_time)
                        st.write(f"**Entry Time:** {entry_time_riyadh.strftime('%Y-%m-%d %H:%M')} (Riyadh)")
                        st.write(f"**Exit Time:** {exit_time_str} (Riyadh)" if exit_time_riyadh else "**Exit Time:** N/A")
                    
                    with col2:
                        if trade.quantity:
                            st.write(f"**Quantity:** {trade.quantity}")
                        if trade.profit_loss:
                            st.write(f"**Total P&L:** ${trade.profit_loss:.2f}")
                    
                    if trade.notes:
                        st.write("**Exit Notes:**")
                        st.info(trade.notes)
                    
                    st.divider()
                    
                    if f"delete_confirm_{trade.id}" not in st.session_state:
                        st.session_state[f"delete_confirm_{trade.id}"] = False
                    
                    col_del1, col_del2 = st.columns([3, 1])
                    with col_del2:
                        if not st.session_state[f"delete_confirm_{trade.id}"]:
                            if st.button("🗑️ Delete Trade", key=f"delete_btn_{trade.id}", type="secondary"):
                                st.session_state[f"delete_confirm_{trade.id}"] = True
                                st.rerun()
                        else:
                            st.warning("⚠️ Are you sure? This cannot be undone!")
                            col_confirm1, col_confirm2 = st.columns(2)
                            with col_confirm1:
                                if st.button("✅ Yes, Delete", key=f"confirm_delete_{trade.id}", type="primary"):
                                    try:
                                        session.delete(trade)
                                        session.commit()
                                        st.session_state[f"delete_confirm_{trade.id}"] = False
                                        st.success(f"✅ Deleted {trade.symbol} trade!")
                                        st.rerun()
                                    except Exception as e:
                                        session.rollback()
                                        st.error(f"❌ Error deleting trade: {str(e)}")
                            with col_confirm2:
                                if st.button("❌ Cancel", key=f"cancel_delete_{trade.id}"):
                                    st.session_state[f"delete_confirm_{trade.id}"] = False
                                    st.rerun()
        
        st.divider()
        st.subheader("Model Performance")
        
        # Manual ML training trigger for existing trades
        if total_trades >= 10:
            model_perf = session.query(ModelPerformance).filter(ModelPerformance.is_active == True).all()
            
            if not model_perf:
                st.warning(f"⚠️ You have {total_trades} trades but ML models haven't been trained yet!")
                st.write("This happens when trades were added before ML was configured.")
                
                if st.button("🤖 Train ML Models Now", type="primary"):
                    with st.spinner("Training ML models on your existing trades..."):
                        try:
                            from ml_engine import MLTradingEngine
                            ml_engine = MLTradingEngine()
                            
                            # Train with lower threshold for manual trigger (min 10 trades)
                            success = ml_engine.train_models(min_trades=10)
                            
                            if success:
                                st.success(f"✅ ML models trained successfully on {total_trades} trades!")
                                st.info("🔄 Refresh the page to see Model Performance and Indicator Analysis")
                            else:
                                st.error("❌ Training failed - make sure your trades have indicator data at entry")
                        except Exception as e:
                            st.error(f"❌ Error training models: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        
        model_perf = session.query(ModelPerformance).filter(ModelPerformance.is_active == True).all()
        
        if model_perf:
            for model in model_perf:
                with st.expander(f"{model.model_name} - Accuracy: {model.accuracy*100:.1f}%"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Precision", f"{model.precision*100:.1f}%")
                    with col2:
                        st.metric("Recall", f"{model.recall*100:.1f}%")
                    with col3:
                        st.metric("F1 Score", f"{model.f1_score*100:.1f}%")
                    
                    training_time_riyadh = convert_to_riyadh_time(model.training_date)
                    st.write(f"Training Date: {training_time_riyadh.strftime('%Y-%m-%d %H:%M')} (Riyadh)")
                    st.write(f"Total Trades Used: {model.total_trades}")
            
            # Manual retrain button with cooldown protection
            st.divider()
            st.write("**🔄 Manual Retraining**")
            st.write("Retrain models manually on all historical trades (automatic retraining happens every 10 trades).")
            
            # Check cooldown (15 minutes)
            most_recent_model = max(model_perf, key=lambda x: x.training_date)
            minutes_since_last_train = (datetime.utcnow() - most_recent_model.training_date).total_seconds() / 60
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if minutes_since_last_train < 15:
                    remaining_minutes = int(15 - minutes_since_last_train)
                    st.warning(f"⏳ Cooldown active: Wait {remaining_minutes} more minutes before manual retrain")
                else:
                    st.info(f"✅ Ready to retrain on {total_trades} trades")
            
            with col2:
                retrain_button_disabled = (minutes_since_last_train < 15 or total_trades < 10)
                
                if st.button("🔄 Retrain Now", type="secondary", disabled=retrain_button_disabled):
                    with st.spinner("Retraining ML models and backfilling indicator graphs..."):
                        try:
                            from ml_engine import MLTradingEngine
                            ml_engine = MLTradingEngine()
                            
                            success = ml_engine.train_models(min_trades=10)
                            
                            if success:
                                trades_processed = ml_engine.backfill_indicator_performance()
                                st.success(f"✅ Models retrained on {total_trades} trades!")
                                st.success(f"📊 Indicator performance analyzed for {trades_processed} trades!")
                                st.info("🔄 Refresh the page to see indicator graphs and updated metrics")
                            else:
                                st.error("❌ Retraining failed - ensure trades have indicator data")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        # Indicator Performance Analysis Dashboard
        st.subheader("🎯 Indicator Performance Analysis")
        
        from database import IndicatorPerformance
        indicator_perf = session.query(IndicatorPerformance).order_by(IndicatorPerformance.accuracy_rate.desc()).all()
        
        if indicator_perf:
            st.write("**Discover which indicators are most accurate for your trading!**")
            
            # Create indicator performance table
            perf_data = []
            for ind in indicator_perf:
                perf_data.append({
                    'Indicator': ind.indicator_name,
                    'Correct': ind.correct_count,
                    'Wrong': ind.wrong_count,
                    'Accuracy %': f"{ind.accuracy_rate:.1f}%",
                    'Weight': f"{ind.weight_multiplier:.2f}x",
                    'Total Signals': ind.total_signals
                })
            
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)
            
            # Visual bar chart of accuracy rates
            st.write("**Indicator Accuracy Rates:**")
            
            # Prepare data for chart
            indicators = [ind.indicator_name for ind in indicator_perf]
            accuracies = [ind.accuracy_rate for ind in indicator_perf]
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=indicators,
                    y=accuracies,
                    marker_color=['green' if acc >= 50 else 'red' for acc in accuracies],
                    text=[f"{acc:.1f}%" for acc in accuracies],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Indicator Accuracy Comparison",
                xaxis_title="Indicator",
                yaxis_title="Accuracy %",
                yaxis=dict(range=[0, 100]),
                height=400,
                showlegend=False
            )
            
            # Add 50% reference line
            fig.add_hline(y=50, line_dash="dash", line_color="blue", 
                         annotation_text="50% (Random)", annotation_position="right")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            best_ind = max(indicator_perf, key=lambda x: x.accuracy_rate)
            worst_ind = min(indicator_perf, key=lambda x: x.accuracy_rate)
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**🏆 Best Performer:** {best_ind.indicator_name} ({best_ind.accuracy_rate:.1f}%)")
            with col2:
                st.error(f"**📉 Weakest:** {worst_ind.indicator_name} ({worst_ind.accuracy_rate:.1f}%)")
        else:
            st.info("Complete some trades to see individual indicator performance!")
    
    # DIAGNOSTIC: Show indicator capture status (always show for completed trades)
    if trades and not indicator_perf:
        total_trades = len(trades)
        
        st.divider()
        st.subheader("🔍 Diagnostic: Indicator Data Status")
        
        # Only count COMPLETED trades with indicators (same filter as trades list)
        trades_with_indicators = session.query(Trade).filter(
            Trade.exit_price.isnot(None),
            Trade.indicators_at_entry.isnot(None),
            Trade.indicators_at_entry != None
        ).count()
        
        trades_without_indicators = total_trades - trades_with_indicators
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("With Indicators", trades_with_indicators, delta_color="normal")
        with col3:
            st.metric("Without Indicators", trades_without_indicators, delta_color="inverse")
        
        if trades_without_indicators > 0:
            st.warning(f"""
            ⚠️ **{trades_without_indicators} trades missing indicator data!**
            
            **Why graphs aren't showing:**
            - Your older trades were created before the indicator capture fix
            - The system needs trades with indicator data to generate graphs
            
            **Solution:**
            1. **Close new trades** - Latest fix now captures indicators automatically
            2. **After 2-3 new trades with indicators** - Graphs will start appearing
            3. **Manual retrain** - Click "Retrain Now" button above to update the system
            
            The indicator capture fix is working now - your next trades will have full data! ✅
            """)
        else:
            st.success("✅ All trades have indicator data - graphs should appear soon!")
    else:
        st.info("No completed trades yet. Start trading to see analytics!")
    
    session.close()

elif menu == "Model Training":
    st.header("🤖 AI Model Training")
    
    st.write("Train the AI models on your completed trades to improve prediction accuracy.")
    
    session = get_session()
    trades_count = session.query(Trade).filter(
        Trade.exit_price.isnot(None),
        Trade.outcome.isnot(None)
    ).count()
    session.close()
    
    st.info(f"Available trades for training: {trades_count}")
    
    if trades_count < 30:
        st.warning(f"⚠️ Minimum 30 trades required for training. You have {trades_count} trades.")
    
    min_trades = st.number_input("Minimum trades for training", min_value=10, value=30, step=5)
    
    if st.button("Train Models", type="primary", disabled=trades_count < 10):
        with st.spinner("Training AI models... This may take a few minutes."):
            ml_engine = MLTradingEngine()
            success = ml_engine.train_models(min_trades=min_trades)
            
            if success:
                st.success("✅ Models trained successfully!")
                st.balloons()
            else:
                st.error("❌ Training failed. Make sure you have enough completed trades.")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "AI-powered trading platform with real-time analysis, "
    "ML predictions, and automated position monitoring."
)
