# AI-Powered Trading Analysis Platform

## Overview
This AI-powered platform provides real-time market analysis, machine learning-based trading recommendations, active position monitoring, and automated learning for crypto, forex, and precious metals markets. Its purpose is to deliver accurate trading insights and improve decision-making through continuous learning and risk management.

## User Preferences
I want the agent to use simple language and provide detailed explanations for complex concepts. I prefer an iterative development approach where major changes are discussed and approved before implementation. The agent should prioritize stability and performance. I like functional programming paradigms where appropriate, and I prefer that the agent does not make changes to the `models/indicator_weights.json` file directly without explicit instruction.

## System Architecture
The platform is built with a modular architecture comprising a Database Layer (SQLAlchemy), API Integrations (OKX, Twelve Data), Technical Analysis, Whale Tracker, ML Engine (Random Forest + XGBoost ensemble), Position Monitor, and Divergence Timing Intelligence, all orchestrated by a Streamlit UI. It features real-time data processing, comprehensive technical indicators including candlestick patterns, and advanced ML models that learn from trade outcomes. The UI/UX is designed for clarity, with dynamic price formatting and an emphasis on actionable insights. The system includes an auto-migration system for zero-downtime database updates and a divergence timing tracker that learns historical patterns to predict when price reversals will occur.

## External Dependencies
- **TWELVE_DATA_API_KEY**: Primary market data (crypto, forex, metals)
- **OKX_API_KEY**: Optional - OKX has public endpoints and serves as automatic fallback for crypto when Twelve Data fails
- **SQLAlchemy**: ORM for database interactions
- **Streamlit**: For the user interface
- **pandas-ta**: For technical analysis calculations

## Key Features

### Divergence Timing Intelligence (New!)
The platform now includes sophisticated divergence timing analysis that tracks when divergences are detected and when they resolve, building historical intelligence to answer: "Will this divergence reverse quickly or slowly?"

**Architecture:**
- `divergence_events` table: Tracks each divergence from detection to resolution
- `divergence_stats` table: Stores aggregated timing statistics per indicator/timeframe
- Auto-resolution detection: Monitors price movements and marks divergences as resolved when price moves 2%+ in predicted direction
- Timeframe-relative speed classification: FAST (<2 candles), ACTIONABLE (2-6 candles), SLOW (>6 candles)

**Workflow:**
1. **Detection**: When divergence appears (OBV, RSI, Stochastic, MFI), system logs it to database
2. **Resolution**: Position monitor checks active divergences every 15 minutes, marks resolved when criteria met
3. **Analytics**: Background job runs daily to calculate average resolution time, success rate, and speed class
4. **Display**: UI shows timing intelligence ("Typically resolves in 4 candles (4 hours) - Scalp possible")

**Use Case Example:**
- Bearish OBV divergence detected on 1H chart
- Historical data shows it typically resolves in 5 hours (ACTIONABLE - scalp window exists)
- Trader can take quick LONG position for 2-3 hours before reversal hits
- System warns if divergence is FAST (<2hrs) - avoid trade, wait for reversal

This solves the critical problem: "Price is still going up despite divergence downtrend" by quantifying the lag time between divergence detection and price reversal.

## Recent Changes
- 2025-11-16: **📊 Critical Fix: Indicator Graph Backfill**
  - **Root Cause**: Manual "Retrain Now" only trained ML models but didn't populate IndicatorPerformance table
  - **Impact**: User had 29 trades with indicators but graphs showed empty
  - **Fix**: Added `backfill_indicator_performance()` method to ml_engine.py
    - Clears existing IndicatorPerformance table (prevents duplicates)
    - Iterates all completed trades with indicators_at_entry
    - Calls `_track_indicator_performance(trade)` for each historical trade
    - Returns count of trades processed for user feedback
  - **Updated Manual Retrain**: Now calls backfill after training models
    - Shows two success messages: "Models retrained" + "Indicator performance analyzed"
    - Instructs user to refresh page to see graphs
  - **Result**: Clicking "Retrain Now" now populates graphs with all historical trade data
  - Modified files: `ml_engine.py`, `app.py`

- 2025-11-15: **📊 Heikin-Ashi Candles + API Rate Limit Optimization + OKX Fallback + Auto-Fill Fix + Indicator Graph Fix**
  - **Heikin-Ashi Candles**: Replaced regular candlesticks with Heikin-Ashi for clearer trend visualization
    - Filters out market noise and makes trend direction easier to identify
    - Smoother price action reduces false signals
    - Chart title updated to "(Heikin-Ashi)" for clarity
    - Zero changes to technical indicators or ML logic - purely visual enhancement
  - **OKX Automatic Fallback**: When Twelve Data fails for crypto pairs, OKX is used automatically
    - Works without requiring OKX_API_KEY (public endpoints)
    - Ensures crypto data is always available even when Twelve Data is down or rate-limited
    - Logs fallback attempts for transparency
  - **API Rate Limit Fix**: Improved error handling for Twelve Data API rate limits
    - Surface actual API error messages instead of generic failures
    - Log "⚠️ Rate limit hit - wait 1 minute" for user visibility
    - Prevents silent failures when hitting 8 req/min limit
  - **Scheduler Optimization**: Reduced position monitoring from 15min → 30min intervals
    - 50% fewer API calls (2 checks/hour vs 4 checks/hour)
    - Prevents 24 req/min bursts that exceed Twelve Data limits
    - Maintains daily divergence analytics unchanged
  - **Auto-Fill Fix**: Fixed "Track This Position" form to pre-populate predicted values
    - Entry price, stop loss, take profit now auto-fill from AI predictions
    - Direction auto-selects LONG or SHORT based on prediction
    - Fixed Streamlit widget state persistence issue
    - Stale values cleared for HOLD predictions
  - **Indicator Graph Fix**: Fixed indicator performance tracking to include all indicators
    - Neutral indicators now counted (previously skipped)
    - Relaxed thresholds: RSI 45/55, MFI 45/55, Stochastic 45/55, CCI at 0
    - Neutral signals count as "correct" since they don't oppose the trade
    - Graph will now populate with real data from trades
    - Added diagnostic tool to show which trades have/lack indicator data
    - Displays explanation when graphs empty due to missing historical indicators
    - New trades automatically capture indicators for future graph population
  - Modified files: `app.py`, `api_integrations.py`, `scheduler.py`, `ml_engine.py`, `replit.md`

- 2025-11-14: **🎓 ML Learning Enhancements (Production-Ready)**
  - **Manual Retrain Button**: Added "🔄 Retrain Now" button in Performance Analytics
    - 15-minute cooldown protection to prevent training abuse
    - Minimum 10 trades requirement (down from 30)
    - Real-time feedback with success/error messages
    - Enables retroactive training on existing production trades
  - **ML Learning Explanation**: Added user-facing documentation in Market Analysis
    - Collapsible expander explaining Individual Learning (every trade)
    - Explains Bulk Retraining (every 10 trades at milestones)
    - Shows how ML improves predictions over time
    - Non-technical language for clarity
  - **Training Threshold Fix**: Reduced `min_trades` from 30→10
    - Automatic retraining now triggers at 10, 20, 30, 40... trades
    - Enables faster ML learning for new users
    - All training uses ALL historical data (wins + losses)
  - **Scaler Safety Check**: Added guard in `predict()` to prevent NotFittedError
    - Falls back to rule-based prediction if scaler not fitted
    - Graceful degradation ensures no crashes
  - **Indicator Responsibility Graph**: Already existed showing wins (green) vs losses (red)
  - Architect-validated and deployment-ready
  - Modified files: `ml_engine.py`, `app.py`

- 2025-11-14: **⚡ Critical Performance & ML Fixes (Deployment-Ready)**
  - **ML Learning Flow Redesign**: Implemented dual learning approach requested by user
    - **Individual Learning**: Updates indicator weights from EVERY trade (wins AND losses) immediately
    - **Bulk Retraining**: Retrains models every 10 trades (10→20→30→40...)
    - At 10 trades: Train on all 10 (wins + losses)
    - At 20 trades: Train on all 20 (wins + losses)
    - At 30 trades: Train on all 30 (wins + losses)
    - Continues every 10 trades using ALL historical data
    - Prevents duplicate retraining with persistent milestone tracking
  - **Database Indexes**: Added indexes to frequently queried columns (symbol, outcome, timestamps, is_active) for performance at scale
  - **Connection Pooling**: Implemented PostgreSQL connection pooling (pool_size=10, max_overflow=20, pool_recycle=3600) to prevent connection exhaustion
  - **SHORT P&L Fix**: Fixed SHORT position P&L calculation to normalize to $10k hypothetical position when quantity missing, enabling fair analytics comparison
  - **Note Preservation**: All position close operations preserve user-provided notes while adding normalization notices
  - **Database Cache Fix**: Fixed Streamlit resource caching to properly key on DATABASE_URL - eliminates stale database connections when switching between dev/prod
  - **Auto-Migration**: Database indexes automatically applied via migration function in `init_db()` for zero-downtime updates
  - All fixes validated by architect review - no regressions, ready for deployment
  - Modified files: `ml_engine.py`, `database.py`, `position_monitor.py`, `app.py`

- 2025-11-14: **🚨 Global Alert System + Auto-Check Position Monitoring**
  - **CRITICAL FIX**: Addressed SOL position loss where stop loss was hit without early warning
  - Implemented hybrid monitoring approach: Auto-check on page view + background scheduler backup
  - **Global Alert Banner**: HIGH severity alerts now appear at top of ALL tabs (Market Analysis, Trading Signals, etc.)
  - Users see critical warnings no matter which page they're viewing - no need to manually check Position Tracker
  - **Auto-Check on Position Tracker**: Automatically checks all positions when viewing Active Positions tab
  - Eliminates need to manually click "Check All Positions" button for instant feedback
  - Background 15-min scheduler continues running as backup safety net
  - Modified `app.py`: Added `check_global_alerts()` function and global banner display logic
  - Surgical fix - zero changes to existing monitoring logic, purely additive display layer
  - Solves "I was on Market Analysis and didn't see the warning" problem completely

- 2025-11-13: **🛡️ Priority-Based Risk Management in Position Monitor**
  - Fixed critical issue where monitoring alerts were ignored in recommendations
  - Implemented 3-tier priority system: HIGH severity alerts (approaching SL, OBV flip) → SL/TP triggers → Indicator signals
  - System now returns EXIT when price reaches 60% distance to stop loss, regardless of bullish indicators
  - Prevents scenarios where "HOLD" recommendation leads to stop loss being hit
  - Example: Gold position approaching SL now triggers "EXIT - ⚠️ RISK ALERT" instead of "HOLD - Still bullish"
  - Modified `position_monitor.py`: Added monitoring_alerts parameter to `_generate_recommendation()` method
  - Risk management now takes absolute priority over indicator optimism

- 2025-11-13: **📊 Indicator Capture System for Pattern Analysis**
  - Implemented automatic capture of all 12+ technical indicators when trades are opened
  - Modified `position_monitor.py`: Added optional `indicators` parameter to `add_position()` function
  - Modified `app.py`: Store `latest_indicators` in session state and pass to `add_position()` from both Market Analysis and Trading Signals sections
  - Data flow: UI indicators → `ActivePosition.indicators_snapshot` → `Trade.indicators_at_entry` on close
  - Enables future automated pattern analysis after 10-15 trades to identify winning/losing indicator patterns
  - Surgical fix with zero logic changes to existing ML/trading algorithms - purely additive data capture
  - Manual position adds (without indicators) still work - handles None gracefully

- 2025-11-12: **🔍 Custom Trading Pair Search**
  - Added "custom" market type option to all analysis sections (Market Analysis, Trading Signals, Position Tracker)
  - Users can now analyze ANY trading pair not in predefined lists (e.g., AAPL/USD, TSLA/USD, LTC/USD, etc.)
  - Text input field appears when "custom" is selected, with automatic uppercase conversion
  - Custom symbols mapped to "forex" for API compatibility (Twelve Data treats most symbols as forex pairs)
  - Validation prevents empty symbol submissions
  - Consistent implementation across all three sections: analysis, signals, and position tracking
  
- 2025-11-12: **⏱️ Divergence Timing Intelligence System**
  - Added divergence event tracking system with full lifecycle monitoring (detection → resolution)
  - Implemented timeframe-relative speed classification (FAST/ACTIONABLE/SLOW based on candle count)
  - Background analytics job calculates historical timing statistics (avg resolution time, success rate)
  - UI displays divergence timing intelligence in Smart Money section with actionable recommendations
  - Position monitor auto-resolves divergences when 2% price movement threshold hit
  - Scheduler runs nightly analytics to update timing statistics from resolved events
  - New database tables: divergence_events, divergence_stats
  - Zero logic changes to existing ML/trading algorithms - purely additive intelligence layer