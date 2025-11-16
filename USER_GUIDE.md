# AI Trading Platform - User Guide

## Getting Started

### Step 1: Set Up API Keys

Before using the platform, you need to obtain and configure API keys:

#### OKX API Key (for Cryptocurrency)
1. Create an account at [OKX.com](https://www.okx.com)
2. Navigate to Account → API Management
3. Create a new API key (read-only permissions are sufficient)
4. Copy the API key

#### Twelve Data API Key (for Forex & Metals)
1. Sign up at [TwelveData.com](https://twelvedata.com)
2. Access your dashboard and copy your API key
3. Free tier provides 800 API calls per day

#### Add Keys to Replit
1. Open the Secrets tab (lock icon) in Replit
2. Add:
   - `OKX_API_KEY` = [your OKX API key]
   - `TWELVE_DATA_API_KEY` = [your Twelve Data key]

### Step 2: Understanding the Platform

The platform has 5 main sections:

## 1. Market Analysis

**Purpose**: Analyze any trading pair with comprehensive technical indicators

**How to Use**:
1. Select market type (crypto/forex/metals)
2. Choose a trading pair
3. Select timeframe (1H, 4H, 1D)
4. Click "Analyze Market"

**What You'll See**:
- Interactive candlestick chart with indicators
- Current price, RSI, ADX, MFI metrics
- Technical signals (bullish/bearish/neutral)
- Support and resistance levels
- Whale activity (for crypto only)
- Smart money signals

**Interpreting Signals**:
- 🟢 Green = Bullish/Oversold/Support
- 🔴 Red = Bearish/Overbought/Resistance  
- 🟡 Yellow = Neutral/Hold

## 2. Trading Signals

**Purpose**: Get AI-powered Long/Short recommendations

**How to Use**:
1. Select market and pair
2. Click "Get AI Recommendation"
3. Review the signal and confidence score

**Understanding Recommendations**:
- **Signal**: LONG (buy), SHORT (sell), or HOLD (wait)
- **Confidence**: Percentage certainty (higher is better)
- **Entry Price**: Suggested entry point
- **Stop Loss**: Risk management exit point
- **Take Profit**: Target profit exit point

**Important Notes**:
- Models need minimum 30 completed trades to generate predictions
- Before that, system will show "Insufficient data"
- Confidence >70% indicates strong signal
- Always verify with Market Analysis before entering

## 3. Position Tracker

**Purpose**: Monitor your active trades with automated analysis

### Adding a Position

1. Go to "Add Position" tab
2. Fill in:
   - Market type and pair
   - Trade type (LONG or SHORT)
   - Entry price
   - Quantity (optional but recommended)
   - Stop loss (optional)
   - Take profit (optional)
3. Click "Add Position"

**What Happens**:
- System saves your position
- Monitors it every 15 minutes automatically
- Analyzes trend changes
- Provides Hold/Exit recommendations

### Checking Positions

1. Go to "Active Positions" tab
2. Click "Check All Positions"
3. Review recommendations for each position

**Recommendation Types**:
- **HOLD**: Position looks healthy, keep it open
- **EXIT**: Conditions suggest closing position

**Exit Reasons**:
- Stop loss hit
- Take profit reached
- Trend reversal detected
- Smart money movement against position
- Approaching support/resistance

### Closing a Position

1. Go to "Close Position" tab
2. Select the position
3. Enter exit price
4. Mark outcome as "win" or "loss"
5. Click "Close Position"

**Why This Matters**:
- Builds your trade history
- Trains the AI models
- Improves future predictions

## 4. Performance Analytics

**Purpose**: Track your trading performance and model accuracy

**Metrics Displayed**:
- Total trades executed
- Win rate percentage
- Recent trade history
- Model performance (accuracy, precision, recall)

**Using Analytics**:
- Monitor your progress toward 60-70% target
- Identify patterns in winning vs losing trades
- Track model improvement over time

## 5. Model Training

**Purpose**: Train AI models on your completed trades

**When to Train**:
- After accumulating 30+ completed trades
- When you want to improve predictions
- After a series of new trades

**How to Train**:
1. Navigate to Model Training section
2. Review available trades count
3. Click "Train Models"
4. Wait for training to complete (1-3 minutes)

**What Happens**:
- System analyzes all completed trades
- Extracts patterns from indicators
- Trains Random Forest and XGBoost models
- Saves models for future predictions
- Updates performance metrics

**Expected Results**:
- Initial accuracy: 50-60%
- With quality data: 60-70% target
- Improves with more diverse trades

## Best Practices

### For Market Analysis
1. **Check Multiple Timeframes**: View 1H, 4H, and 1D for complete picture
2. **Look for Confluence**: Multiple bullish/bearish signals are stronger
3. **Watch Volume**: High volume confirms trends
4. **Note Support/Resistance**: Price often reacts at these levels

### For Trading Signals
1. **Verify Confidence**: Only act on >60% confidence signals
2. **Cross-Reference**: Compare AI signal with Market Analysis
3. **Consider Context**: Check whale activity and smart money
4. **Risk Management**: Always use stop losses

### For Position Monitoring
1. **Add All Trades**: Even external trades help train the system
2. **Be Accurate**: Enter exact entry/exit prices
3. **Mark Outcomes Honestly**: Accurate data = better predictions
4. **Review Recommendations**: System checks every 15 minutes

### For Model Training
1. **Quality Over Quantity**: 50 good trades better than 200 random ones
2. **Diverse Markets**: Trade different pairs for robust models
3. **Consistent Strategy**: Models learn your approach
4. **Regular Retraining**: Train after every 20-30 new trades

## Understanding Technical Indicators

### Trend Indicators
- **SMA/EMA**: Price above = uptrend, below = downtrend
- **ADX**: >25 = strong trend, <25 = weak trend
- **MACD**: Above signal = bullish, below = bearish

### Momentum Indicators
- **RSI**: >70 overbought, <30 oversold
- **Stochastic**: >80 overbought, <20 oversold
- **CCI**: >100 overbought, <-100 oversold

### Volume Indicators
- **OBV**: Rising = accumulation, falling = distribution
- **MFI**: Money flow strength (like RSI with volume)

### Volatility Indicators
- **Bollinger Bands**: Price touches upper = overbought, lower = oversold
- **ATR**: Higher values = more volatile market

## Common Scenarios

### Scenario 1: Finding a Trade Setup
1. Go to Market Analysis
2. Scan multiple pairs for bullish/bearish confluence
3. Check Trading Signals for AI confirmation
4. Verify support/resistance levels
5. Enter position if signals align

### Scenario 2: Managing an Active Trade
1. Add position to Position Tracker
2. System monitors every 15 minutes
3. Check dashboard for updates
4. Follow Hold/Exit recommendations
5. Close when target reached or trend reverses

### Scenario 3: Improving AI Accuracy
1. Trade consistently for 30+ positions
2. Record all trades accurately
3. Run Model Training
4. Check Performance Analytics
5. Continue trading and retraining

## Troubleshooting

### "Failed to fetch market data"
- **Cause**: API key missing or invalid
- **Solution**: Check Replit Secrets, verify API keys

### "No clear signal / HOLD recommendation"
- **Cause**: Models not trained yet or low confidence
- **Solution**: Complete 30+ trades and train models

### "Unable to fetch current price"
- **Cause**: API rate limit or network issue
- **Solution**: Wait a few minutes and retry

### "Not enough trades for training"
- **Cause**: Fewer than minimum required trades
- **Solution**: Complete more trades (30 minimum)

## API Rate Limits

### OKX
- Public endpoints: ~20 requests/minute
- Sufficient for normal usage
- Exceeded: Wait 1 minute

### Twelve Data (Free Tier)
- 8 requests/minute
- 800 requests/day
- Exceeded: Upgrade or wait until next day

## Tips for Success

### Week 1-2: Learning Phase
- Focus on understanding each indicator
- Practice with market analysis
- Don't trade real money yet
- Build initial trade history

### Week 3-4: Data Collection
- Make 30-50 trades across different markets
- Record every trade accurately
- Train models once you hit 30 trades
- Start seeing pattern recognition

### Month 2+: Optimization
- Retrain models regularly
- Focus on high-confidence signals (>70%)
- Track your personal win rate
- Refine your strategy based on analytics

### Long-term Success
- Combine AI signals with your judgment
- Never risk more than you can lose
- Diversify across markets
- Keep learning and adapting

## Risk Disclaimer

⚠️ **Important**: This platform provides analysis tools, not financial advice.

- Trading involves substantial risk of loss
- Past performance doesn't guarantee future results
- Always do your own research
- Only trade with money you can afford to lose
- The 60-70% success rate is a target, not a guarantee
- Results depend on data quality and market conditions

## Support

If you encounter issues:
1. Check API keys are correctly set
2. Review error messages in console
3. Verify you have sufficient API credits
4. Ensure database is accessible
5. Restart the application if needed

---

**Remember**: The platform learns from YOUR trades. The more quality data you provide, the better the predictions become. Start small, trade wisely, and let the system improve over time.
