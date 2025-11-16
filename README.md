# 📈 AI-Powered Trading Analysis Platform

A comprehensive real-time trading analysis platform for cryptocurrency, forex, and precious metals markets. Features ML-based predictions, automated position monitoring, and continuous learning from trade outcomes.

## 🚀 Features

### Market Analysis
- **Real-time Data**: Live market data from OKX (crypto) and Twelve Data (forex/metals)
- **12+ Technical Indicators**: RSI, MACD, Stochastic, OBV, MFI, CCI, ADX, SMA, EMA, Bollinger Bands, ATR
- **Support & Resistance**: Automated detection using pivot points and price action analysis
- **Interactive Charts**: Candlestick charts with all indicators overlaid

### AI-Powered Predictions
- **Ensemble ML Models**: Random Forest + XGBoost for robust predictions
- **Long/Short Signals**: Clear buy/sell recommendations with confidence scores
- **Entry/Exit Prices**: Suggested entry, stop-loss, and take-profit levels
- **Risk Management**: Automated position sizing based on volatility (ATR)

### Position Monitoring
- **15-Minute Updates**: Automatic position checks every 15 minutes
- **Trend Change Detection**: Real-time alerts when market conditions change
- **Hold/Exit Recommendations**: Smart position management based on technical analysis
- **P&L Tracking**: Live profit/loss monitoring for all active positions

### Whale & Smart Money Tracking
- **Volume Analysis**: Detect unusual trading volumes
- **Large Transaction Detection**: Identify whale movements
- **Order Book Analysis**: Monitor buy/sell walls and large orders
- **Smart Money Signals**: Track accumulation and distribution patterns

### Automated Learning
- **Continuous Training**: Models retrain automatically from completed trades
- **Performance Tracking**: Monitor accuracy, precision, recall, and F1 scores
- **Trade Database**: Store all trades with indicators for analysis
- **Target: 60-70% Success Rate**: Through continuous improvement

## 📊 Supported Markets

### Cryptocurrency (vs USD)
BTC, ETH, XRP, SOL, ADA, DOGE, MATIC, DOT, AVAX, LINK

### Forex (vs USD)
EUR, GBP, JPY, CHF, AUD, CAD, NZD

### Precious Metals (vs USD)
Gold (XAU), Silver (XAG), Platinum (XPT), Palladium (XPD)

## 🔧 Setup Instructions

### 1. API Keys Required

You need two API keys to use this platform:

#### OKX API (for cryptocurrency data)
1. Sign up at [OKX.com](https://www.okx.com)
2. Go to Account > API Management
3. Create a new API key (read-only permissions are sufficient)
4. Copy your API key

#### Twelve Data API (for forex and metals)
1. Sign up at [TwelveData.com](https://twelvedata.com)
2. Get your free API key from the dashboard
3. Copy your API key

### 2. Add API Keys to Replit

1. Click on the "Secrets" tab in Replit (lock icon in the sidebar)
2. Add two secrets:
   - Key: `OKX_API_KEY`, Value: [your OKX API key]
   - Key: `TWELVE_DATA_API_KEY`, Value: [your Twelve Data API key]

### 3. Start Using the Platform

The platform will automatically start when you run the Repl. Navigate through the different sections:

1. **Market Analysis**: Analyze any trading pair with full technical indicators
2. **Trading Signals**: Get AI-powered Long/Short recommendations
3. **Position Tracker**: Add and monitor your active trades
4. **Performance Analytics**: View your trading statistics and model performance
5. **Model Training**: Train AI models on your completed trades

## 📖 How to Use

### Analyzing a Market

1. Go to "Market Analysis"
2. Select market type (crypto/forex/metals)
3. Choose a trading pair
4. Select timeframe (1H, 4H, 1D)
5. Click "Analyze Market"
6. Review technical indicators, support/resistance, and whale activity

### Getting Trade Recommendations

1. Go to "Trading Signals"
2. Select your market and pair
3. Click "Get AI Recommendation"
4. Review the AI's Long/Short signal with confidence score
5. Note the suggested entry, stop-loss, and take-profit prices

### Monitoring Positions

1. Go to "Position Tracker" > "Add Position"
2. Enter your trade details (symbol, type, entry price, etc.)
3. The system will check your position every 15 minutes
4. Review recommendations on the "Active Positions" tab
5. Close positions when you exit the trade

### Training the AI Models

1. Complete at least 30 trades (enter and close them in Position Tracker)
2. Go to "Model Training"
3. Click "Train Models"
4. Wait for training to complete
5. Check improved accuracy in Performance Analytics

## 🎯 Trading Workflow

1. **Scan Markets**: Use Market Analysis to identify opportunities
2. **Get Signal**: Check AI Trading Signals for entry recommendation
3. **Enter Position**: Add the trade to Position Tracker
4. **Monitor**: System checks every 15 minutes and alerts on trend changes
5. **Exit**: Follow Hold/Exit recommendations or manual judgment
6. **Learn**: System automatically trains on your completed trades

## 📈 Performance Metrics

The platform tracks:
- **Win Rate**: Percentage of profitable trades
- **Model Accuracy**: How often predictions are correct
- **Average Profit/Loss**: Expected value per trade
- **Confidence Scores**: Prediction certainty levels

## ⚠️ Important Notes

### This is a Tool, Not Financial Advice
- The platform provides analysis and suggestions, not financial advice
- Always do your own research before trading
- Never risk more than you can afford to lose
- Past performance doesn't guarantee future results

### API Rate Limits
- OKX: Public endpoints have rate limits
- Twelve Data: Free tier has limited API calls per day
- Consider upgrading for more frequent updates

### Model Training
- Requires minimum 30 completed trades
- More trades = better accuracy
- Models improve over time with quality trade data
- Target success rate: 60-70%

## 🔒 Security

- API keys stored securely in Replit Secrets
- Database credentials managed via environment variables
- No sensitive data exposed in code or logs
- Read-only API permissions recommended

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **ML Models**: scikit-learn (Random Forest), XGBoost
- **Technical Analysis**: pandas-ta
- **Charting**: Plotly
- **Database**: PostgreSQL with SQLAlchemy
- **APIs**: OKX API, Twelve Data API

## 📝 Database Schema

- **trades**: Completed trades with P&L and indicators
- **active_positions**: Currently monitored positions
- **market_data**: Historical market snapshots
- **model_performance**: ML model metrics
- **whale_activity**: Detected whale movements

## 🚦 Status Indicators

- 🟢 **Green**: Bullish signal, oversold, or strong uptrend
- 🔴 **Red**: Bearish signal, overbought, or strong downtrend  
- 🟡 **Yellow**: Neutral or hold signal

## 💡 Tips for Success

1. **Start Small**: Test with small positions to build trade history
2. **Track Everything**: Log all trades accurately for model training
3. **Multiple Timeframes**: Check multiple timeframes before entering
4. **Risk Management**: Always use stop-losses
5. **Be Patient**: Let the system gather data and improve over time
6. **Review Signals**: Combine multiple indicators, not just one
7. **Watch Whales**: Pay attention to large volume movements

## 🤝 Support

For issues or questions:
1. Check the API keys are correctly set in Replit Secrets
2. Verify API rate limits haven't been exceeded
3. Review the Performance Analytics for model status
4. Check the console logs for error messages

## 📄 License

This is a trading analysis tool for educational and research purposes.

---

**Disclaimer**: Trading involves substantial risk of loss. This platform is provided as-is without any warranties. Use at your own risk.
