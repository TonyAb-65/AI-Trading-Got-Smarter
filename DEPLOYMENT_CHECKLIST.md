# 🚀 GitHub & Streamlit Cloud Deployment Checklist

## Phase 1: Pre-Deployment Preparation

### Files to Create/Update
- [ ] **Create `requirements.txt`** in project root (see GITHUB_DEPLOY.md for exact content)
- [ ] **Update `.streamlit/config.toml`** - Use content from `.streamlit/config.streamlit-cloud.toml`
- [ ] **Verify `.env.example`** exists (template for users)
- [ ] **Verify `.gitignore`** includes `.env`, `*.db`, `__pycache__/`, `models/*.pkl`

### Files to Include in GitHub Repository
```
├── app.py                          ✅ Main Streamlit app
├── ml_engine.py                    ✅ ML engine (Random Forest + XGBoost)
├── position_monitor.py             ✅ Position tracking
├── technical_indicators.py         ✅ Technical analysis
├── api_integrations.py             ✅ API handlers
├── database.py                     ✅ Database layer
├── scheduler.py                    ✅ Background jobs
├── whale_tracker.py                ✅ Smart money tracking
├── divergence_analytics.py         ✅ Divergence timing
├── divergence_logger.py            ✅ Divergence logging
├── divergence_resolver.py          ✅ Divergence resolution
├── config.py                       ✅ Configuration (if exists)
├── requirements.txt                ⚠️  CREATE THIS (see GITHUB_DEPLOY.md)
├── .streamlit/
│   └── config.toml                 ⚠️  UPDATE THIS (remove port/address)
├── .env.example                    ✅ Environment template
├── .gitignore                      ✅ Git ignore rules
├── README.md                       ✅ Project description
├── USER_GUIDE.md                   ✅ User documentation
├── GITHUB_DEPLOY.md                ✅ Deployment guide
└── DEPLOYMENT_CHECKLIST.md         ✅ This file
```

### Files to EXCLUDE from GitHub
- [ ] **.env** (contains actual secrets) - NEVER commit
- [ ] **trading_platform.db** (SQLite database) - too large, regenerated
- [ ] **models/*.pkl** (ML models) - regenerated on deployment
- [ ] **models/*.json** (training metadata) - regenerated
- [ ] **__pycache__/** (Python cache) - auto-generated
- [ ] **.pythonlibs/** (Replit-specific) - not needed
- [ ] **uv.lock** (Replit-specific) - not needed

## Phase 2: API Keys & Secrets Setup

### Required Secrets
- [ ] **TWELVE_DATA_API_KEY** obtained from https://twelvedata.com/pricing
  - Free tier: 800 API calls/day
  - Sufficient for normal usage (2 checks/hour)

### Optional Secrets
- [ ] **DATABASE_URL** (PostgreSQL connection string)
  - Recommended: Neon.tech free tier (500MB)
  - Alternative: Supabase, ElephantSQL
  - If omitted: Uses SQLite (data lost on restart)
  
- [ ] **OKX_API_KEY** (for whale tracking)
  - Optional: Public endpoints work without it
  - Get at: https://www.okx.com/account/my-api

## Phase 3: Code Verification

### Compatibility Checks
- [ ] **No Replit-specific code** (grep for "replit", "REPL_ID")
- [ ] **All secrets use os.getenv()** (no hardcoded keys)
- [ ] **Models directory auto-creates** (lines 31-32 in ml_engine.py)
- [ ] **Database has SQLite fallback** (database.py line 160)
- [ ] **All imports are standard libraries** (no custom modules)

### File Path Checks
- [ ] **Relative paths only** (no `/home/runner/...`)
- [ ] **os.path.join() used** for cross-platform compatibility
- [ ] **Auto-create directories** (models, logs if needed)

## Phase 4: GitHub Push

### Git Commands
```bash
# Step 1: Initialize (if not already)
git init
git status  # Check what will be committed

# Step 2: Add files
git add .
git status  # Verify .env is NOT in the list

# Step 3: Commit
git commit -m "Initial commit: AI Trading Platform"

# Step 4: Create GitHub repo and link
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main

# Step 5: Push
git push -u origin main
```

### Verify on GitHub
- [ ] Repository created successfully
- [ ] All Python files visible
- [ ] `.env` file NOT present (check carefully!)
- [ ] `.env.example` IS present
- [ ] `requirements.txt` exists and correct
- [ ] README.md displays properly

## Phase 5: Streamlit Cloud Deployment

### App Configuration
1. **Go to:** https://share.streamlit.io/
2. **Login** with GitHub account
3. **New app** button
4. **Select repository:** YOUR_USERNAME/YOUR_REPO
5. **Branch:** main
6. **Main file path:** app.py
7. **Python version:** 3.12 (or latest)

### Add Secrets
1. App deployed → Click **Settings**
2. **Secrets** tab
3. Add in TOML format:

```toml
TWELVE_DATA_API_KEY = "your_actual_key_here"

# Optional: Add if using PostgreSQL
DATABASE_URL = "postgresql://user:password@host:port/dbname"

# Optional: Add if using OKX
OKX_API_KEY = "your_okx_key_here"
```

4. **Save** secrets
5. **Reboot** app

### First Launch Verification
- [ ] App loads without errors
- [ ] "TWELVE_DATA_API_KEY required" warning gone
- [ ] Market Analysis tab displays
- [ ] Can analyze a symbol (e.g., BTC/USD)
- [ ] Charts render properly
- [ ] No database errors in logs

## Phase 6: Functional Testing

### Core Features Test
- [ ] **Market Analysis** - Analyze BTC/USD or any symbol
  - Technical indicators display
  - Heikin-Ashi chart renders
  - AI prediction shows (LONG/SHORT/HOLD)
  
- [ ] **Trading Signals** - Check multiple timeframes
  - Signals generate for different symbols
  - Smart Money section displays
  
- [ ] **Position Tracking** - Track a position
  - Can manually add position
  - Position saves to database
  - Monitoring works (check after 30 min)
  
- [ ] **Performance Analytics** - View after adding trades
  - Trade history displays
  - P&L calculations correct
  - ML model metrics visible

### ML Features Test
- [ ] **Indicator Capture** - New positions save indicators
- [ ] **Learning** - Close position triggers learning
- [ ] **Retraining** - "Retrain Now" button works
- [ ] **Graphs** - Indicator performance graphs populate

## Phase 7: Production Database Setup (Optional but Recommended)

### Neon.tech Setup
1. **Sign up:** https://neon.tech
2. **Create project:** "trading-platform"
3. **Copy connection string** (postgres://...)
4. **Add to Streamlit secrets** as `DATABASE_URL`
5. **Reboot app**
6. **Verify:** Check Position Tracker - data persists after restart

### Database Migration
- [ ] Old SQLite data exported (if needed)
- [ ] PostgreSQL connection tested
- [ ] Tables auto-created on first run
- [ ] Indexes created (check logs for "✅ Database indexes migration complete")

## Phase 8: Ongoing Maintenance

### Monitoring
- [ ] Check Streamlit Cloud logs daily (first week)
- [ ] Monitor API usage at Twelve Data dashboard
- [ ] Verify position monitoring runs every 30 minutes
- [ ] Check database storage usage (if PostgreSQL)

### Updates and Fixes
When agent provides surgical fixes:
1. **Get module file** (e.g., `ml_engine.py`)
2. **Copy entire content** from agent
3. **Paste into GitHub file** directly
4. **Commit:** `git commit -m "Fix: [description]"`
5. **Push:** `git push`
6. **Streamlit auto-deploys** (wait 1-2 minutes)

### Backup Strategy
- [ ] **Code:** Already on GitHub ✅
- [ ] **Database:** Use PostgreSQL provider backups
- [ ] **API Keys:** Stored securely in Streamlit secrets
- [ ] **ML Models:** Retrain from historical trades (no backup needed)

## Common Issues and Solutions

### Issue: "No module named 'X'"
**Fix:** Check `requirements.txt` spelling (pandas-ta, not pandas_ta)

### Issue: Database connection errors
**Fix:** Verify DATABASE_URL format in secrets (include port, correct password)

### Issue: API rate limits
**Fix:** Platform uses 2 checks/hour (960 calls/month) - well within free tier

### Issue: Graphs not showing
**Fix:** Close 2-3 positions with indicators, click "Retrain Now", refresh page

### Issue: App slow to load
**Fix:** Normal - ML models retrain on first run after deployment

## Success Criteria

### Deployment Successful When:
✅ App loads on Streamlit Cloud URL  
✅ No errors in Streamlit Cloud logs  
✅ Can analyze any crypto/forex/metal symbol  
✅ Charts render with Heikin-Ashi candles  
✅ Positions can be tracked and monitored  
✅ ML learning works from trade outcomes  
✅ Graphs populate after retraining  
✅ Data persists between sessions (with PostgreSQL)  

### Platform Fully Operational When:
✅ 10+ completed trades in database  
✅ ML models trained and making predictions  
✅ Indicator performance graphs showing  
✅ Position monitoring running automatically  
✅ No API rate limit errors  
✅ All features accessible and working  

## Support Resources

- **Streamlit Docs:** https://docs.streamlit.io/
- **Streamlit Community:** https://discuss.streamlit.io/
- **GitHub Docs:** https://docs.github.com/
- **Twelve Data API:** https://twelvedata.com/docs
- **This Guide:** GITHUB_DEPLOY.md

---

**Ready to Deploy?** Start with Phase 1 and work through each section systematically. Good luck! 🚀
