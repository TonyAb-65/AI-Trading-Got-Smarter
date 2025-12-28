import os
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, inspect, text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import streamlit as st

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    market_type = Column(String(20), nullable=False, index=True)
    trade_type = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    entry_time = Column(DateTime, default=datetime.utcnow, index=True)
    exit_time = Column(DateTime, index=True)
    quantity = Column(Float)
    profit_loss = Column(Float)
    profit_loss_percentage = Column(Float)
    outcome = Column(String(10), index=True)
    exit_type = Column(String(50))
    indicators_at_entry = Column(JSON)
    indicators_at_exit = Column(JSON)
    model_confidence = Column(Float)
    m2_entry_quality = Column(Float)
    consolidation_score = Column(Float)  # For M2 learning - range detection at entry
    notes = Column(Text)
    
    __table_args__ = (
        Index('idx_symbol_outcome', 'symbol', 'outcome'),
        Index('idx_entry_exit_time', 'entry_time', 'exit_time'),
    )

class ActivePosition(Base):
    __tablename__ = 'active_positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    market_type = Column(String(20), nullable=False)
    trade_type = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, default=datetime.utcnow)
    current_price = Column(Float)
    quantity = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    timeframe = Column(String(10), default='1H')
    last_check_time = Column(DateTime, index=True)
    current_recommendation = Column(String(10))
    indicators_snapshot = Column(JSON)
    m2_entry_quality = Column(Float)
    is_active = Column(Boolean, default=True, index=True)
    last_obv_slope = Column(Float)
    monitoring_alerts = Column(JSON)

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    market_type = Column(String(20), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)
    indicators = Column(JSON)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )

class ModelPerformance(Base):
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False)
    version = Column(String(20))
    training_date = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    avg_profit = Column(Float)
    avg_loss = Column(Float)
    parameters = Column(JSON)
    is_active = Column(Boolean, default=True)

class WhaleActivity(Base):
    __tablename__ = 'whale_activity'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    transaction_type = Column(String(10))
    volume = Column(Float)
    price = Column(Float)
    impact_score = Column(Float)
    notes = Column(Text)

class IndicatorPerformance(Base):
    __tablename__ = 'indicator_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    indicator_name = Column(String(50), nullable=False, unique=True)
    correct_count = Column(Integer, default=0)
    wrong_count = Column(Integer, default=0)
    accuracy_rate = Column(Float, default=0.0)
    weight_multiplier = Column(Float, default=1.0)
    last_updated = Column(DateTime, default=datetime.utcnow)
    total_signals = Column(Integer, default=0)

class DivergenceEvent(Base):
    __tablename__ = 'divergence_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, nullable=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    indicator = Column(String(20), nullable=False)
    divergence_type = Column(String(10), nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow)
    detection_price = Column(Float, nullable=False)
    resolved_at = Column(DateTime, nullable=True)
    resolution_price = Column(Float, nullable=True)
    resolution_candles = Column(Integer, nullable=True)
    resolution_outcome = Column(String(20), nullable=True)
    status = Column(String(20), default='active')

class DivergenceStats(Base):
    __tablename__ = 'divergence_stats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    indicator = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    divergence_type = Column(String(10), nullable=False)
    avg_resolution_candles = Column(Float, default=0.0)
    avg_resolution_minutes = Column(Float, default=0.0)
    median_resolution_candles = Column(Float, default=0.0)
    p90_resolution_candles = Column(Float, default=0.0)
    speed_class = Column(String(20), default='unknown')
    success_rate = Column(Float, default=0.0)
    sample_size = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow)

class MLModel(Base):
    __tablename__ = 'ml_models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False, unique=True)
    model_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(Integer, default=1)

class CustomPair(Base):
    __tablename__ = 'custom_pairs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True)
    market_type = Column(String(20), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)

@st.cache_resource(hash_funcs={str: lambda x: x})
def get_engine(_database_url=None):
    """
    Get SQLAlchemy engine with proper connection pooling.
    Cache is keyed on database URL to prevent stale connections.
    """
    database_url = _database_url or os.getenv('DATABASE_URL')
    if not database_url:
        # Use home directory for SQLite (persists on Streamlit Cloud)
        home_dir = Path.home()
        db_path = home_dir / 'trading_platform.db'
        database_url = f'sqlite:///{db_path}'
        print(f"💾 DATABASE_URL not set, using persistent SQLite at: {db_path}")
    
    if database_url.startswith('sqlite'):
        engine = create_engine(database_url, connect_args={'check_same_thread': False})
    else:
        # PostgreSQL with conservative connection pooling (Streamlit Cloud has limited file descriptors)
        engine = create_engine(
            database_url,
            pool_pre_ping=True,          # Test connections before using
            pool_size=3,                  # Max 3 permanent connections (reduced from 10)
            max_overflow=5,               # Allow 5 additional overflow (reduced from 20)
            pool_recycle=1800,            # Recycle connections every 30 mins (was 1 hour)
            pool_timeout=10,              # Wait max 10 seconds for a connection
            echo=False                    # Disable SQL logging for performance
        )
    
    return engine

def ensure_active_position_timeframe_column(engine):
    """Auto-add timeframe column to active_positions if missing (backward compatibility)"""
    try:
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns('active_positions')]
        
        if 'timeframe' not in columns:
            print("⚙️ Auto-migration: Adding timeframe column to active_positions...")
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE active_positions ADD COLUMN timeframe VARCHAR(10) DEFAULT '1H'"
                ))
            print("✅ Timeframe column added successfully!")
        
        if 'last_obv_slope' not in columns:
            print("⚙️ Auto-migration: Adding last_obv_slope column to active_positions...")
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE active_positions ADD COLUMN last_obv_slope FLOAT"
                ))
            print("✅ last_obv_slope column added successfully!")
        
        if 'monitoring_alerts' not in columns:
            print("⚙️ Auto-migration: Adding monitoring_alerts column to active_positions...")
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE active_positions ADD COLUMN monitoring_alerts JSON"
                ))
            print("✅ monitoring_alerts column added successfully!")
        
        if 'm2_entry_quality' not in columns:
            print("⚙️ Auto-migration: Adding m2_entry_quality column to active_positions...")
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE active_positions ADD COLUMN m2_entry_quality FLOAT"
                ))
            print("✅ m2_entry_quality column added successfully!")
        
        # Check trades table for m2_entry_quality
        trades_columns = [col['name'] for col in inspector.get_columns('trades')]
        if 'm2_entry_quality' not in trades_columns:
            print("⚙️ Auto-migration: Adding m2_entry_quality column to trades...")
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE trades ADD COLUMN m2_entry_quality FLOAT"
                ))
            print("✅ m2_entry_quality column added to trades successfully!")
        
        # Check trades table for consolidation_score (M2 learning feature)
        if 'consolidation_score' not in trades_columns:
            print("⚙️ Auto-migration: Adding consolidation_score column to trades...")
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE trades ADD COLUMN consolidation_score FLOAT"
                ))
            print("✅ consolidation_score column added to trades successfully!")
        
        if all(col in columns for col in ['timeframe', 'last_obv_slope', 'monitoring_alerts', 'm2_entry_quality']):
            print("✅ All columns exist - database is up to date")
    except Exception as e:
        print(f"Warning: Could not auto-migrate columns: {e}")

def add_database_indexes(engine):
    """
    Add performance indexes to existing databases (backward compatibility).
    These indexes dramatically improve query performance for 100+ trades.
    """
    try:
        inspector = inspect(engine)
        
        # Check if tables exist first
        tables = inspector.get_table_names()
        if 'trades' not in tables:
            return
        
        # Get existing indexes
        existing_indexes = inspector.get_indexes('trades')
        index_names = [idx['name'] for idx in existing_indexes]
        
        print("⚙️ Adding database indexes for performance...")
        
        with engine.begin() as conn:
            # Composite index for symbol + outcome queries (common in analytics)
            if 'idx_symbol_outcome' not in index_names:
                conn.execute(text(
                    "CREATE INDEX idx_symbol_outcome ON trades(symbol, outcome)"
                ))
                print("   ✅ Added idx_symbol_outcome (trades)")
            
            # Composite index for time-based queries
            if 'idx_entry_exit_time' not in index_names:
                conn.execute(text(
                    "CREATE INDEX idx_entry_exit_time ON trades(entry_time, exit_time)"
                ))
                print("   ✅ Added idx_entry_exit_time (trades)")
            
            # MarketData composite index
            market_data_indexes = inspector.get_indexes('market_data')
            market_data_index_names = [idx['name'] for idx in market_data_indexes]
            
            if 'idx_symbol_timestamp' not in market_data_index_names:
                conn.execute(text(
                    "CREATE INDEX idx_symbol_timestamp ON market_data(symbol, timestamp)"
                ))
                print("   ✅ Added idx_symbol_timestamp (market_data)")
        
        print("✅ Database indexes migration complete!")
        
    except Exception as e:
        print(f"Warning: Could not add indexes: {e}")
        print("   (This is normal for new databases - indexes will be created with tables)")

def init_db():
    try:
        database_url = os.getenv('DATABASE_URL')
        engine = get_engine(database_url)
        Base.metadata.create_all(engine)
        ensure_active_position_timeframe_column(engine)
        add_database_indexes(engine)
        return engine
    except Exception as e:
        print(f"Database initialization error: {e}")
        raise

# Module-level session factory - initialized once at import
# This prevents worker threads from falling back to SQLite when DATABASE_URL is not visible
_cached_database_url = os.getenv('DATABASE_URL')
_cached_engine = None
_SessionLocal = None

def _get_cached_engine():
    """Get or create the cached engine using the DATABASE_URL captured at import time."""
    global _cached_engine
    if _cached_engine is None:
        # Use the URL captured at module load time, not os.getenv() which fails in worker threads
        _cached_engine = get_engine(_cached_database_url)
        print(f"🔌 Database engine initialized: {'PostgreSQL' if _cached_database_url else 'SQLite (fallback)'}")
    return _cached_engine

def _get_session_factory():
    """Get or create the cached session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=_get_cached_engine())
    return _SessionLocal

def get_session():
    """Get a new database session using the cached engine.
    
    IMPORTANT: This uses a module-level cached engine to prevent worker thread issues
    where os.getenv('DATABASE_URL') returns None.
    """
    SessionFactory = _get_session_factory()
    return SessionFactory()

from contextlib import contextmanager

@contextmanager
def get_db_session():
    """Context manager for database sessions - ensures proper cleanup.
    
    Usage:
        with get_db_session() as session:
            trades = session.query(Trade).all()
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
