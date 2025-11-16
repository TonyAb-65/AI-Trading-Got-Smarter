API_CONFIG = {
    'okx': {
        'rate_limit_calls_per_minute': 20,
        'timeout': 10,
        'retry_attempts': 3,
        'retry_delay': 2
    },
    'twelve_data': {
        'rate_limit_calls_per_minute': 8,
        'timeout': 10,
        'retry_attempts': 3,
        'retry_delay': 2
    }
}

ML_CONFIG = {
    'min_trades_for_training': 30,
    'test_size': 0.2,
    'random_state': 42,
    'cross_validation_folds': 5,
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'eval_metric': 'logloss'
    }
}

POSITION_MONITORING = {
    'check_interval_minutes': 15,
    'trend_reversal_threshold': 4,
    'stop_loss_percentage': 3.0,
    'take_profit_percentage': 5.0,
    'atr_multiplier_stop': 2.0,
    'atr_multiplier_target': 3.0
}

TECHNICAL_INDICATORS = {
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'stochastic_k': 14,
    'stochastic_d': 3,
    'adx_period': 14,
    'adx_threshold': 25,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'mfi_period': 14,
    'cci_period': 20,
    'atr_period': 14
}

WHALE_DETECTION = {
    'volume_threshold_multiplier': 3.0,
    'large_order_percentile': 0.2,
    'orderbook_imbalance_threshold': 1.5
}
