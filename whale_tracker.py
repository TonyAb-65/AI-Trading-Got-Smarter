import pandas as pd
import numpy as np
from datetime import datetime

class WhaleTracker:
    def __init__(self, df, orderbook_data=None):
        self.df = df.copy()
        self.orderbook_data = orderbook_data
        
    def detect_whale_movements(self, volume_threshold_multiplier=3.0):
        df = self.df.copy()
        
        if 'Volume_SMA' not in df.columns and len(df) >= 20:
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
        
        whale_signals = []
        
        if 'Volume_SMA' in df.columns:
            df['volume_ratio'] = df['volume'] / df['Volume_SMA']
            
            recent_data = df.tail(50)
            
            for idx, row in recent_data.iterrows():
                if pd.notna(row['volume_ratio']) and row['volume_ratio'] > volume_threshold_multiplier:
                    price_change = ((row['close'] - row['open']) / row['open']) * 100
                    
                    impact_score = row['volume_ratio'] * abs(price_change)
                    
                    if row['close'] > row['open']:
                        transaction_type = 'BUY'
                    else:
                        transaction_type = 'SELL'
                    
                    whale_signals.append({
                        'timestamp': row['timestamp'],
                        'transaction_type': transaction_type,
                        'volume': row['volume'],
                        'volume_ratio': round(row['volume_ratio'], 2),
                        'price': row['close'],
                        'price_change_pct': round(price_change, 2),
                        'impact_score': round(impact_score, 2)
                    })
        
        return sorted(whale_signals, key=lambda x: x['impact_score'], reverse=True)[:10]
    
    def detect_smart_money(self):
        df = self.df.copy()
        signals = []
        
        if len(df) < 20:
            return signals
        
        recent = df.tail(20)
        
        avg_volume = recent['volume'].mean()
        current_volume = recent.iloc[-1]['volume']
        
        price_trend = 'up' if recent.iloc[-1]['close'] > recent.iloc[-10]['close'] else 'down'
        
        volume_trend = 'increasing' if current_volume > avg_volume * 1.5 else 'normal'
        
        if volume_trend == 'increasing':
            if price_trend == 'up':
                signals.append({
                    'type': 'accumulation',
                    'description': 'Smart money accumulation detected - Rising prices with high volume',
                    'confidence': 'high' if current_volume > avg_volume * 2 else 'medium',
                    'timestamp': recent.iloc[-1]['timestamp']
                })
            else:
                signals.append({
                    'type': 'distribution',
                    'description': 'Smart money distribution detected - Falling prices with high volume',
                    'confidence': 'high' if current_volume > avg_volume * 2 else 'medium',
                    'timestamp': recent.iloc[-1]['timestamp']
                })
        
        if self.orderbook_data:
            signals.extend(self._analyze_orderbook())
        
        return signals
    
    def _analyze_orderbook(self):
        signals = []
        
        if not self.orderbook_data or 'bids' not in self.orderbook_data or 'asks' not in self.orderbook_data:
            return signals
        
        bids = self.orderbook_data['bids']
        asks = self.orderbook_data['asks']
        
        if not bids or not asks:
            return signals
        
        total_bid_volume = sum([bid[1] for bid in bids[:10]])
        total_ask_volume = sum([ask[1] for ask in asks[:10]])
        
        if total_bid_volume > total_ask_volume * 1.5:
            signals.append({
                'type': 'buy_wall',
                'description': f'Strong buy wall detected - Bid volume {round(total_bid_volume/total_ask_volume, 2)}x ask volume',
                'confidence': 'high',
                'timestamp': datetime.utcnow()
            })
        elif total_ask_volume > total_bid_volume * 1.5:
            signals.append({
                'type': 'sell_wall',
                'description': f'Strong sell wall detected - Ask volume {round(total_ask_volume/total_bid_volume, 2)}x bid volume',
                'confidence': 'high',
                'timestamp': datetime.utcnow()
            })
        
        large_bids = [bid for bid in bids if bid[1] > total_bid_volume * 0.2]
        large_asks = [ask for ask in asks if ask[1] > total_ask_volume * 0.2]
        
        if large_bids:
            signals.append({
                'type': 'large_bid',
                'description': f'Large bid order detected at {large_bids[0][0]}',
                'confidence': 'medium',
                'timestamp': datetime.utcnow()
            })
        
        if large_asks:
            signals.append({
                'type': 'large_ask',
                'description': f'Large ask order detected at {large_asks[0][0]}',
                'confidence': 'medium',
                'timestamp': datetime.utcnow()
            })
        
        return signals
    
    def get_volume_profile(self):
        df = self.df.copy()
        
        if len(df) < 20:
            return None
        
        recent = df.tail(50)
        
        volume_avg = recent['volume'].mean()
        volume_current = recent.iloc[-1]['volume']
        volume_max = recent['volume'].max()
        
        return {
            'current_volume': round(volume_current, 2),
            'average_volume': round(volume_avg, 2),
            'max_volume': round(volume_max, 2),
            'volume_vs_avg': round((volume_current / volume_avg) * 100, 2),
            'volume_percentile': round((recent['volume'] <= volume_current).sum() / len(recent) * 100, 2)
        }
