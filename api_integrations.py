import requests
import os
from datetime import datetime, timedelta
import pandas as pd
import time

class OKXClient:
    def __init__(self, api_key=None):
        self.base_url = "https://www.okx.com"
        self.api_key = api_key
        
    def get_market_data(self, symbol, interval='1H', limit=100):
        try:
            okx_symbol = symbol.replace('/', '-').replace('USD', 'USDT')
            
            endpoint = f"{self.base_url}/api/v5/market/candles"
            params = {
                'instId': okx_symbol,
                'bar': interval,
                'limit': limit
            }
            
            headers = {}
            if self.api_key:
                headers['OK-ACCESS-KEY'] = self.api_key
            
            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['code'] == '0' and data['data']:
                df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume_coins', 'volCcy', 'volCcyQuote', 'confirm'])
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
                df[['open', 'high', 'low', 'close', 'volCcyQuote']] = df[['open', 'high', 'low', 'close', 'volCcyQuote']].astype(float)
                df['volume'] = df['volCcyQuote']
                df = df.sort_values('timestamp')
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            else:
                return None
                
        except Exception as e:
            print(f"OKX API Error: {e}")
            return None
    
    def get_orderbook(self, symbol, depth=20):
        try:
            endpoint = f"{self.base_url}/api/v5/market/books"
            params = {
                'instId': symbol,
                'sz': depth
            }
            
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['code'] == '0' and data['data']:
                return {
                    'bids': [[float(x[0]), float(x[1])] for x in data['data'][0]['bids']],
                    'asks': [[float(x[0]), float(x[1])] for x in data['data'][0]['asks']]
                }
            return None
            
        except Exception as e:
            print(f"OKX Orderbook Error: {e}")
            return None
    
    def get_ticker(self, symbol):
        try:
            okx_symbol = symbol.replace('/', '-').replace('USD', 'USDT')
            
            endpoint = f"{self.base_url}/api/v5/market/ticker"
            params = {'instId': okx_symbol}
            
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['code'] == '0' and data['data']:
                ticker = data['data'][0]
                return {
                    'symbol': symbol,
                    'last_price': float(ticker['last']),
                    'volume_24h': float(ticker['vol24h']),
                    'high_24h': float(ticker['high24h']),
                    'low_24h': float(ticker['low24h']),
                    'bid': float(ticker['bidPx']),
                    'ask': float(ticker['askPx'])
                }
            return None
            
        except Exception as e:
            print(f"OKX Ticker Error: {e}")
            return None

class TwelveDataClient:
    def __init__(self, api_key):
        self.base_url = "https://api.twelvedata.com"
        self.api_key = api_key
        
    def get_market_data(self, symbol, interval='1h', outputsize=100):
        try:
            endpoint = f"{self.base_url}/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize,
                'apikey': self.api_key
            }
            
            print(f"Fetching {symbol} at {interval} interval...")
            response = requests.get(endpoint, params=params, timeout=10)
            
            data = response.json()
            print(f"API Response Status: {response.status_code}")
            print(f"API Response Keys: {list(data.keys())}")
            
            if response.status_code == 429:
                print(f"⚠️ Rate limit hit - wait 1 minute")
                return None
            
            if response.status_code != 200:
                error_msg = data.get('message', data.get('status', f'HTTP {response.status_code}'))
                print(f"TwelveData API Error: {error_msg}")
                return None
            
            if 'values' in data and data['values']:
                df = pd.DataFrame(data['values'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                
                rename_cols = {
                    'datetime': 'timestamp',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close'
                }
                
                if 'volume' in df.columns:
                    rename_cols['volume'] = 'volume'
                
                df = df.rename(columns=rename_cols)
                
                df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
                
                if 'volume' in df.columns:
                    df['volume'] = df['volume'].astype(float)
                else:
                    df['volume'] = 0.0
                
                df = df.sort_values('timestamp')
                print(f"Successfully fetched {len(df)} candles")
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            else:
                error_msg = data.get('message', data.get('status', 'Unknown error'))
                print(f"TwelveData Error: {error_msg}")
                print(f"Full response: {data}")
                return None
                
        except Exception as e:
            print(f"TwelveData API Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_quote(self, symbol):
        try:
            endpoint = f"{self.base_url}/quote"
            params = {
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'close' in data:
                return {
                    'symbol': symbol,
                    'last_price': float(data['close']),
                    'volume': float(data.get('volume', 0)),
                    'high': float(data.get('high', 0)),
                    'low': float(data.get('low', 0)),
                    'open': float(data.get('open', 0))
                }
            return None
            
        except Exception as e:
            print(f"TwelveData Quote Error: {e}")
            return None

def get_market_data_unified(symbol, market_type, interval='1H', limit=100):
    twelve_data_key = os.getenv('TWELVE_DATA_API_KEY')
    
    if twelve_data_key:
        twelve_client = TwelveDataClient(api_key=twelve_data_key)
        interval_map = {
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1H': '1h',
            '4H': '4h',
            '1D': '1day',
            '1W': '1week'
        }
        td_interval = interval_map.get(interval, '1h')
        df = twelve_client.get_market_data(symbol, td_interval, limit)
        
        # If Twelve Data failed, try OKX fallback for crypto (OKX has public endpoints)
        if df is None and market_type == 'crypto':
            print(f"⚠️ Twelve Data failed for {symbol}. Trying OKX fallback...")
            okx_client = OKXClient(api_key=os.getenv('OKX_API_KEY'))
            df = okx_client.get_market_data(symbol, interval, limit)
            if df is not None:
                print(f"✓ Using OKX data for {symbol}")
                return df
            else:
                print(f"❌ OKX also failed for {symbol}")
        
        if df is not None and market_type == 'crypto' and 'volume' in df.columns:
            if df['volume'].sum() == 0:
                print(f"TwelveData returned {symbol} without volume data. Trying OKX for crypto volume...")
                okx_client = OKXClient(api_key=os.getenv('OKX_API_KEY'))
                okx_df = okx_client.get_market_data(symbol, interval, limit)
                if okx_df is not None and okx_df['volume'].sum() > 0:
                    print(f"✓ Using OKX data for {symbol} (has volume)")
                    return okx_df
                else:
                    print(f"⚠ OKX also has no volume for {symbol}, using TwelveData anyway")
        
        return df
    
    # Only use OKX if configured
    okx_key = os.getenv('OKX_API_KEY')
    if market_type == 'crypto' and okx_key:
        okx_client = OKXClient(api_key=okx_key)
        return okx_client.get_market_data(symbol, interval, limit)
    
    return None

def get_current_price(symbol, market_type):
    twelve_data_key = os.getenv('TWELVE_DATA_API_KEY')
    
    if twelve_data_key:
        twelve_client = TwelveDataClient(api_key=twelve_data_key)
        quote = twelve_client.get_quote(symbol)
        
        # If Twelve Data failed, try OKX fallback for crypto (OKX has public endpoints)
        if quote is None and market_type == 'crypto':
            print(f"⚠️ Twelve Data quote failed for {symbol}. Trying OKX fallback...")
            okx_client = OKXClient(api_key=os.getenv('OKX_API_KEY'))
            ticker = okx_client.get_ticker(symbol)
            if ticker:
                print(f"✓ Using OKX price for {symbol}")
                return ticker['last_price']
            else:
                print(f"❌ OKX also failed for {symbol}")
        
        return quote['last_price'] if quote else None
    
    # Only use OKX if configured
    okx_key = os.getenv('OKX_API_KEY')
    if market_type == 'crypto' and okx_key:
        okx_client = OKXClient(api_key=okx_key)
        ticker = okx_client.get_ticker(symbol)
        return ticker['last_price'] if ticker else None
    
    return None
