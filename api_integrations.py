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

class AlphaVantageClient:
    def __init__(self, api_key):
        self.base_url = "https://www.alphavantage.co/query"
        self.api_key = api_key
        
    def get_market_data(self, symbol, interval='1H', limit=100):
        try:
            # Alpha Vantage FREE tier only supports Daily/Weekly (no intraday)
            # Force daily data for all requests since intraday is premium-only
            interval_map = {
                '1m': 'daily',   # Fallback to daily (intraday is premium)
                '5m': 'daily',   # Fallback to daily (intraday is premium)
                '15m': 'daily',  # Fallback to daily (intraday is premium)
                '30m': 'daily',  # Fallback to daily (intraday is premium)
                '1H': 'daily',   # Fallback to daily (intraday is premium)
                '4H': 'daily',   # Fallback to daily
                '1D': 'daily',
                '1W': 'weekly'
            }
            
            av_interval = interval_map.get(interval, 'daily')
            
            if av_interval == 'daily':
                function = 'FX_DAILY'
            elif av_interval == 'weekly':
                function = 'FX_WEEKLY'
            else:
                function = 'FX_DAILY'
            
            from_symbol = symbol.split('/')[0]
            to_symbol = symbol.split('/')[1] if '/' in symbol else 'USD'
            
            params = {
                'function': function,
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'apikey': self.api_key,
                'outputsize': 'full'  # Get full dataset (more data)
            }
            
            print(f"📡 Fetching {symbol} from Alpha Vantage ({function} - free tier: daily only)...")
            response = requests.get(self.base_url, params=params, timeout=15)
            
            data = response.json()
            
            if 'Error Message' in data:
                print(f"❌ Alpha Vantage Error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                print(f"⚠️ Alpha Vantage API limit: {data['Note']}")
                return None
            
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key or not data[time_series_key]:
                print(f"❌ No data found in Alpha Vantage response")
                return None
            
            time_series = data[time_series_key]
            
            rows = []
            for timestamp_str, values in time_series.items():
                rows.append({
                    'timestamp': pd.to_datetime(timestamp_str),
                    'open': float(values.get('1. open', 0)),
                    'high': float(values.get('2. high', 0)),
                    'low': float(values.get('3. low', 0)),
                    'close': float(values.get('4. close', 0)),
                    'volume': 0.0
                })
            
            if not rows:
                return None
            
            df = pd.DataFrame(rows)
            df = df.sort_values('timestamp')
            df = df.tail(limit)
            
            print(f"✅ Alpha Vantage: Fetched {len(df)} candles for {symbol}")
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Alpha Vantage API Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_quote(self, symbol):
        try:
            from_symbol = symbol.split('/')[0]
            to_symbol = symbol.split('/')[1] if '/' in symbol else 'USD'
            
            params = {
                'function': 'CURRENCY_EXCHANGE_RATE',
                'from_currency': from_symbol,
                'to_currency': to_symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'Realtime Currency Exchange Rate' in data:
                rate_data = data['Realtime Currency Exchange Rate']
                return {
                    'symbol': symbol,
                    'last_price': float(rate_data['5. Exchange Rate']),
                    'volume': 0.0,
                    'high': 0.0,
                    'low': 0.0,
                    'open': 0.0
                }
            
            print(f"❌ Alpha Vantage Quote Error: {data}")
            return None
            
        except Exception as e:
            print(f"Alpha Vantage Quote Error: {e}")
            return None

def get_market_data_unified(symbol, market_type, interval='1H', limit=100):
    twelve_data_key = os.getenv('TWELVE_DATA_API_KEY')
    
    # Auto-detect crypto symbols in custom/forex mode
    CRYPTO_TICKERS = ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE', 'ADA', 'DOT', 'LINK', 'UNI', 'MATIC', 'SOL', 'AVAX', 'ATOM', 'ALGO', 'FIL', 'AAVE', 'COMP', 'MKR', 'SNX', 'CRV', 'SUSHI', 'YFI']
    symbol_base = symbol.split('/')[0].upper() if '/' in symbol else symbol.upper()
    if market_type == 'forex' and symbol_base in CRYPTO_TICKERS:
        market_type = 'crypto'
        print(f"🔍 Auto-detected crypto symbol: {symbol} → routing to crypto API")
    
    # For CRYPTO: Use OKX first (free), Twelve Data as fallback
    if market_type == 'crypto':
        print(f"🔵 Crypto detected: Using OKX as primary source for {symbol}")
        okx_client = OKXClient(api_key=os.getenv('OKX_API_KEY'))
        df = okx_client.get_market_data(symbol, interval, limit)
        
        if df is not None:
            print(f"✓ Using OKX data for {symbol}")
            return df
        else:
            # OKX failed, try Twelve Data fallback
            if twelve_data_key:
                print(f"⚠️ OKX failed for {symbol}. Trying Twelve Data fallback...")
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
                if df is not None:
                    print(f"✓ Using Twelve Data for {symbol}")
                    return df
                else:
                    print(f"❌ Both OKX and Twelve Data failed for {symbol}")
            else:
                print(f"❌ OKX failed and no Twelve Data API key configured")
            return None
    
    # For FOREX & METALS: Use Twelve Data as primary, Alpha Vantage as fallback
    if twelve_data_key:
        print(f"🟢 Forex/Metal detected: Using Twelve Data for {symbol}")
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
        
        if df is not None:
            print(f"✓ Using Twelve Data for {symbol}")
            return df
        else:
            # Twelve Data failed, try Alpha Vantage fallback
            alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if alpha_vantage_key:
                print(f"⚠️ Twelve Data failed for {symbol}. Trying Alpha Vantage fallback...")
                alpha_client = AlphaVantageClient(api_key=alpha_vantage_key)
                df = alpha_client.get_market_data(symbol, interval, limit)
                if df is not None:
                    print(f"✓ Using Alpha Vantage for {symbol}")
                    return df
                else:
                    print(f"❌ Both Twelve Data and Alpha Vantage failed for {symbol}")
            else:
                print(f"❌ Twelve Data failed and no Alpha Vantage API key configured")
            return None
    
    # If no Twelve Data key, try Alpha Vantage directly
    alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if alpha_vantage_key:
        print(f"🟡 No Twelve Data key - Using Alpha Vantage for {symbol}")
        alpha_client = AlphaVantageClient(api_key=alpha_vantage_key)
        df = alpha_client.get_market_data(symbol, interval, limit)
        if df is not None:
            print(f"✓ Using Alpha Vantage for {symbol}")
            return df
    
    return None

def get_current_price(symbol, market_type):
    twelve_data_key = os.getenv('TWELVE_DATA_API_KEY')
    
    # Auto-detect crypto symbols in custom/forex mode
    CRYPTO_TICKERS = ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE', 'ADA', 'DOT', 'LINK', 'UNI', 'MATIC', 'SOL', 'AVAX', 'ATOM', 'ALGO', 'FIL', 'AAVE', 'COMP', 'MKR', 'SNX', 'CRV', 'SUSHI', 'YFI']
    symbol_base = symbol.split('/')[0].upper() if '/' in symbol else symbol.upper()
    if market_type == 'forex' and symbol_base in CRYPTO_TICKERS:
        market_type = 'crypto'
    
    # For CRYPTO: Use OKX first (free), Twelve Data as fallback
    if market_type == 'crypto':
        print(f"🔵 Crypto detected: Using OKX as primary source for {symbol} price")
        okx_client = OKXClient(api_key=os.getenv('OKX_API_KEY'))
        ticker = okx_client.get_ticker(symbol)
        
        if ticker:
            print(f"✓ Using OKX price for {symbol}")
            return ticker['last_price']
        else:
            # OKX failed, try Twelve Data fallback
            if twelve_data_key:
                print(f"⚠️ OKX failed for {symbol}. Trying Twelve Data fallback...")
                twelve_client = TwelveDataClient(api_key=twelve_data_key)
                quote = twelve_client.get_quote(symbol)
                if quote:
                    print(f"✓ Using Twelve Data price for {symbol}")
                    return quote['last_price']
                else:
                    print(f"❌ Both OKX and Twelve Data failed for {symbol}")
            else:
                print(f"❌ OKX failed and no Twelve Data API key configured")
            return None
    
    # For FOREX & METALS: Use Twelve Data as primary, Alpha Vantage as fallback
    if twelve_data_key:
        print(f"🟢 Forex/Metal detected: Using Twelve Data for {symbol} price")
        twelve_client = TwelveDataClient(api_key=twelve_data_key)
        quote = twelve_client.get_quote(symbol)
        
        if quote:
            print(f"✓ Using Twelve Data price for {symbol}")
            return quote['last_price']
        else:
            # Twelve Data failed, try Alpha Vantage fallback
            alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if alpha_vantage_key:
                print(f"⚠️ Twelve Data failed for {symbol}. Trying Alpha Vantage fallback...")
                alpha_client = AlphaVantageClient(api_key=alpha_vantage_key)
                quote = alpha_client.get_quote(symbol)
                if quote:
                    print(f"✓ Using Alpha Vantage price for {symbol}")
                    return quote['last_price']
                else:
                    print(f"❌ Both Twelve Data and Alpha Vantage failed for {symbol}")
            else:
                print(f"❌ Twelve Data failed and no Alpha Vantage API key configured")
            return None
    
    # If no Twelve Data key, try Alpha Vantage directly
    alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if alpha_vantage_key:
        print(f"🟡 No Twelve Data key - Using Alpha Vantage for {symbol} price")
        alpha_client = AlphaVantageClient(api_key=alpha_vantage_key)
        quote = alpha_client.get_quote(symbol)
        if quote:
            print(f"✓ Using Alpha Vantage price for {symbol}")
            return quote['last_price']
    
    return None
