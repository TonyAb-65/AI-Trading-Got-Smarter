import requests
import os
from datetime import datetime, timedelta

_alert_cooldowns = {}
ALERT_COOLDOWN_MINUTES = 5

def send_telegram_alert(symbol, position_type, entry_price, current_price, pnl_percentage, 
                       severity, alert_message, recommendation, reason):
    """
    Send trading alert to Telegram.
    Only called for HIGH severity alerts to avoid spam.
    Implements 5-minute cooldown per symbol to prevent continuous alerts.
    """
    global _alert_cooldowns
    
    token = os.getenv('SMARTTRADING_BOT_TOKEN')
    chat_id = os.getenv('SMARTTRADING_CHAT_ID')
    
    if not token or not chat_id:
        print("⚠️ Telegram not configured - skipping notification")
        return False
    
    now = datetime.now()
    cooldown_key = f"{symbol}_{position_type}"
    
    if cooldown_key in _alert_cooldowns:
        last_alert_time, last_message = _alert_cooldowns[cooldown_key]
        time_since_last = (now - last_alert_time).total_seconds() / 60
        
        if time_since_last < ALERT_COOLDOWN_MINUTES:
            if alert_message == last_message:
                print(f"⏸️ Skipping duplicate alert for {symbol} (cooldown: {ALERT_COOLDOWN_MINUTES - time_since_last:.1f} min remaining)")
                return False
            else:
                print(f"📨 New alert type for {symbol} - sending despite cooldown")
    
    severity_emoji = "🚨" if severity == "HIGH" else "⚠️"
    pnl_emoji = "📉" if pnl_percentage < 0 else "📈"
    
    message = f"""
{severity_emoji} <b>TRADING ALERT - {symbol}</b>

<b>Position:</b> {position_type} @ ${entry_price:,.2f}
<b>Current:</b> ${current_price:,.2f} ({pnl_percentage:+.2f}%) {pnl_emoji}

{alert_message}

<b>M1 Recommendation:</b> {recommendation}
<b>Reason:</b> {reason}

📍 Check Position Tracker immediately
🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message.strip(),
        'parse_mode': 'HTML'
    }
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            _alert_cooldowns[cooldown_key] = (now, alert_message)
            print(f"✅ Telegram alert sent for {symbol} (next alert in {ALERT_COOLDOWN_MINUTES} min)")
            return True
        else:
            print(f"❌ Telegram API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram notification failed: {e}")
        return False

def clear_alert_cooldown(symbol, position_type):
    """Clear cooldown for a specific position (e.g., when position is closed)"""
    global _alert_cooldowns
    cooldown_key = f"{symbol}_{position_type}"
    if cooldown_key in _alert_cooldowns:
        del _alert_cooldowns[cooldown_key]
        print(f"🔄 Alert cooldown cleared for {symbol}")

def test_telegram_connection():
    """Test if Telegram bot is configured correctly"""
    token = os.getenv('SMARTTRADING_BOT_TOKEN')
    chat_id = os.getenv('SMARTTRADING_CHAT_ID')
    
    if not token or not chat_id:
        return {
            'success': False,
            'message': 'Telegram credentials not configured'
        }
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': '✅ Smart Trading Alert System Connected!\n\nYou will receive HIGH priority alerts here.',
        'parse_mode': 'HTML'
    }
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            return {
                'success': True,
                'message': 'Telegram connected successfully! Check your phone.'
            }
        else:
            return {
                'success': False,
                'message': f'Telegram API error: {response.status_code}'
            }
    except Exception as e:
        return {
            'success': False,
            'message': f'Connection failed: {str(e)}'
        }
