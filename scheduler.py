import threading
import time
from datetime import datetime, timedelta
from position_monitor import PositionMonitor
from ml_engine import MLTradingEngine
from divergence_analytics import update_all_divergence_stats

class BackgroundScheduler:
    def __init__(self):
        self.ml_engine = MLTradingEngine()
        self.monitor = PositionMonitor(ml_engine=self.ml_engine)
        self.is_running = False
        self.thread = None
        self.last_analytics_run = None
        
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            print("Background scheduler started - Position monitoring every 30 minutes")
    
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("Background scheduler stopped")
    
    def _run(self):
        while self.is_running:
            try:
                self._check_positions()
                self._run_analytics_if_needed()
                time.sleep(1800)
            except Exception as e:
                print(f"Scheduler error: {e}")
                time.sleep(1800)
    
    def _check_positions(self):
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking active positions...")
            results = self.monitor.check_active_positions()
            
            if results:
                for result in results:
                    if result['status'] == 'success':
                        print(f"  {result['symbol']}: {result['recommendation']} - {result['reason']}")
                    else:
                        print(f"  {result['symbol']}: Error - {result.get('message', 'Unknown error')}")
            else:
                print("  No active positions to monitor")
                
        except Exception as e:
            print(f"Error checking positions: {e}")
    
    def _run_analytics_if_needed(self):
        """Run divergence analytics once per day"""
        try:
            now = datetime.now()
            
            # Run if never run before or if it's been more than 24 hours
            if self.last_analytics_run is None or (now - self.last_analytics_run).total_seconds() > 86400:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Running divergence analytics...")
                updated = update_all_divergence_stats()
                self.last_analytics_run = now
                if updated > 0:
                    print(f"  Updated {updated} divergence timing stats")
                else:
                    print(f"  No divergence stats updated (need more data)")
        except Exception as e:
            print(f"Error running divergence analytics: {e}")

_scheduler_instance = None

def get_scheduler():
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = BackgroundScheduler()
    return _scheduler_instance
