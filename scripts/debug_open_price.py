import yfinance as yf
import pandas as pd

def check_monday_open():
    print("Fetching SI=F data to verify Monday Open price...")
    
    # Fetch data including today
    ticker = yf.Ticker("SI=F")
    # Get recent daily data
    hist = ticker.history(period="5d", interval="1d")
    
    print("\nRecent Daily Candles:")
    print(hist[['Open', 'High', 'Low', 'Close']].tail())
    
    if not hist.empty:
        last_date = hist.index[-1]
        last_open = hist.iloc[-1]['Open']
        print(f"\nLatest Date: {last_date}")
        print(f"Latest Open: {last_open}")
        
        # User claimed Sunday open was ~88.54
        print(f"User Sunday target: ~88.54")
        
if __name__ == "__main__":
    check_monday_open()
