#!/usr/bin/env python3
"""
Fetch today's opening prices for all assets.
Useful for getting the current day's open even if the day hasn't closed yet.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

ASSETS = {
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'US10Y': '^TNX',
    'DXY': 'DX-Y.NYB',
    'SP500': '^GSPC',
    'VIX': '^VIX'
}

def get_todays_open():
    """Fetch today's opening prices."""
    print(f"[{datetime.now()}] Fetching today's opening prices...\n")
    
    # Fetch 1 day of data with 1-minute intervals to get today's open
    # Fetch one by one to avoid yfinance bulk download tuple/MultiIndex ambiguity
    try:
        print("Today's Opening Prices:")
        print("=" * 60)
        
        results = {}
        
        for name, ticker in ASSETS.items():
            try:
                # Download just this ticker
                # Use 2d period to ensure we get 'today' or last close
                ticker_data = yf.download(ticker, period="5d", interval="1d", progress=False)
                
                # Check for tuple return (Data, Meta)
                if isinstance(ticker_data, tuple):
                    ticker_data = ticker_data[0]
                
                if ticker_data.empty:
                    print(f"{name:8} : No data")
                    continue
                    
                # Fix columns if MultiIndex (happens even on single tickers sometimes)
                if isinstance(ticker_data.columns, pd.MultiIndex):
                     ticker_data.columns = ticker_data.columns.droplevel(0) if 'Close' not in ticker_data.columns else ticker_data.columns
                
                if 'Open' not in ticker_data.columns:
                     print(f"{name:8} : No 'Open' column")
                     continue

                # Get last row
                today = ticker_data.iloc[-1]
                
                open_price = float(today['Open'])
                close_price = float(today['Close'])
                
                results[name] = open_price
                
                if pd.isna(close_price):
                    status = "(Market Open)"
                    change_str = "N/A"
                else:
                    change = ((close_price - open_price) / open_price) * 100
                    status = "(Market Closed)"
                    change_str = f"{change:+.2f}%"
                
                print(f"{name:8} : Open: ${open_price:,.2f} {status} | Change: {change_str}")
                    
            except Exception as e:
                print(f"{name:8} : Error - {e}")
        
        print("=" * 60)
        return results
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return {}

if __name__ == "__main__":
    get_todays_open()
