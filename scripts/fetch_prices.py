#!/usr/bin/env python3
"""
Daily script to fetch the LATEST prices and append them to data/prices.csv.
Uses a Hybrid Approach:
- Gold/Silver: Fetched from Binance (PAXG/XAG) for 24/7 accuracy.
- Macro (Yields/DXY): Fetched from Yahoo Finance.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import sys

# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Configuration
YAHOO_ASSETS = {
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'US10Y': '^TNX',
    'DXY': 'DX-Y.NYB',
    'SP500': '^GSPC',
    'VIX': '^VIX'
}

DATA_DIR = os.path.join(os.path.dirname(script_dir), 'data')
PRICES_FILE = os.path.join(DATA_DIR, 'prices.csv')

def fetch_prices():
    print(f"[{datetime.now()}] Fetching latest Daily prices (Yahoo Finance)...")
    
    new_data = []

    # Fetch Yahoo Data (Unified) - Latest 5 days
    print("\n--- Fetching Latest Data from Yahoo Finance ---")
    tickers = list(YAHOO_ASSETS.values())
    
    # Optional: fetch one by one if group download is problematic, but group is usually efficient if few tickers.
    # However, to be safe and consistent with initialize, let's allow group download but if it fails we might need fallback.
    # For now, let's try standard bulk download but with a small sleep before starting if running in a loop context (not here).
    
    try:
        data = yf.download(tickers, period="5d", interval="1d", group_by='ticker', progress=False)
        
        for name, ticker in YAHOO_ASSETS.items():
            # Check if ticker is in columns (MultiIndex)
            if ticker in data.columns.levels[0]:
                df = data[ticker].reset_index()
                
                # Standardize columns
                date_col = 'Date' if 'Date' in df.columns else 'Datetime'
                
                if 'Close' not in df.columns:
                    continue
                    
                for _, row in df.iterrows():
                    ts = row[date_col]
                    price = row['Close']
                    if pd.notna(price):
                        new_data.append({
                            'timestamp': ts,
                            'asset': name,
                            'price': float(price),
                            'volume': float(row['Volume']) if 'Volume' in row else 0.0,
                            'high': float(row['High']) if 'High' in row else 0.0,
                            'low': float(row['Low']) if 'Low' in row else 0.0,
                            'open': float(row['Open']) if 'Open' in row else 0.0
                        })
                # Print latest for confirmation
                try:
                    latest_price = df['Close'].dropna().iloc[-1]
                    print(f"  ✓ {name}: {latest_price:.4f} (Yahoo)")
                except IndexError:
                    print(f"  ⚠️ No recent data for {name}")

    except Exception as e:
        print(f"Error fetching Yahoo data: {e}")

    # 3. Append to Database (Upsert)
    if new_data:
        # Load existing
        if os.path.exists(PRICES_FILE):
            existing_df = pd.read_csv(PRICES_FILE)
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
        else:
            existing_df = pd.DataFrame()
            
        new_df = pd.DataFrame(new_data)
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
        
        # Combine
        combined_df = pd.concat([existing_df, new_df])
        
        # Deduplicate (Keep LAST/Newest version of a row)
        combined_df = combined_df.sort_values(['timestamp', 'asset'])
        combined_df = combined_df.drop_duplicates(subset=['timestamp', 'asset'], keep='last')
        
        # Save
        cols = ['timestamp', 'asset', 'price', 'volume', 'high', 'low', 'open']
        for col in cols:
            if col not in combined_df.columns:
                combined_df[col] = 0.0
                
        combined_df[cols].to_csv(PRICES_FILE, index=False)
        print(f"\n✅ Updated {PRICES_FILE}")
        print(f"Total Records: {len(combined_df)}")
        print(combined_df.groupby('asset').tail(1))
        
    else:
        print("No new data fetched.")

if __name__ == "__main__":
    fetch_prices()
