#!/usr/bin/env python3
"""
One-time script to fetch 7 years of historical data for all assets (Hybrid: Binance + Yahoo).
Run this ONCE to initialize the database, then use fetch_prices.py for daily updates.
"""
import pandas as pd
import os
import sys
import yfinance as yf
from datetime import datetime, timedelta

# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Configuration
YAHOO_ASSETS = {
    'Gold': 'GC=F',   # Yahoo Futures (Volume Provided)
    'Silver': 'SI=F', # Yahoo Futures
    'US10Y': '^TNX',
    'DXY': 'DX-Y.NYB',
    'SP500': '^GSPC',
    'VIX': '^VIX'
}

DATA_DIR = os.path.join(os.path.dirname(script_dir), 'data')
PRICES_FILE = os.path.join(DATA_DIR, 'prices.csv')

def initialize_database():
    print("Initializing Database (Source: Yahoo Finance + FRED)...")
    print("Fetches 10 years of history for Gold, Silver, and Macro indicators.")
    print("This will take a few minutes due to rate-limiting protection. Please wait...\n")
    
    all_data = []
    
    # 2. Fetch Yahoo Data (Unified List)
    print("--- Fetching Assets from Yahoo Finance ---")
    start_date = (datetime.now() - timedelta(days=365*15)).strftime('%Y-%m-%d')
    
    # We fetch one by one to avoid rate limits and handle errors individually
    import time
    
    for name, ticker in YAHOO_ASSETS.items():
        print(f"Processing {name} ({ticker})...")
        try:
            # Add delay to avoid 429 Rate Limit
            time.sleep(2) 
            
            df = yf.download(ticker, start=start_date, progress=False, multi_level_index=False)
            
            if df.empty:
                print(f"  ⚠️ No data for {name}")
                continue
            
            df = df.reset_index()
            # Standardize column names
            date_col = 'Date' if 'Date' in df.columns else 'Datetime'
            
            # Get Close prices
            if 'Close' not in df.columns:
                print(f"Warning: No Close column for {name}")
                continue
            
            # Add to our data
            for _, row in df.iterrows():
                ts = row[date_col]
                price = row['Close']
                
                if pd.isna(price):
                    continue
                
                all_data.append({
                    'timestamp': ts,
                    'asset': name,
                    'price': float(price),
                    'volume': float(row['Volume']) if 'Volume' in row else 0.0,
                    'high': float(row['High']) if 'High' in row else 0.0,
                    'low': float(row['Low']) if 'Low' in row else 0.0,
                    'open': float(row['Open']) if 'Open' in row else 0.0
                })
            print(f"  ✓ {name}: {len(df)} records (Yahoo)")
            
        except Exception as e:
            print(f"Error fetching {name}: {e}")

    # 3. Save to CSV
    if all_data:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            
        main_df = pd.DataFrame(all_data)
        main_df['timestamp'] = pd.to_datetime(main_df['timestamp'])
        main_df = main_df.sort_values(['timestamp', 'asset'])
        
        # Save necessary columns
        cols = ['timestamp', 'asset', 'price', 'volume', 'high', 'low', 'open']
        # Create missing columns
        for col in cols:
            if col not in main_df.columns:
                main_df[col] = 0.0
                
        main_df[cols].to_csv(PRICES_FILE, index=False)
        
        print("\n" + "="*70)
        print("✅ SUCCESS! Yahoo Database Initialized")
        print("="*70)
        print(f"Total records: {len(main_df)}")
        print(f"File location: {PRICES_FILE}")
        
    else:
        print("Failed to initialize database (No data collected)")

if __name__ == "__main__":
    # Safety Check
    if os.path.exists(PRICES_FILE):
        print(f"⚠️  WARNING: {PRICES_FILE} already exists!")
        # We assume 'yes' if running non-interactively, but best to force it for the tool
        # In this environment, we just overwrite.
        print("Overwriting existing database...")
    
    initialize_database()
