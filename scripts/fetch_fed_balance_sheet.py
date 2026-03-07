#!/usr/bin/env python3
"""
Fetch Federal Reserve Balance Sheet data.

Key metrics:
- WALCL: Total Assets (overall balance sheet size)
- WSHOSHO: Securities Held Outright (QE holdings)

This shows the Fed's money printing activity - crucial for gold/silver.
"""

import pandas as pd
import os
from datetime import datetime, timedelta

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fed_balance_sheet.csv')
API_KEY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fred_api_key.txt')

def fetch_with_fred(api_key, years=10):
    """Fetch using FRED API."""
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        
        start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
        
        print(f"Fetching Fed Balance Sheet (last {years} years)...")
        
        # Total Assets
        total_assets = fred.get_series('WALCL', observation_start=start_date)
        
        # Securities Held Outright
        securities = fred.get_series('WSHOSHO', observation_start=start_date)
        
        df = pd.DataFrame({
            'date': total_assets.index,
            'total_assets': total_assets.values,
            'securities_held': securities.values
        })
        
        # Calculate YoY change (key indicator of QE/QT)
        df['assets_yoy_change'] = df['total_assets'].pct_change(periods=52) * 100  # 52 weeks = 1 year
        
        return df
        
    except Exception as e:
        print(f"FRED API error: {e}")
        return None

def fetch_with_pandas_datareader(years=10):
    """Fetch using pandas_datareader."""
    try:
        import pandas_datareader.data as web
        
        start_date = datetime.now() - timedelta(days=365*years)
        end_date = datetime.now()
        
        print(f"Fetching via pandas_datareader (last {years} years)...")
        
        total_assets = web.DataReader('WALCL', 'fred', start_date, end_date)
        securities = web.DataReader('WSHOSHO', 'fred', start_date, end_date)
        
        df = pd.DataFrame({
            'date': total_assets.index,
            'total_assets': total_assets['WALCL'].values,
            'securities_held': securities['WSHOSHO'].values
        })
        
        df['assets_yoy_change'] = df['total_assets'].pct_change(periods=52) * 100
        
        return df
        
    except Exception as e:
        print(f"pandas_datareader error: {e}")
        return None

def fetch_fed_balance_sheet(years=10):
    """Main function to fetch Fed balance sheet data."""
    print(f"[{datetime.now()}] Fetching Federal Reserve Balance Sheet...\n")
    
    df = None
    
    # Try FRED API if key exists
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, 'r') as f:
            api_key = f.read().strip()
        
        if api_key:
            print("Using FRED API...")
            df = fetch_with_fred(api_key, years)
    
    # Try pandas_datareader
    if df is None:
        print("\nTrying pandas_datareader...")
        df = fetch_with_pandas_datareader(years)
    
    # Manual instructions
    if df is None:
        print("\n" + "="*70)
        print("❌ Automatic fetch failed.")
        print("="*70)
        print("\nTo get Fed balance sheet data:")
        print("1. Get a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html")
        print(f"2. Save it to: {API_KEY_FILE}")
        print("3. Run this script again")
        print("="*70)
        return
    
    # Save
    df.to_csv(DATA_FILE, index=False)
    
    print(f"\n{'='*70}")
    print("✅ Fed Balance Sheet data fetched!")
    print(f"{'='*70}")
    print(f"Records: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"File: {DATA_FILE}")
    
    latest = df.iloc[-1]
    print(f"\nLatest Fed Balance Sheet:")
    print(f"  Date: {latest['date'].date()}")
    print(f"  Total Assets: ${latest['total_assets']:.0f} billion")
    print(f"  Securities Held: ${latest['securities_held']:.0f} billion")
    if pd.notna(latest['assets_yoy_change']):
        print(f"  YoY Change: {latest['assets_yoy_change']:+.1f}%")
        if latest['assets_yoy_change'] > 0:
            print(f"  Status: 🟢 QE (Quantitative Easing) - Bullish for Gold")
        else:
            print(f"  Status: 🔴 QT (Quantitative Tightening) - Bearish for Gold")
    
    # Show peak and current
    peak_idx = df['total_assets'].idxmax()
    peak = df.loc[peak_idx]
    print(f"\nPeak Balance Sheet:")
    print(f"  Date: {peak['date'].date()}")
    print(f"  Total Assets: ${peak['total_assets']:.0f} billion")
    
    change_from_peak = ((latest['total_assets'] - peak['total_assets']) / peak['total_assets']) * 100
    print(f"  Change from Peak: {change_from_peak:+.1f}%")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    fetch_fed_balance_sheet(years=10)
