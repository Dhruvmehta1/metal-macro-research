#!/usr/bin/env python3
"""
Fetch Global Central Bank Balance Sheets.

Central Banks:
- Fed (US Federal Reserve) - WALCL
- ECB (European Central Bank) - ECBASSETS
- BOJ (Bank of Japan) - JPNASSETS
- PBOC (People's Bank of China) - CHNASSETS
- BOE (Bank of England) - UKASSETS

Shows global liquidity trends - crucial for gold/silver.
"""

import pandas as pd
import os
from datetime import datetime, timedelta

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'central_bank_balance_sheets.csv')
API_KEY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fred_api_key.txt')

# FRED series codes for central bank assets
CENTRAL_BANKS = {
    'Fed': 'WALCL',           # US Federal Reserve
    'ECB': 'ECBASSETS',       # European Central Bank
    'BOJ': 'JPNASSETS',       # Bank of Japan
    'PBOC': 'CHNASSETS',      # People's Bank of China
    'BOE': 'UKASSETS'         # Bank of England
}

def fetch_with_fred(api_key, years=10):
    """Fetch using FRED API."""
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        
        start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
        
        print(f"Fetching Global Central Bank Balance Sheets (last {years} years)...")
        
        all_data = {}
        
        for bank, series_id in CENTRAL_BANKS.items():
            try:
                print(f"  Fetching {bank} ({series_id})...")
                data = fred.get_series(series_id, observation_start=start_date)
                all_data[bank] = data
            except Exception as e:
                print(f"  Warning: Could not fetch {bank}: {e}")
        
        if not all_data:
            return None
        
        # Combine into single DataFrame
        df = pd.DataFrame(all_data)
        df.index.name = 'date'
        df = df.reset_index()
        
        # Calculate total global liquidity
        numeric_cols = [col for col in df.columns if col != 'date']
        df['Total_Global'] = df[numeric_cols].sum(axis=1, skipna=True)
        
        # Calculate YoY changes
        for bank in numeric_cols:
            df[f'{bank}_yoy'] = df[bank].pct_change(periods=52) * 100  # 52 weeks
        
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
        
        all_data = {}
        
        for bank, series_id in CENTRAL_BANKS.items():
            try:
                print(f"  Fetching {bank} ({series_id})...")
                data = web.DataReader(series_id, 'fred', start_date, end_date)
                all_data[bank] = data[series_id]
            except Exception as e:
                print(f"  Warning: Could not fetch {bank}: {e}")
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data)
        df.index.name = 'date'
        df = df.reset_index()
        
        # Calculate total global liquidity
        numeric_cols = [col for col in df.columns if col != 'date']
        df['Total_Global'] = df[numeric_cols].sum(axis=1, skipna=True)
        
        # Calculate YoY changes
        for bank in numeric_cols:
            df[f'{bank}_yoy'] = df[bank].pct_change(periods=52) * 100
        
        return df
        
    except Exception as e:
        print(f"pandas_datareader error: {e}")
        return None

def fetch_central_banks(years=10):
    """Main function to fetch central bank balance sheets."""
    print(f"[{datetime.now()}] Fetching Global Central Bank Balance Sheets...\n")
    
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
        print("\nTo get central bank data:")
        print("1. Get a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html")
        print(f"2. Save it to: {API_KEY_FILE}")
        print("3. Run this script again")
        print("="*70)
        return
    
    # Save
    df.to_csv(DATA_FILE, index=False)
    
    print(f"\n{'='*70}")
    print("✅ Global Central Bank Balance Sheets fetched!")
    print(f"{'='*70}")
    print(f"Records: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"File: {DATA_FILE}")
    
    # Show latest data
    latest = df.iloc[-1]
    print(f"\nLatest Balance Sheets (as of {latest['date'].date()}):")
    print("-" * 70)
    
    for bank in CENTRAL_BANKS.keys():
        if bank in df.columns and pd.notna(latest[bank]):
            value = latest[bank]
            yoy_col = f'{bank}_yoy'
            yoy = latest[yoy_col] if yoy_col in df.columns and pd.notna(latest[yoy_col]) else None
            
            status = ""
            if yoy is not None:
                if yoy > 0:
                    status = "🟢 QE"
                else:
                    status = "🔴 QT"
            
            print(f"  {bank:5} : ${value:,.0f}B | YoY: {yoy:+.1f}% {status}" if yoy else f"  {bank:5} : ${value:,.0f}B")
    
    if 'Total_Global' in df.columns and pd.notna(latest['Total_Global']):
        print("-" * 70)
        print(f"  TOTAL : ${latest['Total_Global']:,.0f}B (Global Liquidity)")
        
        total_yoy = latest['Total_Global_yoy'] if 'Total_Global_yoy' in df.columns else None
        if total_yoy and pd.notna(total_yoy):
            print(f"  Global YoY: {total_yoy:+.1f}%")
            if total_yoy > 0:
                print(f"  Status: 🟢 Global QE - Bullish for Gold")
            else:
                print(f"  Status: 🔴 Global QT - Bearish for Gold")
    
    # Show peak
    if 'Total_Global' in df.columns:
        peak_idx = df['Total_Global'].idxmax()
        peak = df.loc[peak_idx]
        print(f"\nPeak Global Liquidity:")
        print(f"  Date: {peak['date'].date()}")
        print(f"  Total: ${peak['Total_Global']:,.0f}B")
        
        change_from_peak = ((latest['Total_Global'] - peak['Total_Global']) / peak['Total_Global']) * 100
        print(f"  Change from Peak: {change_from_peak:+.1f}%")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    fetch_central_banks(years=10)
