#!/usr/bin/env python3
"""
Fetch 10-Year Breakeven Inflation Rate and calculate Real Yields.

Real Yield = Nominal 10Y Yield - Breakeven Inflation Rate

Data from FRED:
- T10YIE: 10-Year Breakeven Inflation Rate (market's inflation expectation)
"""

import pandas as pd
import os
from datetime import datetime, timedelta

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'breakeven_inflation.csv')
API_KEY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fred_api_key.txt')

def fetch_with_fred(api_key, years=10):
    """Fetch using FRED API."""
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        
        start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
        
        print(f"Fetching 10-Year Breakeven Inflation Rate (last {years} years)...")
        breakeven = fred.get_series('T10YIE', observation_start=start_date)
        
        df = pd.DataFrame({
            'date': breakeven.index,
            'breakeven_inflation': breakeven.values
        })
        
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
        breakeven = web.DataReader('T10YIE', 'fred', start_date, end_date)
        
        df = pd.DataFrame({
            'date': breakeven.index,
            'breakeven_inflation': breakeven['T10YIE'].values
        })
        
        return df
        
    except Exception as e:
        print(f"pandas_datareader error: {e}")
        return None

def fetch_breakeven_inflation(years=10):
    """Main function to fetch breakeven inflation data."""
    print(f"[{datetime.now()}] Fetching 10-Year Breakeven Inflation Rate...\n")
    
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
        print("\nTo get breakeven inflation data:")
        print("1. Get a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html")
        print(f"2. Save it to: {API_KEY_FILE}")
        print("3. Run this script again")
        print("="*70)
        return
    
    # Save
    df.to_csv(DATA_FILE, index=False)
    
    print(f"\n{'='*70}")
    print("✅ Breakeven inflation data fetched!")
    print(f"{'='*70}")
    print(f"Records: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"File: {DATA_FILE}")
    
    latest = df.iloc[-1]
    print(f"\nLatest Breakeven Inflation:")
    print(f"  Date: {latest['date'].date()}")
    print(f"  Rate: {latest['breakeven_inflation']:.2f}%")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    fetch_breakeven_inflation(years=10)
