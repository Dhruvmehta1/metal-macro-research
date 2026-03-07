#!/usr/bin/env python3
"""
Fetch US GDP data for the past 10 years.
GDP is released quarterly, so 10 years = ~40 data points.

Data sources (in order of preference):
1. FRED API (requires free API key from https://fred.stlouisfed.org/docs/api/api_key.html)
2. pandas_datareader (no API key needed, but less reliable)
3. Manual CSV download fallback

GDP Series:
- GDP: Nominal GDP (current dollars)
- GDPC1: Real GDP (inflation-adjusted, 2017 dollars)
"""

import pandas as pd
import os
from datetime import datetime, timedelta

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'gdp.csv')
API_KEY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fred_api_key.txt')

def fetch_gdp_with_fred(api_key):
    """Fetch GDP using FRED API (most reliable)."""
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        
        # Fetch both nominal and real GDP
        start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        
        print("Fetching Nominal GDP (current dollars)...")
        gdp_nominal = fred.get_series('GDP', observation_start=start_date)
        
        print("Fetching Real GDP (2017 dollars)...")
        gdp_real = fred.get_series('GDPC1', observation_start=start_date)
        
        # Combine into DataFrame
        df = pd.DataFrame({
            'date': gdp_nominal.index,
            'gdp_nominal': gdp_nominal.values,
            'gdp_real': gdp_real.values
        })
        
        # Calculate YoY growth rate
        df['gdp_growth_yoy'] = df['gdp_real'].pct_change(periods=4) * 100  # 4 quarters = 1 year
        
        return df
        
    except Exception as e:
        print(f"FRED API error: {e}")
        return None

def fetch_gdp_with_pandas_datareader():
    """Fetch GDP using pandas_datareader (no API key needed)."""
    try:
        import pandas_datareader.data as web
        
        start_date = datetime.now() - timedelta(days=365*10)
        end_date = datetime.now()
        
        print("Fetching GDP via pandas_datareader...")
        gdp_nominal = web.DataReader('GDP', 'fred', start_date, end_date)
        gdp_real = web.DataReader('GDPC1', 'fred', start_date, end_date)
        
        df = pd.DataFrame({
            'date': gdp_nominal.index,
            'gdp_nominal': gdp_nominal['GDP'].values,
            'gdp_real': gdp_real['GDPC1'].values
        })
        
        df['gdp_growth_yoy'] = df['gdp_real'].pct_change(periods=4) * 100
        
        return df
        
    except Exception as e:
        print(f"pandas_datareader error: {e}")
        return None

def fetch_gdp():
    """Main function to fetch GDP data."""
    print(f"[{datetime.now()}] Fetching US GDP data (10 years, quarterly)...\n")
    
    df = None
    
    # Method 1: Try FRED API if key exists
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, 'r') as f:
            api_key = f.read().strip()
        
        if api_key:
            print("Using FRED API (most reliable)...")
            df = fetch_gdp_with_fred(api_key)
    
    # Method 2: Try pandas_datareader
    if df is None:
        print("\nTrying pandas_datareader (no API key needed)...")
        df = fetch_gdp_with_pandas_datareader()
    
    # Method 3: Manual instructions
    if df is None:
        print("\n" + "="*70)
        print("❌ Automatic fetch failed. Manual download required:")
        print("="*70)
        print("\nOption 1 (Recommended): Get a free FRED API key")
        print("1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Sign up (free)")
        print("3. Copy your API key")
        print(f"4. Save it to: {API_KEY_FILE}")
        print("5. Run this script again")
        print("\nOption 2: Manual CSV download")
        print("1. Visit: https://fred.stlouisfed.org/series/GDP")
        print("2. Click 'Download' → 'CSV'")
        print("3. Save to metals_macro_system/data/gdp.csv")
        print("="*70)
        return
    
    # Save to CSV
    df.to_csv(DATA_FILE, index=False)
    
    print(f"\n{'='*70}")
    print("✅ GDP data fetched successfully!")
    print(f"{'='*70}")
    print(f"Records: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"File: {DATA_FILE}")
    
    print(f"\nLatest GDP data:")
    latest = df.iloc[-1]
    print(f"  Date: {latest['date'].date()}")
    print(f"  Nominal GDP: ${latest['gdp_nominal']:.1f} billion")
    print(f"  Real GDP: ${latest['gdp_real']:.1f} billion (2017 dollars)")
    if pd.notna(latest['gdp_growth_yoy']):
        print(f"  YoY Growth: {latest['gdp_growth_yoy']:+.2f}%")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    fetch_gdp()
