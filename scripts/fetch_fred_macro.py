#!/usr/bin/env python3
"""
Fetch Macro Economic Data directly from St. Louis Fed (FRED) API.
Datasets:
- Growth:     GDP (Quarterly)
- Inflation:  T10YIE (10-Year Breakeven Inflation Rate)
- Real Yield: DFII10 (Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis, Inflation-Indexed)
- Nominal:    DGS10 (10-Year Treasury Constant Maturity Rate)

Output:
- data/macro_data_fred.csv
"""

from fredapi import Fred
import pandas as pd
import os
import sys
from datetime import datetime

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'macro_data_fred.csv')

# API Key Config
# Check environment variable first
api_key = os.environ.get('FRED_API_KEY')

def fetch_fred_data():
    if not api_key:
        print("❌ Error: FRED_API_KEY environment variable not set.")
        print("Please export your key: export FRED_API_KEY='your_key_here'")
        return

    print(f"[{datetime.now()}] Fetching Macro Data from FRED API...")
    try:
        fred = Fred(api_key=api_key)
    except Exception as e:
        print(f"❌ Error initializing FRED client: {e}")
        return

    # 1. Fetch GDP (Quarterly)
    print("  Fetching Growth (GDP)...")
    try:
        gdp = fred.get_series('GDP')
        gdp.name = 'GDP'
        print(f"    ✓ GDP: {len(gdp)} records (Last: {gdp.index[-1].date()})")
    except Exception as e:
        print(f"    ❌ Error fetching GDP: {e}")
        gdp = pd.Series(name='GDP')

    # 2. Fetch Breakeven Inflation (Daily)
    print("  Fetching Inflation (T10YIE)...")
    try:
        inf = fred.get_series('T10YIE')
        inf.name = 'Breakeven_10Y'
        print(f"    ✓ Inflation: {len(inf)} records (Last: {inf.index[-1].date()})")
    except Exception as e:
        print(f"    ❌ Error fetching Inflation: {e}")
        inf = pd.Series(name='Breakeven_10Y')

    # 3. Fetch Real Yields (Daily)
    print("  Fetching Real Yields (DFII10)...")
    try:
        ry = fred.get_series('DFII10')
        ry.name = 'Real_Yield_10Y'
        print(f"    ✓ Real Yields: {len(ry)} records (Last: {ry.index[-1].date()})")
    except Exception as e:
        print(f"    ❌ Error fetching Real Yields: {e}")
        ry = pd.Series(name='Real_Yield_10Y')

    # 4. Fetch Nominal 10Y (Daily)
    print("  Fetching Nominal 10Y (DGS10)...")
    try:
        nom = fred.get_series('DGS10')
        nom.name = 'Nominal_10Y'
        print(f"    ✓ Nominal 10Y: {len(nom)} records (Last: {nom.index[-1].date()})")
    except Exception as e:
        print(f"    ❌ Error fetching Nominal 10Y: {e}")
        nom = pd.Series(name='Nominal_10Y')

    # --- Merge & Clean ---
    print("\nMerging datasets...")
    
    # Convert Series to DataFrame for merging
    df_list = [inf, ry, nom, gdp]
    # Filter out empty series
    df_list = [s for s in df_list if not s.empty]
    
    if not df_list:
        print("❌ No data fetched. Exiting.")
        return

    merged = pd.concat(df_list, axis=1)
    merged.index.name = 'date'
    
    # Filter history
    merged = merged[merged.index >= '2000-01-01']
    
    # Forward Fill GDP (Macro regime component)
    if 'GDP' in merged.columns:
        merged['GDP'] = merged['GDP'].ffill()
    
    # Forward Fill others
    merged = merged.ffill()
    
    # Drop rows where we don't have core regime data
    # We prioritize valid Inflation and Yield data. GDP is slow moving.
    required_cols = [c for c in ['Breakeven_10Y', 'Real_Yield_10Y'] if c in merged.columns]
    if required_cols:
        merged = merged.dropna(subset=required_cols)
    
    # Save
    merged.to_csv(OUTPUT_FILE)
    print(f"\n✅ Stats:")
    print(merged.describe().loc[['count', 'mean', 'min', 'max']])
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(merged.tail())

if __name__ == "__main__":
    fetch_fred_data()
