#!/usr/bin/env python3
"""
Calculate 10 years of historical Real Yields.

Real Yield = Nominal 10Y Treasury Yield - 10Y Breakeven Inflation Rate

This creates a historical dataset of real yields that can be used for analysis.
"""

import pandas as pd
import os
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICES_FILE = os.path.join(BASE_DIR, 'data', 'prices.csv')
BREAKEVEN_FILE = os.path.join(BASE_DIR, 'data', 'breakeven_inflation.csv')
REAL_YIELD_FILE = os.path.join(BASE_DIR, 'data', 'real_yields.csv')

def calculate_real_yields():
    """Calculate historical real yields."""
    print(f"[{datetime.now()}] Calculating 10 years of Real Yields...\n")
    
    # Check if required files exist
    if not os.path.exists(PRICES_FILE):
        print(f"❌ Error: {PRICES_FILE} not found!")
        print("Run 'initialize_historical_data.py' first.")
        return
    
    if not os.path.exists(BREAKEVEN_FILE):
        print(f"❌ Error: {BREAKEVEN_FILE} not found!")
        print("Run 'fetch_breakeven_inflation.py' first.")
        return
    
    # Load data
    print("Loading nominal yields...")
    df_prices = pd.read_csv(PRICES_FILE)
    df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'])
    
    # Filter for US10Y only
    df_nominal = df_prices[df_prices['asset'] == 'US10Y'].copy()
    df_nominal = df_nominal[['timestamp', 'price']].rename(columns={'price': 'nominal_yield'})
    df_nominal['date'] = df_nominal['timestamp'].dt.date
    
    print("Loading breakeven inflation...")
    df_breakeven = pd.read_csv(BREAKEVEN_FILE)
    df_breakeven['date'] = pd.to_datetime(df_breakeven['date']).dt.date
    
    # Merge on date (both datasets have daily data)
    print("Merging and calculating real yields...")
    df_merged = pd.merge(
        df_nominal,
        df_breakeven,
        on='date',
        how='inner'
    )
    
    # Calculate real yield
    df_merged['real_yield'] = df_merged['nominal_yield'] - df_merged['breakeven_inflation']
    
    # Select final columns
    df_real = df_merged[['date', 'nominal_yield', 'breakeven_inflation', 'real_yield']].copy()
    df_real = df_real.sort_values('date')
    
    # Save
    df_real.to_csv(REAL_YIELD_FILE, index=False)
    
    print(f"\n{'='*70}")
    print("✅ Real Yields calculated successfully!")
    print(f"{'='*70}")
    print(f"Records: {len(df_real):,}")
    print(f"Date range: {df_real['date'].min()} to {df_real['date'].max()}")
    print(f"File: {REAL_YIELD_FILE}")
    
    # Show latest
    latest = df_real.iloc[-1]
    print(f"\nLatest Real Yield:")
    print(f"  Date: {latest['date']}")
    print(f"  Nominal 10Y: {latest['nominal_yield']:.2f}%")
    print(f"  Breakeven Inflation: {latest['breakeven_inflation']:.2f}%")
    print(f"  Real Yield: {latest['real_yield']:.2f}%")
    
    # Show statistics
    print(f"\n10-Year Statistics:")
    print(f"  Real Yield Average: {df_real['real_yield'].mean():.2f}%")
    print(f"  Real Yield Min: {df_real['real_yield'].min():.2f}% ({df_real.loc[df_real['real_yield'].idxmin(), 'date']})")
    print(f"  Real Yield Max: {df_real['real_yield'].max():.2f}% ({df_real.loc[df_real['real_yield'].idxmax(), 'date']})")
    
    # Count negative real yield periods
    negative_days = len(df_real[df_real['real_yield'] < 0])
    pct_negative = (negative_days / len(df_real)) * 100
    print(f"  Negative Real Yield Days: {negative_days:,} ({pct_negative:.1f}%)")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    calculate_real_yields()
