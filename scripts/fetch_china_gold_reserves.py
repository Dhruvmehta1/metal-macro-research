#!/usr/bin/env python3
"""
Fetch China (PBOC) Gold Reserves Data

Sources (in order of preference):
1. World Gold Council - Official gold reserves data
2. IMF International Financial Statistics
3. Manual CSV from MacroMicro or Trading Economics

China is the world's largest gold consumer and PBOC has been aggressively
accumulating gold reserves - crucial for gold/silver price analysis.
"""

import pandas as pd
import os
from datetime import datetime
import requests

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'china_gold_reserves.csv')

def fetch_from_world_gold_council():
    """
    Try to fetch from World Gold Council's public data.
    They provide XLSX downloads of official gold reserves.
    """
    try:
        # World Gold Council official gold reserves URL (public data)
        # This URL provides latest official gold reserves in XLSX format
        url = "https://www.gold.org/goldhub/data/monthly-central-bank-statistics"
        
        print("Attempting to fetch from World Gold Council...")
        print("Note: This may require manual download if API access is restricted.")
        print(f"Visit: {url}")
        
        # For now, return None - user may need to manually download
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def fetch_from_trading_economics():
    """
    Try to scrape from Trading Economics public page.
    """
    try:
        url = "https://tradingeconomics.com/china/gold-reserves"
        
        print("Attempting to fetch from Trading Economics...")
        
        # Try to read tables from the page
        tables = pd.read_html(url)
        
        if tables:
            # Find the table with historical data
            for table in tables:
                if 'Date' in table.columns or 'date' in str(table.columns).lower():
                    print(f"Found data table with {len(table)} records")
                    return table
        
        return None
        
    except Exception as e:
        print(f"Trading Economics error: {e}")
        return None

def create_manual_instructions():
    """Provide manual download instructions."""
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD REQUIRED FOR PBOC GOLD RESERVES")
    print("="*70)
    print("\nChina's gold reserves data is not available via free API.")
    print("Please manually download from one of these sources:\n")
    
    print("Option 1: World Gold Council (Most Reliable)")
    print("  1. Visit: https://www.gold.org/goldhub/data/monthly-central-bank-statistics")
    print("  2. Download 'Latest World Official Gold Reserves' (XLSX)")
    print("  3. Filter for China")
    print(f"  4. Save as CSV to: {DATA_FILE}\n")
    
    print("Option 2: MacroMicro (Easy CSV Download)")
    print("  1. Visit: https://en.macromicro.me/charts/1357/china-central-bank-gold-reserves")
    print("  2. Click 'Download' → CSV")
    print(f"  3. Save to: {DATA_FILE}\n")
    
    print("Option 3: Trading Economics")
    print("  1. Visit: https://tradingeconomics.com/china/gold-reserves")
    print("  2. Click 'Download' (may require free account)")
    print(f"  3. Save to: {DATA_FILE}\n")
    
    print("Expected CSV format:")
    print("  date,gold_reserves_tonnes")
    print("  2020-01-01,1948.3")
    print("  2020-02-01,1948.3")
    print("  ...")
    
    print("\nKey Facts:")
    print("  - China is the world's largest gold consumer")
    print("  - PBOC has been accumulating gold since 2018")
    print("  - Latest reserves: ~2,300 tonnes (as of 2025)")
    print("  - This data is CRITICAL for gold price analysis")
    
    print("="*70 + "\n")

def check_existing_data():
    """Check if manual data already exists."""
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            print(f"\n✅ Existing China gold reserves data found!")
            print(f"Records: {len(df)}")
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            
            if len(df.columns) >= 2:
                latest = df.iloc[-1]
                print(f"\nLatest data:")
                for col in df.columns:
                    print(f"  {col}: {latest[col]}")
            
            return True
        except Exception as e:
            print(f"Error reading existing file: {e}")
            return False
    return False

def fetch_china_gold_reserves():
    """Main function."""
    print(f"[{datetime.now()}] Fetching China (PBOC) Gold Reserves...\n")
    
    # Check if data already exists
    if check_existing_data():
        print("\nData already exists. To update, manually download new data.")
        return
    
    # Try automated methods
    df = fetch_from_trading_economics()
    
    if df is not None:
        # Save
        df.to_csv(DATA_FILE, index=False)
        print(f"\n✅ China gold reserves data saved to: {DATA_FILE}")
        return
    
    # If automated methods fail, provide manual instructions
    create_manual_instructions()

if __name__ == "__main__":
    fetch_china_gold_reserves()
