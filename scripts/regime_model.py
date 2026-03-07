
import pandas as pd
import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICES_FILE = os.path.join(BASE_DIR, 'data', 'prices.csv')
MACRO_FILE = os.path.join(BASE_DIR, 'data', 'macro_data_fred.csv')

def calculate_slope(series, window=20):
    """Calculates the slope of a rolling linear regression over 'window' days."""
    return series.pct_change(window, fill_method=None) * 100

def detect_regime():
    """
    Classifies the daily macro environment into 4 Regimes based on
    Growth (GDP) and Inflation (Breakeven).
    """
    
    # 1. Load Data
    if not os.path.exists(PRICES_FILE):
        print("Prices file not found.")
        return None
    if not os.path.exists(MACRO_FILE):
        print("Macro data file not found (Run fetch_fred_macro.py first).")
        return None
        
    df = pd.read_csv(PRICES_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Pivot to get Asset Prices (if needed, mainly for reference)
    df_pivot = df.pivot(index='timestamp', columns='asset', values='price')
    
    # Load Macro Data
    macro = pd.read_csv(MACRO_FILE, index_col='date', parse_dates=True)
    
    # Merge
    # Use LEFT JOIN to keep latest price date
    merged = df_pivot.join(macro, how='left').sort_index()
    
    # Forward Fill missing Macro data
    merged = merged.ffill()
    
    # 2. Calculate Trends
    # Growth Proxy: GDP (Quarterly, filled daily)
    # Using 60-day ROC (approx 1 quarter) to smooth the step-function of quarterly data
    merged['growth_momentum'] = merged['GDP'].pct_change(60, fill_method=None)
    
    # Inflation Proxy: 10Y Breakeven
    # Using 20-day absolute change (1 month)
    merged['inflation_momentum'] = merged['Breakeven_10Y'].diff(20)
    
    # 3. Define Regimes
    def get_regime(row):
        g = row['growth_momentum']
        i = row['inflation_momentum']
        
        # Tweak: >= 0 is better because GDP is slow. 
        # If GDP is flat but high, we don't want to call it "Stagflation" immediately.
        # Ideally we check levels, but for now >= 0 momentum on quarterly forward-fill is safer.
        if g >= 0 and i > 0:
            return "REFLATION (Boom)"
        elif g >= 0 and i <= 0:
            return "GOLDILOCKS"
        elif g <= 0 and i > 0:
            return "STAGFLATION"
        elif g <= 0 and i <= 0:
            return "DEFLATION (Bust)"
        else:
            return "Unclassified"

    merged['regime'] = merged.apply(get_regime, axis=1)
    
    # Drop rows with NaN regimes
    merged = merged.dropna(subset=['growth_momentum', 'inflation_momentum'])
    
    # Keep only relevant columns
    result = merged[['GDP', 'Breakeven_10Y', 'growth_momentum', 'inflation_momentum', 'regime']]
    
    return result

if __name__ == "__main__":
    regimes = detect_regime()
    if regimes is not None:
        print("Regime Detection Complete.")
        print("\nLast 10 Days:")
        print(regimes[['regime', 'growth_momentum', 'inflation_momentum']].tail(10))
        
        print("\nRegime Distribution (2016-Present):")
        print(regimes['regime'].value_counts(normalize=True))
