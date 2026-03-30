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

def detect_regime(lag_days=30):
    """
    Classifies the daily macro environment into 4 Regimes based on
    Growth (GDP) and Inflation (Breakeven).

    Parameters:
    - lag_days: Number of days to lag macro data to simulate real-time availability.
                GDP is typically released with a 30-day lag after quarter end.
                This prevents using data that wasn't available at prediction time.
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

    # Pivot to get Asset Prices
    df_pivot = df.pivot(index='timestamp', columns='asset', values='price')

    # Load Macro Data
    macro = pd.read_csv(MACRO_FILE, index_col='date', parse_dates=True)

    # IMPORTANT: Apply publication lag to macro data
    # This shifts macro data forward in time to simulate when it was actually available
    if len(macro) > 0:
        macro_lagged = macro.copy()
        macro_lagged.index = macro_lagged.index + pd.Timedelta(days=lag_days)
    else:
        macro_lagged = macro

    # Merge with LEFT JOIN to keep all price dates
    merged = df_pivot.join(macro_lagged, how='left').sort_index()

    # DO NOT forward-fill GDP/breakeven - they should only be available on release dates
    # We only ffill within the same release period (already handled by the lag)

    # 2. Calculate Trends
    # Growth Proxy: GDP (Quarterly)
    # Using 60-day ROC to smooth quarterly data
    merged['growth_momentum'] = merged['GDP'].pct_change(60, fill_method=None)

    # Inflation Proxy: 10Y Breakeven
    # Using 20-day absolute change (1 month)
    merged['inflation_momentum'] = merged['Breakeven_10Y'].diff(20)

    # 3. Define Regimes
    def get_regime(row):
        g = row['growth_momentum']
        i = row['inflation_momentum']

        if pd.isna(g) or pd.isna(i):
            return "Unclassified"

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

    # Keep all rows - regime will be "Unclassified" where data isn't available yet
    # This is more honest than dropping rows
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
