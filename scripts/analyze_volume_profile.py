import pandas as pd
import yfinance as yf
import numpy as np
import sys
import os

# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def analyze_volume_profile(lookback_days=10, bins=50):
    print(f"Analyzing Volume Profile (Last {lookback_days} Days)...")
    print("Sources: Gold (Yahoo Futures), Silver (Yahoo Futures)")
    
    results = {}
    
    # 1. Gold (Yahoo Futures)
    process_asset(results, 'Gold', 'GC=F', source='yahoo', lookback=lookback_days, bins=bins)
    
    # 2. Silver (Yahoo Futures)
    process_asset(results, 'Silver', 'SI=F', source='yahoo', lookback=lookback_days, bins=bins)
            
    return results

def process_asset(results, name, ticker, source, lookback, bins):
    print(f"\nProcessing {name} ({ticker})...")
    close_prices = []
    volumes = []
    
    try:
        if source == 'yahoo':
            # yfinance 1h data
            # buffer days to ensure we get enough hourly bars
            df = yf.download(ticker, period=f"{lookback}d", interval="1h", progress=False)
            if not df.empty:
                # Handle MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    if ticker in df.columns.levels[0]: # If grouped by ticker
                         close_prices = df['Close'][ticker]
                         volumes = df['Volume'][ticker]
                    else: # If flat but multi-level (rare with 1 ticker but possible)
                         close_prices = df['Close']
                         volumes = df['Volume']
                else:
                    close_prices = df['Close']
                    volumes = df['Volume']
                    
                # Clean NaNs
                close_prices = close_prices.dropna()
                volumes = volumes.dropna()
                
                # Align lengths
                common_idx = close_prices.index.intersection(volumes.index)
                close_prices = close_prices.loc[common_idx]
                volumes = volumes.loc[common_idx]

        if len(close_prices) == 0:
            print(f"  Warning: No data for {name}")
            return

        # Fix FutureWarning: calling float on a single element Series
        # Use .item() to get scalar if it's a Series, or just float() if it's already a scalar
        last_val = close_prices.iloc[-1]
        if hasattr(last_val, 'item'):
            current_price = float(last_val.item())
        else:
            current_price = float(last_val)
        
        # --- Volume Profile Logic (Common) ---
        hist, bin_edges = np.histogram(close_prices, bins=bins, weights=volumes)
        
        # POC
        max_vol_idx = hist.argmax()
        poc_price = (bin_edges[max_vol_idx] + bin_edges[max_vol_idx+1]) / 2
        
        # Value Area (70%)
        total_volume = hist.sum()
        value_area_vol = total_volume * 0.70
        
        sorted_indices = np.argsort(hist)[::-1]
        accumulated_vol = 0
        va_indices = []
        
        for idx in sorted_indices:
            accumulated_vol += hist[idx]
            va_indices.append(idx)
            if accumulated_vol >= value_area_vol:
                break
        
        va_min_idx = min(va_indices)
        va_max_idx = max(va_indices)
        
        val_price = bin_edges[va_min_idx]
        vah_price = bin_edges[va_max_idx+1]
        
        # Bias
        if current_price > vah_price:
            bias = "Bullish Breakout (Above Value Area)"
        elif current_price < val_price:
            bias = "Bearish Breakdown (Below Value Area)"
        else:
            bias = "Neutral / Ranging (Inside Value Area)"

        results[name] = {
            'current_price': current_price,
            'poc': poc_price,
            'vah': vah_price,
            'val': val_price,
            'bias': bias
        }
        
        print(f"  Price: ${current_price:.2f}")
        print(f"  POC:   ${poc_price:.2f}")
        print(f"  VAH:   ${vah_price:.2f}")
        print(f"  VAL:   ${val_price:.2f}")
        print(f"  Bias:  {bias}")

    except Exception as e:
        print(f"Error analyzing {name}: {e}")

if __name__ == "__main__":
    analyze_volume_profile()
