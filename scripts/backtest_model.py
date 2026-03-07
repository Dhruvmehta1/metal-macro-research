#!/usr/bin/env python3
"""
Backtest Logic for Metals Macro System.
Walk-forward validation over the last 1 year.
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from predict_prices import prepare_data, calculate_bayesian_probability, calculate_expected_range

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'predictions_log.csv')

def run_backtest(start_warmup_days=365):
    print(f"Generating Historical Predictions Log...")
    
    # 1. Load ALL Data
    df_all, _ = prepare_data()
    
    # Ensure sorted
    df_all = df_all.sort_values('date')
    
    results = []
    
    # Start after one year of data (Warmup)
    start_idx = start_warmup_days
    if start_idx >= len(df_all):
        print("Not enough data for requested warmup period.")
        return

    print(f"Generating log from {df_all.iloc[start_idx]['date']} to {df_all.iloc[-1]['date']}")
    print(f"Target File: {OUTPUT_FILE}")
    
    for i in range(start_idx, len(df_all)):
        # History (Training Set) is everything before today
        df_train = df_all.iloc[:i] 
        
        # Current Day (for Reference/Features)
        current = df_all.iloc[i]
        
        current_conditions = {
            'real_yield_bin': current['real_yield_bin'],
            'dxy_bin': current['dxy_bin'],
            'vix_bin': current['vix_bin'],
            'fed_bs_bin': current['fed_bs_bin'],
            'Gold_vol_bin': current['Gold_vol_bin'],
            'Silver_vol_bin': current['Silver_vol_bin'],
            'Gold_rsi_bin': current['Gold_rsi_bin'],
            'Silver_rsi_bin': current['Silver_rsi_bin'],
            'Gold_trend_bin': current['Gold_trend_bin'],
            'Silver_trend_bin': current['Silver_trend_bin'],
            'regime': current.get('regime', 'Unclassified') # PASS REGIME
        }
        
        for asset in ['Gold', 'Silver']:
            # Run Model
            p_up, p_down, n_samples, used_vol = calculate_bayesian_probability(df_train, asset, current_conditions)
            lower, mean, upper = calculate_expected_range(df_train, asset, current_conditions)
            
            if p_up is not None and mean is not None:
                results.append({
                    'date': current['date'],
                    'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Simulated timestamp
                    'asset': asset,
                    'prob_up': round(p_up, 4),
                    'prob_down': round(p_down, 4),
                    'expected_return': round(mean, 4),
                    'range_lower': round(lower, 4),
                    'range_upper': round(upper, 4),
                    'samples': n_samples,
                    'model_version': 'hybrid_v1_backtest'
                })
                
    
    # Save Results
    df_res = pd.DataFrame(results)
    
    # 1. Save Full Predictions Log
    df_res.to_csv(OUTPUT_FILE, index=False)
    print(f"\nPredictions Log saved to {OUTPUT_FILE}")
    print(f"Total Entries: {len(df_res)}")

    # 2. Save Backtest Scorecard (Verification)
    # We add verification columns locally
    scorecard = []
    
    # Reload data to ensure we have actuals available easily (or just use df_res joined with df_all)
    # We can iterate the results we just made.
    
    BACKTEST_FILE = os.path.join(BASE_DIR, 'data', 'backtest_results.csv')
    
    print("\nGenerating Backtest Scorecard...")
    
    for idx, row in df_res.iterrows():
        # Find actuals in df_all
        date_match = df_all[df_all['date'] == row['date']]
        if not date_match.empty:
            actual_row = date_match.iloc[0]
            asset = row['asset']
            
            # Use 'Gold_return', 'Gold_direction' etc.
            if pd.notna(actual_row[f'{asset}_return']):
                actual_ret = actual_row[f'{asset}_return']
                actual_dir_val = actual_row[f'{asset}_direction']
                
                # Check prediction
                predicted_dir = 1 if row['prob_up'] > 0.5 else 0
                is_correct = (predicted_dir == actual_dir_val)
                
                scorecard.append({
                    'date': row['date'],
                    'asset': asset,
                    'prob_up': row['prob_up'],
                    'predicted_dir': 'UP' if predicted_dir == 1 else 'DOWN',
                    'actual_dir': 'UP' if actual_dir_val == 1 else 'DOWN',
                    'actual_return': round(actual_ret, 4),
                    'is_correct': is_correct,
                    'samples': row['samples']
                })
    
    if scorecard:
        df_score = pd.DataFrame(scorecard)
        df_score.to_csv(BACKTEST_FILE, index=False)
        print(f"Backtest Scorecard saved to {BACKTEST_FILE}")
        print(f"Verified Entries: {len(df_score)}")
        
        # --- Advanced Metrics Calculation ---
        print("\n" + "="*60)
        print("PERFORMANCE REPORT (Long/Short Strategy)")
        print("="*60)
        
        metrics = {}
        for asset in ['Gold', 'Silver']:
            asset_res = df_score[df_score['asset'] == asset].copy()
            total = len(asset_res)
            
            if total > 0:
                # 1. Calculate PnL per trade
                # If Correct: PnL = abs(Actual Return)
                # If Wrong: PnL = -abs(Actual Return)
                # This assumes we capture the full day's move directionally
                asset_res['pnl'] = asset_res.apply(
                    lambda row: abs(row['actual_return']) if row['is_correct'] else -abs(row['actual_return']), 
                    axis=1
                )
                
                # 2. Win Rate
                wins = asset_res[asset_res['pnl'] > 0]
                losses = asset_res[asset_res['pnl'] < 0] # Zeros excluded from loss count typically
                win_rate = (len(wins) / total) * 100
                
                # 3. Payoff Ratio (Avg Win / Avg Loss)
                avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
                avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
                payoff = avg_win / avg_loss if avg_loss > 0 else 0
                
                # 4. Expected Value (Per Trade)
                ev = asset_res['pnl'].mean()
                
                # 5. Sharpe Ratio (Annualized)
                # Sharpe = (Mean Return / Std Dev) * sqrt(252)
                # Assuming Risk Free Rate = 0 for simplicity of "Strategy vs Cash"
                std_dev = asset_res['pnl'].std()
                sharpe = (ev / std_dev) * np.sqrt(252) if std_dev > 0 else 0
                
                # 6. Max Drawdown
                asset_res['cumulative_pnl'] = asset_res['pnl'].cumsum()
                asset_res['peak'] = asset_res['cumulative_pnl'].cummax()
                asset_res['drawdown'] = asset_res['cumulative_pnl'] - asset_res['peak']
                max_drawdown = asset_res['drawdown'].min()
                
                # 7. Total Return
                total_return = asset_res['pnl'].sum()
                
                print(f"[{asset}]")
                print(f"  Win Rate:      {win_rate:.1f}%")
                print(f"  Payoff Ratio:  {payoff:.2f} (Avg Win: {avg_win:.2f}% / Avg Loss: {avg_loss:.2f}%)")
                print(f"  Sharpe Ratio:  {sharpe:.2f}")
                print(f"  Exp Value:     {ev:.3f}% per trade")
                print(f"  Total Return:  {total_return:.1f}% (uncompounded)")
                print(f"  Max Drawdown:  {max_drawdown:.1f}%")
                print("-" * 30)

                # Store metrics for JSON export
                metrics[asset] = {
                    "Win Rate": f"{win_rate:.1f}%",
                    "Payoff Ratio": f"{payoff:.2f}",
                    "Sharpe Ratio": f"{sharpe:.2f}",
                    "Max Drawdown": f"{max_drawdown:.1f}%"
                }
                
        # Save metrics to JSON
        import json
        with open(os.path.join(BASE_DIR, 'data', 'performance_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Performance metrics saved to {os.path.join(BASE_DIR, 'data', 'performance_metrics.json')}")
                
    else:
        print("No verification data available (maybe futures missing?)")
        
    print("="*60)

if __name__ == "__main__":
    run_backtest(365) # Warmup 1 year, predict everything after
