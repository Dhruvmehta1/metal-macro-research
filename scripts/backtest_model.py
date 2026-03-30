#!/usr/bin/env python3
"""
Backtest Logic for Metals Macro System.

Walk-forward validation with NO lookahead bias:
- At each step, indicators are recalculated using ONLY historical data
- Targets are realized returns from the NEXT day (not pre-computed with future knowledge)
- All features available at prediction time only
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

from predict_prices import (
    load_base_data,
    calculate_technical_indicators,
    calculate_macro_features,
    bin_features,
    calculate_bayesian_probability,
    calculate_expected_range,
    calculate_rules_and_risk,
    calculate_sentiment_for_date,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'predictions_log.csv')
BACKTEST_FILE = os.path.join(BASE_DIR, 'data', 'backtest_results.csv')
NEWS_FILE = os.path.join(BASE_DIR, 'data', 'news.csv')


def get_realized_return(df_base, current_idx, asset):
    """
    Get the actual realized return for a given day from the price data.

    This is the return from close at day t to close at day t+1.
    We can use this in backtest evaluation because we're evaluating
    what happened AFTER the prediction was made, not using future
    data to make the prediction.

    Returns None if next day's price isn't available.
    """
    price_col = f"price_{asset}"
    if price_col not in df_base.columns:
        return None

    if current_idx + 1 >= len(df_base):
        return None  # No next day data

    price_today = df_base.iloc[current_idx][price_col]
    price_tomorrow = df_base.iloc[current_idx + 1][price_col]

    if pd.isna(price_today) or pd.isna(price_tomorrow) or price_today == 0:
        return None

    return ((price_tomorrow - price_today) / price_today) * 100


def prepare_walk_forward_data(df_base, current_idx):
    """
    Prepare data for a single walk-forward step.

    At index i:
    - Training data = rows 0 to i-1 (with targets realized at row i)
    - Current features = row i (no target yet - that's what we're predicting)

    This ensures NO lookahead bias - we only know what actually happened,
    not what will happen.
    """
    # Training set: all data BEFORE current day
    df_train = df_base.iloc[:current_idx].copy()

    # Current day: features only (target unknown at prediction time)
    df_current = df_base.iloc[current_idx:current_idx+1].copy()

    if len(df_train) < 100:
        return None, None  # Not enough training data

    # Calculate indicators for training data (WITH targets - they're in the past)
    for asset in ["Gold", "Silver"]:
        df_train = calculate_technical_indicators(df_train, asset, include_target=True)

    # Calculate indicators for current day (NO target - we're predicting it)
    for asset in ["Gold", "Silver"]:
        df_current = calculate_technical_indicators(df_current, asset, include_target=False)

    # Macro features
    df_train = calculate_macro_features(df_train)
    df_current = calculate_macro_features(df_current)

    # Bin features
    df_train = bin_features(df_train)
    df_current = bin_features(df_current)

    # Forward fill macro data (simulates real-time availability)
    for col in ["real_yield", "regime", "growth_momentum", "inflation_momentum"]:
        if col in df_train.columns:
            df_train[col] = df_train[col].ffill()
        if col in df_current.columns:
            df_current[col] = df_current[col].ffill()

    df_train["regime"] = df_train["regime"].fillna("Unclassified")
    df_current["regime"] = df_current["regime"].fillna("Unclassified")

    # Drop training rows without valid targets
    df_train = df_train.dropna(subset=["Gold_return", "Silver_return"])

    if len(df_train) < 50:
        return None, None  # Not enough valid training data

    return df_train, df_current.iloc[-1] if len(df_current) > 0 else None


def get_current_conditions(row, sentiment_data=None):
    """Extract current conditions dict from a DataFrame row."""
    if row is None:
        return None

    if sentiment_data is None:
        sentiment_data = {'Gold': {'score': 0, 'label': 'Neutral'}, 'Silver': {'score': 0, 'label': 'Neutral'}}

    conditions = {
        'regime': row.get('regime', 'Unclassified'),
        'real_yield_bin': row.get('real_yield_bin', 'unknown'),
        'dxy_bin': row.get('dxy_bin', 'flat'),
        'vix_bin': row.get('vix_bin', 'medium'),
        'fed_bs_bin': row.get('fed_bs_bin', 'qt_mild'),
        'Gold_vol_bin': row.get('Gold_vol_bin', 'normal'),
        'Silver_vol_bin': row.get('Silver_vol_bin', 'normal'),
        'Gold_rsi_bin': row.get('Gold_rsi_bin', 'neutral'),
        'Silver_rsi_bin': row.get('Silver_rsi_bin', 'neutral'),
        'Gold_trend_bin': row.get('Gold_trend_bin', 'chop'),
        'Silver_trend_bin': row.get('Silver_trend_bin', 'chop'),
        'Gold_bb_bin': row.get('Gold_bb_bin', 'range_bound'),
        'Silver_bb_bin': row.get('Silver_bb_bin', 'range_bound'),
        'sentiment': sentiment_data
    }
    return conditions


TRANSACTION_COST_PCT = 0.05  # Round-trip cost per trade (spread + commissions), as %


def run_backtest(start_warmup_days=252):
    """
    Run walk-forward backtest.

    Parameters:
    - start_warmup_days: Number of days to use as initial training (default ~1 trading year)
    """
    print("="*60)
    print("WALK-FORWARD BACKTEST (No Lookahead Bias)")
    print(f"Transaction Cost: {TRANSACTION_COST_PCT:.2f}% per trade (round-trip)")
    print("="*60)

    # 1. Load ALL raw data
    print("\nLoading base data...")
    df_base = load_base_data()

    if len(df_base) < start_warmup_days + 50:
        print(f"Error: Not enough data for backtest. Have {len(df_base)}, need {start_warmup_days + 50}")
        return

    print(f"Loaded {len(df_base)} days of raw price data")
    print(f"Backtest period: {df_base.iloc[start_warmup_days]['date']} to {df_base.iloc[-1]['date']}")

    # Load news data for sentiment analysis (if available)
    news_df = None
    if os.path.exists(NEWS_FILE):
        try:
            news_df = pd.read_csv(NEWS_FILE)
            if not news_df.empty:
                news_df["timestamp"] = pd.to_datetime(news_df["timestamp"], errors="coerce")
                news_df = news_df.dropna(subset=["timestamp"])
            else:
                news_df = None
        except Exception as e:
            print(f"Warning: Could not load news data: {e}")
            news_df = None

    results = []
    scorecard = []

    # 2. Walk-forward loop
    print("\nRunning walk-forward validation...")
    for current_idx in range(start_warmup_days, len(df_base)):
        current_date = df_base.iloc[current_idx]['date']

        # Progress indicator
        if (current_idx - start_warmup_days) % 50 == 0:
            print(f"  Processing {current_date}...")

        # Prepare data with NO lookahead
        df_train, current_row = prepare_walk_forward_data(df_base, current_idx)

        if df_train is None or current_row is None:
            continue

        # Calculate sentiment using ONLY news available up to current date
        sentiment_data = calculate_sentiment_for_date(news_df, current_date)

        current_conditions = get_current_conditions(current_row, sentiment_data)

        # Generate predictions for each asset
        for asset in ["Gold", "Silver"]:
            # Calculate Bayesian probability
            p_up, p_down, n_samples, used_vol = calculate_bayesian_probability(
                df_train, asset, current_conditions
            )

            # Calculate expected range
            lower, mean, upper = calculate_expected_range(
                df_train, asset, current_conditions
            )

            if p_up is not None and mean is not None:
                # Store prediction
                results.append({
                    'date': current_date,
                    'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'asset': asset,
                    'prob_up': round(p_up, 4),
                    'prob_down': round(p_down, 4),
                    'expected_return': round(mean, 4),
                    'range_lower': round(lower, 4),
                    'range_upper': round(upper, 4),
                    'samples': n_samples,
                    'model_version': 'walkforward_v1'
                })

                # Evaluate prediction against ACTUAL realized return
                # Use get_realized_return which calculates return from t to t+1
                actual_return = get_realized_return(df_base, current_idx, asset)

                if actual_return is not None and not pd.isna(actual_return):
                    actual_direction = 1 if actual_return > 0 else 0

                    # Prediction: UP if prob > 0.5
                    predicted_direction = 1 if p_up > 0.5 else 0
                    is_correct = (predicted_direction == actual_direction)

                    scorecard.append({
                        'date': current_date,
                        'asset': asset,
                        'prob_up': round(p_up, 4),
                        'predicted_dir': 'UP' if predicted_direction == 1 else 'DOWN',
                        'actual_dir': 'UP' if actual_direction == 1 else 'DOWN',
                        'actual_return': round(actual_return, 4),
                        'is_correct': is_correct,
                        'samples': n_samples
                    })

    # 3. Save results
    print("\nSaving results...")

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(OUTPUT_FILE, index=False)
        print(f"Predictions log saved to {OUTPUT_FILE} ({len(df_results)} entries)")

    if scorecard:
        df_score = pd.DataFrame(scorecard)
        df_score.to_csv(BACKTEST_FILE, index=False)
        print(f"Backtest scorecard saved to {BACKTEST_FILE} ({len(df_score)} entries)")

        # 4. Calculate performance metrics
        print("\n" + "="*60)
        print("PERFORMANCE REPORT (Out-of-Sample)")
        print("="*60)

        metrics = {}
        for asset in ['Gold', 'Silver']:
            asset_df = df_score[df_score['asset'] == asset].copy()
            total = len(asset_df)

            if total > 0:
                # Win Rate
                wins = asset_df[asset_df['is_correct'] == True]
                win_rate = (len(wins) / total) * 100

                # PnL calculation: Long-only strategy with transaction costs
                # If we predict UP: gain actual_return MINUS round-trip cost
                # If we predict DOWN: cash (no position, no cost)
                def calc_pnl(row):
                    if row['predicted_dir'] == 'UP':
                        return row['actual_return'] - TRANSACTION_COST_PCT  # Long position minus cost
                    else:
                        return 0  # Cash/skip

                asset_df['pnl'] = asset_df.apply(calc_pnl, axis=1)

                # Average win/loss
                winning_trades = asset_df[asset_df['pnl'] > 0]
                losing_trades = asset_df[asset_df['pnl'] < 0]

                avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
                avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
                payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0

                # Expected value per trade
                ev = asset_df['pnl'].mean()

                # Sharpe ratio (annualized)
                std_pnl = asset_df['pnl'].std()
                sharpe = (ev / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0

                # Max drawdown
                asset_df['cumulative_pnl'] = asset_df['pnl'].cumsum()
                asset_df['peak'] = asset_df['cumulative_pnl'].cummax()
                asset_df['drawdown'] = asset_df['cumulative_pnl'] - asset_df['peak']
                max_drawdown = asset_df['drawdown'].min()

                # Total return
                total_return = asset_df['pnl'].sum()

                print(f"\n[{asset}]")
                print(f"  Total Trades:  {total}")
                print(f"  Win Rate:      {win_rate:.1f}%")
                print(f"  Avg Win:       {avg_win:+.3f}%")
                print(f"  Avg Loss:      {avg_loss:+.3f}%")
                print(f"  Payoff Ratio:  {payoff_ratio:.2f}")
                print(f"  Exp Value:     {ev:+.3f}% per trade")
                print(f"  Sharpe Ratio:  {sharpe:.2f} (ann.)")
                print(f"  Total Return:  {total_return:+.1f}%")
                print(f"  Max Drawdown:  {max_drawdown:+.1f}%")

                metrics[asset] = {
                    "Win Rate": f"{win_rate:.1f}%",
                    "Avg Win": f"{avg_win:+.3f}%",
                    "Avg Loss": f"{avg_loss:+.3f}%",
                    "Payoff Ratio": f"{payoff_ratio:.2f}",
                    "Sharpe Ratio": f"{sharpe:.2f}",
                    "Max Drawdown": f"{max_drawdown:+.1f}%",
                    "Total Return": f"{total_return:+.1f}%",
                    "Total Trades": total,
                    "Transaction Cost": f"{TRANSACTION_COST_PCT:.2f}% per trade"
                }

        # Save metrics to JSON
        import json
        metrics_file = os.path.join(BASE_DIR, 'data', 'performance_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\nPerformance metrics saved to {metrics_file}")

        print("\n" + "="*60)
        print("NOTE: This is an OUT-OF-SAMPLE test with NO lookahead bias.")
        print("All indicators were calculated using only historical data")
        print("available at each prediction point.")
        print(f"Transaction cost of {TRANSACTION_COST_PCT:.2f}% applied per trade.")
        print("="*60)

    else:
        print("No scorecard data generated - check data quality")


if __name__ == "__main__":
    run_backtest(start_warmup_days=252)  # ~1 trading year warmup
