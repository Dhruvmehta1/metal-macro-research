#!/usr/bin/env python3
"""
Bayesian Prediction Model for Gold/Silver Price Movements

Uses:
1. Bayes' Theorem - Calculate probability of price going up/down
2. Central Limit Theorem - Estimate expected price range

Features:
- Real Yield levels
- DXY trends
- VIX volatility
- Fed balance sheet (QE/QT status)
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICES_FILE = os.path.join(BASE_DIR, "data", "prices.csv")
REAL_YIELD_FILE = os.path.join(BASE_DIR, "data", "real_yields.csv")
GDP_FILE = os.path.join(BASE_DIR, "data", "gdp.csv")
CHINA_GOLD_FILE = os.path.join(BASE_DIR, "data", "china_gold_reserves.csv")
FED_BS_FILE = os.path.join(BASE_DIR, "data", "central_bank_balance_sheets.csv")
NEWS_FILE = os.path.join(BASE_DIR, "data", "news.csv")


def bin_real_yield(value):
    """Classify real yield into bins."""
    if value < 0:
        return "negative"
    elif value < 1.5:
        return "low_positive"  # Neutral/Supportive
    elif value < 2.5:
        return "medium_positive"  # Restrictive (Current Norm)
    else:
        return "high_positive"  # > 2.5% (Crushing)


def bin_dxy_change(value):
    """Classify DXY change."""
    if value < -0.5:
        return "dump_strong"
    elif value < -0.1:
        return "down"
    elif value > 0.5:
        return "squeeze_strong"
    elif value > 0.1:
        return "up"
    else:
        return "flat"  # Noise (-0.1 to +0.1)


def bin_vix(value):
    """Classify VIX level."""
    if value < 15:
        return "low"
    elif value < 25:
        return "medium"
    else:
        return "high"


def bin_china_gold_trend(value):
    """Classify China gold reserves monthly change."""
    if value > 10:
        return "aggressive_buying"
    elif value > 0:
        return "buying"
    else:
        return "no_buying"


def bin_fed_balance_sheet(value):
    """Classify Fed balance sheet YoY change (QE/QT)."""
    if value > 2:
        return "qe_strong"
    elif value > 0:
        return "qe_mild"
    elif value > -2:
        return "qt_mild"
    else:
        return "qt_strong"


def bin_volume_trend(value):
    """Classify Relative Volume (Current / 20-Day Avg)."""
    if value > 1.5:
        return "high"  # > 150% of avg
    elif value < 0.6:
        return "low"  # < 60% of avg
    else:
        return "normal"  # 60% - 150%


def bin_rsi(value):
    """Classify RSI (14)."""
    if pd.isna(value):
        return "neutral"
    if value > 70:
        return "overbought"
    elif value < 30:
        return "oversold"
    else:
        return "neutral"


def bin_trend_sma(row, asset):
    """Classify Trend based on Price vs SMA50 vs SMA200."""
    price = row[f"price_{asset}"]
    sma50 = row[f"{asset}_sma50"]
    sma200 = row[f"{asset}_sma200"]

    if pd.isna(sma50) or pd.isna(sma200):
        return "chop"

    if price > sma50 and sma50 > sma200:
        return "bullish_trend"
    elif price < sma50 and sma50 < sma200:
        return "bearish_trend"
    else:
        return "chop"  # Entangled


def bin_bb_pct_b(value):
    """Classify Bollinger Band %B."""
    if pd.isna(value):
        return "squeeze"
    if value > 1.0:
        return "breakout_high"
    elif value < 0.0:
        return "breakout_low"
    else:
        return "range_bound"


def prepare_data():
    """Load and prepare all data for analysis."""
    print("Loading historical data...")

    # Load prices
    df_prices = pd.read_csv(PRICES_FILE)
    df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"])
    df_prices["date"] = df_prices["timestamp"].dt.date

    # Pivot to get one row per date
    # We need both PRICE and VOLUME this time
    df_pivot = df_prices.pivot_table(
        index="date", columns="asset", values=["price", "volume"], aggfunc="last"
    ).reset_index()

    # Flatten columns (MultiIndex)
    new_cols = ["date"]
    if isinstance(df_pivot.columns, pd.MultiIndex):
        # Format: price_Gold, volume_Gold
        for col in df_pivot.columns:
            if col[0] == "date" or col[1] == "":
                continue
            new_cols.append(f"{col[0]}_{col[1]}")
    else:
        new_cols = df_pivot.columns  # Already flattened? Unlikely given pivot_table

    df_pivot.columns = ["date"] + [c for c in new_cols if c != "date"]

    # --- MERGE REGIME MODEL ---
    try:
        from regime_model import detect_regime

        regimes = detect_regime()
        if regimes is not None:
            # Regimes are indexed by timestamp. We need to match 'date'.
            regimes["date"] = regimes.index.date
            df_pivot = pd.merge(
                df_pivot,
                regimes[["date", "regime", "growth_momentum", "inflation_momentum"]],
                on="date",
                how="left",
            )
    except ImportError:
        print("Warning: Regime model import failed.")
        df_pivot["regime"] = "Unclassified"

    # -- Calculate Features for Each Asset --
    for asset in ["Gold", "Silver"]:
        # Returns
        # TARGET DEFINITION: Next Day's Close vs Today's Close
        # pct_change() calculates (Close_t - Close_t-1) / Close_t-1
        # shift(-1) moves the return from (t+1) to (t)
        # Result: Return realized at Close of t+1, relative to Close of t.
        df_pivot[f"{asset}_return"] = (
            df_pivot[f"price_{asset}"].pct_change(fill_method=None).shift(-1) * 100
        )
        df_pivot[f"{asset}_direction"] = (df_pivot[f"{asset}_return"] > 0).astype(int)

        # Volume Trend (RVOL)
        # 1. Fill 0 volume with NaN to avoid skewing average
        vol_col = f"volume_{asset}"
        if vol_col in df_pivot.columns:
            df_pivot[vol_col] = df_pivot[vol_col].replace(0, np.nan)
            # 2. Daily RVOL (Volume / 20-day SMA)
            df_pivot[f"{asset}_rvol"] = (
                df_pivot[vol_col] / df_pivot[vol_col].rolling(window=20).mean()
            )
        else:
            df_pivot[f"{asset}_rvol"] = 1.0  # Default if missing

    # Calculate DXY change
    if "price_DXY" in df_pivot.columns:
        df_pivot["DXY_change"] = (
            df_pivot["price_DXY"].pct_change(fill_method=None) * 100
        )
    else:
        df_pivot["DXY_change"] = 0.0

    # --- Technical Indicators (RSI, Trend, Bollinger) ---
    for asset in ["Gold", "Silver"]:
        price_col = f"price_{asset}"
        if price_col not in df_pivot.columns:
            continue

        # 1. RSI (14) - Momentum
        delta = df_pivot[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_pivot[f"{asset}_rsi"] = 100 - (100 / (1 + rs))

        # 2. Trend (SMA 50 vs 200) - Direction
        df_pivot[f"{asset}_sma50"] = df_pivot[price_col].rolling(window=50).mean()
        df_pivot[f"{asset}_sma200"] = df_pivot[price_col].rolling(window=200).mean()

        # 3. Bollinger Bands (20, 2) - Volatility/Reversion
        rolling_mean = df_pivot[price_col].rolling(window=20).mean()
        rolling_std = df_pivot[price_col].rolling(window=20).std()
        df_pivot[f"{asset}_bb_upper"] = rolling_mean + (rolling_std * 2)
        df_pivot[f"{asset}_bb_lower"] = rolling_mean - (rolling_std * 2)
        # %B (Position within bands)
        df_pivot[f"{asset}_pct_b"] = (
            df_pivot[price_col] - df_pivot[f"{asset}_bb_lower"]
        ) / (df_pivot[f"{asset}_bb_upper"] - df_pivot[f"{asset}_bb_lower"])

    # Load real yields
    df_real = pd.read_csv(REAL_YIELD_FILE)
    df_real["date"] = pd.to_datetime(df_real["date"]).dt.date

    # Merge real yields (Use LEFT JOIN to keep price data even if yield is missing for today)
    df = pd.merge(df_pivot, df_real[["date", "real_yield"]], on="date", how="left")

    # Forward fill real yield for the very last day if missing (Macro data often lags 1 day)
    df["real_yield"] = df["real_yield"].ffill()

    # Load and merge China gold reserves
    if os.path.exists(CHINA_GOLD_FILE):
        df_china = pd.read_csv(CHINA_GOLD_FILE)
        df_china["date"] = pd.to_datetime(df_china["date"]).dt.date
        df = pd.merge(
            df, df_china[["date", "monthly_change_tonnes"]], on="date", how="left"
        )
        df["monthly_change_tonnes"] = df["monthly_change_tonnes"].fillna(0)
    else:
        df["monthly_change_tonnes"] = 0

    # Load and merge Fed balance sheet
    if os.path.exists(FED_BS_FILE):
        df_fed = pd.read_csv(FED_BS_FILE)
        df_fed["date"] = pd.to_datetime(df_fed["date"]).dt.date
        df = pd.merge(df, df_fed[["date", "Fed_yoy"]], on="date", how="left")
        df["Fed_yoy"] = df["Fed_yoy"].fillna(0)
    else:
        df["Fed_yoy"] = 0

    # Create feature bins
    df["real_yield_bin"] = df["real_yield"].apply(bin_real_yield)
    df["dxy_bin"] = df["DXY_change"].apply(bin_dxy_change)
    if "price_VIX" in df.columns:
        df["vix_bin"] = df["price_VIX"].apply(bin_vix)
        df["VIX_val"] = df["price_VIX"]  # Helper for display
    else:
        df["vix_bin"] = "medium"
        df["VIX_val"] = 20.0

    df["china_gold_bin"] = df["monthly_change_tonnes"].apply(bin_china_gold_trend)
    df["fed_bs_bin"] = df["Fed_yoy"].apply(bin_fed_balance_sheet)

    # Bin Volume and Technicals
    for asset in ["Gold", "Silver"]:
        df[f"{asset}_vol_bin"] = df[f"{asset}_rvol"].apply(bin_volume_trend)
        df[f"{asset}_rsi_bin"] = df[f"{asset}_rsi"].apply(bin_rsi)
        df[f"{asset}_trend_bin"] = df.apply(
            lambda row: bin_trend_sma(row, asset), axis=1
        )
        df[f"{asset}_bb_bin"] = df[f"{asset}_pct_b"].apply(bin_bb_pct_b)

    # Remove NaN rows for TRAINING (Historical probabilities need outcomes)
    # BUT, we need the LATEST row for PREDICTION (Current conditions) which has NaN return.

    # 1. Training Data (History)
    df_train = df.dropna(
        subset=["Gold_return", "Silver_return", "real_yield", "DXY_change"]
    )

    # 2. Latest Data (Inference)
    # We take the very last row of the full feature-rich DF, assuming it's the latest date.
    # We only care that the INPUT features (Macro, Tech) are present.
    last_row = df.iloc[-1]

    print(f"Prepared {len(df_train)} days of historical training data")
    return df_train, last_row


def calculate_rules_and_risk(df, asset, current_conditions):
    """
    New Engine: Regime -> Rules -> Risk.
    Returns:
        - trade_signal (Buy/Sell/Hold)
        - risk_metrics (Skew, Tail Risk, Volatility)
        - rule_explanation (Why?)
    """

    # 1. Primary Filter: REGIME
    regime = current_conditions.get("regime", "Unclassified")

    # Filter history by Regime
    if regime != "Unclassified" and regime in df["regime"].values:
        history = df[df["regime"] == regime].copy()
    else:
        history = df.copy()  # Fallback

    if len(history) < 10:
        return "NEUTRAL", None, f"Insufficient Data for Regime: {regime}"

    # 2. Secondary Filter: Technicals (if enough data)
    # Refine by "Trend State" to see if momentum aligns with regime
    trend_key = f"{asset}_trend_bin"
    trend = current_conditions.get(trend_key, "chop")
    history_trend = history[history[trend_key] == trend]

    if len(history_trend) > 10:
        history = history_trend
        used_trend = True
    else:
        used_trend = False

    # 3. Calculate Risk Metrics
    returns = history[f"{asset}_return"]

    # Win Rate (Directional Prob)
    prob_up = (returns > 0).mean()

    # Skew (Expected Value)
    # (Avg Win * Prob Win) - (Avg Loss * Prob Loss)
    avg_win = returns[returns > 0].mean() if not returns[returns > 0].empty else 0
    avg_loss = abs(returns[returns < 0].mean()) if not returns[returns < 0].empty else 0
    skew_ev = (avg_win * prob_up) - (avg_loss * (1 - prob_up))

    # Tail Risk (Prob of > 1% Drop for Gold, > 2% for Silver)
    cutoff = -1.0 if asset == "Gold" else -2.0
    tail_risk_prob = (returns < cutoff).mean()

    # 4. Generate Rule-Based Output (The "Playbook")
    rationale = f"Regime is {regime}."
    if used_trend:
        rationale += f" Trend is {trend}."
    else:
        rationale += " (Trend ignored due to low sample size)."

    # --- SENTIMENT OVERRIDE CHECK ---
    sent_score = current_conditions.get("sentiment", {}).get(asset, {}).get("score", 0)
    sent_label = current_conditions.get("sentiment", {}).get(asset, {}).get("label", "Neutral")
    
    is_sentiment_strong = abs(sent_score) > 0.25
    
    if is_sentiment_strong:
         rationale += f" [NEWS ALERT: Strong {sent_label} Sentiment ({sent_score:.2f})]"

    signal = "NEUTRAL"

    # --- RULES MATRIX ---
    if regime == "REFLATION (Boom)" or regime == "STAGFLATION":
        # Gold likes these (Inflationary)
        if trend in ["bull_strong", "bull_weak"]:
            signal = "STRONG LONG"
        elif skew_ev > 0:
            signal = "LONG (Dip Buy)"
        else:
            signal = "NEUTRAL (Chop)"

    elif regime == "DEFLATION (Bust)":
        # Cash is king, Gold volatile but distinct from other commodities
        if trend == "bull_strong":
            signal = "LONG (Safety Bid)"
        else:
            signal = "CASH / SHORT"

        # Worst for Gold (Stocks up, Yields stable/up)
        if trend == "bear_strong":
            signal = "STRONG SHORT"
        else:
            signal = "SHORT / AVOID"
    
    # --- APPLY SENTIMENT OVERRIDE ---
    if is_sentiment_strong:
        if sent_label == "Bearish" and "LONG" in signal:
            signal = "NEUTRAL (News Warning)"
            rationale += " -> Technical Bullishness damped by Bearish News."
        elif sent_label == "Bullish" and "SHORT" in signal:
            signal = "NEUTRAL (News Warning)"
            rationale += " -> Technical Bearishness damped by Bullish News."
        elif sent_label == "Bearish" and "SHORT" in signal:
            signal = "STRONG SHORT (News Confirmed)"
        elif sent_label == "Bullish" and "LONG" in signal:
            signal = "STRONG LONG (News Confirmed)"
    # ---------------------

    # --- BAYESIAN BELIEF UPDATE ---
    # Prior = prob_up (Historical Win Rate)
    # Evidence = sent_score
    # Weight = 0.2 (Max sentiment can shift prob by +/- 20%)
    
    # Only apply if sentiment is significant
    sentiment_impact = 0.0
    if abs(sent_score) > 0.05:
        weight = 0.20
        # If sentiment is Bullish (>0), it adds to Prob Up.
        # If sentiment is Bearish (<0), it subtracts.
        sentiment_impact = sent_score * weight
        
        # Explain the shift
        shift_pct = sentiment_impact * 100
        rationale += f" [Bayesian Update: Win Rate {shift_pct:+.1f}% due to Sentiment]"

    prob_up_posterior = prob_up + sentiment_impact
    prob_up_posterior = max(0.01, min(0.99, prob_up_posterior)) # Clip to 1-99%

    metrics = {
        "prob_up": prob_up_posterior,
        "prob_up_prior": prob_up, # Keep original for reference
        "drivers": f"Base: {prob_up:.2f} + Sentiment: {sentiment_impact:+.2f}",
        "skew_ev": skew_ev,
        "tail_risk": tail_risk_prob,
        "samples": len(history),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }

    return signal, metrics, rationale


def calculate_bayesian_probability(df, asset, current_conditions):
    """
    Legacy Adapter: Returns directional probability using the new engine logic.
    """
    signal, metrics, rationale = calculate_rules_and_risk(df, asset, current_conditions)
    if metrics:
        return metrics["prob_up"], 1 - metrics["prob_up"], metrics["samples"], True
    return 0.5, 0.5, 0, False


def calculate_bayesian_probability_legacy(df, asset, current_conditions):
    """
    Calculate P(Up | Conditions) using Bayes' Theorem.
    Now includes VOLUME + TECHNICALS.
    """
    # Prior probability
    p_up = df[f"{asset}_direction"].mean()
    p_down = 1 - p_up

    # Filter for matching conditions
    # Note: We filter by GLOBAL Macro + ASSET SPECIFIC Features
    mask = (
        (df["real_yield_bin"] == current_conditions["real_yield_bin"])
        & (df["dxy_bin"] == current_conditions["dxy_bin"])
        & (df["vix_bin"] == current_conditions["vix_bin"])
        & (df[f"{asset}_vol_bin"] == current_conditions[f"{asset}_vol_bin"])
        & (df[f"{asset}_rsi_bin"] == current_conditions[f"{asset}_rsi_bin"])
        & (df[f"{asset}_trend_bin"] == current_conditions[f"{asset}_trend_bin"])
    )

    df_conditions = df[mask]

    # Fallback Mechanism (Tiered Degradation)
    used_full_model = True

    # Level 1: If < 5 samples, drop RSI/Trend (keep Macro + Volume)
    if len(df_conditions) < 5:
        mask = (
            (df["real_yield_bin"] == current_conditions["real_yield_bin"])
            & (df["dxy_bin"] == current_conditions["dxy_bin"])
            & (df["vix_bin"] == current_conditions["vix_bin"])
            & (df[f"{asset}_vol_bin"] == current_conditions[f"{asset}_vol_bin"])
        )
        df_conditions = df[mask]
        used_full_model = False

    # Level 2: If still < 5 samples, drop Volume (keep Macro)
    if len(df_conditions) < 5:
        mask = (
            (df["real_yield_bin"] == current_conditions["real_yield_bin"])
            & (df["dxy_bin"] == current_conditions["dxy_bin"])
            & (df["vix_bin"] == current_conditions["vix_bin"])
        )
        df_conditions = df[mask]

    if len(df_conditions) < 5:
        print(f"  Warning: Only {len(df_conditions)} samples (insufficient)")
        return None, None, 0, False

    # P(Conditions | Up)
    df_up = df[df[f"{asset}_direction"] == 1]
    mask_up = mask[df[f"{asset}_direction"] == 1]  # Re-use the working mask
    p_conditions_given_up = mask_up.sum() / len(df_up) if len(df_up) > 0 else 0

    # P(Conditions | Down)
    df_down = df[df[f"{asset}_direction"] == 0]
    mask_down = mask[df[f"{asset}_direction"] == 0]
    p_conditions_given_down = mask_down.sum() / len(df_down) if len(df_down) > 0 else 0

    # P(Conditions)
    p_conditions = len(df_conditions) / len(df)

    # Bayes' Theorem
    if p_conditions > 0:
        p_up_given_conditions = (p_conditions_given_up * p_up) / p_conditions
        p_down_given_conditions = (p_conditions_given_down * p_down) / p_conditions
    else:
        p_up_given_conditions = p_up
        p_down_given_conditions = p_down

    return (
        p_up_given_conditions,
        p_down_given_conditions,
        len(df_conditions),
        used_full_model,
    )


def calculate_expected_range(df, asset, current_conditions, confidence=0.95):
    """
    Calculate expected price range using Central Limit Theorem.
    """
    # Try with Volume + Technicals First
    mask = (
        (df["real_yield_bin"] == current_conditions["real_yield_bin"])
        & (df["dxy_bin"] == current_conditions["dxy_bin"])
        & (df["vix_bin"] == current_conditions["vix_bin"])
        & (df[f"{asset}_vol_bin"] == current_conditions[f"{asset}_vol_bin"])
        & (df[f"{asset}_rsi_bin"] == current_conditions[f"{asset}_rsi_bin"])
        & (df[f"{asset}_trend_bin"] == current_conditions[f"{asset}_trend_bin"])
    )

    df_conditions = df[mask]

    # Fallback 1: Drop Technicals (Keep Volume)
    if len(df_conditions) < 5:
        mask = (
            (df["real_yield_bin"] == current_conditions["real_yield_bin"])
            & (df["dxy_bin"] == current_conditions["dxy_bin"])
            & (df["vix_bin"] == current_conditions["vix_bin"])
            & (df[f"{asset}_vol_bin"] == current_conditions[f"{asset}_vol_bin"])
        )
        df_conditions = df[mask]

    # Fallback 2: Drop Volume (Original Macro Only)
    if len(df_conditions) < 5:
        mask = (
            (df["real_yield_bin"] == current_conditions["real_yield_bin"])
            & (df["dxy_bin"] == current_conditions["dxy_bin"])
            & (df["vix_bin"] == current_conditions["vix_bin"])
        )
        df_conditions = df[mask]

    if len(df_conditions) < 5:
        return None, None, None

    # Sample statistics
    returns = df_conditions[f"{asset}_return"].values
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    # Prediction Interval
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin = z_score * std_return

    lower = mean_return - margin
    upper = mean_return + margin

    return lower, mean_return, upper


def predict_prices():
    """Main prediction function."""
    print(
        f"[{datetime.now()}] Running Bayesian Price Prediction (w/ Volume & Technicals)...\n"
    )

    # Load data
    df_train, latest = prepare_data()

    # Get current conditions (latest data is now explicitly returned)
    # latest = df.iloc[-1] # Removed as prepare_data handles this uniqueness

    # Load Sentiment
    sentiment_data = {
        "Gold": {"score": 0.0, "label": "Neutral", "items": []},
        "Silver": {"score": 0.0, "label": "Neutral", "items": []}
    }

    if os.path.exists(NEWS_FILE):
        try:
            df_news = pd.read_csv(NEWS_FILE)
            if not df_news.empty:
                # Filter for Last 24-48 Hours
                # Use mixed format inference and coerce errors to NaT
                df_news["timestamp"] = pd.to_datetime(df_news["timestamp"], errors='coerce')
                df_news = df_news.dropna(subset=["timestamp"])
                
                limit_time = pd.Timestamp.now() - pd.Timedelta(hours=48)
                df_recent = df_news[df_news["timestamp"] > limit_time]
                
                # --- PROCESS GOLD SENTIMENT ---
                # Direct Gold News + Inverse Economy News
                gold_news = df_recent[df_recent["asset"] == "Gold"]
                economy_news = df_recent[df_recent["asset"] == "Economy"]
                
                # Formula: Gold_Score = (Gold_News_Avg) - (0.5 * Economy_News_Avg)
                # Strong Economy is usually bad for Gold (rates up).
                score_g_raw = gold_news["sentiment"].mean() if not gold_news.empty else 0
                score_eco = economy_news["sentiment"].mean() if not economy_news.empty else 0
                
                final_gold_score = score_g_raw - (0.5 * score_eco)

                # Determine Label
                if final_gold_score > 0.2:
                    label_g = "Bullish"
                elif final_gold_score < -0.2:
                    label_g = "Bearish"
                else:
                    label_g = "Neutral"
                    
                sentiment_data["Gold"] = {
                    "score": final_gold_score, 
                    "label": label_g,
                    "items": gold_news.head(3)["headline"].tolist() + economy_news.head(2)["headline"].tolist()
                }

                # --- PROCESS SILVER SENTIMENT (Correlated to Gold + Industrial) ---
                # Silver is Gold's little brother but also industrial.
                # Formula: Silver_Score = (Silver_News_Avg) + (0.5 * Gold_Score) 
                silver_news = df_recent[df_recent["asset"] == "Silver"]
                score_s_raw = silver_news["sentiment"].mean() if not silver_news.empty else 0
                
                final_silver_score = score_s_raw + (0.5 * final_gold_score)
                
                if final_silver_score > 0.2:
                    label_s = "Bullish"
                elif final_silver_score < -0.2:
                    label_s = "Bearish"
                else:
                    label_s = "Neutral"

                sentiment_data["Silver"] = {
                    "score": final_silver_score, 
                    "label": label_s,
                    "items": silver_news.head(3)["headline"].tolist()
                }

        except Exception as e:
            print(f"Error processing news: {e}")

    current_conditions = {
        "regime": latest.get("regime", "Unclassified"),
        "real_yield_bin": latest["real_yield_bin"],
        "dxy_bin": latest["dxy_bin"],
        "vix_bin": latest["vix_bin"],
        "fed_bs_bin": latest["fed_bs_bin"],
        "Gold_vol_bin": latest["Gold_vol_bin"],
        "Silver_vol_bin": latest["Silver_vol_bin"],
        "Gold_rsi_bin": latest["Gold_rsi_bin"],
        "Silver_rsi_bin": latest["Silver_rsi_bin"],
        "Gold_trend_bin": latest["Gold_trend_bin"],
        "Silver_trend_bin": latest["Silver_trend_bin"],
        "Gold_bb_bin": latest["Gold_bb_bin"],
        "Silver_bb_bin": latest["Silver_bb_bin"],
        "real_yield": latest["real_yield"],
        "dxy_change": latest["DXY_change"],
        "vix": latest["VIX_val"],
        "fed_bs": latest["Fed_yoy"],
        "sentiment": sentiment_data # Pass full object
    }

    regime_desc = current_conditions["regime"]
    print("Current Market Conditions:")
    print(
        f"  REGIME: {regime_desc} (Growth: {latest.get('growth_momentum', 0):.2f}% | Inflation: {latest.get('inflation_momentum', 0):.2f}%)"
    )
    print(
        f"  Real Yield: {current_conditions['real_yield']:.2f}% ({current_conditions['real_yield_bin']})"
    )
    print(
        f"  DXY Change: {current_conditions['dxy_change']:+.2f}% ({current_conditions['dxy_bin']})"
    )
    print(f"  VIX: {current_conditions['vix']:.2f} ({current_conditions['vix_bin']})")
    print(
        f"  Fed Balance Sheet: {current_conditions['fed_bs']:+.1f}% YoY ({current_conditions['fed_bs_bin']})"
    )
    print(f"  News Sentiment (Gold): {sentiment_data['Gold']['score']:.2f} ({sentiment_data['Gold']['label']})")
    print(f"  News Sentiment (Silver): {sentiment_data['Silver']['score']:.2f} ({sentiment_data['Silver']['label']})")
    print("\nVolume & Technicals:")
    print(
        f"  Gold Volume: {latest['Gold_rvol']:.2f}x ({current_conditions['Gold_vol_bin']})"
    )
    print(
        f"  Gold RSI: {latest['Gold_rsi']:.1f} ({latest['Gold_rsi_bin']}) | Trend: {latest['Gold_trend_bin']} | BB: {latest['Gold_bb_bin']}"
    )
    print(
        f"  Silver Volume: {latest['Silver_rvol']:.2f}x ({current_conditions['Silver_vol_bin']})"
    )
    print(
        f"  Silver RSI: {latest['Silver_rsi']:.1f} ({latest['Silver_rsi_bin']}) | Trend: {latest['Silver_trend_bin']} | BB: {latest['Silver_bb_bin']}"
    )
    print()

    # Calculate Target Date (Next US Trading Day)
    today = datetime.now()
    target_date = today.strftime("%Y-%m-%d")
    weekday = today.weekday()
    if weekday == 5:  # Saturday -> Monday
        target_date = (today + timedelta(days=2)).strftime("%Y-%m-%d")
    elif weekday == 6:  # Sunday -> Monday
        target_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    print("=" * 70)
    print(f"PREDICTIONS for Trading Session: {target_date} (1-Day Horizon)")
    print("=" * 70)

    print("Fetching today's opening prices...\n")
    try:
        from get_todays_open import get_todays_open

        todays_open = get_todays_open()
    except Exception as e:
        print(f"Warning: Could not fetch today's opening prices: {e}")
        todays_open = {}

    for asset in ["Gold", "Silver"]:
        print(f"\n{asset}:")

        # New Engine: Rule-Based Logic
        # df_train contains history, current_conditions uses 'latest' data
        signal, metrics, rationale = calculate_rules_and_risk(
            df_train, asset, current_conditions
        )

        if metrics:
            print(f"  SIGNAL: {signal}")
            print(f"  Rationale: {rationale}")

            # Risk Metrics
            skew_label = "BULLISH" if metrics["skew_ev"] > 0 else "BEARISH"
            tail_label = "HIGH" if metrics["tail_risk"] > 0.3 else "NORMAL"

            print("  RISK PROFILE:")
            print(f"    Skew (EV): {metrics['skew_ev']:+.3f}% ({skew_label})")
            print(
                f"    Tail Risk (Crash Prob): {metrics['tail_risk'] * 100:.1f}% ({tail_label})"
            )
            print(
                f"    Win Rate ({regime_desc}): {metrics['prob_up'] * 100:.1f}% (Adjusted) vs {metrics['prob_up_prior'] * 100:.1f}% (Base)"
            )

            # Legacy expected range (still useful)
            lower, mean, upper = calculate_expected_range(
                df_train, asset, current_conditions
            )
            if lower is not None:
                print(f"  Expected Return: {mean:+.2f}%")
                print(f"  Range (95%): {lower:+.2f}% to {upper:+.2f}%")

                # Targets
                latest_price = latest[f"price_{asset}"]
                target_price = latest_price * (1 + mean / 100)
                range_low = latest_price * (1 + lower / 100)
                range_high = latest_price * (1 + upper / 100)

                print(f"\n  From Close (${latest_price:,.2f}):")
                print(f"    Target: ${target_price:,.2f}")
                print(f"    Range: ${range_low:,.2f} - ${range_high:,.2f}")
        else:
            print(f"  Direction: Insufficient history for {regime_desc}")

    print("\n" + "=" * 70)
    print("\nNote: Signals are rule-based derivations from the Macro Regime.")
    print("Risk metrics estimate the shape of the probability distribution.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    predict_prices()
