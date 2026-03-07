
import pandas as pd
import os
from datetime import datetime, timedelta
import sys

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICES_FILE = os.path.join(BASE_DIR, 'data', 'prices.csv')
CALENDAR_FILE = os.path.join(BASE_DIR, 'data', 'calendar.csv')
NEWS_FILE = os.path.join(BASE_DIR, 'data', 'news.csv')
GDP_FILE = os.path.join(BASE_DIR, 'data', 'gdp.csv')
BREAKEVEN_FILE = os.path.join(BASE_DIR, 'data', 'breakeven_inflation.csv')
REPORT_FILE = os.path.join(BASE_DIR, 'data', 'daily_report.txt')

def get_price_change(df, asset):
    """Calculates % change for an asset from ~24h ago (or last available trading price)."""
    # Filter for asset
    asset_df = df[df['asset'] == asset].copy()
    if asset_df.empty:
        return "N/A"
    
    # Ensure sorted
    asset_df['timestamp'] = pd.to_datetime(asset_df['timestamp'])
    asset_df = asset_df.sort_values('timestamp')
    
    if len(asset_df) < 2:
        return "N/A"
    
    # Get latest price
    latest_price = asset_df.iloc[-1]['price']
    latest_time = asset_df.iloc[-1]['timestamp']
    
    # Try to find price from ~24h ago
    target_time = latest_time - timedelta(hours=24)
    
    # Find records that are BEFORE the target time (not after)
    # This ensures we're comparing to an older price, not the same one
    older_records = asset_df[asset_df['timestamp'] <= target_time]
    
    if not older_records.empty:
        # Use the most recent record before 24h ago
        old_price = older_records.iloc[-1]['price']
    else:
        # No data from 24h ago (e.g., weekend gap)
        # Use the second-to-last record if it's at least 12h old
        if len(asset_df) >= 2:
            second_latest_time = asset_df.iloc[-2]['timestamp']
            time_gap = latest_time - second_latest_time
            
            if time_gap >= timedelta(hours=12):
                # Significant gap (weekend/holiday), use second-to-last
                old_price = asset_df.iloc[-2]['price']
            else:
                # Recent data, go back further to find ~24h
                # Find record closest to 24h that's not the latest
                asset_df_excl_latest = asset_df.iloc[:-1]
                asset_df_excl_latest['time_diff'] = (asset_df_excl_latest['timestamp'] - target_time).abs()
                old_price = asset_df_excl_latest.loc[asset_df_excl_latest['time_diff'].idxmin()]['price']
        else:
            return "N/A"
    
    pct_change = ((latest_price - old_price) / old_price) * 100
    return f"{pct_change:+.2f}%"

def generate_report():
    report_lines = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    report_lines.append(f"=== Daily Report: {today_str} ===\n")
    
    # 1. Prices
    report_lines.append("MARKET SNAPSHOT (24h Change):")
    if os.path.exists(PRICES_FILE):
        df_prices = pd.read_csv(PRICES_FILE)
        df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'])
        
        assets = ['US10Y', 'DXY', 'Gold', 'Silver', 'SP500', 'VIX']
        for asset in assets:
            change = get_price_change(df_prices, asset)
            report_lines.append(f"{asset}: {change}")
        
        # Calculate and display Real Yield
        if os.path.exists(BREAKEVEN_FILE):
            try:
                df_breakeven = pd.read_csv(BREAKEVEN_FILE)
                df_breakeven['date'] = pd.to_datetime(df_breakeven['date'])
                
                # Get latest nominal 10Y yield
                us10y_data = df_prices[df_prices['asset'] == 'US10Y'].sort_values('timestamp')
                if not us10y_data.empty:
                    latest_nominal = us10y_data.iloc[-1]['price']
                    
                    # Get latest breakeven inflation
                    latest_breakeven = df_breakeven.iloc[-1]['breakeven_inflation']
                    
                    # Calculate real yield
                    real_yield = latest_nominal - latest_breakeven
                    
                    report_lines.append(f"Real Yield (10Y): {real_yield:.2f}% (Nominal: {latest_nominal:.2f}% - Inflation: {latest_breakeven:.2f}%)")
            except Exception as e:
                report_lines.append(f"Real Yield: Error calculating ({e})")
    else:
        report_lines.append("No price data found.")
    
    report_lines.append("")
    
    # 2. GDP Data
    report_lines.append("US GDP (Latest):")
    if os.path.exists(GDP_FILE):
        try:
            df_gdp = pd.read_csv(GDP_FILE)
            if not df_gdp.empty:
                latest_gdp = df_gdp.iloc[-1]
                gdp_date = latest_gdp['date']
                gdp_nominal = latest_gdp['gdp_nominal']
                gdp_real = latest_gdp['gdp_real']
                gdp_growth = latest_gdp['gdp_growth_yoy']
                
                report_lines.append(f"Quarter: {gdp_date}")
                report_lines.append(f"Nominal: ${gdp_nominal:.1f}B | Real: ${gdp_real:.1f}B | YoY: {gdp_growth:+.2f}%")
            else:
                report_lines.append("No GDP data available.")
        except Exception as e:
            report_lines.append(f"Error reading GDP data: {e}")
    else:
        report_lines.append("No GDP data found. Run 'fetch_gdp.py' to initialize.")
    
    report_lines.append("")
    
    
    # 3. Predictions and Logging
    PREDICTIONS_LOG_FILE = os.path.join(BASE_DIR, 'data', 'predictions_log.csv')
    log_entries = []
    
    # Calculate Target Date (Next US Trading Day)
    today = datetime.now()
    target_date = today.strftime('%Y-%m-%d')
    weekday = today.weekday()
    if weekday == 5: # Saturday -> Monday
         target_date = (today + timedelta(days=2)).strftime('%Y-%m-%d')
    elif weekday == 6: # Sunday -> Monday
         target_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
         
    report_lines.append(f"PREDICTIONS for Trading Session: {target_date} (1-Day Horizon)")
    try:
        # Import prediction functions
        sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
        # Reloading to ensure we get the new function
        import predict_prices
        from importlib import reload
        reload(predict_prices)
        from predict_prices import prepare_data, calculate_bayesian_probability, calculate_expected_range, calculate_rules_and_risk
        
        # Prepare data
        df, latest = prepare_data()
        
        current_conditions = {
            'regime': latest.get('regime', 'Unclassified'),
            'real_yield_bin': latest['real_yield_bin'],
            'dxy_bin': latest['dxy_bin'],
            'vix_bin': latest['vix_bin'],
            'fitch_gold_bin': latest['china_gold_bin'],
            'china_gold_bin': latest['china_gold_bin'],
            'fed_bs_bin': latest['fed_bs_bin'],
            'Gold_vol_bin': latest['Gold_vol_bin'],
            'Silver_vol_bin': latest['Silver_vol_bin'],
            'Gold_rsi_bin': latest['Gold_rsi_bin'],
            'Silver_rsi_bin': latest['Silver_rsi_bin'],
            'Gold_trend_bin': latest['Gold_trend_bin'],
            'Silver_trend_bin': latest['Silver_trend_bin'],
            'Gold_bb_bin': latest['Gold_bb_bin'],
            'Silver_bb_bin': latest['Silver_bb_bin']
        }
        
        # Predictions for Gold and Silver
        for asset in ['Gold', 'Silver']:
            try:
                 signal, metrics, rationale = calculate_rules_and_risk(df, asset, current_conditions)
            except NameError:
                 # Fallback if function not found (e.g. earlier version of predict_prices)
                 p_up, p_down, n_samples, used_vol = calculate_bayesian_probability(df, asset, current_conditions)
                 signal = f"{p_up*100:.0f}% UP"
                 metrics = {'prob_up': p_up, 'samples': n_samples, 'skew_ev': 0, 'tail_risk': 0}
                 rationale = "Legacy Probability Model"

            lower, mean, upper = calculate_expected_range(df, asset, current_conditions, confidence=0.95)
            
            if metrics and mean is not None:
                skew_label = "BULL" if metrics.get('skew_ev', 0) > 0 else "BEAR"
                
                # New Format Line
                report_lines.append(f"{asset}: {signal} | Skew: {metrics.get('skew_ev',0):+.3f}% ({skew_label})")
                report_lines.append(f"  Rationale: {rationale}")
                report_lines.append(f"  Exp Ret: {mean:+.2f}% | Range: {lower:+.2f}% to {upper:+.2f}% | Win Rate: {metrics.get('prob_up',0)*100:.1f}%")
                
                # Display Historical Metrics
                try:
                    import json
                    with open(os.path.join(BASE_DIR, 'data', 'performance_metrics.json'), 'r') as f:
                        metrics = json.load(f)
                        if asset in metrics:
                            m = metrics[asset]
                            report_lines.append(f"  [Hist. Scorecard] Win: {m['Win Rate']} | Payoff: {m['Payoff Ratio']} | Sharpe: {m['Sharpe Ratio']}")
                except Exception:
                    pass
                
                # Prepare Log Entry
                # Extract p_up/p_down from metrics or calculate
                p_up_val = metrics.get('prob_up', 0.5)
                p_down_val = 1.0 - p_up_val
                
                log_entries.append({
                    'date': target_date,
                    'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'asset': asset,
                    'prob_up': round(p_up_val, 4),
                    'prob_down': round(p_down_val, 4),
                    'expected_return': round(mean, 4),
                    'range_lower': round(lower, 4),
                    'range_upper': round(upper, 4),
                    'samples': metrics.get('samples', 0),
                    'model_version': f'regime_v1_{signal.split()[0]}'
                })
            else:
                report_lines.append(f"{asset}: Insufficient data for prediction")
                
        # Save to CSV (Upsert Logic)
        if log_entries:
            df_log = pd.DataFrame(log_entries)
            if os.path.exists(PREDICTIONS_LOG_FILE):
                existing_log = pd.read_csv(PREDICTIONS_LOG_FILE)
                # Append
                combined_log = pd.concat([existing_log, df_log])
                # Deduplicate based on date+asset, keeping LAST (newest)
                combined_log = combined_log.drop_duplicates(subset=['date', 'asset'], keep='last')
                combined_log.to_csv(PREDICTIONS_LOG_FILE, index=False)
            else:
                df_log.to_csv(PREDICTIONS_LOG_FILE, index=False)
            print(f"Predictions saved to {PREDICTIONS_LOG_FILE}")
                
    except Exception as e:
        report_lines.append(f"Prediction error: {e}")
    
    report_lines.append("")

    # --- 6. Volume Profile Analysis ---
    report_lines.append("\nVOLUME PROFILE (Support/Resistance):")
    try:
        from analyze_volume_profile import analyze_volume_profile
        
        # New function returns a dict of results directly
        vp_results = analyze_volume_profile(lookback_days=10)
        
        if vp_results:
            for name, data in vp_results.items():
                report_lines.append(f"{name}:")
                report_lines.append(f"  Price: ${data['current_price']:,.2f} | Bias: {data['bias']}")
                report_lines.append(f"  Resistance (VAH): ${data['vah']:,.2f}")
                report_lines.append(f"  Magnet (POC):     ${data['poc']:,.2f}")
                report_lines.append(f"  Support (VAL):    ${data['val']:,.2f}")
        else:
             report_lines.append("No Volume Profile data available.")

    except ImportError as e:
        report_lines.append(f"Volume Profile Error: {e}")
    except Exception as e:
        report_lines.append(f"Volume Profile Analysis Failed: {e}")

    report_lines.append("")
    
    # 4. Events Today
    report_lines.append("EVENTS TODAY:")
    if os.path.exists(CALENDAR_FILE):
        df_cal = pd.read_csv(CALENDAR_FILE)
        # Filter for today's date (string match)
        today_events = df_cal[df_cal['date'] == today_str]
        
        if not today_events.empty:
            for _, row in today_events.iterrows():
                # Format: Time - Event (Actual / Forecast)
                val_str = f"Act: {row['actual']} / Fcst: {row['forecast']}" if pd.notna(row['actual']) else f"Fcst: {row['forecast']}"
                report_lines.append(f"- {row['time']} {row['currency']} {row['event']} [{val_str}]")
        else:
            report_lines.append("No collected events for today.")
    else:
        report_lines.append("No calendar data found.")
    
    report_lines.append("")
    
    # 5. Important News (Last 24h)
    report_lines.append("IMPORTANT NEWS (Last 24h):")
    if os.path.exists(NEWS_FILE):
        df_news = pd.read_csv(NEWS_FILE)
        # Convert to datetime (feedparser format is messy, but usually parseable)
        # We might need safe parsing. For now, let's just take the last 10 entries if parsing fails,
        # or try pd.to_datetime with coerce.
        
        # Simple approach: just show the top 5 most recent entries from the file (assuming we append)
        # Better: Filter by timestamp if possible.
        
        # Let's take last 5
        recent_news = df_news.tail(5)
        for _, row in recent_news.iterrows():
            report_lines.append(f"- {row['headline']} ({row['source']})")
    else:
        report_lines.append("No news data found.")
        
    report_lines.append("\n" + "-"*30 + "\n")
    
    # Write to file
    final_report = "\n".join(report_lines)
    
    # Append to daily_report.txt as requested ("log each day's report")
    with open(REPORT_FILE, 'a') as f:
        f.write(final_report)
        
    print("Report generated and appended to daily_report.txt")
    print(final_report)

if __name__ == "__main__":
    generate_report()
