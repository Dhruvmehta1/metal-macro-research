# Metals Macro System

Automated data collection system for Gold & Silver trading decisions with 7-year historical database.

## Quick Start

### First Time Setup (One-Time Only)
Initialize the 7-year historical database:
```bash
python3 scripts/initialize_historical_data.py
```
This will take 30-60 seconds and download ~15,000 records (7 years of daily data for 6 assets).

### Daily Usage
After initialization, run this daily to fetch only new data:
```bash
python3 scripts/run_all.py
```

### View the Daily Report
```bash
tail -20 data/daily_report.txt
```

## System Overview

This system:
- **Historical Data**: 7 years of daily price data for all assets
- **Daily Updates**: Fetches only new data (last 3 days) to stay current
- **Economic Calendar**: Tracks major events (GDP, CPI, FOMC, NFP, etc.)
- **Macro News**: Filters news by keywords (fed, inflation, gold, etc.)
- **Daily Reports**: Generates summary with 24h changes, events, and news

## Folder Structure
```
metal-macro-research/
├── data/
│   ├── prices.csv          # 7-year historical + daily updates (~15k records)
│   ├── calendar.csv        # Economic events
│   ├── news.csv            # Filtered news headlines
│   └── daily_report.txt    # Daily summary reports (appended)
│
├── scripts/
│   ├── initialize_historical_data.py  # ONE-TIME: Fetch 7 years of data
│   ├── fetch_prices.py                # Daily: Fetch new data only
│   ├── fetch_calendar.py              # Daily: Economic calendar
│   ├── fetch_news.py                  # Daily: News aggregation
│   ├── generate_report.py             # Daily: Generate summary
│   └── run_all.py                     # Master script (run daily)
│
└── README.md
```

## Data Sources

### Market Data (Yahoo Finance)
| Asset | Ticker | Description |
|-------|--------|-------------|
| Gold | GC=F | Gold Futures |
| Silver | SI=F | Silver Futures |
| US10Y | ^TNX | 10-Year Treasury Yield |
| DXY | DX-Y.NYB | US Dollar Index |
| SP500 | ^GSPC | S&P 500 Index |
| VIX | ^VIX | Volatility Index |

**Historical**: 7 years of daily data  
**Updates**: Daily (fetches last 3 days to catch any gaps)

### Economic Calendar
**Keywords**: GDP, CPI, Core PCE, FOMC, NFP, Retail Sales, Personal Income/Spending, Fed, Rate Decision, Employment, Inflation

**Note**: Calendar uses RSS feeds. For comprehensive data, manually check:
- [Federal Reserve Calendar](https://www.federalreserve.gov/newsevents/calendar.htm)
- [Investing.com Economic Calendar](https://www.investing.com/economic-calendar/)

### News Sources
**RSS Feeds**: Investing.com (Commodities, Forex, Economy), Reuters (Business, Top News)

**Keywords**: fed, rate, yield, inflation, gold, silver, central bank, china economy, war, sanctions, solar, mining

## Scripts

### 1. initialize_historical_data.py (ONE-TIME)
**Run once** to build the 7-year historical database.

```bash
python3 scripts/initialize_historical_data.py
```

- Fetches 7 years of **daily** data for all 6 assets
- Creates ~15,000 records (2,500 per asset)
- Takes 30-60 seconds
- **WARNING**: Overwrites existing prices.csv

### 2. fetch_prices.py (DAILY)
Fetches only **new data** and appends to historical database.

```bash
python3 scripts/fetch_prices.py
```

- Checks latest timestamp in database
- Fetches last 3 days (to catch any gaps)
- Deduplicates and appends
- Fast (~5 seconds)

### 3. run_all.py (DAILY)
Master script that runs everything.

```bash
python3 scripts/run_all.py
```

Executes in order:
1. `fetch_prices.py` - Update price database
2. `fetch_calendar.py` - Get economic events
3. `fetch_news.py` - Aggregate news
4. `generate_report.py` - Generate daily summary

## Sample Report Output
```
=== Daily Report: 2026-01-19 ===

MARKET SNAPSHOT (24h Change):
US10Y: +1.71%
DXY: -0.25%
Gold: +1.52%
Silver: +3.24%
SP500: -0.08%
VIX: +21.00%

EVENTS TODAY:
- Check Federal Reserve calendar manually

IMPORTANT NEWS (Last 24h):
- JPM favors AngloGold, Fresnillo
- BofA sees lower NZD/USD on USD resilience

------------------------------
```

## Automation

### Daily Cron Job (macOS/Linux)
Run automatically every day at 9 AM:

```bash
crontab -e
```

Add:
```
0 9 * * * python3 scripts/run_all.py >> logs/cron.log 2>&1
```

## Data Management

### Database Size
- **Initial**: ~15,000 records (7 years daily)
- **Growth**: +6 records per day (1 per asset)
- **Annual growth**: ~2,200 records
- **File size**: ~500 KB (CSV)

### Rebuilding Historical Data
If you need to rebuild from scratch:

```bash
# Backup existing data
mv data/prices.csv data/prices_backup.csv

# Reinitialize
python3 scripts/initialize_historical_data.py
```

## Customization

### Add More Assets
Edit `ASSETS` in both `initialize_historical_data.py` and `fetch_prices.py`:
```python
ASSETS = {
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'Copper': 'HG=F',  # Add copper
    'Platinum': 'PL=F',  # Add platinum
}
```

Then re-run initialization.

### Change Historical Period
Edit `initialize_historical_data.py`:
```python
# Change from 7y to 10y
data = yf.download(tickers_str, period="10y", interval="1d", ...)
```

### Add More News Sources
Edit `RSS_FEEDS` in `fetch_news.py`:
```python
RSS_FEEDS = [
    "https://www.investing.com/rss/news_25.rss",
    "https://your-custom-feed.com/rss",
]
```

## Troubleshooting

### "No historical data found"
Run the initialization script first:
```bash
python3 scripts/initialize_historical_data.py
```

### Rate Limiting
If you see `YFRateLimitError`, wait 10-15 minutes. The system uses batch downloads to minimize this.

### Missing Data for Specific Days
The system automatically fetches the last 3 days on each run to catch any gaps from weekends/holidays.

### Percentage Changes Look Wrong
The system compares to the last available trading price (handles weekends/holidays). If markets were closed, it compares to Friday's close.

## Dependencies
All installed:
- yfinance (v1.0)
- pandas
- beautifulsoup4
- feedparser
- requests
- lxml
- cloudscraper

## Next Steps
1. **Initialize database** (one-time): Run `initialize_historical_data.py`
2. **Set up automation**: Add cron job for daily 9 AM execution
3. **Customize keywords**: Tune news/calendar filters for your needs
4. **Review daily**: Check `daily_report.txt` each morning

## Support
- [yfinance documentation](https://github.com/ranaroussi/yfinance)
- [pandas documentation](https://pandas.pydata.org/)
