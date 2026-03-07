
import pandas as pd
import os
from datetime import datetime
import time
import ssl
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf

# Fix SSL issue for Mac
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

# Initialize VADER
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')
    
sia = SentimentIntensityAnalyzer()

# Configuration
ASSETS = {
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'Economy': '^GSPC',  # S&P 500 as proxy for general market/economy news
    'Dollar': 'DX=F'
}

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'news.csv')

def fetch_news():
    print(f"[{datetime.now()}] Fetching news via yfinance...")
    
    all_news = []
    
    for asset_name, ticker in ASSETS.items():
        try:
            print(f"Fetching news for {asset_name} ({ticker})...")
            t = yf.Ticker(ticker)
            news_items = t.news
            
            if not news_items:
                print(f"  No news found for {asset_name}")
                continue
                
            for item in news_items:
                # yfinance news structure:
                # {'uuid': '...', 'title': '...', 'publisher': '...', 'link': '...', 'providerPublishTime': ...}
                
                title = item.get('title', '')
                # Some providers don't give a summary, so we use title mainly. 
                # If there's a 'summary' field (rare in new API), use it.
                # Note: yfinance `news` result structure varies slightly but 'title' is reliable.
                
                link = item.get('link', '')
                provider_publish_time = item.get('providerPublishTime', time.time())
                published_dt = datetime.fromtimestamp(provider_publish_time)
                
                # Calculate Sentiment (Compound Score: -1 to 1)
                score = sia.polarity_scores(title)['compound']
                
                all_news.append({
                    'timestamp': published_dt,
                    'date': published_dt.date(),
                    'asset': asset_name,
                    'source': item.get('publisher', 'Yahoo'),
                    'headline': title,
                    'link': link,
                    'sentiment': score
                })
                
        except Exception as e:
            print(f"Error fetching {asset_name}: {e}")

    # DataFrame
    new_df = pd.DataFrame(all_news)
    
    if new_df.empty:
        print("No news found.")
        return

    # Load Existing to Avoid Duplicates
    if os.path.exists(DATA_FILE):
        try:
            existing_df = pd.read_csv(DATA_FILE)
            # Ensure timestamp is datetime
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
            existing_df['date'] = existing_df['timestamp'].dt.date
            
            combined_df = pd.concat([existing_df, new_df])
        except Exception as e:
            print(f"Error reading existing file, starting fresh: {e}")
            combined_df = new_df
    else:
        combined_df = new_df
    
    # Dedup by headline and asset (same news might appear for multiple assets, that's fine, but not duplicates for same asset)
    # Actually simpler to dedup by headline globally? No, "Market Crash" might be relevant for both Gold and Silver separately for filtering.
    combined_df = combined_df.drop_duplicates(subset=['headline', 'asset'])
    
    # Sort by time
    combined_df = combined_df.sort_values(by='timestamp', ascending=False)
    
    # Keep only last 30 days of news to prevent bloat
    # combined_df = combined_df.head(500) 
    
    combined_df.to_csv(DATA_FILE, index=False)
    print(f"Saved {len(combined_df)} news items to {DATA_FILE}")

if __name__ == "__main__":
    fetch_news()
