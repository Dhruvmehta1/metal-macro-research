
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime, timedelta
import feedparser

# Configuration
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'calendar.csv')

# Alternative: Use FXStreet Economic Calendar RSS or MarketWatch
# Since Investing.com requires JS, we'll use a simpler approach:
# 1. Try FXStreet RSS
# 2. Or create a manual list of known upcoming events (user can update)

CALENDAR_RSS = "https://www.fxstreet.com/feeds/calendar"

# Keywords to filter
IMPORTANT_EVENTS = [
    'GDP', 'CPI', 'Core PCE', 'FOMC', 'NFP', 'Nonfarm Payrolls', 
    'Retail Sales', 'Personal Income', 'Personal Spending',
    'Fed', 'Rate', 'Decision', 'Minutes', 'Employment', 'Inflation',
    'PMI', 'ISM', 'Jobless Claims', 'PPI'
]

def fetch_calendar():
    print(f"[{datetime.now()}] Fetching economic calendar...")
    
    events_data = []
    
    # Method 1: Try FXStreet RSS
    try:
        print("Trying FXStreet calendar RSS...")
        feed = feedparser.parse(CALENDAR_RSS)
        
        for entry in feed.entries:
            title = entry.title
            published = getattr(entry, 'published', str(datetime.now()))
            link = entry.link
            
            # Check if relevant
            is_relevant = any(kw.lower() in title.lower() for kw in IMPORTANT_EVENTS)
            
            if is_relevant:
                # Parse date from published
                try:
                    pub_date = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %z")
                    date_str = pub_date.strftime("%Y-%m-%d")
                    time_str = pub_date.strftime("%H:%M")
                except:
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    time_str = ""
                
                events_data.append({
                    'date': date_str,
                    'time': time_str,
                    'currency': 'USD',  # Assume USD for most macro events
                    'event': title,
                    'actual': '',
                    'forecast': '',
                    'previous': ''
                })
        
        print(f"Found {len(events_data)} events from RSS")
        
    except Exception as e:
        print(f"RSS fetch failed: {e}")
    
    # Method 2: Fallback - Create a manual upcoming events list
    # This is a simple fallback that users can manually update
    if not events_data:
        print("Using fallback: checking for known weekly events...")
        today = datetime.now()
        
        # Common weekly events (user should update this manually or we can scrape from Fed calendar)
        weekly_events = []
        
        # Example: NFP is first Friday of month
        # FOMC meetings are scheduled (can be hardcoded for the year)
        # For now, just create a placeholder
        
        events_data.append({
            'date': today.strftime("%Y-%m-%d"),
            'time': '',
            'currency': 'USD',
            'event': 'Check Federal Reserve calendar manually',
            'actual': '',
            'forecast': '',
            'previous': ''
        })

    # DataFrame
    new_df = pd.DataFrame(events_data)
    
    if new_df.empty:
        print("No events found.")
        return

    # Save/Append
    if os.path.exists(DATA_FILE):
        try:
            existing_df = pd.read_csv(DATA_FILE)
            combined_df = pd.concat([existing_df, new_df])
        except:
            combined_df = new_df
    else:
        combined_df = new_df
        
    combined_df = combined_df.drop_duplicates(subset=['date', 'event'])
    combined_df = combined_df.sort_values(['date', 'time'])
    
    combined_df.to_csv(DATA_FILE, index=False)
    print(f"✓ Saved {len(combined_df)} events to {DATA_FILE}")

if __name__ == "__main__":
    fetch_calendar()
