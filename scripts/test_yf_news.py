
import yfinance as yf
import json

def test_news():
    tickers = ["GC=F", "SI=F", "^GSPC"] # Gold, Silver, S&P 500
    for ticker in tickers:
        print(f"\n--- Output for {ticker} ---")
        try:
            t = yf.Ticker(ticker)
            news = t.news
            if news:
                print(f"Found {len(news)} news items.")
                print(json.dumps(news[0], indent=2))
            else:
                print("No news found.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_news()
