CRYPTO_PANIC_RSS = "https://cryptopanic.com/news/rss"
COINDESK_RSS = "https://www.coindesk.com/arc/outboundfeeds/rss/"
FOREX_FACTORY_RSS = "https://www.forexfactory.com/calendar.php?week=this&day=this&month=this&rss=1"

import requests
import feedparser
from datetime import datetime
from log_utils import setup_logger

# === Method 1: Use CryptoPanic RSS (Free RSS feed)
CRYPTO_PANIC_RSS = "https://cryptopanic.com/news/rss/"

# === Method 2: Use CoinDesk RSS
COINDESK_RSS = "https://feeds.feedburner.com/CoinDesk"

# === Method 3: Use ForexFactory RSS for macroeconomic events
FOREX_FACTORY_RSS = "https://www.forexfactory.com/rss.php"


logger = setup_logger(__name__)


def fetch_headlines_from_rss(url, limit=10):
    try:
        feed = feedparser.parse(url)
        entries = feed.entries[:limit]
        headlines = [f"{entry.title}: {entry.link}" for entry in entries]
        return headlines
    except Exception as e:
        logger.warning("Failed to fetch RSS: %s", e, exc_info=True)
        return []


def get_combined_headlines():
    headlines = []
    headlines += fetch_headlines_from_rss(CRYPTO_PANIC_RSS)
    headlines += fetch_headlines_from_rss(COINDESK_RSS)
    headlines += fetch_headlines_from_rss(FOREX_FACTORY_RSS)
    return headlines[:15]


if __name__ == "__main__":
    headlines = get_combined_headlines()
    logger.info("Latest News Headlines (Auto-Fetched):")
    for h in headlines:
        logger.info("- %s", h)
