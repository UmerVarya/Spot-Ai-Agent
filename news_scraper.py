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


def get_combined_headlines(limit: int | None = 60):
    """Return a blended list of crypto and macro headlines.

    Args:
        limit: Maximum number of headlines to return. ``None`` disables
            truncation and returns the full corpus collected.

    Returns:
        List of headline strings ordered by the order they were fetched.
    """

    headlines = []
    headlines += fetch_headlines_from_rss(CRYPTO_PANIC_RSS, limit=limit or 50)
    headlines += fetch_headlines_from_rss(COINDESK_RSS, limit=limit or 50)
    headlines += fetch_headlines_from_rss(FOREX_FACTORY_RSS, limit=limit or 50)

    if limit is None:
        return headlines

    return headlines[:limit]


if __name__ == "__main__":
    headlines = get_combined_headlines()
    logger.info("Latest News Headlines (Auto-Fetched):")
    for h in headlines:
        logger.info("- %s", h)
