import pandas as pd
import aiohttp

BINANCE_BASE = "https://api.binance.com"
INTERVAL = "1m"      # change if you prefer
LIMIT = 200          # ~200 minutes of candles

async def fetch_candles_async(symbol: str) -> pd.DataFrame | None:
    """Fetch OHLCV candles from Binance's REST API.

    This helper issues a HTTP GET against ``/api/v3/klines`` (the standard
    REST endpoint rather than the websocket feed) and returns the result as a
    pandas DataFrame sorted by timestamp with open/high/low/close/volume
    columns. REST polling is generally simpler to operate when you only need
    periodic candle snapshots, while maintaining a websocket stream is more
    beneficial for ultra-low-latency or tick-level strategies that demand a
    continuous push of updates.
    """
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": INTERVAL, "limit": LIMIT}
    timeout = aiohttp.ClientTimeout(total=15)

    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.get(url, params=params) as r:
            if r.status != 200:
                # You can add logging here if you like
                return None
            data = await r.json()

    if not data:
        return None

    # Binance kline schema:
    # [ openTime, open, high, low, close, volume, closeTime, quoteAssetVolume,
    #   trades, takerBuyBase, takerBuyQuote, ignore ]
    df = pd.DataFrame(
        data,
        columns=[
            "open_time","open","high","low","close","volume","close_time",
            "qav","trades","taker_base","taker_quote","ignore"
        ],
    )

    # Cast numeric columns
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build timestamp and set as DatetimeIndex (required for resample)
    df["timestamp"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    # If your pipeline expects tz-naive, uncomment:
    # df.index = df.index.tz_localize(None)

    return df[["open","high","low","close","volume"]]
