import os
import pandas as pd

WARMUP_BARS = int(os.getenv("RTSC_REST_WARMUP_BARS", "300"))


def _to_df(raw):
    if not raw:
        return pd.DataFrame(
            columns=[
                "open_time",
                "close_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base",
                "taker_buy_quote",
                "taker_sell_base",
                "taker_sell_quote",
            ]
        )
    df = pd.DataFrame(
        raw,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in [
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["taker_sell_base"] = (df["volume"].fillna(0.0) - df["taker_buy_base"]).clip(lower=0.0)
    df["taker_sell_quote"] = (
        df["quote_asset_volume"] - df["taker_buy_quote"]
    ).clip(lower=0.0)

    return df[
        [
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "taker_sell_base",
            "taker_sell_quote",
        ]
    ].dropna()


def rest_backfill_klines(binance, symbol, interval="1m", bars=WARMUP_BARS):
    raw = binance.get_klines(symbol=symbol, interval=interval, limit=bars)
    return _to_df(raw)


def rest_fetch_latest_closed(binance, symbol, interval="1m"):
    raw = binance.get_klines(symbol=symbol, interval=interval, limit=2)
    return _to_df(raw)
