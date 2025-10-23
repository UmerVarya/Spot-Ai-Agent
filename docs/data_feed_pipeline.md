# Data Feed Pipeline: REST vs. WebSocket Responsibilities

The trading agent layers its market data and decision flow so that each stage
uses the transport best suited to its latency and throughput requirements.
This document captures the reference split between REST polling and
WebSocket streaming for every phase of the pipeline.

## 1. Macro & Risk Guardrails — REST / Scheduled Fetches

Macro indicators and risk guardrails change slowly relative to trade
execution. Scheduled REST requests (or cron-like jobs) retrieve drawdown
limits, BTC dominance, curated news, and composite sentiment readings such as
the Fear & Greed index. The results are cached in a shared **risk context** so
all downstream scans operate on the same snapshot. WebSockets would not add
value here because macro data is not tick-level.

## 2. Universe Construction & Signal Caching — Mostly REST with Light Streams

The universe builder relies on REST to pull the initial asset list, liquidity
and volume snapshots, and the one-minute OHLCV history required before live
streaming starts. Once the watchlist is prepared, the agent subscribes to
lightweight `@kline_1m` or `@miniTicker` WebSocket channels so the
`RealTimeSignalCache` stays fresh between the scheduled REST refresh cycles.

## 3. Signal Evaluation & Scoring — Cached REST + WebSocket Updates

Signals are computed from the cached data maintained by the WebSocket feeds.
REST only intervenes on cache misses or to backfill short indicator windows
(typically 30–100 bars). REST establishes the context; WebSockets keep it
current in near-real time while scoring logic executes.

## 4. Decision Stack & Risk Vetting — REST + Internal Logic

The decision stack (LLMs, ML models, and deterministic vetos) consumes the
cached snapshots assembled upstream. No additional WebSocket subscriptions are
required because decisions are made on the consolidated view rather than on
millisecond updates.

## 5. Trade Setup Parameters — REST with Optional WebSocket Preview

Execution planning pulls a depth snapshot via REST (`get_order_book(limit=100)`)
when sizing entries and refining stops or targets. If the strategy needs to
inspect micro-imbalance or spoofing in the final seconds before entry, it can
open a temporary order-book stream and close it immediately after.

## 6. Live Trade Management & Exit Logic — WebSocket Dominant

Once a trade is active the agent switches to WebSocket-first monitoring.
Streams delivering price ticks, order-book deltas, and other high-frequency
signals drive trailing stops, exit logic, and intra-trade adjustments. REST
remains available as a backup fail-safe but is no longer the primary data
source until the position is closed.

## 7. Post-Trade & Logging — REST Wrap-Up

After exiting a trade, REST endpoints provide the official record of balances,
realised PnL, and confirmation receipts. Metrics derived from the live
WebSocket session (e.g., cumulative volume delta, book imbalance trends) are
summarised and stored with the trade record.
