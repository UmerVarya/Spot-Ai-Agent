# Trade Record Format

Completed trades are appended to a unified CSV. By default this file lives at `/home/ubuntu/spot_data/trades/completed_trades.csv`, but the path can be changed with the `TRADE_HISTORY_FILE` environment variable (alias `COMPLETED_TRADES_FILE`).

Each row recorded by `trade_storage.log_trade_result` contains the following columns. All timestamps use ISO 8601 in UTC (e.g. `2025-09-05T09:56:35Z`). The `timestamp` column is when the result was logged and is distinct from the trade's `entry_time` and `exit_time`.

| Column | Description |
| --- | --- |
| `trade_id` | Unique identifier for the trade. May be an exchange ID or generated UUID. |
| `timestamp` | Time the trade result was written to the log. |
| `symbol` | Trading pair, such as `WLFIUSDT`. |
| `direction` | `long` for buys or `short` for sells. |
| `entry_time` | Time the position was opened. |
| `exit_time` | Time the position was closed. |
| `entry` | Entry price for the base asset. |
| `exit` | Exit price for the base asset. |
| `size` | Notional amount in quote currency allocated to the trade. |
| `notional` | Same as `size`. Retained for backward compatibility. |
| `fees` | Total commissions paid on exit. |
| `slippage` | Slippage incurred on exit. |
| `pnl` | Net profit or loss in quote currency after fees and slippage. |
| `pnl_pct` | Profit/loss as a percentage of `notional`. |
| `outcome` | Outcome code (see table below). |
| `outcome_desc` | Human readable description of the outcome. |
| `strategy` | Name of the strategy responsible for the trade. |
| `session` | Market session or timeframe identifier. |
| `confidence` | Confidence score supplied by the strategy. |
| `btc_dominance` | BTC dominance value at entry. |
| `fear_greed` | Fear & Greed index value at entry. |
| `sentiment_bias` | Aggregate sentiment classification. |
| `sentiment_confidence` | Confidence in `sentiment_bias`. |
| `score` | Additional score or strength metric. |
| `pattern` | Detected chart pattern. |
| `narrative` | Free-form narrative explaining the trade. |
| `llm_decision` | Whether the LLM approved the trade. |
| `llm_confidence` | Confidence returned by the LLM. |
| `llm_error` | Indicates the LLM encountered an error. |
| `volatility` | Measured volatility at entry. |
| `htf_trend` | Higher time frame trend assessment. |
| `order_imbalance` | Order flow imbalance metric. |
| `macro_indicator` | Macro indicator value. |

A `duration_min` field is **not** stored. It can be derived as the difference between `exit_time` and `entry_time` if needed.

## Outcome Codes

The `outcome` field uses the following codes:

| Code | Description |
| --- | --- |
| `tp1_partial` | Exited 50% at TP1 |
| `tp2_partial` | Exited additional 30% at TP2 |
| `tp4` | Final Exit (TP4 ride) |
| `tp4_sl` | Stopped out after TP3 |
| `sl` | Stopped Out (SL) |
| `early_exit` | Early Exit |
| `tp1` | Take Profit 1 |
| `tp2` | Take Profit 2 |
| `tp3` | Take Profit 3 |
| `time_exit` | Time-based Exit |

Partial exits are denoted with a `_partial` suffix. Manual closures may appear as `early_exit` in the log.

## Example Header

```
trade_id,timestamp,symbol,direction,entry_time,exit_time,entry,exit,size,notional,fees,slippage,pnl,pnl_pct,outcome,outcome_desc,strategy,session,confidence,btc_dominance,fear_greed,sentiment_bias,sentiment_confidence,score,pattern,narrative,llm_decision,llm_confidence,llm_error,volatility,htf_trend,order_imbalance,macro_indicator
```

A single trade may produce multiple rows if partial take-profits occur. The loader collapses these into one row using `_deduplicate_history` while retaining `tp1_partial`/`tp2_partial` flags.
