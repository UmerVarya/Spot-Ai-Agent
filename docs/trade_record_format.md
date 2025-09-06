# Trade Record Format

This repository stores each completed trade as one row in `completed_trades.csv`.
The file is expected to be well-structured and contain the key fields listed
below.  All timestamps use ISO 8601 in UTC (e.g. `2025-09-05T09:56:35Z`).

| Field | Description |
| --- | --- |
| `trade_id` | Unique identifier for the trade. May be an exchange ID or generated UUID. |
| `symbol` | Trading pair, such as `WLFIUSDT`. |
| `direction` | `long` for buys or `short` for sells. |
| `entry_time` | Time the position was opened. |
| `exit_time` | Time the position was closed. |
| `entry` | Entry price for the base asset. |
| `exit` | Exit price for the base asset. |
| `size` | Quantity of the base asset traded. Must be positive. |
| `notional` | Total value committed in quote currency (`entry` × `size`). |
| `pnl` | Net profit or loss in quote currency after fees and slippage. |
| `pnl_pct` | Profit/loss as a percentage of `notional`. |
| `outcome` | Short outcome code such as `tp1`, `tp2`, `sl`, `manual`. |
| `outcome_desc` | Human readable description of the outcome. |
| `duration_min` | Derived field measuring minutes between `entry_time` and `exit_time`. |
| `strategy` | Name of the strategy responsible for the trade. |

A clean CSV header for a trade log might look like:

```
trade_id,symbol,direction,entry_time,exit_time,entry,exit,size,notional,pnl,pnl_pct,outcome,outcome_desc,duration_min,strategy
```

Additional contextual fields (such as session or confidence scores) may appear
in the log, but they should remain well named and consistently populated.
