# Trade Record Format

Completed trades are appended to a unified CSV. By default this file lives at `/home/ubuntu/spot_data/trades/historical_trades.csv`, but the path can be changed with the `TRADE_HISTORY_FILE` environment variable (alias `COMPLETED_TRADES_FILE`).

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
| `size` | Position size. By default this is a notional amount in quote currency, but if the `SIZE_AS_NOTIONAL` environment variable is set to `false` it instead represents the asset quantity. |
| `notional` | Dollar value committed to the trade. When `SIZE_AS_NOTIONAL` is `true`, this is identical to `size`; otherwise it is calculated as `entry * size`. |
| `fees` | Total commissions paid on exit. |
| `slippage` | Slippage incurred on exit. |
| `pnl` | Net profit or loss in quote currency after fees and slippage. |
| `pnl_pct` | Profit/loss as a percentage of `notional`. |
| `outcome` | Outcome code (see table below). |
| `outcome_desc` | Human readable description of the outcome. |
| `exit_reason` | Explanation of how or why the trade closed (e.g. TP1 hit, trailing stop). |
| `strategy` | Name of the strategy responsible for the trade. |
| `session` | Market session or timeframe identifier. |
| `confidence` | Confidence score supplied by the strategy. Missing values are stored as `N/A`. |
| `btc_dominance` | BTC dominance value at entry (or `N/A` when unavailable). |
| `fear_greed` | Fear & Greed index value at entry. |
| `sentiment_bias` | Aggregate sentiment classification. |
| `sentiment_confidence` | Confidence in `sentiment_bias`. |
| `score` | Additional score or strength metric. |
| `technical_indicator_score` | Normalised 0–10 score summarising core technical indicators. |
| `pattern` | Detected chart pattern. |
| `narrative` | Free-form narrative explaining the trade. |
| `llm_decision` | Textual advice or rationale returned by the LLM advisor. |
| `llm_approval` | Boolean flag indicating if the LLM approved the setup. |
| `llm_confidence` | Numeric confidence score returned by the LLM advisor. |
| `llm_error` | Indicates the LLM encountered an error. |
| `volatility` | Measured volatility at entry. |
| `htf_trend` | Higher time frame trend assessment. |
| `order_imbalance` | Order flow imbalance metric. |
| `order_flow_score` | Composite [-1, 1] order-flow score combining microstructure features. |
| `order_flow_flag` | Legacy ±1/0 directional flag derived from the score (kept for compatibility). |
| `order_flow_state` | Textual classification of order-flow pressure (e.g. `buyers in control`). |
| `cvd` | Normalised cumulative volume delta observed over the recent window. |
| `cvd_change` | Short-term change in cumulative volume delta (last bar vs. average). |
| `cvd_divergence` | Signed divergence score where positive values indicate CVD strength without matching price highs. |
| `cvd_absorption` | Magnitude of sell-side absorption (CVD higher high while price fails to break out). |
| `cvd_accumulation` | Magnitude of buy-side accumulation (CVD lower low while price holds support). |
| `taker_buy_ratio` | Normalised taker-buy to total volume ratio. |
| `trade_imbalance` | Net taker flow imbalance for the latest bar. |
| `aggressive_trade_rate` | Normalised rate of aggressive trades relative to recent history. |
| `spoofing_intensity` | Spoofing detection score derived from order-book withdrawals. |
| `spoofing_alert` | Binary flag (0/1) when spoofing intensity crosses the alert threshold. |
| `volume_ratio` | Recent volume ratio versus the 5-bar average (bounded to [-1, 1]). |
| `price_change_pct` | Percentage price change between the last close and the reference open. |
| `spread_bps` | Bid/ask spread at decision time expressed in basis points. |
| `macro_indicator` | Macro indicator value. |
| `tp1_partial` | `true` when a TP1 partial exit occurred; otherwise `false`. |
| `tp2_partial` | `true` when a TP2 partial exit occurred; otherwise `false`. |
| `pnl_tp1` | Profit or loss realised at the TP1 partial exit. |
| `pnl_tp2` | Profit or loss realised at the TP2 partial exit. |
| `size_tp1` | Size closed at the TP1 partial exit. |
| `size_tp2` | Size closed at the TP2 partial exit. |
| `notional_tp1` | Notional value closed at the TP1 partial exit. |
| `notional_tp2` | Notional value closed at the TP2 partial exit. |
| `auction_state` | Auction regime classification (e.g. `balanced`, `out_of_balance_trend`). |
| `volume_profile_leg_type` | Indicates whether the profile was derived from an impulse or reclaim leg. |
| `volume_profile_poc` | Point of control price from the captured volume profile. |
| `volume_profile_lvns` | JSON array of low-volume node (LVN) prices. |
| `volume_profile_bin_width` | Price step used when constructing the histogram bins. |
| `volume_profile_snapshot` | JSON blob containing the full volume-profile summary (POC, LVNs, histogram bins, metadata). |
| `lvn_entry_level` | LVN level that triggered the entry (when available). |
| `lvn_stop` | Stop level derived from the LVN structure (when available). |
| `poc_target` | Target price derived from the POC or reclaim objective. |
| `orderflow_state_detail` | Detailed order-flow state captured at entry. |
| `orderflow_features` | JSON blob of the order-flow feature vector (CVD, imbalance metrics, spoofing intensity, etc.). |
| `orderflow_snapshot` | Combined JSON snapshot mirroring the in-memory `orderflow_analysis` structure (state + features). |

The Streamlit dashboard surfaces the stage-specific `size_tp*`,
`notional_tp*` and `pnl_tp*` columns so each partial take-profit's
contribution is visible when present.

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
trade_id,timestamp,symbol,direction,entry_time,exit_time,entry,exit,size,notional,fees,slippage,pnl,pnl_pct,outcome,outcome_desc,exit_reason,strategy,session,confidence,btc_dominance,fear_greed,sentiment_bias,sentiment_confidence,score,pattern,narrative,llm_decision,llm_approval,llm_confidence,llm_error,technical_indicator_score,volatility,htf_trend,order_imbalance,order_flow_score,order_flow_flag,order_flow_state,cvd,cvd_change,taker_buy_ratio,trade_imbalance,aggressive_trade_rate,spoofing_intensity,spoofing_alert,volume_ratio,price_change_pct,spread_bps,macro_indicator,tp1_partial,tp2_partial,pnl_tp1,pnl_tp2,size_tp1,size_tp2,notional_tp1,notional_tp2,auction_state,volume_profile_leg_type,volume_profile_poc,volume_profile_lvns,volume_profile_bin_width,volume_profile_snapshot,lvn_entry_level,lvn_stop,poc_target,orderflow_state_detail,orderflow_features,orderflow_snapshot
```

A single trade may produce multiple rows if partial take-profits occur. Rows are grouped by `trade_id` (falling back to `entry_time`, `symbol` and `strategy` when absent) and collapsed into one summary row by `_deduplicate_history`. PnL, size and notional values are summed for the whole trade, while per-stage fields such as `pnl_tp1`, `pnl_tp2`, `size_tp1`, `size_tp2`, `notional_tp1` and `notional_tp2` detail the contribution of each partial exit alongside the `tp1_partial`/`tp2_partial` flags.

## Legacy files

Logs created before the unified schema lacked the canonical header row, which caused downstream tools to mislabel fields (for example the dashboard rendering numeric IDs under the **Symbol** column). When `trade_storage.log_trade_result` encounters one of these legacy files it now moves it aside as `historical_trades.csv.legacy-<timestamp>` and starts a fresh log with the correct header. Likewise, `load_trade_history_df` ignores CSVs whose first line does not match the canonical header to prevent misaligned data from reaching the dashboard.
