# Spot AI Agent

This repository implements the Spot AI Super Agent – a research bot for scanning the crypto market, evaluating signals and managing paper trades. The project now supports asynchronous price fetching, optional dashboard service and basic backtesting utilities.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the following environment variables as needed:

- `BINANCE_API_KEY` / `BINANCE_API_SECRET`
- `DATA_DIR` – optional override for trade storage. Defaults to
  `/home/ubuntu/spot_data/trades`.
- `RUN_DASHBOARD` – set to `1` to launch the Streamlit dashboard from the agent,
  or `0` to rely on the separate `spot-ai-dashboard` service

## Sentiment Fusion

Headline sentiment is computed using a two-model stack:

* **FinBERT** quickly converts individual headlines into class probabilities
  (positive/neutral/negative).  The expected value of these probabilities is
  mapped to ``s_fb \in [-1, 1]`` with confidence ``c_fb``.
* **FinLlama** aggregates the headlines into a discrete signal ``s_fl \in
  \{-1,0,1\}`` (bearish/neutral/bullish) with confidence ``c_fl`` and a short
  rationale.

The fused score ``0.55*s_fb + 0.45*s_fl`` is considered bullish above ``+0.15``
and bearish below ``-0.15``; otherwise the outlook is neutral.  This fused
sentiment is passed to the Groq LLM for macro context and final arbitration.

### Persistent data

All runtime data is written to real directories under
`/home/ubuntu/spot_data` so nothing relies on writable symlinks.  The key
files are:

```
/home/ubuntu/spot_data/logs/spot_ai.log          # agent log output
/home/ubuntu/spot_data/trades/active_trades.json # open positions
/home/ubuntu/spot_data/trades/completed_trades.csv # unified trade log
/home/ubuntu/spot_data/trades/rejected_trades.csv
```

Symlinks back into the repository are created only for read-only
compatibility with existing tools, but all writes happen directly in the
`spot_data` tree.

Systemd unit files for the agent and dashboard are provided in the
`systemd/` directory.  Both services explicitly grant write access to
`/home/ubuntu/spot_data` via `ReadWritePaths` so historical trade data
persists across restarts.

## Running

**Trading loop**

```bash
python agent.py
```

**Dashboard only**

```bash
streamlit run dashboard.py
```

**Backtester**

```bash
python backtest.py
```

Upload a CSV of trade logs on the dashboard's *Backtest* tab to visualise equity curves and return distributions.

## Architecture

```
Trader (agent.py) --> Trade Storage (JSON/DB) <-- Dashboard (dashboard.py)
                               \
                                --> Backtester (backtest.py)
```

## Testing

Unit tests use `pytest`:

```bash
pytest
```
