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
- `DATA_DIR` – directory for persistent trade logs and `spot_ai.log`
- `DATABASE_URL` – optional PostgreSQL connection string for trade storage
- `RUN_DASHBOARD` – set to `1` to launch the Streamlit dashboard from the agent

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
The agent logs activity and uncaught exceptions to `spot_ai.log` in `DATA_DIR` for later analysis.

## Architecture

```
Trader (agent.py) --> Trade Storage (JSON/DB) <-- Dashboard (dashboard.py)
                               \
                                --> Backtester (backtest.py)
```

## Render deployment

The included `render.yaml` defines two Render services:

- **spot-ai-trader** – a background worker that runs the trading loop.
- **spot-ai-dashboard** – a web service exposing the Streamlit dashboard on the `$PORT` Render provides.

Deploying them separately keeps the trading loop running even if the dashboard crashes and avoids port health check issues.

## Testing

Unit tests use `pytest`:

```bash
pytest
```
