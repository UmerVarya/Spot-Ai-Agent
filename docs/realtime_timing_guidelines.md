# Real-time timing guidelines

Operational timing around the Binance WebSocket feeds influences both signal
quality and the stability of the agent.  The notes below capture the current
best practices baked into the runtime so future changes preserve the same
behaviour.

## Treat Binance's close flag as canonical

- The trading loop waits for the `kline` payload's close flag (`payload.get("x")`)
  before scheduling a fresh technical analysis refresh.  This ensures the
  candle boundaries line up with Binance server time instead of any local clock
  skew, and the final close event is treated as the authoritative bar finish.
  The callback wiring that enforces this lives in `agent.py` where the
  WebSocket bridge hands closed bars to the real-time cache refresher.

## Keep the WebSocket heartbeat healthy

- `WSPriceBridge` already issues heartbeats and exponential backoff reconnects.
  It uses the `websockets` client's built-in ping/pong (15 second interval and
  timeout) and doubles the reconnect delay up to 30 seconds when remote errors
  occur.  Leave these defaults intact when evolving the bridge so the stream is
  resilient to transient connectivity issues without overwhelming Binance with
  retries.

## Debounce expensive signal recomputations

- Mini ticker and book ticker updates can arrive several times per second; they
  are routed to the trade manager to check stop-loss and take-profit triggers in
  near real time.  The heavier technical indicator refresh only runs on the
  closed-kline events above, keeping CPU usage in check and preventing noisy
  intra-bar ticks from thrashing the cache.  When introducing new intrabar
  consumers, follow the same pattern—treat frequent updates as position
  management inputs, and keep TA refreshes gated on the `kline` close signal.

