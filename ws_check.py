import asyncio, json
from os import getenv
from websockets import __version__ as WEBSOCKETS_VERSION
from websockets.legacy.client import connect as ws_connect

DEFAULT_STREAM = "wss://stream.binance.com:9443/stream?streams=!miniTicker@arr"
url = getenv("WS_CHECK_URL") or getenv("WS_BRIDGE_URL") or getenv("WS_URL") or DEFAULT_STREAM
IDLE_RECV_TIMEOUT = max(10, int(getenv("WS_IDLE_RECV_TIMEOUT", "90")))


async def main() -> None:
    print(f"Connecting to {url}")
    async with ws_connect(
        url,
        ping_interval=None,
        ping_timeout=None,
        max_queue=None,
        open_timeout=20,
        close_timeout=3,
        extra_headers=[("User-Agent", "Mozilla/5.0"), ("Accept-Encoding", "identity")],
        compression=None,
    ) as ws:
        print(
            "Connected.",
            "websockets",
            WEBSOCKETS_VERSION,
            "connect_func",
            ws_connect.__module__,
        )
        print("Protocol:", ws.__class__.__module__, ws.__class__.__name__)
        try:
            if hasattr(ws, "ping"):
                ws.ping = lambda *a, **k: None
            if hasattr(ws, "pong"):
                ws.pong = lambda *a, **k: None
        except Exception:
            pass
        for i in range(3):
            msg = await asyncio.wait_for(ws.recv(), timeout=IDLE_RECV_TIMEOUT)
            try:
                json.loads(msg)
            except Exception:
                pass
            print("MSG", i + 1, "ok.")
        print("Closed cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
