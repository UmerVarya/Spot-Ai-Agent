import asyncio, json, os
from websockets.legacy.client import connect as ws_connect

DEFAULT_STREAM = "wss://stream.binance.com:9443/stream?streams=!miniTicker@arr"
url = os.getenv("WS_CHECK_URL") or os.getenv("WS_BRIDGE_URL") or os.getenv("WS_URL") or DEFAULT_STREAM


async def main() -> None:
    print(f"Connecting to {url}")
    async with ws_connect(
        url,
        ping_interval=None,
        ping_timeout=None,
        max_queue=None,
        open_timeout=20,
        close_timeout=3,
        extra_headers=[("User-Agent","Mozilla/5.0"), ("Accept-Encoding","identity")],
        compression=None,
    ) as ws:
        try:
            if hasattr(ws, "ping"):
                ws.ping = lambda *a, **k: None
            if hasattr(ws, "pong"):
                ws.pong = lambda *a, **k: None
        except Exception:
            pass
        print("Connected.")
        for i in range(3):
            msg = await ws.recv()
            try:
                json.loads(msg)
            except Exception:
                pass
            print("MSG", i + 1, "ok.")
        print("Closed cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
