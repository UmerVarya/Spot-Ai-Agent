import logging
import re


class GlobalQuietFilter(logging.Filter):
    """Drops all repetitive Binance/RTSC/market spam, keeps warmup, errors, and trade logs."""
    DROP_PATTERNS = [
        r"REST mirror", r"REST fetch OK", r"REST cache update",
        r"ENTER _refresh", r"Refreshing symbol", r"market_queue_drop",
        r"submitted to bg loop", r"runner\(", r"api\.binance\.com",
    ]
    KEEP_PATTERNS = [r"RTSC warm", r"ERROR", r"CRITICAL", r"trade", r"scan", r"PnL"]

    def filter(self, record):
        msg = record.getMessage()
        # Keep warmup/errors/trades
        if any(re.search(p, msg) for p in self.KEEP_PATTERNS):
            return True
        # Drop spam
        if any(re.search(p, msg) for p in self.DROP_PATTERNS):
            return False
        return True


# Apply to root logger so it covers all submodules
root = logging.getLogger()
if not any(isinstance(f, GlobalQuietFilter) for f in root.filters):
    root.addFilter(GlobalQuietFilter())
