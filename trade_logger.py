"""
Logger for final trade outcomes (deprecated).

This module now delegates trade result logging to ``trade_storage``.
``log_trade_result`` is provided only for backward compatibility and will
emit a ``DeprecationWarning`` when used.  All paths resolve to the unified
completed trades log managed by ``trade_storage``.
"""

import os
import warnings
from log_utils import ensure_symlink
from trade_storage import (
    COMPLETED_TRADES_FILE,
    log_trade_result as _storage_log_trade_result,
)


TRADE_LEARNING_LOG_FILE = COMPLETED_TRADES_FILE
TRADE_LOG_FILE = COMPLETED_TRADES_FILE

module_dir = os.path.dirname(os.path.abspath(__file__))
ensure_symlink(TRADE_LEARNING_LOG_FILE, os.path.join(module_dir, "trade_learning_log.csv"))
ensure_symlink(TRADE_LEARNING_LOG_FILE, os.path.join(module_dir, "trade_logs.csv"))


def log_trade_result(*args, **kwargs):
    """Deprecated wrapper for compatibility.

    Calls :func:`trade_storage.log_trade_result` after emitting a
    ``DeprecationWarning``.  New code should import and use
    ``log_trade_result`` directly from ``trade_storage``.
    """

    warnings.warn(
        "trade_logger.log_trade_result is deprecated; import from trade_storage instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return _storage_log_trade_result(*args, **kwargs)
