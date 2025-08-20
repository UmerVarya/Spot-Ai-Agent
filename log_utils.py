import logging
from logging.handlers import RotatingFileHandler
import os

# Use the actual data path rather than relying on a potentially restricted
# symlink. This avoids PermissionError in hardened environments and ensures
# the agent reads and writes logs directly from the data storage location.
LOG_FILE = "/home/ubuntu/spot_data/logs/spot_ai.log"


def _ensure_symlink(target: str, link: str) -> None:
    try:
        if os.path.islink(link) or os.path.exists(link):
            return
        os.symlink(target, link)
    except OSError:
        pass


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_ensure_symlink(LOG_FILE, os.path.join(_REPO_ROOT, "spot_ai.log"))


def setup_logger(name: str) -> logging.Logger:
    """Configure and return a module-level logger.

    Logs are written to both console and a rotating file to persist
    information for debugging. Subsequent calls with the same name
    return the already configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Ensure the log directory exists before creating the handler
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    # Rotating file handler keeps last 5 logs of ~1MB each
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def read_logs(tail: int = 100) -> str:
    """Return the last ``tail`` lines from the log file.

    This helper enables agents or diagnostic tools to ingest recent
    log output for learning or analysis.  If the log file does not
    exist, an empty string is returned.

    Parameters
    ----------
    tail : int, optional
        The number of lines from the end of the log to return. Defaults
        to 100.

    Returns
    -------
    str
        The concatenated log lines.
    """
    if not os.path.exists(LOG_FILE):
        return ""
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    if tail <= 0:
        return "".join(lines)
    return "".join(lines[-tail:])
