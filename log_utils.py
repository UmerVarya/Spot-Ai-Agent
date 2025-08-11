import logging
from logging.handlers import RotatingFileHandler
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), 'spot_ai.log')


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
    # Rotating file handler keeps last 5 logs of ~1MB each
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
