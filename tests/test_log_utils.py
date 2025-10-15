import logging
from importlib import reload

import log_utils


def _reset_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def test_setup_logger_filters_ping_frame(tmp_path, monkeypatch):
    # Ensure a clean module state for the logger initialisation.
    module = reload(log_utils)
    monkeypatch.setattr(module, "LOG_FILE", str(tmp_path / "spot_ai.log"), raising=False)

    logger = module.setup_logger("test_log_utils_filter")

    # ``setup_logger`` should attach the shared filter to both handlers.
    for handler in logger.handlers:
        assert module._NOISE_FILTER in handler.filters

    normal_record = logger.makeRecord(
        logger.name, logging.INFO, __file__, 0, "Normal log entry", (), None
    )
    noisy_record = logger.makeRecord(
        logger.name,
        logging.WARNING,
        __file__,
        0,
        "Websocket reconnect requested: Sending ping frame",
        (),
        None,
    )

    assert module._NOISE_FILTER.filter(normal_record) is True
    assert module._NOISE_FILTER.filter(noisy_record) is False

    _reset_logger(logger)
