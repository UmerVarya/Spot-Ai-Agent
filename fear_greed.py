try:  # pragma: no cover - optional dependency
    import requests
except ModuleNotFoundError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
from log_utils import setup_logger

logger = setup_logger(__name__)

def get_fear_greed_index():
    if requests is None:
        return 50
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        index_value = int(data['data'][0]['value'])
        return index_value
    except Exception as e:
        logger.warning("Failed to fetch Fear & Greed Index: %s", e, exc_info=True)
        return 50
