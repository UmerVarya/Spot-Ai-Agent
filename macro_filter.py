try:  # pragma: no cover - optional dependency
    import requests
except ModuleNotFoundError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

def get_btc_dominance():
    if requests is None:
        return 50.0  # fallback average value when HTTP client unavailable
    try:
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        btc_dominance = data['data']['market_cap_percentage']['btc']
        return round(btc_dominance, 2)
    except Exception:
        return 50.0  # fallback average value

def get_fear_greed_index():
    if requests is None:
        return 50  # fallback average value when HTTP client unavailable
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        index_value = int(data['data'][0]['value'])
        return index_value
    except Exception:
        return 50  # fallback average value

def get_macro_context():
    btc_d = get_btc_dominance()
    fg_index = get_fear_greed_index()

    # Interpretation logic (can be enhanced over time)
    macro_sentiment = "neutral"
    if btc_d > 52 or fg_index < 30:
        macro_sentiment = "risk_off"
    elif btc_d < 48 and fg_index > 60:
        macro_sentiment = "risk_on"

    return {
        "btc_dominance": btc_d,
        "fear_greed": fg_index,
        "macro_sentiment": macro_sentiment
    }
