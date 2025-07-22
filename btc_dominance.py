import requests

_last_dominance = None

def get_btc_dominance():
    global _last_dominance
    try:
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url, timeout=10)
        data = response.json()
        btc_dominance = data['data']['market_cap_percentage']['btc']
        _last_dominance = round(btc_dominance, 2)
        return _last_dominance
    except Exception as e:
        print(f"⚠️ Failed to fetch BTC dominance: {e}")
        if _last_dominance is not None:
            print("⚠️ Using cached BTC dominance value.")
            return _last_dominance
        return 50.0
