import requests

def get_btc_dominance():
    try:
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url, timeout=10)
        data = response.json()
        btc_dominance = data['data']['market_cap_percentage']['btc']
        return round(btc_dominance, 2)
    except Exception as e:
        print(f"⚠️ Failed to fetch BTC dominance: {e}")
        return 50.0
