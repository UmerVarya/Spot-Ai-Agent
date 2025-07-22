import requests

def get_fear_greed_index():
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, timeout=10)
        data = response.json()
        index_value = int(data['data'][0]['value'])
        return index_value
    except Exception as e:
        print(f"⚠️ Failed to fetch Fear & Greed Index: {e}")
        return 50
