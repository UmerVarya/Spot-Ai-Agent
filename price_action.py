import numpy as np
import pandas as pd

def detect_support_resistance_zones(df, window=20, tolerance=0.003):
    support_zones = []
    resistance_zones = []

    for i in range(window, len(df) - window):
        low = df['low'][i]
        high = df['high'][i]

        is_support = True
        is_resistance = True

        for j in range(i - window, i + window):
            if df['low'][j] < low:
                is_support = False
            if df['high'][j] > high:
                is_resistance = False

        if is_support:
            support_zones.append(low)
        if is_resistance:
            resistance_zones.append(high)

    # Remove duplicates using tolerance
    def clean_zones(zones):
        zones = list(zones)
        zones.sort()
        cleaned = []
        for zone in zones:
            if not cleaned or abs(zone - cleaned[-1]) > tolerance * zone:
                cleaned.append(zone)
        return cleaned

    return {
        "support": clean_zones(support_zones),
        "resistance": clean_zones(resistance_zones)
    }

def is_price_near_zone(price, zones_dict, zone_type='support', proximity=0.005):
    zones = zones_dict.get(zone_type, [])
    # Ensure zones is a list of floats
    if isinstance(zones, (pd.Series, np.ndarray)):
        zones = zones.tolist()
    for zone in zones:
        if abs(price - zone) / price < proximity:
            return True
    return False