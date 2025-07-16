import pandas as pd

def detect_candlestick_patterns(df):
    patterns = {
        "hammer": False,
        "inverted_hammer": False,
        "bullish_engulfing": False,
        "morning_star": False,
        "piercing_line": False,
        "three_white_soldiers": False,
        "tweezer_bottom": False,
        "bullish_harami": False,
        "dragonfly_doji": False,
        "marubozu_bullish": False,
        "rising_three_method": False,
        "gap_up_bullish": False,
        # Optional bearish filters (for rejection/skip logic)
        "shooting_star": False,
        "bearish_engulfing": False,
        "evening_star": False,
        "three_black_crows": False
    }

    try:
        df = df.copy()
        df['open'] = pd.to_numeric(df['open'])
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])

        c = df.iloc[-1]
        p1 = df.iloc[-2]
        p2 = df.iloc[-3] if len(df) >= 3 else None

        body = abs(c['close'] - c['open'])
        range_ = c['high'] - c['low']
        upper_wick = c['high'] - max(c['close'], c['open'])
        lower_wick = min(c['close'], c['open']) - c['low']

        # Bullish patterns
        if lower_wick > 2 * body and body > 0:
            patterns["hammer"] = True

        if upper_wick > 2 * body and body > 0:
            patterns["inverted_hammer"] = True

        if p1['close'] < p1['open'] and c['close'] > c['open'] and c['close'] > p1['open'] and c['open'] < p1['close']:
            patterns["bullish_engulfing"] = True

        if p2 is not None:
            if p2['close'] < p2['open'] and p1['close'] < p1['open'] and c['close'] > c['open'] and c['close'] > p1['open']:
                patterns["morning_star"] = True

        if p1['close'] < p1['open'] and c['close'] > (p1['open'] + p1['close']) / 2:
            patterns["piercing_line"] = True

        if len(df) >= 4:
            last_3 = df.iloc[-3:]
            if all(x['close'] > x['open'] for _, x in last_3.iterrows()) and \
               last_3.iloc[1]['close'] > last_3.iloc[0]['close'] and \
               last_3.iloc[2]['close'] > last_3.iloc[1]['close']:
                patterns["three_white_soldiers"] = True

        if abs(c['low'] - p1['low']) < 0.0001 and c['close'] > c['open'] and p1['close'] < p1['open']:
            patterns["tweezer_bottom"] = True

        if p1['close'] < p1['open'] and c['open'] > p1['close'] and c['close'] < p1['open']:
            patterns["bullish_harami"] = True

        if upper_wick == 0 and lower_wick > 2 * body:
            patterns["dragonfly_doji"] = True

        if upper_wick == 0 and lower_wick == 0 and body / range_ > 0.95:
            patterns["marubozu_bullish"] = True

        if len(df) >= 6:
            first = df.iloc[-5]
            mid = df.iloc[-4:-1]
            last = df.iloc[-1]
            if first['close'] > first['open'] and last['close'] > last['open']:
                if all(x['close'] < x['open'] for _, x in mid.iterrows()):
                    patterns["rising_three_method"] = True

        if p1['close'] < p1['open'] and c['open'] > p1['close'] and c['close'] > c['open']:
            patterns["gap_up_bullish"] = True

        # Optional: Bearish patterns for filter
        if upper_wick > 2 * body and body > 0:
            patterns["shooting_star"] = True

        if p1['close'] > p1['open'] and c['close'] < c['open'] and c['open'] > p1['close'] and c['close'] < p1['open']:
            patterns["bearish_engulfing"] = True

        if p2 is not None:
            if p2['close'] > p2['open'] and p1['close'] > p1['open'] and c['close'] < c['open'] and c['close'] < p1['open']:
                patterns["evening_star"] = True

        if len(df) >= 4:
            last_3 = df.iloc[-3:]
            if all(x['close'] < x['open'] for _, x in last_3.iterrows()) and \
               last_3.iloc[1]['close'] < last_3.iloc[0]['close'] and \
               last_3.iloc[2]['close'] < last_3.iloc[1]['close']:
                patterns["three_black_crows"] = True

        return patterns

    except Exception as e:
        print(f"⚠️ Candlestick detection error: {e}")
        return patterns
