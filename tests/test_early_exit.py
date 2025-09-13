import pandas as pd
import trade_manager

def test_should_exit_early_weighted(monkeypatch):
    def fake_indicators(df):
        return {
            'rsi': 40,
            'macd': -0.5,
            'ema_20': 95,
            'ema_50': 96,
            'vwap': 97,
            'atr': 2,
        }
    monkeypatch.setattr(trade_manager, 'calculate_indicators', fake_indicators)
    price_data = pd.DataFrame({
        'open': [100, 100],
        'high': [101, 101],
        'low': [99, 97],
        'close': [100, 94],
        'volume': [1000, 1100],
    })
    trade = {'entry': 100, 'direction': 'long'}
    should_exit, reason = trade_manager.should_exit_early(trade, 98, price_data)
    assert should_exit is True
    assert 'score' in reason.lower()


def test_should_exit_early_atr(monkeypatch):
    def fake_indicators(df):
        return {
            'rsi': 55,
            'macd': 0.1,
            'ema_20': 100,
            'ema_50': 101,
            'vwap': 102,
            'atr': 5,
        }
    monkeypatch.setattr(trade_manager, 'calculate_indicators', fake_indicators)
    price_data = pd.DataFrame({
        'open': [100, 100],
        'high': [106, 106],
        'low': [94, 94],
        'close': [100, 105],
        'volume': [1000, 1100],
    })
    trade = {'entry': 100, 'direction': 'long'}
    should_exit, reason = trade_manager.should_exit_early(trade, 94, price_data)
    assert should_exit is True
    assert 'atr' in reason.lower()
