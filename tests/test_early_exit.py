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


def test_atr_guard_requires_min_drawdown(monkeypatch):
    monkeypatch.setattr(trade_manager, "_ATR_TIMEFRAME_WARNING_EMITTED", False)

    def fake_indicators(df):
        return {
            'atr': 120,
            'rsi': 55,
            'macd': 0.1,
            'ema_20': 1,
            'ema_50': 1,
            'vwap': 1,
        }

    monkeypatch.setattr(trade_manager, 'calculate_indicators', fake_indicators)
    price_data = pd.DataFrame({
        'open': [91000, 91020],
        'high': [91050, 91080],
        'low': [90900, 90890],
        'close': [91010, 91030],
        'volume': [100, 110],
    })
    trade = {'entry': 91000, 'direction': 'long', 'symbol': 'BTCUSDT'}
    should_exit, reason = trade_manager.should_exit_early(trade, 90890, price_data)
    assert should_exit is False
    assert reason is None


def test_atr_guard_triggers_on_percentage(monkeypatch):
    monkeypatch.setattr(trade_manager, "_ATR_TIMEFRAME_WARNING_EMITTED", False)

    def fake_indicators(df):
        return {
            'atr': 0.004,
            'rsi': 50,
            'macd': 0.0,
            'ema_20': 1,
            'ema_50': 1,
            'vwap': 1,
        }

    monkeypatch.setattr(trade_manager, 'calculate_indicators', fake_indicators)
    price_data = pd.DataFrame({
        'open': [1.0, 1.0],
        'high': [1.01, 1.01],
        'low': [0.995, 0.995],
        'close': [1.0, 0.998],
        'volume': [1000, 1100],
    })
    trade = {'entry': 1.0, 'direction': 'long', 'symbol': 'ALTUSDT'}
    should_exit, reason = trade_manager.should_exit_early(trade, 0.995, price_data)
    assert should_exit is True
    assert 'atr drawdown' in reason.lower()
