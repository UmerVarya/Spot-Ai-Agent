import trade_manager
import trade_utils
from datetime import datetime, timedelta

import pandas as pd
import pytest

from trade_constants import (
    TRAIL_INITIAL_ATR,
    TRAIL_TIGHT_ATR,
    TRAIL_LOCK_IN_RATIO,
    TP1_TRAILING_ONLY_STRATEGY,
)


def test_update_stop_loss_calls_util(monkeypatch):
    calls = {}
    emails = {}

    def fake_update(symbol, qty, price, existing, tp):
        calls['args'] = (symbol, qty, price, existing, tp)
        return '123'

    def fake_email(subject, details):
        emails['args'] = (subject, details)

    monkeypatch.setattr(trade_manager, 'update_stop_loss_order', fake_update)
    monkeypatch.setattr(trade_manager, 'send_email', fake_email)
    trade = {
        'symbol': 'BTCUSDT',
        'position_size': 1,
        'size': 100,
        'sl': 100,
        'tp1': 110,
        'status': {'tp1': False},
        'take_profit_strategy': 'atr_trailing',
        'trailing_active': False,
    }
    trade_manager._update_stop_loss(trade, 90)
    assert trade['sl'] == 90
    assert trade['sl_order_id'] == '123'
    assert calls['args'] == ('BTCUSDT', 1.0, 90, None, None)
    assert emails['args'][0] == 'SL Updated: BTCUSDT'
    assert emails['args'][1]['new_sl'] == 90


def test_update_stop_loss_order_places_and_cancels(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.cancel_args = None
            self.oco_kwargs = None

        def cancel_oco_order(self, symbol, orderListId):
            self.cancel_args = (symbol, orderListId)

        def create_oco_order(self, **kwargs):
            self.oco_kwargs = kwargs
            return {'orderListId': 789}

    dummy = DummyClient()
    monkeypatch.setattr(trade_utils, 'client', dummy)
    order_id = trade_utils.update_stop_loss_order(
        'BTCUSDT', 0.5, 25000.0, existing_order_id=1, take_profit_price=30000.0
    )
    assert order_id == 789
    assert dummy.cancel_args == ('BTCUSDT', 1)
    assert dummy.oco_kwargs['symbol'] == 'BTCUSDT'
    assert dummy.oco_kwargs['price'] == 30000.0
    assert dummy.oco_kwargs['stopPrice'] == 25000.0


def test_trailing_activation_moves_stop_to_lock_price(monkeypatch):
    now = datetime.utcnow()
    entry = 100.0
    tp1 = 115.0
    atr_value = 5.0
    lock_price = entry + (tp1 - entry) * TRAIL_LOCK_IN_RATIO
    trade = {
        'symbol': 'BTCUSDT',
        'direction': 'long',
        'entry': entry,
        'position_size': 1.0,
        'sl': entry - 10.0,
        'tp1': tp1,
        'tp2': 120.0,
        'tp3': 130.0,
        'status': {'tp1': False, 'tp2': False, 'tp3': False},
        'take_profit_strategy': 'atr_trailing',
        'trailing_active': False,
        'atr_at_entry': atr_value,
        'entry_time': (now - timedelta(minutes=5)).isoformat(),
    }

    def fake_load():
        return [trade]

    saved = {}

    def fake_save(trades):
        saved['trades'] = trades

    def fake_price_data(symbol):
        idx = pd.DatetimeIndex([now - timedelta(minutes=1)])
        return pd.DataFrame(
            {'close': [116.0], 'high': [116.0], 'low': [112.0]}, index=idx
        )

    def fake_commission(symbol, quantity, maker):
        return 0.0

    def fake_slippage(price, direction):
        return price

    def fake_calc_indicators(price_data):
        return {
            'adx': 25,
            'macd': 0.4,
            'macd_signal': 0.1,
            'kc_lower': 0,
            'atr': atr_value,
        }

    def fake_macro():
        return {'bias': 'neutral', 'confidence': 0}

    sl_calls = {}

    def fake_update_sl(tr, new_sl):
        sl_calls['new_sl'] = new_sl
        tr['sl'] = new_sl

    monkeypatch.setattr(trade_manager, 'load_active_trades', fake_load)
    monkeypatch.setattr(trade_manager, 'save_active_trades', fake_save)
    monkeypatch.setattr(trade_manager, 'get_price_data', fake_price_data)
    monkeypatch.setattr(trade_manager, 'estimate_commission', fake_commission)
    monkeypatch.setattr(trade_manager, 'simulate_slippage', fake_slippage)
    monkeypatch.setattr(trade_manager, 'calculate_indicators', fake_calc_indicators)
    monkeypatch.setattr(trade_manager, 'analyze_macro_sentiment', fake_macro)
    monkeypatch.setattr(trade_manager, '_update_stop_loss', fake_update_sl)
    monkeypatch.setattr(trade_manager, '_update_rl', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, 'log_trade_result', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, 'send_email', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, 'get_order_book', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, 'plan_execution', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, 'detect_sell_pressure', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, 'detect_aggression', lambda *args, **kwargs: None)

    trade_manager.manage_trades()

    expected_lock = pytest.approx(lock_price, rel=1e-6)
    assert sl_calls['new_sl'] == expected_lock
    saved_trade = saved['trades'][0]
    assert saved_trade['sl'] == expected_lock
    assert saved_trade['trailing_active'] is True
    assert saved_trade['trail_multiplier'] == pytest.approx(TRAIL_INITIAL_ATR)


def test_tp2_threshold_tightens_trailing_multiplier(monkeypatch):
    now = datetime.utcnow()
    entry = 100.0
    tp1 = 115.0
    tp2 = 120.0
    atr_value = 5.0
    lock_price = entry + (tp1 - entry) * TRAIL_LOCK_IN_RATIO
    trade = {
        'symbol': 'BTCUSDT',
        'direction': 'long',
        'entry': entry,
        'position_size': 0.5,
        'initial_size': 0.5,
        'sl': lock_price,
        'tp1': tp1,
        'tp2': tp2,
        'tp3': 130.0,
        'status': {'tp1': True, 'tp2': False, 'tp3': False},
        'take_profit_strategy': 'atr_trailing',
        'trailing_active': True,
        'trail_multiplier': TRAIL_INITIAL_ATR,
        'trail_high': tp1,
        'locked_profit_price': lock_price,
        'atr_at_entry': atr_value,
        'entry_time': (now - timedelta(minutes=15)).isoformat(),
    }

    def fake_load():
        return [trade]

    saved = {}

    def fake_save(trades):
        saved['trades'] = trades

    def fake_price_data(symbol):
        idx = pd.DatetimeIndex([now - timedelta(minutes=1)])
        return pd.DataFrame(
            {'close': [121.0], 'high': [121.0], 'low': [118.0]}, index=idx
        )

    def fake_commission(symbol, quantity, maker):
        return 0.0

    def fake_slippage(price, direction):
        return price

    def fake_calc_indicators(price_data):
        return {
            'adx': 28,
            'macd': 0.5,
            'macd_signal': 0.2,
            'kc_lower': 0,
            'atr': atr_value,
        }

    def fake_macro():
        return {'bias': 'neutral', 'confidence': 0}

    sl_calls = {}

    def fake_update_sl(tr, new_sl):
        sl_calls['new_sl'] = new_sl
        tr['sl'] = new_sl

    monkeypatch.setattr(trade_manager, 'load_active_trades', fake_load)
    monkeypatch.setattr(trade_manager, 'save_active_trades', fake_save)
    monkeypatch.setattr(trade_manager, 'get_price_data', fake_price_data)
    monkeypatch.setattr(trade_manager, 'estimate_commission', fake_commission)
    monkeypatch.setattr(trade_manager, 'simulate_slippage', fake_slippage)
    monkeypatch.setattr(trade_manager, 'calculate_indicators', fake_calc_indicators)
    monkeypatch.setattr(trade_manager, 'analyze_macro_sentiment', fake_macro)
    monkeypatch.setattr(trade_manager, '_update_stop_loss', fake_update_sl)
    monkeypatch.setattr(trade_manager, '_update_rl', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, 'log_trade_result', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, 'send_email', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, 'get_order_book', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, 'plan_execution', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, 'detect_sell_pressure', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, 'detect_aggression', lambda *args, **kwargs: None)

    trade_manager.manage_trades()

    saved_trade = saved['trades'][0]
    assert saved_trade['status']['tp2'] is True
    assert saved_trade['trail_multiplier'] == pytest.approx(TRAIL_TIGHT_ATR)


def test_tp1_trailing_only_activation_and_ratcheting(monkeypatch):
    trade = {
        'symbol': 'BTCUSDT',
        'entry': 100.0,
        'sl': 95.0,
        'tp1_price': 105.0,
        'tp1_triggered': False,
        'trail_mode': False,
        'max_price': 100.0,
        'take_profit_strategy': TP1_TRAILING_ONLY_STRATEGY,
    }
    sl_updates = []

    def fake_update_sl(tr, new_sl):
        sl_updates.append(new_sl)
        tr['sl'] = new_sl

    monkeypatch.setattr(trade_manager, '_update_stop_loss', fake_update_sl)
    armed = trade_manager._tp1_trailing_only_activate(
        trade,
        tp_price=105.0,
        current_price=105.0,
    )
    assert armed == pytest.approx(102.5)
    assert trade['trail_mode'] is True
    assert trade['tp1_triggered'] is True

    sl_move = trade_manager._tp1_trailing_only_update(trade, current_price=108.0)
    assert sl_move == pytest.approx(107.2)
    assert trade['max_price'] == pytest.approx(108.0)
    assert sl_updates == [pytest.approx(102.5), pytest.approx(107.2)]
