import trade_manager
import trade_utils
from datetime import datetime, timedelta

import pandas as pd
import pytest


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
    trade = {'symbol': 'BTCUSDT', 'position_size': 1, 'size': 100, 'sl': 100, 'tp1': 110, 'status': {'tp1': False}}
    trade_manager._update_stop_loss(trade, 90)
    assert trade['sl'] == 90
    assert trade['sl_order_id'] == '123'
    assert calls['args'] == ('BTCUSDT', 1.0, 90, None, 110)
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


def test_profit_riding_trails_stop_loss(monkeypatch):
    trade = {
        'symbol': 'BTCUSDT',
        'direction': 'long',
        'entry': 100.0,
        'position_size': 1.0,
        'sl': 120.0,
        'tp1': 110.0,
        'tp2': 120.0,
        'tp3': 130.0,
        'status': {'tp1': True, 'tp2': True, 'tp3': True},
        'profit_riding': True,
        'trail_tp_pct': 0.05,
        'next_trail_tp': 150.0,
        'entry_time': datetime.utcnow().isoformat() + 'Z',
    }

    def fake_load():
        return [trade]

    saved = {}

    def fake_save(trades):
        saved['trades'] = trades

    def fake_price_data(symbol):
        now = datetime.utcnow()
        idx = pd.DatetimeIndex([now - timedelta(minutes=1)])
        return pd.DataFrame(
            {'close': [151.0], 'high': [151.0], 'low': [151.0]}, index=idx
        )

    def fake_commission(symbol, quantity, maker):
        return 0.0

    def fake_slippage(price, direction):
        return price

    def fake_calc_indicators(price_data):
        return {'adx': 25, 'macd': 0, 'macd_signal': 0, 'kc_lower': 0, 'atr': 1}

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
    monkeypatch.setattr(trade_manager, 'log_trade_result', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, '_update_stop_loss', fake_update_sl)
    monkeypatch.setattr(trade_manager, '_update_rl', lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, 'send_email', lambda *args, **kwargs: None)

    trade_manager.manage_trades()

    assert sl_calls['new_sl'] == pytest.approx(150.0)
    saved_trade = saved['trades'][0]
    assert saved_trade['sl'] == pytest.approx(150.0)
    assert saved_trade['next_trail_tp'] == pytest.approx(150.0 * 1.05)
