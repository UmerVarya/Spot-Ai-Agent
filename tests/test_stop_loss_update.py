import trade_manager
import trade_utils


def test_update_stop_loss_calls_util(monkeypatch):
    calls = {}

    def fake_update(symbol, qty, price, existing):
        calls['args'] = (symbol, qty, price, existing)
        return '123'

    monkeypatch.setattr(trade_manager, 'update_stop_loss_order', fake_update)
    trade = {'symbol': 'BTCUSDT', 'size': 1, 'sl': 100}
    trade_manager._update_stop_loss(trade, 90)
    assert trade['sl'] == 90
    assert trade['sl_order_id'] == '123'
    assert calls['args'] == ('BTCUSDT', 1.0, 90, None)


def test_update_stop_loss_order_places_and_cancels(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.cancel_args = None
            self.create_kwargs = None

        def cancel_order(self, symbol, orderId):
            self.cancel_args = (symbol, orderId)

        def create_order(self, **kwargs):
            self.create_kwargs = kwargs
            return {'orderId': 456}

    dummy = DummyClient()
    monkeypatch.setattr(trade_utils, 'client', dummy)
    order_id = trade_utils.update_stop_loss_order('BTCUSDT', 0.5, 25000.0, existing_order_id=1)
    assert order_id == 456
    assert dummy.cancel_args == ('BTCUSDT', 1)
    assert dummy.create_kwargs['symbol'] == 'BTCUSDT'
    assert dummy.create_kwargs['stopPrice'] == 25000.0
