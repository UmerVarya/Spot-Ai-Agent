import trade_manager
import trade_utils


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
