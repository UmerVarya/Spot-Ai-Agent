import copy

import pytest

import trade_manager


def test_execute_exit_trade_pnl_and_slippage(monkeypatch):
    trade = {
        'symbol': 'BTCUSDT',
        'direction': 'long',
        'entry': 100.0,
        'position_size': 1.0,
        'size': 100.0,
        'entry_time': '2024-01-01T00:00:00Z',
    }

    monkeypatch.setattr(trade_manager, 'estimate_commission', lambda *_, **__: 0.001)
    monkeypatch.setattr(trade_manager, 'simulate_slippage', lambda price, direction=None: price - 0.05)

    recorded = {}

    def fake_log_trade_result(trade_payload, **kwargs):
        recorded['trade'] = copy.deepcopy(trade_payload)
        recorded['fees'] = kwargs['fees']
        recorded['slippage'] = kwargs['slippage']

    monkeypatch.setattr(trade_manager, 'log_trade_result', fake_log_trade_result)

    qty, total_fees, total_slippage = trade_manager.execute_exit_trade(
        trade,
        exit_price=99.0,
        reason='test',
        outcome='manual_exit',
        quantity=1.0,
        exit_time='2024-01-01T00:10:00Z',
    )

    assert qty == pytest.approx(1.0)
    assert total_slippage == pytest.approx(0.05)
    expected_fees = (99.0 - 0.05) * 0.001
    assert total_fees == pytest.approx(expected_fees)

    gross_expected = (99.0 - 100.0) * 1.0
    net_expected = gross_expected - expected_fees - total_slippage

    assert trade['gross_pnl'] == pytest.approx(gross_expected)
    assert trade['net_pnl'] == pytest.approx(net_expected)
    assert recorded['trade']['gross_pnl'] == pytest.approx(gross_expected)
    assert recorded['trade']['net_pnl'] == pytest.approx(net_expected)
    assert recorded['fees'] == pytest.approx(total_fees)
    assert recorded['slippage'] == pytest.approx(total_slippage)
