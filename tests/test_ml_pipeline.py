import numpy as np
import pandas as pd
import pytest

import ml_model
import sequence_model
import rl_policy
import trade_storage


def _build_trade_history(rows: int = 80) -> pd.DataFrame:
    base_time = pd.Timestamp('2024-01-01')
    records = []
    sessions = ['Asia', 'Europe', 'US']
    for i in range(rows):
        entry = base_time + pd.Timedelta(hours=i)
        exit_time = entry + pd.Timedelta(hours=4)
        outcome = 'tp1' if i % 2 else 'loss'
        records.append({
            'entry_time': entry.isoformat(),
            'exit_time': exit_time.isoformat(),
            'score': 55 + (i % 10),
            'confidence': 5 + (i % 4),
            'session': sessions[i % len(sessions)],
            'btc_dominance': 45 + (i % 5),
            'fear_greed': 40 + (i % 10),
            'sentiment_confidence': 6 + (i % 3),
            'sentiment_bias': 'bullish' if i % 3 else 'bearish',
            'pattern': f'pattern_{i % 5}',
            'volatility': 18 + (i % 6),
            'htf_trend': 4 + (i % 3),
            'order_imbalance': (-1) ** i * 5,
            'macro_indicator': 2 + (i % 2),
            'macd': 1.2 + (i % 3) * 0.1,
            'rsi': 55 + (i % 5),
            'sma': 1.1 + (i % 4) * 0.05,
            'atr': 0.8 + (i % 4) * 0.02,
            'volume': 1_000_000 + i * 1500,
            'order_flow_score': 0.2 * ((-1) ** i),
            'order_flow_flag': -1 if i % 4 == 0 else 1,
            'cvd': 0.15 * ((-1) ** i),
            'cvd_change': 0.1 * ((-1) ** (i + 1)),
            'cvd_divergence': 0.05 * ((-1) ** i),
            'cvd_absorption': 0.2 if i % 3 == 0 else 0.0,
            'cvd_accumulation': 0.15 if i % 4 == 0 else 0.05,
            'taker_buy_ratio': 0.3 * ((-1) ** i),
            'trade_imbalance': 0.25 * ((-1) ** (i + 1)),
            'aggressive_trade_rate': 0.05 * ((-1) ** i),
            'spoofing_intensity': 0.1 * ((-1) ** i),
            'spoofing_alert': 1 if i % 7 == 0 else 0,
            'volume_ratio': 0.05 * ((-1) ** i),
            'price_change_pct': 0.02 * ((-1) ** (i + 1)),
            'spread_bps': 12 + (i % 5),
            'llm_approval': bool(i % 3),
            'llm_confidence': 7 + (i % 2),
            'outcome': outcome,
            'btc_dominance_change': 0.2,
            'htf_bias': 'bull',
            'fear_greed_rolling': 50 + (i % 4),
            'sentiment_score': 0.1 * ((-1) ** i),
        })
    return pd.DataFrame(records)


def test_train_model_generates_metrics(tmp_path, monkeypatch):
    history_df = _build_trade_history()
    history_path = tmp_path / 'history.csv'
    history_df.to_csv(history_path, index=False)

    monkeypatch.setattr(ml_model, 'LOG_FILE', str(history_path))
    monkeypatch.setattr(ml_model, 'MODEL_JSON', str(tmp_path / 'ml_model.json'))
    monkeypatch.setattr(ml_model, 'MODEL_PKL', str(tmp_path / 'ml_model.pkl'))
    monkeypatch.setattr(trade_storage, 'TRADE_HISTORY_FILE', str(history_path))

    diagnostics = ml_model.train_model()
    assert diagnostics is not None
    if ml_model.SKLEARN_AVAILABLE:
        assert diagnostics.validation_metrics
    else:
        assert diagnostics.model_type == 'manual'
    report = ml_model.load_model_report()
    if ml_model.SKLEARN_AVAILABLE:
        assert report.get('validation_metrics')
    else:
        assert report.get('model_type') == 'manual'
    prob = ml_model.predict_success_probability(
        score=60,
        confidence=7,
        session='US',
        btc_d=52,
        fg=55,
        sentiment_conf=7,
        pattern='pattern_1',
        llm_approval=True,
        llm_confidence=8,
        feature_overrides={'volatility': 0.25},
        micro_features={
            'order_imbalance': 4.0,
            'order_flow_score': 0.3,
            'order_flow_flag': 1.0,
            'cvd': 0.1,
            'cvd_change': -0.05,
            'cvd_divergence': 0.12,
            'cvd_absorption': 0.0,
            'cvd_accumulation': 0.3,
            'taker_buy_ratio': 0.2,
            'trade_imbalance': -0.1,
            'aggressive_trade_rate': 0.15,
            'spoofing_intensity': 0.05,
            'spoofing_alert': 0.0,
            'volume_ratio': 0.1,
            'price_change_pct': 0.03,
            'spread_bps': 15.0,
            'macro_indicator': 20.0,
            'htf_trend': 6.0,
            'volatility': 22.0,
            'sent_bias': 'bullish',
        },
    )
    assert 0.0 <= prob <= 1.0


def test_sequence_model_training(tmp_path, monkeypatch):
    rows = 160
    ts = pd.date_range('2024-01-01', periods=rows, freq='H')
    df = pd.DataFrame({
        'timestamp': ts,
        'close': np.linspace(100, 120, rows) + np.sin(np.linspace(0, 6, rows)),
        'feature_a': np.cos(np.linspace(0, 3, rows)),
        'feature_b': np.linspace(0.1, 0.9, rows),
    })
    monkeypatch.setattr(sequence_model, 'SEQ_PKL', str(tmp_path / 'sequence.pkl'))

    diagnostics = sequence_model.train_sequence_model(df, window_size=8)
    if not sequence_model.SKLEARN_AVAILABLE:
        pytest.skip("scikit-learn not available for sequence model test")
    assert diagnostics is not None
    assert diagnostics.metrics
    artefact = sequence_model._load_sequence_model()
    assert artefact is not None
    assert artefact['metrics']
    report = sequence_model.load_sequence_model_report()
    assert report.get('model_type')
    window = df.tail(8)
    pred = sequence_model.predict_next_return(window)
    assert isinstance(pred, float)


def test_rl_policy_adaptive_behaviour(tmp_path, monkeypatch):
    monkeypatch.setattr(rl_policy, 'POLICY_FILE', str(tmp_path / 'rl_policy.json'))
    sizer = rl_policy.RLPositionSizer(epsilon=0.5, epsilon_decay=0.8, min_epsilon=0.1)
    initial_epsilon = sizer.epsilon
    state = 'winning_state'
    action = sizer.select_multiplier(state)
    sizer.update(state, action, reward=3.0)
    assert sizer.state_action_visits[(state, action)] == 1
    assert sizer.epsilon <= initial_epsilon
    summary = sizer.policy_summary()
    assert state in summary
