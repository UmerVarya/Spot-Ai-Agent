import math
import textwrap

import pytest

from trade_utils import compute_performance_metrics


def test_compute_performance_metrics_handles_full_header(tmp_path):
    """Metrics should be computed when the CSV already includes a header."""

    content = textwrap.dedent(
        """
        trade_id,timestamp,symbol,direction,entry_time,exit_time,entry,exit,size,notional,fees,slippage,pnl,pnl_pct,win,outcome,outcome_desc,exit_reason,strategy,session,confidence,btc_dominance,fear_greed,sentiment_bias,sentiment_confidence,score,pattern,narrative,llm_decision,llm_approval,llm_confidence,llm_error,technical_indicator_score,volatility,htf_trend,order_imbalance,macro_indicator
        t1,2024-01-01T00:00:00Z,BTCUSDT,long,2024-01-01T00:00:00Z,2024-01-01T01:00:00Z,100,110,100,100,0,0,10,10,true,tp1,Take Profit 1,TP1 hit,core,asia,0.7,50,40,bullish,0.8,1,pattern,story,approved,true,0.9,false,6.5,0.1,1,0.2,0.3
        t2,2024-01-02T00:00:00Z,BTCUSDT,short,2024-01-02T00:00:00Z,2024-01-02T01:00:00Z,200,180,200,200,0,0,20,10,true,tp1,Take Profit 1,TP1 hit,core,asia,0.7,50,40,bearish,0.8,1,pattern,story,approved,true,0.9,false,6.0,0.1,1,0.2,0.3
        t3,2024-01-03T00:00:00Z,BTCUSDT,long,2024-01-03T00:00:00Z,2024-01-03T01:00:00Z,50,47.5,50,50,0,0,-2.5,-5,false,sl,Stopped Out,Stop loss hit,core,asia,0.7,50,40,neutral,0.8,1,pattern,story,vetoed,false,0.9,false,5.2,0.1,1,0.2,0.3
        """
    ).strip()
    path = tmp_path / "history.csv"
    path.write_text(content)

    perf = compute_performance_metrics(log_file=str(path), lookback=10)

    assert perf
    assert perf["max_drawdown"] == pytest.approx(-0.05, rel=1e-6)
    assert perf["var"] == pytest.approx(-0.05, rel=1e-6)
    assert math.isnan(perf["es"])


def test_compute_performance_metrics_handles_entry_exit_fallback(tmp_path):
    """When percentage columns are absent we fall back to entry/exit prices."""

    content = textwrap.dedent(
        """
        trade_id,timestamp,symbol,direction,entry,exit,size,outcome
        a1,2024-01-01T00:00:00Z,BTCUSDT,long,100,110,100,tp1
        a2,2024-01-02T00:00:00Z,BTCUSDT,short,200,180,200,tp1
        a3,2024-01-03T00:00:00Z,BTCUSDT,long,50,47.5,50,sl
        """
    ).strip()
    path = tmp_path / "minimal.csv"
    path.write_text(content)

    perf = compute_performance_metrics(log_file=str(path), lookback=10)

    assert perf
    assert perf["max_drawdown"] == pytest.approx(-0.05, rel=1e-6)
    assert perf["var"] == pytest.approx(-0.05, rel=1e-6)
    assert math.isnan(perf["es"])
