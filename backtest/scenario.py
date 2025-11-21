from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .engine import BacktestConfig, ResearchBacktester


def run_fee_slippage_scenarios(
    backtester: ResearchBacktester,
    base_config: BacktestConfig,
    fee_bps_values: Iterable[float],
    slippage_bps_values: Iterable[float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for fee, slip in itertools.product(fee_bps_values, slippage_bps_values):
        cfg = BacktestConfig(**{**base_config.__dict__, "fee_bps": fee, "slippage_bps": slip})
        result = backtester.run(cfg)
        rows.append(
            {
                "fee_bps": fee,
                "slippage_bps": slip,
                "sharpe": result.metrics.sharpe,
                "calmar": result.metrics.calmar,
                "total_return": result.metrics.total_return,
                "max_drawdown": result.metrics.max_drawdown,
            }
        )
    return pd.DataFrame(rows)


def run_parameter_scenarios(
    backtester: ResearchBacktester,
    base_config: BacktestConfig,
    param_grid: Dict[str, Iterable[object]],
) -> pd.DataFrame:
    keys = list(param_grid.keys())
    rows: List[Dict[str, object]] = []
    for values in itertools.product(*(param_grid[k] for k in keys)):
        overrides = dict(zip(keys, values))
        cfg_dict = {**base_config.__dict__, **overrides}
        cfg = BacktestConfig(**cfg_dict)
        result = backtester.run(cfg)
        row = {"scenario": str(overrides)}
        row.update({k: v for k, v in overrides.items()})
        row.update(
            {
                "sharpe": result.metrics.sharpe,
                "sortino": result.metrics.sortino,
                "calmar": result.metrics.calmar,
                "total_return": result.metrics.total_return,
                "max_drawdown": result.metrics.max_drawdown,
                "win_rate": result.metrics.win_rate,
                "profit_factor": result.metrics.profit_factor,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
