from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestPresetConfig:
    """Feature flags that tune performance and observability.

    The presets intentionally avoid altering core trading logic; they only
    enable/disable expensive instrumentation and control progress reporting
    cadence so long runs remain responsive.
    """

    name: str
    enable_intrabar_simulation: bool = True
    collect_microstructure: bool = True
    enable_per_bar_debug_logging: bool = False
    enable_slippage_model: bool = True
    enable_assertions: bool = True
    enable_rich_metrics: bool = True
    progress_log_interval_bars: int = 1_000
    processing_chunk_size: int = 1_000


def _preset(
    name: str,
    *,
    intrabar: bool = True,
    microstructure: bool = True,
    per_bar_debug: bool = False,
    slippage: bool = True,
    assertions: bool = True,
    rich_metrics: bool = True,
    progress_interval: int = 1_000,
    chunk_size: int = 1_000,
) -> BacktestPresetConfig:
    return BacktestPresetConfig(
        name=name,
        enable_intrabar_simulation=intrabar,
        collect_microstructure=microstructure,
        enable_per_bar_debug_logging=per_bar_debug,
        enable_slippage_model=slippage,
        enable_assertions=assertions,
        enable_rich_metrics=rich_metrics,
        progress_log_interval_bars=progress_interval,
        processing_chunk_size=chunk_size,
    )


PRESET_QUICK_SMOKE = _preset(
    "Quick smoke test",
    intrabar=False,
    microstructure=False,
    per_bar_debug=False,
    slippage=False,
    assertions=False,
    rich_metrics=False,
    progress_interval=500,
    chunk_size=2_000,
)

PRESET_STANDARD_RESEARCH = _preset(
    "Standard research",
    intrabar=True,
    microstructure=True,
    per_bar_debug=False,
    slippage=True,
    assertions=True,
    rich_metrics=True,
    progress_interval=1_500,
    chunk_size=1_500,
)

PRESET_FULL_AUDIT = _preset(
    "Full audit",
    intrabar=True,
    microstructure=True,
    per_bar_debug=False,
    slippage=True,
    assertions=True,
    rich_metrics=True,
    progress_interval=500,
    chunk_size=750,
)


PRESETS_BY_NAME = {
    PRESET_QUICK_SMOKE.name: PRESET_QUICK_SMOKE,
    PRESET_STANDARD_RESEARCH.name: PRESET_STANDARD_RESEARCH,
    PRESET_FULL_AUDIT.name: PRESET_FULL_AUDIT,
}


DEFAULT_PRESET_NAME = PRESET_STANDARD_RESEARCH.name


def list_presets() -> list[str]:
    return list(PRESETS_BY_NAME.keys())


def get_preset(name: str | None = None) -> BacktestPresetConfig:
    preset_name = DEFAULT_PRESET_NAME if name is None else name
    return PRESETS_BY_NAME[preset_name]


def resolve_preset(name: str | BacktestPresetConfig | None) -> BacktestPresetConfig:
    if isinstance(name, BacktestPresetConfig):
        return name
    if name is None:
        return PRESETS_BY_NAME[DEFAULT_PRESET_NAME]
    return PRESETS_BY_NAME.get(name, PRESET_STANDARD_RESEARCH)
