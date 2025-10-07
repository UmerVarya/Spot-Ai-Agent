import json
import os
from typing import Any, Dict, Tuple

PATTERN_MEMORY_FILE = "pattern_memory.json"

# Prior parameters for the Beta distribution (uninformative prior)
PRIOR_ALPHA: float = 1.0
PRIOR_BETA: float = 1.0


def load_pattern_memory() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load the pattern memory file from disk.

    Returns
    -------
    dict
        Nested dictionary ``{symbol: {pattern: stats_dict}}``.  If the file
        does not exist or is invalid, an empty dictionary is returned.
    """

    if not os.path.exists(PATTERN_MEMORY_FILE):
        return {}
    try:
        with open(PATTERN_MEMORY_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_pattern_memory(memory: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """Persist the in-memory structure back to disk."""

    with open(PATTERN_MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2, sort_keys=True)


def _default_entry() -> Dict[str, Any]:
    """Return a default Bayesian memory entry."""

    total = PRIOR_ALPHA + PRIOR_BETA
    mean = PRIOR_ALPHA / total
    variance = (PRIOR_ALPHA * PRIOR_BETA) / (total * total * (total + 1))
    return {
        "alpha": PRIOR_ALPHA,
        "beta": PRIOR_BETA,
        "wins": 0,
        "losses": 0,
        "trades": 0,
        "posterior_mean": mean,
        "posterior_variance": variance,
        # Maintain backwards compatibility with prior interface
        "confidence": round(mean * 10.0, 2),
    }


def _migrate_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade legacy memory entries to the Bayesian representation."""

    if "alpha" in entry and "beta" in entry:
        alpha = float(entry.get("alpha", PRIOR_ALPHA))
        beta = float(entry.get("beta", PRIOR_BETA))
        wins = int(entry.get("wins", max(0, round(alpha - PRIOR_ALPHA))))
        losses = int(entry.get("losses", max(0, round(beta - PRIOR_BETA))))
    else:
        wins = int(entry.get("wins", 0))
        losses = int(entry.get("losses", 0))
        alpha = PRIOR_ALPHA + max(wins, 0)
        beta = PRIOR_BETA + max(losses, 0)

    trades = max(wins + losses, int(round(alpha + beta - PRIOR_ALPHA - PRIOR_BETA)))
    total = alpha + beta
    mean = alpha / total if total else 0.5
    variance = 0.0
    if total > 0:
        variance = (alpha * beta) / (total * total * (total + 1))

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "wins": wins,
        "losses": losses,
        "trades": trades,
        "posterior_mean": float(mean),
        "posterior_variance": float(variance),
        "confidence": round(max(0.0, min(mean * 10.0, 10.0)), 2),
    }


def _get_entry(memory: Dict[str, Dict[str, Dict[str, Any]]], symbol: str, pattern_name: str) -> Dict[str, Any]:
    """Return a mutable entry for ``(symbol, pattern)`` with migration."""

    symbol_key = symbol or "UNKNOWN"
    pattern_key = pattern_name or "UNKNOWN"
    symbol_bucket = memory.setdefault(symbol_key, {})
    entry = symbol_bucket.get(pattern_key)
    if entry is None:
        entry = _default_entry()
    else:
        entry = _migrate_entry(entry)
    symbol_bucket[pattern_key] = entry
    return entry


def recall_pattern_confidence(symbol: str, pattern_name: str) -> float:
    """Return a 0-10 confidence score derived from the posterior mean."""

    memory = load_pattern_memory()
    symbol_bucket = memory.get(symbol or "UNKNOWN", {})
    entry = symbol_bucket.get(pattern_name or "UNKNOWN")
    if not entry:
        return _default_entry()["confidence"]
    entry = _migrate_entry(entry)
    return entry.get("confidence", 5.0)


def get_pattern_posterior_stats(symbol: str, pattern_name: str) -> Dict[str, float]:
    """Return Bayesian posterior statistics for the pattern.

    The resulting dictionary contains ``mean`` (posterior mean probability of a
    profitable outcome), ``variance`` (posterior variance), ``alpha`` and
    ``beta`` (updated Beta parameters) and ``trades`` (number of recorded
    outcomes).
    """

    memory = load_pattern_memory()
    entry = _get_entry(memory, symbol, pattern_name)
    return {
        "mean": float(entry["posterior_mean"]),
        "variance": float(entry["posterior_variance"]),
        "alpha": float(entry["alpha"]),
        "beta": float(entry["beta"]),
        "trades": float(entry.get("trades", entry.get("wins", 0) + entry.get("losses", 0))),
    }


def update_pattern_memory(symbol: str, pattern_name: str, outcome: str) -> Dict[str, Any]:
    """Update the Bayesian posterior with a new trade outcome.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g. ``"BTCUSDT"``).
    pattern_name : str
        The detected pattern.
    outcome : str
        Either ``"win"`` or ``"loss"``.  Any other value is treated as a
        neutral result and does not update the distribution.

    Returns
    -------
    dict
        The updated entry for ``(symbol, pattern)`` containing posterior
        statistics.
    """

    memory = load_pattern_memory()
    entry = _get_entry(memory, symbol, pattern_name)

    outcome_lc = (outcome or "").strip().lower()
    if outcome_lc not in {"win", "loss"}:
        return entry

    if outcome_lc == "win":
        entry["alpha"] = float(entry["alpha"]) + 1.0
        entry["wins"] = int(entry.get("wins", 0)) + 1
    else:
        entry["beta"] = float(entry["beta"]) + 1.0
        entry["losses"] = int(entry.get("losses", 0)) + 1

    entry["trades"] = int(entry.get("wins", 0) + entry.get("losses", 0))

    total = float(entry["alpha"]) + float(entry["beta"])
    if total <= 0:
        mean = 0.5
        variance = 0.0
    else:
        mean = float(entry["alpha"]) / total
        variance = (float(entry["alpha"]) * float(entry["beta"])) / (total * total * (total + 1.0))

    entry["posterior_mean"] = float(mean)
    entry["posterior_variance"] = float(variance)
    entry["confidence"] = round(max(0.0, min(mean * 10.0, 10.0)), 2)

    save_pattern_memory(memory)
    return entry


def get_pattern_feature_vector(symbol: str, pattern_name: str) -> Tuple[float, float]:
    """Return ``(posterior_mean, posterior_variance)`` for ML pipelines."""

    stats = get_pattern_posterior_stats(symbol, pattern_name)
    return stats["mean"], stats["variance"]
