"""Retrieval‑augmented trade memory utilities.

This module converts completed trade records into dense vector embeddings so
that similar past trades can be recalled and injected into prompts for the
LLM-based decision maker.  Set the environment variable
``ENABLE_TRADE_EMBEDDINGS=1`` to enable embeddings in production.  When the
embedding model is unavailable (or the flag is disabled), the retriever
falls back to a lightweight string summary of the most recent trades.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd
from trade_storage import TRADE_HISTORY_FILE

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - gracefully degrade when numpy is absent
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - degrade when embeddings unavailable
    SentenceTransformer = None  # type: ignore


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_ENV_FLAG = "ENABLE_TRADE_EMBEDDINGS"

# Path to the completed trades log used for memory retrieval
LOG_FILE = TRADE_HISTORY_FILE

# In-memory cache of embeddings to avoid recomputation on every call
_EMBEDDING_CACHE: Dict[str, Any] = {
    "mtime": None,
    "embeddings": None,
    "df": None,
}
_EMBEDDING_MODEL: Optional[Any] = None


def _get_embedding_model() -> Optional[Any]:
    """Return a lazily-initialised sentence-transformer model."""

    global _EMBEDDING_MODEL
    if SentenceTransformer is None:
        return None
    if os.environ.get(EMBEDDING_ENV_FLAG, "0").lower() not in {"1", "true", "yes"}:
        return None
    if _EMBEDDING_MODEL is None:
        try:
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception:
            _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


def _build_trade_document(row: pd.Series) -> str:
    """Convert a trade row into a textual document for embedding."""

    tokens: list[str] = []
    for key, value in row.items():
        if pd.isna(value):
            continue
        if isinstance(value, (float, int)):
            tokens.append(f"{key}={float(value):.4f}")
        else:
            tokens.append(f"{key}={str(value)}")
    return " | ".join(tokens)


def _load_trade_embeddings(df: pd.DataFrame, file_mtime: float) -> Optional[Tuple[pd.DataFrame, Any]]:
    """Compute or retrieve cached trade embeddings for ``df``."""

    if np is None or _get_embedding_model() is None:
        return None

    global _EMBEDDING_CACHE
    if _EMBEDDING_CACHE["mtime"] == file_mtime:
        cached_df = _EMBEDDING_CACHE["df"]
        cached_embeddings = _EMBEDDING_CACHE["embeddings"]
        if cached_df is not None and cached_embeddings is not None:
            return cached_df, cached_embeddings

    model = _get_embedding_model()
    if model is None:
        return None

    working_df = df.copy().reset_index(drop=True)
    documents = [_build_trade_document(row) for _, row in working_df.iterrows()]
    try:
        embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
    except Exception:
        return None

    _EMBEDDING_CACHE = {
        "mtime": file_mtime,
        "df": working_df,
        "embeddings": embeddings,
    }
    return working_df, embeddings


def _format_trade_summary(row: pd.Series, similarity: float) -> str:
    """Create a compact human-readable summary for the LLM prompt."""

    ts = row.get("timestamp", "?")
    symbol = row.get("symbol", "?")
    pattern = row.get("pattern", "?")
    outcome = row.get("outcome", "open")
    confidence = row.get("confidence")
    pnl = row.get("pnl", row.get("return", ""))
    note = row.get("thesis") or row.get("narrative") or row.get("notes")
    extras: list[str] = []
    if confidence not in (None, "") and not pd.isna(confidence):
        extras.append(f"conf {confidence}")
    if pnl not in (None, "") and not pd.isna(pnl):
        extras.append(f"pnl {pnl}")
    extras.append(f"sim {similarity:.2f}")
    header = f"- [{ts}] {symbol} {pattern} → {outcome}"
    if extras:
        header += " | " + ", ".join(extras)
    if isinstance(note, str) and note.strip():
        trimmed = note.strip().replace("\n", " ")
        if len(trimmed) > 140:
            trimmed = trimmed[:137] + "..."
        header += f"\n  note: {trimmed}"
    return header


def _fallback_recent_trades(
    df: pd.DataFrame,
    max_entries: int,
    *,
    symbol: str,
    pattern: str,
) -> str:
    """Simple fallback summary using the most recent rows.

    When embeddings are unavailable we still honour the caller provided
    ``symbol``/``pattern`` filters so that the returned context remains relevant.
    """

    working_df = df.copy()

    if symbol and "symbol" in working_df.columns:
        working_df = working_df[working_df["symbol"].astype(str) == symbol]

    normalized_pattern = (pattern or "").strip().lower()
    if normalized_pattern and "pattern" in working_df.columns:
        working_df = working_df[
            working_df["pattern"].astype(str).str.lower() == normalized_pattern
        ]

    if "timestamp" in working_df.columns:
        working_df = working_df.sort_values(by="timestamp", ascending=False)
    working_df = working_df.head(max_entries)

    summaries: list[str] = []
    for _, row in working_df.iterrows():
        ts = row.get("timestamp", "?")
        pattern = row.get("pattern", "?")
        outcome = row.get("outcome", "open")
        conf = row.get("confidence", "?")
        summaries.append(f"[{ts}] {pattern} → {outcome} (conf {conf})")
    return "; ".join(summaries) if summaries else "No prior trades on record."


def _select_similar_trades(
    df: pd.DataFrame,
    embeddings: Any,
    query_embedding: Any,
    max_entries: int,
    symbol: str,
    pattern: str,
) -> Iterable[Tuple[int, float]]:
    """Return indices of the most similar trades ordered by cosine similarity."""

    if np is None:
        return []

    similarities = embeddings @ query_embedding  # type: ignore[operator]
    if np.isscalar(similarities):  # pragma: no cover - degenerate case
        similarities = np.array([similarities])

    symbol_bonus = np.zeros_like(similarities)
    pattern_bonus = np.zeros_like(similarities)

    if "symbol" in df.columns:
        symbol_matches = df["symbol"].astype(str) == symbol
        symbol_bonus = np.where(symbol_matches, 0.05, 0.0)

    if pattern and "pattern" in df.columns:
        pattern_matches = df["pattern"].astype(str).str.lower() == pattern.lower()
        pattern_bonus = np.where(pattern_matches, 0.03, 0.0)

    adjusted = similarities + symbol_bonus + pattern_bonus
    top_indices = np.argsort(adjusted)[::-1][:max_entries]
    return [(int(idx), float(adjusted[idx])) for idx in top_indices]


def get_recent_trade_summary(symbol: str, pattern: str, max_entries: int = 3) -> str:
    """Return a similarity-based summary of recent comparable trades."""

    if not os.path.exists(LOG_FILE):
        return "No prior trades on record."
    try:
        df = pd.read_csv(LOG_FILE, engine="python", on_bad_lines="skip", encoding="utf-8")
    except Exception:
        return "(Unable to read trade log.)"

    if df.empty:
        return "No prior trades on record."

    file_mtime = os.path.getmtime(LOG_FILE)
    embedding_bundle = _load_trade_embeddings(df, file_mtime)

    if embedding_bundle is None:
        return _fallback_recent_trades(df, max_entries, symbol=symbol, pattern=pattern)

    embedding_df, embeddings = embedding_bundle
    model = _get_embedding_model()
    if model is None or np is None:
        return _fallback_recent_trades(df, max_entries, symbol=symbol, pattern=pattern)

    query_description = {
        "symbol": symbol,
        "pattern": pattern,
        "intent": "retrieve similar successful trades",
    }
    query_document = " | ".join(f"{k}={v}" for k, v in query_description.items() if v)
    try:
        query_embedding = model.encode([query_document], convert_to_numpy=True, normalize_embeddings=True)[0]
    except Exception:
        return _fallback_recent_trades(df, max_entries, symbol=symbol, pattern=pattern)

    similar_indices = list(
        _select_similar_trades(
            embedding_df,
            embeddings,
            query_embedding,
            max_entries,
            symbol,
            pattern,
        )
    )

    if not similar_indices:
        return _fallback_recent_trades(df, max_entries, symbol=symbol, pattern=pattern)

    summary_lines = [
        _format_trade_summary(embedding_df.iloc[idx], similarity)
        for idx, similarity in similar_indices
    ]
    return "\n".join(summary_lines)
