"""Vector-based retrieval utilities for macro sentiment analysis.

This module builds a light-weight retrieval augmented generation (RAG)
workflow that collects recent news headlines together with cached macro
narratives (e.g. regulatory alerts or notable events).  The documents are
embedded with a transformer encoder and indexed in FAISS for fast
similarity search.  When the embedding stack is unavailable the module
falls back to simple keyword filtering so that the rest of the agent can
continue to operate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence

import numpy as np

from log_utils import setup_logger

try:  # pragma: no cover - optional dependency is runtime guarded
    import faiss  # type: ignore
except Exception:  # pragma: no cover - handled gracefully below
    faiss = None

try:  # pragma: no cover - optional dependency is runtime guarded
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - handled gracefully below
    torch = None
    AutoModel = None
    AutoTokenizer = None


LOGGER = setup_logger(__name__)


@dataclass(frozen=True)
class RetrievedDocument:
    """Container for retrieved snippets."""

    text: str
    score: float
    source: str = "headline"


class NewsVectorStore:
    """Simple FAISS-backed semantic search over news documents."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_documents: int = 200,
    ) -> None:
        self.model_name = model_name
        self.max_documents = max_documents
        self._documents: list[str] = []
        self._sources: list[str] = []
        self._index = None
        self._tokenizer = None
        self._model = None
        self._available = False
        self._initialise()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self._available

    def refresh(self, documents: Sequence[str], sources: Sequence[str] | None = None) -> None:
        """Rebuild the FAISS index with the provided documents."""

        if not self.is_available:
            # Still keep the raw texts for heuristic fallback retrieval.
            self._documents = list(documents)[: self.max_documents]
            self._sources = list(sources or ["headline"] * len(self._documents))[: self.max_documents]
            return

        if not documents:
            self._documents = []
            self._sources = []
            self._index = None
            return

        self._documents = list(documents)[: self.max_documents]
        self._sources = list(sources or ["headline"] * len(documents))[: self.max_documents]

        embeddings = self._encode(self._documents)
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def query(self, text: str, top_k: int = 12) -> List[RetrievedDocument]:
        """Return the most relevant snippets for the supplied query."""

        if not self._documents:
            return []

        top_k = min(top_k, len(self._documents))

        if not self.is_available or not self._index:
            # Fallback: naive keyword matching scored by occurrence count.
            matches: list[RetrievedDocument] = []
            lowered = text.lower()
            for snippet, src in zip(self._documents[:top_k], self._sources[:top_k]):
                score = sum(word in snippet.lower() for word in lowered.split())
                matches.append(RetrievedDocument(text=snippet, score=float(score), source=src))
            return matches

        query_embedding = self._encode([text])
        scores, indices = self._index.search(query_embedding, top_k)
        retrieved: list[RetrievedDocument] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            retrieved.append(
                RetrievedDocument(
                    text=self._documents[idx],
                    score=float(score),
                    source=self._sources[idx],
                )
            )
        return retrieved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialise(self) -> None:
        if faiss is None or torch is None or AutoTokenizer is None or AutoModel is None:
            LOGGER.warning(
                "Semantic retrieval disabled - faiss/transformers not available. Falling back to keyword matching.")
            self._available = False
            return

        try:
            allow_downloads = os.getenv("ENABLE_RAG_DOWNLOADS", "0") == "1"
            load_kwargs = {"local_files_only": not allow_downloads}
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_kwargs)
            self._model = AutoModel.from_pretrained(self.model_name, **load_kwargs)
            self._model.eval()
            self._available = True
        except Exception as exc:  # pragma: no cover - runtime model download
            LOGGER.warning(
                "Failed to load transformer model '%s': %s. Falling back to keyword search.",
                self.model_name,
                exc,
            )
            self._available = False

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        assert self._tokenizer is not None and self._model is not None  # for type checker

        encoded_input = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        )

        with torch.no_grad():
            model_output = self._model(**encoded_input)

        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded_input["attention_mask"].unsqueeze(-1)
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype("float32")


def load_structured_events(path: str = "news_events.json") -> list[str]:
    """Load cached macro events if available."""

    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return [entry.get("event", "").strip() for entry in payload if entry.get("event")]
    except Exception as exc:  # pragma: no cover - defensive I/O handling
        LOGGER.warning("Failed to load structured events from %s: %s", path, exc)
        return []


@lru_cache(maxsize=1)
def _vector_store() -> NewsVectorStore:
    return NewsVectorStore()


def build_retrieval_context(
    query: str,
    documents: Iterable[str],
    sources: Iterable[str] | None = None,
    top_k: int = 12,
) -> str:
    """Construct a context string using semantic retrieval."""

    docs = [doc.strip() for doc in documents if doc and doc.strip()]
    srcs = list(sources) if sources is not None else ["headline"] * len(docs)

    store = _vector_store()
    store.refresh(docs, srcs)

    retrieved = store.query(query, top_k=top_k)

    if not retrieved:
        return "\n".join(docs[:top_k])

    context_lines = []
    for doc in retrieved:
        prefix = f"[{doc.source}]" if doc.source else ""
        context_lines.append(f"{prefix} {doc.text}".strip())
    return "\n".join(context_lines)
