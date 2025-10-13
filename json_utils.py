"""Shared helpers for parsing loosely formatted JSON from LLM responses.

The Spot agent frequently consumes JSON emitted by large language models. In
practice those responses may include Markdown code fences, superfluous text, or
minor formatting glitches.  This module consolidates the defensive parsing
logic so that callers across the codebase benefit from the same sanitisation
steps and fallback strategies.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple
import json
import logging
import re


def strip_markdown_json(text: str) -> str:
    """Remove Markdown fences and leading ``json`` labels from *text*."""

    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
        cleaned = cleaned.lstrip(":").strip()
    return cleaned


def _normalise_candidate(data: Any) -> dict[str, Any] | None:
    """Return a dict when ``data`` is a mapping or list of mappings."""

    if isinstance(data, Mapping):
        return dict(data)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping):
                return dict(item)
    return None


def parse_llm_json_response(
    raw_text: str,
    *,
    defaults: Mapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> Tuple[dict[str, Any], bool]:
    """Parse ``raw_text`` into a dictionary while tolerating noisy formats.

    Parameters
    ----------
    raw_text:
        Raw response returned by the language model.
    defaults:
        Optional key/value pairs that will be used to populate missing fields
        in the parsed output.
    logger:
        Optional logger used for debug messages when JSON parsing fails.

    Returns
    -------
    tuple(dict, bool)
        A tuple containing the parsed dictionary (with ``defaults`` applied)
        and a boolean indicating whether valid JSON content was extracted.
    """

    text = str(raw_text or "").strip()
    base: dict[str, Any] = dict(defaults or {})

    if not text:
        return base, False

    candidates: list[str] = []

    def _add_candidate(value: str) -> None:
        candidate = value.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    _add_candidate(text)
    stripped = strip_markdown_json(text)
    _add_candidate(stripped)

    parsed: dict[str, Any] | None = None

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:  # pragma: no cover - debug logging
            if logger:
                logger.debug("Failed to parse JSON candidate: %s", exc)
            continue
        parsed = _normalise_candidate(data)
        if parsed is not None:
            break

    if parsed is None:
        search_text = stripped or text
        match = re.search(r"\{.*\}", search_text, flags=re.DOTALL)
        if match:
            snippet = match.group(0)
            try:
                parsed = _normalise_candidate(json.loads(snippet))
            except json.JSONDecodeError as exc:  # pragma: no cover - debug logging
                if logger:
                    logger.debug("Regex JSON extraction failed: %s", exc)
        if parsed is None:
            first = search_text.find("{")
            last = search_text.rfind("}")
            if first != -1 and last != -1 and last > first:
                snippet = search_text[first : last + 1]
                try:
                    parsed = _normalise_candidate(json.loads(snippet))
                except json.JSONDecodeError as exc:  # pragma: no cover - debug logging
                    if logger:
                        logger.debug("Bracket slicing JSON parsing failed: %s", exc)

    success = parsed is not None
    if parsed:
        base.update(parsed)
    return base, success

