"""Bitquery on-chain integration for the Spot AI agent."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_BITQUERY_ENDPOINT = "https://graphql.bitquery.io/"


def _bitquery_post(query: str, variables: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("BITQUERY_API_KEY")
    if not api_key:
        logger.debug("Bitquery API key is not set; skipping on-chain request.")
        return None

    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": api_key,
    }
    payload: Dict[str, Any] = {"query": query}
    if variables is not None:
        payload["variables"] = variables

    try:
        response = requests.post(
            _BITQUERY_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=15,
        )
    except Exception as exc:  # pragma: no cover - network call
        logger.warning("Bitquery request failed: %s", exc)
        return None

    if response.status_code != 200:
        snippet = response.text[:300]
        logger.warning(
            "Bitquery returned non-200 status %s: %s", response.status_code, snippet
        )
        return None

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        logger.warning("Bitquery response was not valid JSON: %s", exc)
        return None
    except ValueError as exc:
        logger.warning("Bitquery response JSON decode error: %s", exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Bitquery response payload malformed: %r", data)
        return None

    errors = data.get("errors")
    if errors:
        logger.warning("Bitquery GraphQL errors: %s", errors)
        return None

    payload_data = data.get("data")
    if not isinstance(payload_data, dict):
        logger.warning("Bitquery response missing data field: %r", data)
        return None
    return payload_data


def bitquery_health_check() -> bool:
    """Run a lightweight query to confirm the API key works."""

    query = """
    query HealthCheck {
      bitcoin {
        blocks(limit: 1) {
          count
        }
      }
    }
    """

    data = _bitquery_post(query)
    if not data:
        return False

    bitcoin = data.get("bitcoin")
    if isinstance(bitcoin, list):
        bitcoin = bitcoin[0] if bitcoin else {}
    if not isinstance(bitcoin, dict):
        return False

    blocks = bitcoin.get("blocks")
    if isinstance(blocks, dict):
        entries = [blocks]
    elif isinstance(blocks, list):
        entries = blocks
    else:
        entries = []
    if not entries:
        return False
    count = entries[0].get("count") if isinstance(entries[0], dict) else None
    return isinstance(count, (int, float))


def get_btc_onchain_signal(window_hours: int = 24) -> Optional[Dict[str, Any]]:
    """Return a lightweight BTC on-chain activity snapshot."""

    try:
        window = int(window_hours)
    except (TypeError, ValueError):
        window = 24
    window = max(1, min(window, 24 * 7))
    since = (_dt.datetime.utcnow() - _dt.timedelta(hours=window)).isoformat(timespec="seconds") + "Z"

    query = """
    query BtcTransfers($since: ISO8601DateTime!) {
      bitcoin {
        transactions(date: {since: $since}) {
          count
        }
      }
    }
    """

    variables = {"since": since}
    data = _bitquery_post(query, variables=variables)
    if not data:
        return None

    bitcoin = data.get("bitcoin")
    if isinstance(bitcoin, list):
        bitcoin = bitcoin[0] if bitcoin else {}
    if not isinstance(bitcoin, dict):
        logger.warning("Bitquery BTC payload malformed: %r", bitcoin)
        return None

    transactions = bitcoin.get("transactions")
    if isinstance(transactions, dict):
        entries = [transactions]
    elif isinstance(transactions, list):
        entries = transactions
    else:
        entries = []

    if not entries:
        logger.warning("Bitquery BTC transactions payload missing: %r", bitcoin)
        return None

    first = entries[0]
    count = first.get("count") if isinstance(first, dict) else None

    if count is None:
        logger.warning("Bitquery BTC transactions count missing: %r", first)
        return None

    try:
        transfers = float(count)
    except (TypeError, ValueError):
        logger.warning("Bitquery BTC transactions count invalid: %r", count)
        return None

    return {
        "ok": True,
        "window_hours": window,
        "total_transfers": transfers,
        "total_volume_btc": None,
    }


__all__ = ["bitquery_health_check", "get_btc_onchain_signal"]
