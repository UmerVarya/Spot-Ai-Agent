from __future__ import annotations

from http.client import RemoteDisconnected

from fear_greed import FearGreedIndexFetcher


def test_fetch_remote_disconnected_uses_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    fetcher = FearGreedIndexFetcher(cache_path=cache_path)
    fetcher._write_cache(42)

    def raise_remote():
        raise RemoteDisconnected("boom")

    monkeypatch.setattr(fetcher, "_fetch_remote", raise_remote)

    assert fetcher.fetch() == 42


def test_fetch_remote_disconnected_without_cache_returns_default(monkeypatch, tmp_path):
    fetcher = FearGreedIndexFetcher(cache_path=tmp_path / "cache.json")

    def raise_remote():
        raise RemoteDisconnected("boom")

    monkeypatch.setattr(fetcher, "_fetch_remote", raise_remote)

    assert fetcher.fetch() == 50
