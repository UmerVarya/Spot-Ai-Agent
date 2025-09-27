# Proposed Follow-up Tasks

## Fix a Typo
- **Issue**: The module docstring in `btc_dominance.py` spells "CoinGecko" as "Coingecko".
- **Location**: `btc_dominance.py`, module-level documentation string.
- **Suggested Task**: Correct the proper noun to "CoinGecko" in the docstring.

## Fix a Bug
- **Issue**: `notifier.log_rejection` calls `os.makedirs(os.path.dirname(path), exist_ok=True)`, which raises `FileNotFoundError` when the environment sets `REJECTED_TRADES_FILE` to a bare filename (directory component is an empty string).
- **Location**: `notifier.py`, `log_rejection` implementation.
- **Suggested Task**: Guard against empty directory components before calling `os.makedirs`.

## Correct Documentation/Comment Discrepancy
- **Issue**: The trade record documentation states the `tp1_partial` / `tp2_partial` columns contain the strings ``true`` / ``false``, but `trade_storage.log_trade_result` writes boolean values (`True` / `False`).
- **Location**: `docs/trade_record_format.md` vs. `trade_storage.py` row construction.
- **Suggested Task**: Update the documentation to match the actual boolean values written by the code (or adjust the writer to emit lowercase strings).

## Improve a Test
- **Issue**: `tests/test_trade_sizing.py::test_trade_size_bounds` only asserts behaviour at the minimum and maximum limits. A regression in the interpolation logic could pass these assertions while breaking typical mid-range sizing.
- **Location**: `tests/test_trade_sizing.py`.
- **Suggested Task**: Add an assertion for a mid-range input (e.g., confidence/score halfway between thresholds) to ensure the calculated trade size scales smoothly between the bounds.
