import pandas as pd

import dashboard


def test_recover_unescaped_llm_error(tmp_path):
    csv_text = (
        "trade_id,timestamp,llm_error,volatility\n"
        '1,2024-01-01T00:00:00Z,"{""error"": ""bad, stuff""}",0.25\n'
    )
    path = tmp_path / "history.csv"
    path.write_text(csv_text, encoding="utf-8")

    frame, recovered = dashboard._read_csv_with_recovery(path)

    assert recovered == 0
    assert frame.loc[0, "llm_error"] == '{"error": "bad, stuff"}'
    assert frame.loc[0, "volatility"] == 0.25


def test_recover_malformed_llm_error(tmp_path):
    csv_text = (
        "trade_id,timestamp,llm_error,volatility\n"
        '1,2024-01-01T00:00:00Z,"{"error": "bad, stuff"}",0.25\n'
    )
    path = tmp_path / "history.csv"
    path.write_text(csv_text, encoding="utf-8")

    frame, recovered = dashboard._read_csv_with_recovery(path)

    assert recovered == 1
    assert frame.loc[0, "llm_error"].startswith(dashboard.PARSE_ERROR_TOKEN)
    assert "bad, stuff" in frame.loc[0, "llm_error"]
    assert frame.loc[0, "volatility"] == "0.25"


def test_normalise_time_exit_column():
    df = pd.DataFrame({"time_exit": ["2024-06-01T00:00:00Z"]})
    normalised = dashboard.normalise_history_columns(df)
    assert "exit_time" in normalised.columns
    assert normalised.loc[0, "exit_time"] == "2024-06-01T00:00:00Z"


def test_duplicate_headers_are_mangled(tmp_path):
    csv_text = (
        "trade_id,timestamp,pnl,pnl\n"
        '1,2024-01-01T00:00:00Z,"{""error"": ""bad, stuff""}",10\n'
    )
    path = tmp_path / "history.csv"
    path.write_text(csv_text, encoding="utf-8")

    frame, recovered = dashboard._read_csv_with_recovery(path)

    # ``pd.read_csv`` succeeds here but the duplicate ``pnl`` header is mangled
    # so that downstream consumers (Streamlit/Arrow) do not raise ``ValueError``.
    assert recovered == 0
    assert frame.columns.tolist() == ["trade_id", "timestamp", "pnl", "pnl.1"]
