import pandas as pd
import pytest

from validation import require_columns, require_nonempty


def test_require_columns_passes_when_present():
    df = pd.DataFrame({"Date": [1, 2], "Adj_Close": [10, 11]})
    # Should not raise
    require_columns(df, ["Date", "Adj_Close"], "test_fn")


def test_require_columns_raises_clear_error_when_missing():
    df = pd.DataFrame({"Date": [1, 2]})
    with pytest.raises(ValueError) as exc_info:
        require_columns(df, ["Date", "Adj_Close"], "test_fn")
    assert "Adj_Close" in str(exc_info.value)
    assert "test_fn" in str(exc_info.value)


def test_require_nonempty_passes_for_populated_df():
    df = pd.DataFrame({"a": [1]})
    require_nonempty(df, "test_fn")


def test_require_nonempty_raises_for_empty_df():
    df = pd.DataFrame({"a": []})
    with pytest.raises(ValueError):
        require_nonempty(df, "test_fn")
