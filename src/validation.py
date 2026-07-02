"""
Small shared validation helpers so a missing column fails fast with a clear
message instead of a cryptic KeyError three functions later.
"""

import pandas as pd


def require_columns(df: pd.DataFrame, columns, fn_name: str) -> None:
    """Raise a clear ValueError if any of `columns` is missing from df."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{fn_name}: missing required column(s) {missing}. "
            f"Available columns: {list(df.columns)}. "
            f"Did you forget to call add_returns() first?"
        )


def require_nonempty(df: pd.DataFrame, fn_name: str) -> None:
    """Raise a clear ValueError if df has no rows."""
    if df is None or len(df) == 0:
        raise ValueError(f"{fn_name}: received an empty DataFrame.")
