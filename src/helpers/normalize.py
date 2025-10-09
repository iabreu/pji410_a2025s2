import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd


def make_municipio_key(value: Optional[str]) -> str:
    s = str(value).strip()
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]+", "", s)
    return s


def municipio_key_series(series: pd.Series) -> pd.Series:
    """Vectorized municipio key creation for a pandas Series."""
    return series.apply(make_municipio_key)


def winsorize_series(
    s: pd.Series, lower: float = 0.01, upper: float = 0.99
) -> pd.Series:
    if s is None or len(s) == 0:
        return s
    try:
        vals = s.to_numpy(dtype=float, copy=False)
    except Exception:
        return s
    # If all nan or constant
    non_nan = s.dropna()
    if non_nan.empty:
        return s
    if non_nan.nunique() <= 1:
        return s
    try:
        ql = float(np.nanquantile(vals, lower))
        qu = float(np.nanquantile(vals, upper))
    except Exception:
        return s
    if not np.isfinite(ql) or not np.isfinite(qu) or qu < ql:
        return s
    return s.clip(lower=ql, upper=qu)


def key_to_display_map(df: pd.DataFrame) -> dict:
    if "Municipio_key" not in df.columns or "Municipio" not in df.columns:
        return {}
    # Most frequent display value per key
    vc = (
        df.groupby("Municipio_key")["Municipio"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    return vc
