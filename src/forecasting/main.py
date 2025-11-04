import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.helpers.extract_zip import extract, ZipExtractionError
from src.helpers.load_csv import load_csv
from src.helpers.normalize import (
    municipio_key_series,
    key_to_display_map,
    winsorize_series,
)

logging.basicConfig(level=logging.INFO)


def _pick_date_col(df: pd.DataFrame) -> str:
    """Select appropriate date column for time series."""
    if "Data Início Fiscalização" in df.columns:
        return "Data Início Fiscalização"
    if "Data Notificação" in df.columns:
        return "Data Notificação"
    raise ValueError("Colunas de data não encontradas")


def _prepare_monthly_by_municipio(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate data by month and municipio."""
    df = df.copy()
    date_col = _pick_date_col(df)
    if "Municipio" not in df.columns:
        return pd.DataFrame()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    s = df.dropna(subset=[date_col, "Municipio_key"]).set_index(date_col).sort_index()
    g = (
        s.groupby([pd.Grouper(freq="MS"), "Municipio_key"])
        .size()
        .rename("count")
        .reset_index()
    )
    pivot = g.pivot(
        index=date_col, columns="Municipio_key", values="count"
    ).sort_index()
    pivot = pivot.fillna(0.0)
    pivot = pivot.apply(lambda col: winsorize_series(col, 0.01, 0.99))
    return pivot


def _make_time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    t = np.arange(len(idx), dtype=float)
    month = idx.month.values
    sin_m = np.sin(2 * np.pi * month / 12.0)
    cos_m = np.cos(2 * np.pi * month / 12.0)
    return pd.DataFrame({"t": t, "sin_m": sin_m, "cos_m": cos_m}, index=idx)


def _fit_predict_series(y: pd.Series, horizon: int, alpha: float = 1.0) -> pd.Series:
    y = y.astype(float).copy()
    X = _make_time_features(y.index)
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X.values, y.values)

    last = y.index[-1]
    fut_idx = pd.date_range(
        start=(last + pd.offsets.MonthBegin(1)), periods=horizon, freq="MS"
    )
    Xf = _make_time_features(fut_idx)
    y_fore = model.predict(Xf.values)
    y_fore = np.clip(y_fore, 0.0, None)

    return pd.Series(y_fore, index=fut_idx, name="forecast")


def _safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.Series(y_true).astype(float).values
    y_pred = pd.Series(y_pred).astype(float).values
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _mean_level(y_train: pd.Series, horizon: int) -> pd.Series:
    s = pd.Series(y_train).astype(float)
    lvl = max(0.0, float(s.mean()))
    if isinstance(s.index, pd.DatetimeIndex) and len(s.index) > 0:
        last = s.index[-1]
        fut_idx = pd.date_range(
            start=last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )
        return pd.Series(np.full(horizon, lvl, dtype=float), index=fut_idx)
    return pd.Series(np.full(horizon, lvl, dtype=float))


def _moving_average_level(y_train: pd.Series, horizon: int, window: int) -> pd.Series:
    s = pd.Series(y_train).astype(float)
    lvl = s.rolling(window, min_periods=max(1, window // 2)).mean().iloc[-1]
    if not np.isfinite(lvl):
        lvl = float(s.mean())
    lvl = max(0.0, float(lvl))
    if isinstance(s.index, pd.DatetimeIndex) and len(s.index) > 0:
        last = s.index[-1]
        fut_idx = pd.date_range(
            start=last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )
        return pd.Series(np.full(horizon, lvl, dtype=float), index=fut_idx)
    return pd.Series(np.full(horizon, lvl, dtype=float))


def _seasonal_naive(y_train: pd.Series, horizon: int) -> pd.Series:
    """Seasonal naive: repeat the last 12 months; ensure monthly DatetimeIndex."""
    s = pd.Series(y_train).astype(float)
    vals = s.values
    n = len(vals)
    if n >= 12:
        last12 = np.asarray(vals[-12:], dtype=float)
        reps = int(np.ceil(horizon / 12))
        fc = np.tile(last12, reps)[:horizon]
    else:
        lvl = max(0.0, float(np.nanmean(vals)))
        fc = np.full(horizon, lvl, dtype=float)

    fc = np.clip(fc, 0.0, None)
    if isinstance(s.index, pd.DatetimeIndex) and len(s.index) > 0:
        last = s.index[-1]
        fut_idx = pd.date_range(
            start=last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )
        return pd.Series(fc, index=fut_idx)
    return pd.Series(fc)


def _select_and_forecast(y: pd.Series, horizon: int = 48) -> pd.Series:
    """Choose best simple forecaster by short backtest, then forecast horizon."""
    y = pd.Series(y).astype(float)
    n = y.dropna().shape[0]
    if n < 6:
        lvl = max(0.0, float(y.mean())) if np.isfinite(y.mean()) else 0.0
        return pd.Series(np.full(horizon, lvl, dtype=float))

    valid_h = min(6, max(1, n // 4))
    y_train = y.iloc[:-valid_h]
    y_valid = y.iloc[-valid_h:]

    candidates = [
        ("mean", lambda s, h: _mean_level(s, h)),
        ("ma3", lambda s, h: _moving_average_level(s, h, 3)),
        ("ma6", lambda s, h: _moving_average_level(s, h, 6)),
        ("ma12", lambda s, h: _moving_average_level(s, h, 12)),
        ("seasonal", lambda s, h: _seasonal_naive(s, h)),
    ]

    def _model_alpha(a: float):
        return lambda s, h: pd.Series(
            np.clip(
                np.asarray(_fit_predict_series(pd.Series(s), horizon=h, alpha=a)),
                0.0,
                None,
            )
        )

    candidates.extend(
        [
            ("model_a0.5", _model_alpha(0.5)),
            ("model_a1.0", _model_alpha(1.0)),
            ("model_a2.0", _model_alpha(2.0)),
        ]
    )

    best_err = float("inf")
    best_fn = None
    for _, fn in candidates:
        try:
            pred = fn(y_train, valid_h)
            err = _safe_mape(y_valid, pred)
            if err < best_err:
                best_err = err
                best_fn = fn
        except Exception:
            continue

    if best_fn is None:
        best_fn = lambda s, h: _mean_level(s, h)

    y_fore = best_fn(y, horizon)
    y_fore = pd.Series(np.clip(pd.Series(y_fore).astype(float).values, 0.0, None))
    if isinstance(y.index, pd.DatetimeIndex) and len(y.index) > 0:
        last = y.index[-1]
        fut_idx = pd.date_range(
            start=last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )
        y_fore.index = fut_idx
    return y_fore


def _classify_tendencia(y: pd.Series, forecast: pd.Series) -> str:
    y = pd.Series(y).astype(float)
    forecast = pd.Series(forecast).astype(float)

    y12 = y.tail(12)
    f12 = forecast.iloc[:12]

    if y12.shape[0] < 3 or f12.shape[0] < 3:
        return "estável"

    hist_12m_sum = float(y12.sum())
    fore_12m_sum = float(f12.sum())

    if hist_12m_sum < 1 and fore_12m_sum < 1:
        return "estável"

    if hist_12m_sum < 0.01:
        return "crescente" if fore_12m_sum > 1 else "estável"

    pct_change = ((fore_12m_sum - hist_12m_sum) / hist_12m_sum) * 100

    if abs(pct_change) < 10:
        y36 = y.tail(36) if len(y) >= 36 else y
        baseline_mean = float(y36.mean())
        recent_mean = float(y12.mean())

        if baseline_mean > 0.01:
            long_term_ratio = recent_mean / baseline_mean
            if long_term_ratio > 1.15 and fore_12m_sum >= hist_12m_sum * 0.9:
                return "crescente"
            elif long_term_ratio < 0.85 and fore_12m_sum <= hist_12m_sum * 1.1:
                return "decrescente"

    if pct_change > 10:
        return "crescente"
    elif pct_change < -10:
        return "decrescente"
    else:
        return "estável"


def forecast_and_report(csv_path: str, sql_path: str, db_path: str) -> pd.DataFrame:
    """Generate municipio-level forecasting summary CSV."""
    df = load_csv(csv_path=csv_path, sql_path=sql_path, db_path=db_path)
    if "Municipio_key" not in df.columns and "Municipio" in df.columns:
        df["Municipio_key"] = municipio_key_series(df["Municipio"])
    key_map = key_to_display_map(df)

    monthly_by_muni = _prepare_monthly_by_municipio(df)

    if monthly_by_muni.empty:
        return pd.DataFrame()

    results = []

    for muni_key in monthly_by_muni.columns:
        y = monthly_by_muni[muni_key].astype(float)

        if y.dropna().shape[0] < 6:
            mean_val = float(y.mean()) if np.isfinite(y.mean()) else 0.0
            mean_val = max(0.0, mean_val)
            results.append(
                {
                    "Municipio": key_map.get(
                        muni_key,
                        (
                            df.loc[df["Municipio_key"] == muni_key, "Municipio"].iloc[0]
                            if (df["Municipio_key"] == muni_key).any()
                            else muni_key
                        ),
                    ),
                    "historico_total": int(y.sum()),
                    "historico_12m": int(y.tail(12).sum()),
                    "previsao_12m": int(mean_val * 12),
                    "previsao_24m": int(mean_val * 24),
                    "previsao_48m": int(mean_val * 48),
                    "tendencia": "estável",
                }
            )
            continue

        forecast = _select_and_forecast(y, horizon=48)

        hist_total_f = float(y.sum())
        hist_12m_f = float(y.tail(12).sum())
        prev_12m_f = float(forecast.iloc[:12].sum())
        prev_24m_f = float(forecast.iloc[:24].sum())
        prev_48m_f = float(forecast.sum())

        tendencia = _classify_tendencia(y, forecast)

        results.append(
            {
                "Municipio": key_map.get(
                    muni_key,
                    (
                        df.loc[df["Municipio_key"] == muni_key, "Municipio"].iloc[0]
                        if (df["Municipio_key"] == muni_key).any()
                        else muni_key
                    ),
                ),
                "historico_total": int(hist_total_f),
                "historico_12m": int(hist_12m_f),
                "previsao_12m": int(prev_12m_f),
                "previsao_24m": int(prev_24m_f),
                "previsao_48m": int(prev_48m_f),
                "tendencia": tendencia,
            }
        )

    return pd.DataFrame(results)


def main(event: dict):
    """Execute forecasting analysis and save municipio-level CSV."""
    data_dir = os.path.join(PROJECT_ROOT, event.get("data_folder"))
    zip_path = (
        os.path.join(data_dir, event.get("zip_file")) if event.get("zip_file") else None
    )
    csv_path = os.path.join(data_dir, event.get("csv_file"))
    sql_path = os.path.join(PROJECT_ROOT, "sql", event.get("sql_file"))
    db_path = os.path.join(PROJECT_ROOT, event.get("db_file"))
    password = event.get("zip_password")

    if zip_path and os.path.exists(zip_path):
        try:
            extract(zip_path, password)
        except ZipExtractionError as e:
            logging.warning(
                f"Falha ao extrair zip: {e}. Prosseguindo assumindo que CSV já existe."
            )

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    df_municipio = forecast_and_report(
        csv_path=csv_path, sql_path=sql_path, db_path=db_path
    )

    float_cols = df_municipio.select_dtypes(include=[np.floating]).columns
    if len(float_cols) > 0:
        df_municipio[float_cols] = df_municipio[float_cols].round(2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"forecasting_municipio_{timestamp}.csv")
    df_municipio.to_csv(output_path, index=False)
    logging.info(f"CSV salvo em: {output_path}")


if __name__ == "__main__":
    event = {
        "data_folder": "data",
        "zip_file": "fiscalizacao.csv.zip",
        "csv_file": "fiscalizacao.csv",
        "sql_file": "clean_fiscalizacao.sql",
        "db_file": "fiscalizacao.db",
        "zip_password": "",
    }
    main(event)
