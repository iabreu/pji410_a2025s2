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
    s = df.dropna(subset=[date_col, "Municipio"]).set_index(date_col).sort_index()
    g = (
        s.groupby([pd.Grouper(freq="MS"), "Municipio"])
        .size()
        .rename("count")
        .reset_index()
    )
    pivot = g.pivot(index=date_col, columns="Municipio", values="count").sort_index()
    return pivot.fillna(0.0)


def _make_time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Create time-based features for regression."""
    t = np.arange(len(idx), dtype=float)
    month = idx.month.values
    sin_m = np.sin(2 * np.pi * month / 12.0)
    cos_m = np.cos(2 * np.pi * month / 12.0)
    return pd.DataFrame({"t": t, "sin_m": sin_m, "cos_m": cos_m}, index=idx)


def _fit_predict_series(y: pd.Series, horizon: int, alpha: float = 1.0) -> pd.Series:
    """Fit Ridge model and predict future values."""
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

    return pd.Series(y_fore, index=fut_idx, name="forecast")


def forecast_and_report(csv_path: str, sql_path: str, db_path: str) -> pd.DataFrame:
    """Generate municipio-level forecasting summary CSV."""
    df = load_csv(csv_path=csv_path, sql_path=sql_path, db_path=db_path)
    monthly_by_muni = _prepare_monthly_by_municipio(df)

    if monthly_by_muni.empty:
        return pd.DataFrame()

    results = []

    for muni in monthly_by_muni.columns:
        y = monthly_by_muni[muni].astype(float)

        if y.dropna().shape[0] < 6:
            mean_val = float(y.mean()) if np.isfinite(y.mean()) else 0.0
            results.append(
                {
                    "Municipio": muni,
                    "historico_total": int(y.sum()),
                    "historico_12m": int(y.tail(12).sum()),
                    "previsao_12m": int(mean_val * 12),
                    "previsao_24m": int(mean_val * 24),
                    "previsao_48m": int(mean_val * 48),
                    "tendencia": "estável",
                }
            )
            continue

        forecast = _fit_predict_series(y, horizon=48, alpha=1.0)

        hist_total = int(y.sum())
        hist_12m = int(y.tail(12).sum())
        prev_12m = int(forecast.iloc[:12].sum())
        prev_24m = int(forecast.iloc[:24].sum())
        prev_48m = int(forecast.sum())

        if hist_12m > 0:
            trend_ratio = prev_12m / hist_12m
            if trend_ratio > 1.1:
                tendencia = "crescente"
            elif trend_ratio < 0.9:
                tendencia = "decrescente"
            else:
                tendencia = "estável"
        else:
            tendencia = "estável"

        results.append(
            {
                "Municipio": muni,
                "historico_total": hist_total,
                "historico_12m": hist_12m,
                "previsao_12m": prev_12m,
                "previsao_24m": prev_24m,
                "previsao_48m": prev_48m,
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
