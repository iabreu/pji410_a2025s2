import os
import sys
import logging
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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


def _engineer_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Select and derive features and target for classification."""
    df = df.copy()

    def _days_diff(a, b):
        try:
            return (a - b).dt.days
        except Exception:
            return pd.Series(np.nan, index=df.index)

    if "Data Início Fiscalização" in df.columns and "Data Notificação" in df.columns:
        df["dias_entre_inicio_e_notificacao"] = _days_diff(
            df["Data Notificação"], df["Data Início Fiscalização"]
        ).astype("float")
    else:
        df["dias_entre_inicio_e_notificacao"] = np.nan

    if "Data Notificação" in df.columns and "Data Limite Resolução" in df.columns:
        df["dias_ate_limite"] = _days_diff(
            df["Data Limite Resolução"], df["Data Notificação"]
        ).astype("float")
    else:
        df["dias_ate_limite"] = np.nan

    if "Condição" not in df.columns:
        raise ValueError("Coluna 'Condição' não encontrada no dataset limpo.")
    y = (
        df["Condição"]
        .fillna("desconhecido")
        .apply(lambda s: 1 if str(s).strip().lower() == "vencido" else 0)
    )

    cat_cols = [
        c
        for c in ["Municipio", "Sistema", "Subsistema", "Código Não Conformidade"]
        if c in df.columns
    ]
    num_cols = [c for c in ["dias_entre_inicio_e_notificacao", "dias_ate_limite"]]

    for c in num_cols:
        if c in df.columns:
            df[c] = winsorize_series(df[c], 0.01, 0.99)

    X = df[cat_cols + num_cols]
    return X, y, cat_cols, num_cols


def train_and_report(csv_path: str, sql_path: str, db_path: str) -> pd.DataFrame:
    """Train classification model and generate municipio-level aggregated CSV."""
    df = load_csv(csv_path=csv_path, sql_path=sql_path, db_path=db_path)
    if "Municipio_key" not in df.columns and "Municipio" in df.columns:
        df["Municipio_key"] = municipio_key_series(df["Municipio"])
    key_map = key_to_display_map(df)

    X, y, cat_cols, num_cols = _engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    def _make_ohe():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            try:
                return OneHotEncoder(handle_unknown="ignore", sparse=False)
            except TypeError:
                return OneHotEncoder(handle_unknown="ignore")

    pre = ColumnTransformer(
        transformers=[
            ("cat", _make_ohe(), cat_cols),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=2000, solver="saga")
    pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
    pipe.fit(X_train, y_train)

    y_proba_all = pipe.predict_proba(X)[:, 1]

    required_cols = ["Municipio_key", "Municipio", "Condição"]
    missing = [c for c in required_cols if c not in df.columns]
    df_result = df[required_cols].copy()
    df_result["status_vencido"] = y
    df_result["prob_vencido"] = y_proba_all

    agg = (
        df_result.groupby("Municipio_key")
        .agg(
            total_casos=("Condição", "count"),
            casos_vencidos=("status_vencido", "sum"),
            casos_abertos=("Condição", lambda s: (s == "aberto").sum()),
            casos_baixados=("Condição", lambda s: (s == "baixado").sum()),
            prob_vencido_media=("prob_vencido", "mean"),
        )
        .reset_index()
    )

    agg["prob_vencido_media"] = agg["prob_vencido_media"].round(4)
    agg["Municipio"] = agg["Municipio_key"].map(lambda k: key_map.get(k, k))
    return agg[
        [
            "Municipio",
            "total_casos",
            "casos_abertos",
            "casos_baixados",
            "casos_vencidos",
            "prob_vencido_media",
        ]
    ]


def main(event: dict):
    """Execute resolution classification analysis and save municipio-level CSV."""
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

    df_municipio = train_and_report(
        csv_path=csv_path, sql_path=sql_path, db_path=db_path
    )

    float_cols = df_municipio.select_dtypes(include=[np.floating]).columns
    if len(float_cols) > 0:
        df_municipio[float_cols] = df_municipio[float_cols].round(2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"classification_municipio_{timestamp}.csv")
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
