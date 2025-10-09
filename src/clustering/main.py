import os
import sys
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

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


def cluster_municipios(
    df: pd.DataFrame,
    output_dir: str = ".",
    max_k_cap: int = 10,
    default_k: int = 4,
    normalize_rows: bool = True,
    min_code_count: int = 3,
) -> pd.DataFrame:
    """Cluster municipios by non-conformity distribution and return aggregated CSV data."""
    if not {"Municipio", "Código Não Conformidade"}.issubset(df.columns):
        raise KeyError(
            "Colunas necessárias ausentes: 'Municipio' e 'Código Não Conformidade'"
        )

    if "Municipio_key" not in df.columns:
        df["Municipio_key"] = municipio_key_series(df["Municipio"])
    key_map = key_to_display_map(df)

    matrix = pd.crosstab(df["Municipio_key"], df["Código Não Conformidade"]).astype(
        np.float64
    )

    if min_code_count and min_code_count > 1:
        keep_cols = [c for c in matrix.columns if matrix[c].sum() >= min_code_count]
        matrix = matrix[keep_cols] if keep_cols else matrix

    if normalize_rows and matrix.shape[0] > 0:
        row_sums = matrix.sum(axis=1).replace(0, np.nan)
        matrix = matrix.div(row_sums, axis=0).fillna(0.0)

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return pd.DataFrame()

    max_k = int(min(max_k_cap, max(1, matrix.shape[0])))
    optimal_k = int(min(default_k, max_k))

    if optimal_k <= 1:
        result = pd.DataFrame({"Municipio": matrix.index, "Cluster": 0})
    else:
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(matrix)
        result = pd.DataFrame({"Municipio": matrix.index, "Cluster": clusters})

    raw_counts = pd.crosstab(df["Municipio_key"], df["Código Não Conformidade"]).astype(
        float
    )
    raw_counts = raw_counts.apply(lambda row: winsorize_series(row, 0.0, 0.99), axis=1)

    result["Municipio"] = [key_map.get(k, k) for k in result["Municipio"].tolist()]
    result["total_nao_conformidades"] = result["Municipio"].map(
        pd.Series(
            raw_counts.sum(axis=1).values,
            index=[key_map.get(k, k) for k in raw_counts.index],
        )
    )
    result["total_nao_conformidades"] = (
        result["total_nao_conformidades"].fillna(0).round().astype(int)
    )

    for i in range(1, 4):
        result[f"top_codigo_{i}"] = ""
        result[f"freq_codigo_{i}"] = 0

    for idx, row in result.iterrows():
        muni_display = row["Municipio"]
        muni_key = municipio_key_series(pd.Series([muni_display])).iloc[0]
        if muni_key in raw_counts.index:
            top = raw_counts.loc[muni_key].sort_values(ascending=False).head(3)
            for i, (code, freq) in enumerate(top.items(), 1):
                result.at[idx, f"top_codigo_{i}"] = str(code)
                result.at[idx, f"freq_codigo_{i}"] = int(freq)

    return result[
        [
            "Municipio",
            "Cluster",
            "total_nao_conformidades",
            "top_codigo_1",
            "freq_codigo_1",
            "top_codigo_2",
            "freq_codigo_2",
            "top_codigo_3",
            "freq_codigo_3",
        ]
    ]


def main(event: dict):
    """Execute clustering analysis and save municipio-level CSV."""
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

    df = load_csv(csv_path=csv_path, sql_path=sql_path, db_path=db_path)

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    df_municipio = cluster_municipios(
        df, output_dir=results_dir, max_k_cap=10, default_k=4
    )

    if "Cluster" in df_municipio.columns:
        df_municipio = df_municipio.drop(columns=["Cluster"])

    float_cols = df_municipio.select_dtypes(include=[np.floating]).columns
    if len(float_cols) > 0:
        df_municipio[float_cols] = df_municipio[float_cols].round(2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"clustering_municipio_{timestamp}.csv")
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
