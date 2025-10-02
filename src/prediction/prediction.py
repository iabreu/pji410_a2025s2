import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.helpers.extract_zip import extract
    from src.helpers.load_csv import load_csv
except ImportError:
    from helpers.extract_zip import extract
    from helpers.load_csv import load_csv

RANDOM_STATE = 42

DATE_COLS = [
    "Data Início Fiscalização",
    "Data Notificação",
    "Data Advertencia",
    "Data da Multa",
    "Data Limite Resolução",
    "Data Limite CAC",
]

CATEGORICAL_COLS = [
    "Municipio",
    "Sistema",
    "Subsistema",
    "Código Não Conformidade",
    "Item Não Conforme",
]

NUMERIC_DERIVED = [
    "tempo_ate_notificacao",
    "tempo_ate_advertencia",
    "tempo_ate_multa",
    "tempo_fiscalizacao",
    "tempo_ate_limite_resolucao",
    "tempo_ate_limite_cac",
]


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLS:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")
            if df[col].notna().sum() == 0:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ID Multa" not in df.columns:
        raise KeyError("Coluna 'ID Multa' ausente no dataset.")
    df["tem_multa"] = df["ID Multa"].notnull().astype(int)
    df = _parse_dates(df)

    df["tempo_ate_notificacao"] = (
        df["Data Notificação"] - df["Data Início Fiscalização"]
    ).dt.days
    df["tempo_ate_advertencia"] = (
        df["Data Advertencia"] - df["Data Notificação"]
    ).dt.days
    if "Data da Multa" in df.columns:
        df["tempo_ate_multa"] = (df["Data da Multa"] - df["Data Notificação"]).dt.days
    if (
        "Data Fim Fiscalização" in df.columns
        and "Data Início Fiscalização" in df.columns
    ):
        df["tempo_fiscalizacao"] = (
            df["Data Fim Fiscalização"] - df["Data Início Fiscalização"]
        ).dt.days
    if "Data Limite Resolução" in df.columns:
        df["tempo_ate_limite_resolucao"] = (
            df["Data Limite Resolução"] - df["Data Notificação"]
        ).dt.days
    if "Data Limite CAC" in df.columns:
        df["tempo_ate_limite_cac"] = (
            df["Data Limite CAC"] - df["Data Notificação"]
        ).dt.days

    for c in NUMERIC_DERIVED:
        if c in df.columns:
            df[f"{c}_is_missing"] = df[c].isna().astype(int)
            df[f"{c}_was_negative"] = (df[c] < 0).fillna(False).astype(int)
            df[c] = df[c].clip(lower=0)
        else:
            df[c] = np.nan
            df[f"{c}_is_missing"] = 1
            df[f"{c}_was_negative"] = 0
    return df


def _build_features(df: pd.DataFrame):
    feature_cols = (
        CATEGORICAL_COLS
        + NUMERIC_DERIVED
        + [f"{c}_is_missing" for c in NUMERIC_DERIVED]
        + [f"{c}_was_negative" for c in NUMERIC_DERIVED]
    )
    missing_in_df = [c for c in feature_cols if c not in df.columns]
    if missing_in_df:
        raise ValueError(f"Colunas esperadas ausentes após preparação: {missing_in_df}")
    X = df[feature_cols]
    y = df["tem_multa"]
    return X, y, feature_cols


def _build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            (
                "num",
                "passthrough",
                NUMERIC_DERIVED
                + [f"{c}_is_missing" for c in NUMERIC_DERIVED]
                + [f"{c}_was_negative" for c in NUMERIC_DERIVED],
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )
    return model


def train_and_evaluate(
    zip_path: str,
    zip_password: str,
    csv_path: str,
    sql_path: str,
    db_path: str,
    class_weight: bool = True,
) -> pd.DataFrame:
    """Train model and generate municipio-level aggregated predictions CSV."""
    if zip_path:
        extract(zip_path, zip_password)

    df = load_csv(csv_path=csv_path, sql_path=sql_path, db_path=db_path)
    df = _prepare_dataframe(df)
    X, y, feature_cols = _build_features(df)

    sample_weight = None
    if class_weight:
        pos_freq = y.mean()
        neg_freq = 1 - pos_freq
        sample_weight = np.where(y == 1, neg_freq / pos_freq, 1.0)

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weight, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    model = _build_model()
    model.fit(
        X_train,
        y_train,
        classifier__sample_weight=sw_train if sw_train is not None else None,
    )

    y_prob_all = model.predict_proba(X)[:, 1]
    df_result = df[["Municipio"]].copy()
    df_result["tem_multa"] = y
    df_result["prob_multa"] = y_prob_all

    agg = (
        df_result.groupby("Municipio")
        .agg(
            total_casos=("tem_multa", "count"),
            casos_com_multa=("tem_multa", "sum"),
            prob_multa_media=("prob_multa", "mean"),
        )
        .reset_index()
    )

    agg["casos_sem_multa"] = agg["total_casos"] - agg["casos_com_multa"]
    agg["taxa_multa"] = (agg["casos_com_multa"] / agg["total_casos"] * 100).round(2)
    agg["prob_multa_media"] = agg["prob_multa_media"].round(4)

    high_risk = df_result[df_result["prob_multa"] >= 0.7].groupby("Municipio").size()
    agg["casos_alto_risco"] = agg["Municipio"].map(high_risk).fillna(0).astype(int)

    return agg[
        [
            "Municipio",
            "total_casos",
            "casos_com_multa",
            "casos_sem_multa",
            "taxa_multa",
            "prob_multa_media",
            "casos_alto_risco",
        ]
    ]
