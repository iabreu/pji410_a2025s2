import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.inspection import permutation_importance
import json
from typing import Tuple, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)

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


def _cross_validate(model: Pipeline, X, y, n_splits: int = 5) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics = {
        "roc_auc": [],
        "balanced_accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        if hasattr(model.named_steps["classifier"], "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            if hasattr(model.named_steps["classifier"], "decision_function"):
                df_scores = model.decision_function(X_val)
                mn, mx = df_scores.min(), df_scores.max()
                y_prob = (df_scores - mn) / (mx - mn) if mx > mn else df_scores
            else:
                y_prob = None
        y_pred = model.predict(X_val)
        if y_prob is not None:
            metrics["roc_auc"].append(roc_auc_score(y_val, y_prob))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(y_val, y_pred))
        metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
    return {k: float(np.mean(v)) for k, v in metrics.items() if v}


def _optimal_threshold(
    y_true, y_prob, criterion: str = "balanced_accuracy"
) -> Tuple[float, Dict[str, float]]:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_score = -np.inf
    best_metrics = {}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        score = bal_acc if criterion == "balanced_accuracy" else f1
        if score > best_score:
            best_score = score
            best_t = t
            best_metrics = {
                "threshold": t,
                "balanced_accuracy": bal_acc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
            }
    return best_t, best_metrics


def train_and_evaluate(
    zip_path: str,
    zip_password: str,
    csv_path: str,
    sql_path: str,
    db_path: str,
    class_weight: bool = True,
    cv_folds: int = 5,
) -> Tuple[Pipeline, Dict[str, Any]]:
    extracted = extract(zip_path, zip_password) if zip_path else []
    if zip_path and not extracted:
        raise RuntimeError("Nenhum arquivo extraído do zip.")

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

    cv_metrics = {}
    if cv_folds and cv_folds > 1:
        cv_model = _build_model()
        cv_metrics = _cross_validate(cv_model, X_train, y_train, n_splits=cv_folds)

    if hasattr(model.named_steps["classifier"], "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model.named_steps["classifier"], "decision_function"):
        scores = model.decision_function(X_test)
        mn, mx = scores.min(), scores.max()
        y_prob = (scores - mn) / (mx - mn) if mx > mn else scores
    else:
        y_prob = np.zeros(len(y_test))
    y_pred_default = model.predict(X_test)

    best_t, best_thr_metrics = _optimal_threshold(y_test, y_prob)
    y_pred_opt = (y_prob >= best_t).astype(int)

    try:
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring=(
                "roc_auc"
                if hasattr(model.named_steps["classifier"], "predict_proba")
                else None
            ),
        )
        ohe: OneHotEncoder = model.named_steps["preprocessor"].named_transformers_[
            "cat"
        ]
        try:
            cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
        except Exception:
            cat_feature_names = [f"cat_{i}" for i in range(len(CATEGORICAL_COLS))]
        num_feature_names = (
            NUMERIC_DERIVED
            + [f"{c}_is_missing" for c in NUMERIC_DERIVED]
            + [f"{c}_was_negative" for c in NUMERIC_DERIVED]
        )
        all_feature_names = cat_feature_names + num_feature_names
        importances = sorted(
            [
                {
                    "feature": (
                        all_feature_names[i] if i < len(all_feature_names) else f"f_{i}"
                    ),
                    "importance_mean": float(perm.importances_mean[i]),
                    "importance_std": float(perm.importances_std[i]),
                }
                for i in range(len(perm.importances_mean))
            ],
            key=lambda x: abs(x["importance_mean"]),
            reverse=True,
        )
        top_importances = importances[:25]
    except Exception as e:
        logging.warning(f"Falha ao calcular permutation importance: {e}")
        top_importances = []

    cm_default = confusion_matrix(y_test, y_pred_default)
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    metrics_default = {
        "classification_report": classification_report(
            y_test, y_pred_default, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": cm_default.tolist(),
        "precision": precision_score(y_test, y_pred_default, zero_division=0),
        "recall": recall_score(y_test, y_pred_default, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_default),
        "f1": f1_score(y_test, y_pred_default, zero_division=0),
    }
    metrics_opt = {"confusion_matrix": cm_opt.tolist(), **best_thr_metrics}

    metrics = {
        "features": feature_cols,
        "n_rows": int(len(df)),
        "positive_rate": float(y.mean()),
        "cv": cv_metrics,
        "default_threshold": metrics_default,
        "optimal_threshold": metrics_opt,
        "used_threshold": best_t,
        "permutation_importance_top25": top_importances,
    }

    return model, metrics
