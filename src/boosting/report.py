from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang='pt-BR'>
<head>
  <meta charset='utf-8'/>
  <title>Relatório Boosting Fiscalização</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 1.5rem; }}
    h1 {{ font-size: 1.6rem; }}
    h2 {{ border-bottom: 1px solid #ccc; padding-bottom: .2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
    th, td {{ border: 1px solid #ddd; padding: .4rem .6rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    code {{ background: #f0f0f0; padding: 2px 4px; border-radius: 3px; }}
    .metric-block {{ margin-bottom: 1.2rem; }}
  </style>
</head>
<body>
  <h1>Relatório do Modelo (HistGradientBoosting)</h1>
  <p>Gerado em: {generated_at}</p>
  <h2>Resumo</h2>
  <ul>
    <li>Linhas utilizadas: {n_rows}</li>
    <li>Proporção classe positiva (tem_multa): {positive_rate:.4f}</li>
    <li>Threshold ótimo usado: {used_threshold:.2f}</li>
  </ul>
  <h2>Métricas - Threshold Padrão (0.5)</h2>
  <pre>{classification_report}</pre>
  <table>
    <tr><th>ROC AUC</th><th>Precision</th><th>Recall</th><th>Balanced Accuracy</th><th>F1</th></tr>
    <tr><td>{roc_auc:.4f}</td><td>{precision:.4f}</td><td>{recall:.4f}</td><td>{balanced_accuracy:.4f}</td><td>{f1:.4f}</td></tr>
  </table>
  <h3>Matriz de Confusão (default)</h3>
  <pre>{conf_default}</pre>
  <h2>Métricas - Threshold Ótimo</h2>
  <table>
    <tr><th>Limiar</th><th>Balanced Accuracy</th><th>F1</th><th>Precision</th><th>Recall</th></tr>
    <tr><td>{opt_threshold:.2f}</td><td>{opt_balanced_accuracy:.4f}</td><td>{opt_f1:.4f}</td><td>{opt_precision:.4f}</td><td>{opt_recall:.4f}</td></tr>
  </table>
  <h3>Matriz de Confusão (ótimo)</h3>
  <pre>{conf_opt}</pre>
  <h2>Cross-Validation (médias)</h2>
  <table>
    <tr><th>ROC AUC</th><th>Balanced Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
    <tr><td>{cv_roc_auc}</td><td>{cv_balanced_accuracy}</td><td>{cv_precision}</td><td>{cv_recall}</td><td>{cv_f1}</td></tr>
  </table>
  <h2>Top 25 Importâncias por Permutação</h2>
  <table>
    <tr><th>Feature</th><th>Importance (mean)</th><th>Std</th></tr>
    {importance_rows}
  </table>
  <h2>Lista de Features</h2>
  <pre>{feature_list}</pre>
</body>
</html>
"""


def build_html_report(metrics: Dict[str, Any]) -> str:
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else v

    default = metrics.get("default_threshold", {})
    optimal = metrics.get("optimal_threshold", {})
    cv = metrics.get("cv", {})
    importance = metrics.get("permutation_importance_top25", [])
    importance_rows = "\n".join(
        f"<tr><td>{i['feature']}</td><td>{i['importance_mean']:.5f}</td><td>{i['importance_std']:.5f}</td></tr>"
        for i in importance
    )
    html = HTML_TEMPLATE.format(
        generated_at=datetime.utcnow().isoformat(timespec="seconds"),
        n_rows=metrics.get("n_rows", 0),
        positive_rate=metrics.get("positive_rate", 0.0),
        used_threshold=metrics.get("used_threshold", 0.5),
        classification_report=default.get("classification_report", ""),
        roc_auc=default.get("roc_auc", 0.0),
        precision=default.get("precision", 0.0),
        recall=default.get("recall", 0.0),
        balanced_accuracy=default.get("balanced_accuracy", 0.0),
        f1=default.get("f1", 0.0),
        conf_default=default.get("confusion_matrix", []),
        opt_threshold=optimal.get("threshold", metrics.get("used_threshold", 0.5)),
        opt_balanced_accuracy=optimal.get("balanced_accuracy", 0.0),
        opt_f1=optimal.get("f1", 0.0),
        opt_precision=optimal.get("precision", 0.0),
        opt_recall=optimal.get("recall", 0.0),
        conf_opt=optimal.get("confusion_matrix", []),
        cv_roc_auc=fmt(cv.get("roc_auc", "-")),
        cv_balanced_accuracy=fmt(cv.get("balanced_accuracy", "-")),
        cv_precision=fmt(cv.get("precision", "-")),
        cv_recall=fmt(cv.get("recall", "-")),
        cv_f1=fmt(cv.get("f1", "-")),
        importance_rows=importance_rows,
        feature_list="\n".join(metrics.get("features", [])),
    )
    return html


def save_html_report(metrics: Dict[str, Any], path: str | Path) -> str:
    html = build_html_report(metrics)
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return str(path)


__all__ = ["build_html_report", "save_html_report"]
