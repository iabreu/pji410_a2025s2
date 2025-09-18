import os
import sys
from datetime import datetime
import joblib
import logging
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)


if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.boosting.boosting import train_and_evaluate
from src.boosting.report import save_html_report

logging.basicConfig(level=logging.INFO)


def main(event: dict):
    data_dir = os.path.join(PROJECT_ROOT, event.get("data_folder"))
    zip_path = os.path.join(data_dir, event.get("zip_file"))
    csv_path = os.path.join(data_dir, event.get("csv_file"))
    sql_path = os.path.join(PROJECT_ROOT, "sql", event.get("sql_file"))
    db_path = os.path.join(PROJECT_ROOT, event.get("db_file"))
    password = event.get("zip_password")

    models_dir = os.path.join(PROJECT_ROOT, "models")
    model, metrics = train_and_evaluate(
        zip_path=zip_path,
        zip_password=password,
        csv_path=csv_path,
        sql_path=sql_path,
        db_path=db_path,
    )
    model_path = os.path.join(
        models_dir,
        f"boosting_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
    )
    joblib.dump(model, model_path)
    logging.info(f"Modelo salvo em: {model_path}")

    html_report_path = os.path.join(
        models_dir, f"boosting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    save_html_report(metrics, html_report_path)
    logging.info(f"Relatório HTML salvo em: {html_report_path}")


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
