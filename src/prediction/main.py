import os
import sys
import logging
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction.prediction import train_and_evaluate

logging.basicConfig(level=logging.INFO)


def main(event: dict):
    """Execute prediction analysis and save municipio-level CSV."""
    data_dir = os.path.join(PROJECT_ROOT, event.get("data_folder"))
    zip_path = (
        os.path.join(data_dir, event.get("zip_file")) if event.get("zip_file") else None
    )
    csv_path = os.path.join(data_dir, event.get("csv_file"))
    sql_path = os.path.join(PROJECT_ROOT, "sql", event.get("sql_file"))
    db_path = os.path.join(PROJECT_ROOT, event.get("db_file"))
    password = event.get("zip_password")

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    df_municipio = train_and_evaluate(
        zip_path=zip_path,
        zip_password=password,
        csv_path=csv_path,
        sql_path=sql_path,
        db_path=db_path,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"prediction_municipio_{timestamp}.csv")
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
