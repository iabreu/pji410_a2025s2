import duckdb
import pandas as pd
import os
from typing import Optional


def load_csv(
    csv_path: Optional[str] = None,
    sql_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """Carrega dados processados via SQL/duckdb.

    Parametros podem ser fornecidos externamente; se ausentes são inferidos a partir
    da estrutura do projeto (este arquivo dentro de src/helpers/).
    """
    helpers_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(helpers_dir)
    project_root = os.path.dirname(src_dir)

    if csv_path is None:
        csv_path = os.path.join(project_root, "data", "fiscalizacao.csv")
    if sql_path is None:
        sql_path = os.path.join(project_root, "sql", "clean_fiscalizacao.sql")
    if db_path is None:
        db_path = os.path.join(project_root, "fiscalizacao.db")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV não encontrado em: {csv_path}")
    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"SQL não encontrado em: {sql_path}")

    with open(sql_path, "r") as f:
        sql_raw = f.read()
        sql_query = sql_raw.replace("{{CSV_PATH}}", csv_path).replace(
            "'data/fiscalizacao.csv'", f"'{csv_path}'"
        )

    con = duckdb.connect(database=db_path, read_only=False)
    try:
        con.execute(sql_query)
        data = con.execute("SELECT * FROM fiscalizacao_limpa").fetchdf()
    finally:
        con.close()
    return data
