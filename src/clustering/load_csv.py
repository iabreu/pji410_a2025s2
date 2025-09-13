import duckdb
import pandas as pd

def load_csv():
    con = duckdb.connect(database='fiscalizacao.db', read_only=False)
    sql_query = open('sql/clean_fiscalizacao.sql', 'r').read()
    con.execute(sql_query)
    data = con.execute("SELECT * FROM fiscalizacao_limpa").fetchdf()
    con.close()
    return data