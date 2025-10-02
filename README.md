# pji410_a2025s2

## Objetivos

- Limpar e preparar dados de fiscalização
- Predizer probabilidade de multas
- Agrupar municípios por padrão de não conformidades (clustering)
- Prever casos futuros (forecasting - 48 meses)
- Classificar status de resolução (aberto/baixado)
- Estimar tempo de resolução

## Estrutura do repositório

```
pji410_a2025s2/
├── data/                    # CSV protegido por senha
│   └── fiscalizacao.csv.zip
├── src/                     # Módulos de análise
│   ├── prediction/         # Predição de multas
│   ├── clustering/         # Agrupamento de municípios
│   ├── forecasting/        # Previsão de casos futuros
│   ├── resolution_classification/  # Classificação de status
│   ├── resolution_regression/      # Estimativa de tempo
│   └── helpers/            # Utilitários (extração, carregamento)
├── sql/                    # Queries de limpeza (DuckDB)
│   └── clean_fiscalizacao.sql
└── results/                # CSVs gerados (Municipio como chave)
```

## Dependências

- Python 3.11+
- Bibliotecas: pandas, numpy, scikit-learn, duckdb

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install pandas numpy scikit-learn duckdb
```

## Como executar

Cada módulo gera um CSV em `results/`:

```bash
# Predição de multas
python src/prediction/main.py

# Clustering de municípios
python src/clustering/main.py

# Previsão de casos (48 meses)
python src/forecasting/main.py
```

## Outputs

Todos os CSVs possuem `Municipio` como primeira coluna para facilitar a análise.

### 1. clustering_municipio_YYYYMMDD_HHMMSS.csv

Agrupamento de municípios por padrão de não conformidades.

**Colunas:**

- `Municipio` - nome do município
- `Cluster` - ID do cluster (0, 1, 2, 3...)
- `total_nao_conformidades` - total de não conformidades
- `top_codigo_1`, `top_codigo_2`, `top_codigo_3` - códigos mais frequentes
- `freq_codigo_1`, `freq_codigo_2`, `freq_codigo_3` - frequências dos códigos

### 2. forecasting_municipio_YYYYMMDD_HHMMSS.csv

Previsão de casos futuros (12, 24 e 48 meses) por município.

**Colunas:**

- `Municipio` - nome do município
- `historico_total` - total histórico de casos
- `historico_12m` - casos nos últimos 12 meses
- `previsao_12m` - previsão para próximos 12 meses
- `previsao_24m` - previsão para próximos 24 meses
- `previsao_48m` - previsão para próximos 48 meses
- `tendencia` - tendência (crescente, estável, decrescente)

## Configuração

Os módulos utilizam um dicionário `event` para configuração:

```python
event = {
    "data_folder": "data",
    "zip_file": "fiscalizacao.csv.zip",
    "csv_file": "fiscalizacao.csv",
    "sql_file": "clean_fiscalizacao.sql",
    "db_file": "fiscalizacao.db",
    "zip_password": "",
}
```
