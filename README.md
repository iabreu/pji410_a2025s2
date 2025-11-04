# pji410_a2025s2

## Objetivos

- Limpar e preparar dados para análise
- Agrupar municípios por padrão de não conformidades (clustering)
- Classificar probabilidade de status (vencido) e consolidar por município
- Prever casos futuros (forecasting - 48 meses)

## Estrutura do repositório

```
pji410_a2025s2/
├── data/                    # CSV protegido por senha
│   └── fiscalizacao.csv.zip
├── src/                     # Módulos de análise
│   ├── clustering/         # Agrupamento de municípios
│   ├── forecasting/        # Previsão de casos futuros
│   ├── resolution_classification/  # Classificação de status
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

Cada módulo gera um CSV em `results/`. Você pode chamar diretamente via `python -m` para respeitar imports relativos:

```bash
# Clustering de municípios
python -m src.clustering.main

# Classificação (probabilidade de vencido + agregação)
python -m src.resolution_classification.main

# Previsão de casos (12, 24, 48 meses)
python -m src.forecasting.main
```

## Outputs

Todos os CSVs possuem `Municipio` como primeira coluna para facilitar a análise.

### 1. clustering_municipio_YYYYMMDD_HHMMSS.csv

Agrupamento de municípios por padrão de não conformidades.

**Colunas:**

- `Municipio` - nome do município
- `total_nao_conformidades` - total de não conformidades
- `top_codigo_1`, `top_codigo_2`, `top_codigo_3` - códigos mais frequentes
- `freq_codigo_1`, `freq_codigo_2`, `freq_codigo_3` - frequências dos códigos

### 2. classification_municipio_YYYYMMDD_HHMMSS.csv

Classificação do status com modelo de Regressão Logística; a classe positiva é `vencido`. O CSV agrega por município.

**Colunas:**

- `Municipio` - nome do município
- `total_casos` - total de registros
- `casos_abertos` - contagem de casos com Condição = "aberto"
- `casos_baixados` - contagem de casos com Condição = "baixado"
- `casos_vencidos` - contagem de casos com Condição = "vencido" (também soma do alvo positivo)
- `prob_vencido_media` - média da probabilidade prevista de `vencido` pelo modelo

### 3. forecasting_municipio_YYYYMMDD_HHMMSS.csv

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

Observações:

- Para executar com `python -m`, garanta que está na raiz do projeto.
