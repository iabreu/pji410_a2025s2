# pji410_a2025s2


## Resumo
Aplicação de técnicas de análise de dados e aprendizagem de máquina sobre datasets, com objetivo de identificar padrões e gerar visualizações que apoiem decisões.


## Objetivos

- Preparar e limpar os dados.
- Explorar padrões e gerar visualizações.
- Treinar modelos de machine learning (por ex.:  clustering).

## Estrutura sugerida do repositório

- `data/` — datasets (não adicione dados sensíveis ao repositório público).
- `src/` — módulos Python: pré-processamento, treino, inferência, utilitários.


## Tecnologias e dependências

- Python 3.11+ (usar virtualenv ou venv).
- Bibliotecas principais: pandas, numpy, scikit-learn, matplotlib, jupyter.

Exemplo mínimo de `requirements.txt`:

```text
pandas
numpy
scikit-learn
matplotlib
jupyterlab
```

## Como começar (local)

1. Criar ambiente virtual e instalar dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Colocar os datasets no diretório `data/` (manter cópias seguras fora do repo se sensível).

3. Rodar scripts em `src/`.

## Contrato mínimo (inputs / outputs)

- Input: CSV(s) com registros.
- Output: relatórios com recomendações.
- Possíveis problemas: valores nulos/inconsistentes; previstas etapas de limpeza e validação.

## Próximos passos sugeridos

1. Inspecionar os arquivos na pasta `data/` e documentar colunas e tipos.
2. Criar os scripts na pasta `src/` e definir a(s) tarefa(s) de ML (ex.: classificar gravidade, agrupar tipos de não conformidade).