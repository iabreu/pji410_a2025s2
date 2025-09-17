
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from src.helpers import extract
from src.clustering.load_csv import load_csv

# Carrega dados
extract("fiscalizacao.csv.zip", "univesp")
df = load_csv()

# Cria variável alvo (1 = multa, 0 = sem multa)
df["tem_multa"] = df["ID Multa"].notnull().astype(int)

# Converte colunas de data (DD/MM/YYYY)
date_cols = [
    "Data Início Fiscalização",
    "Data Notificação",
    "Data Advertencia",
    "Data da Multa",
    "Data Limite Resolução",
    "Data Limite CAC"
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors="coerce")

# Cria features temporais
df["tempo_ate_notificacao"] = (df["Data Notificação"] - df["Data Início Fiscalização"]).dt.days
df["tempo_ate_advertencia"] = (df["Data Advertencia"] - df["Data Notificação"]).dt.days

# Seleciona features úteis
features = [
    "Municipio",
    "Sistema",
    "Subsistema",
    "Código Não Conformidade",
    "Item Não Conforme",
    "tempo_ate_notificacao",
    "tempo_ate_advertencia"
]
X = df[features]
y = df["tem_multa"]

# Separação em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Pré-processamento: OneHot para categóricas
categorical_cols = ["Municipio", "Sistema", "Subsistema", "Código Não Conformidade", "Item Não Conforme"]
numeric_cols = ["tempo_ate_notificacao", "tempo_ate_advertencia"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# HistGradientBoostingClassifier
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(random_state=42))
])

# Treina modelo
model.fit(X_train, y_train)

# Avalia
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
