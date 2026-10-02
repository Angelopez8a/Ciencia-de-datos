# Módulo 1 · De la regresión logística a la tarjeta de puntos (scorecard), con datos reales
# Factor = PDO / ln(2)        Offset = Score - Factor * ln(odds)
# Score total = Offset + Factor * ln(momios bueno:malo)
# Repartido entre las n variables (logística que modela P(malo) con variables WoE):
#   Puntos = (-WoE * beta - alpha / n) * Factor + Offset / n
# Datos: 4,454 solicitudes de crédito del curso de minería de datos de la UPC. Evento = "bad".
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "credito_upc.csv"
URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/modeldata/credit_data.csv"
df = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL, index_col=0)
df["malo"] = (df["Status"] == "bad").astype(int)

train, valid = train_test_split(df, test_size=0.3, random_state=0, stratify=df["malo"])
train, valid = train.copy(), valid.copy()

# 1) Atributos aprendidos SOLO con entrenamiento: cortes por cuantiles (nulos aparte) y normalización de Home
def atributos(x_train, x, k=5):
    c = np.unique(np.quantile(x_train.dropna(), np.linspace(0, 1, k + 1)))
    c[0], c[-1] = -np.inf, np.inf                     # los extremos se conglomeran en los bines de las orillas
    nombres = [f"<= {b:g}" if a == -np.inf else f"> {a:g}" if b == np.inf else f"({a:g}, {b:g}]"
               for a, b in zip(c[:-1], c[1:])]
    b = pd.cut(x, c, labels=nombres)
    return b.cat.add_categories("Sin dato").fillna("Sin dato") if x_train.isna().any() else b

for v in ["Seniority", "Income"]:
    train[v], valid[v] = atributos(train[v], train[v]), atributos(train[v], valid[v])
frecuentes = train["Home"].fillna("Sin categoría").value_counts(normalize=True).loc[lambda f: f >= 0.03].index
for d in (train, valid):
    d["Home"] = d["Home"].fillna("Sin categoría").where(lambda h: h.isin(frecuentes), "Otros")
variables = ["Seniority", "Income", "Records", "Home"]

# 2) Mapa WoE con entrenamiento y regresión logística sobre las variables WoE
woe = {}
for v in variables:
    t = train.groupby(v, observed=True)["malo"].agg(malos="sum", total="count")
    t["buenos"] = t["total"] - t["malos"]
    woe[v] = np.log((t["buenos"] / t["buenos"].sum()) / (t["malos"] / t["malos"].sum()))
    train[f"w_{v}"], valid[f"w_{v}"] = train[v].map(woe[v]), valid[v].map(woe[v])
cols = [f"w_{v}" for v in variables]
modelo = LogisticRegression().fit(train[cols], train["malo"])
alpha, betas, n = modelo.intercept_[0], dict(zip(variables, modelo.coef_[0])), len(variables)
print("Intercepto:", round(float(alpha), 4), "| betas:", {v: round(float(b), 4) for v, b in betas.items()})
print(f"AUC en validación: {roc_auc_score(valid['malo'], modelo.predict_proba(valid[cols])[:, 1]):.3f}\n")

# 3) Escala: cada 20 puntos los momios bueno:malo se duplican; 600 puntos = momios 50:1
PDO, SCORE_REF, ODDS_REF = 20, 600, 50
factor = PDO / np.log(2)
offset = SCORE_REF - factor * np.log(ODDS_REF)
print(f"Factor = {PDO}/ln(2) = {factor:.4f} | Offset = {SCORE_REF} - {factor:.4f}*ln({ODDS_REF}) = {offset:.4f}\n")

def puntos(v, atributo):
    return (-woe[v][atributo] * betas[v] - alpha / n) * factor + offset / n

print("SCORECARD (puntos por atributo)")
for v in variables:
    for atributo in woe[v].index:
        print(f"  {v:<10} {atributo:<13} WoE={woe[v][atributo]:>7.3f}  ->  {puntos(v, atributo):6.1f} puntos")

# 4) Calificar a un solicitante real del conjunto de validación
cliente = valid.iloc[0]
score = sum(puntos(v, cliente[v]) for v in variables)
logit_malo = alpha + sum(betas[v] * woe[v][cliente[v]] for v in variables)
print(f"\nSolicitante {cliente.name}: {dict((v, str(cliente[v])) for v in variables)} | desenlace real: {cliente['Status']}")
print(f"Score = {score:.1f} | P(malo) = {1 / (1 + np.exp(-logit_malo)):.4f} | momios bueno:malo = {np.exp(-logit_malo):.2f}"
      f" | Offset + Factor*ln(momios) = {offset + factor * (-logit_malo):.1f}")

# 5) ¿Separa el score a buenos de malos? Score promedio en validación
tabla_puntos = {v: pd.Series({a: puntos(v, a) for a in woe[v].index}) for v in variables}
valid["score"] = sum(valid[v].map(tabla_puntos[v]).astype(float) for v in variables)
print("\nScore promedio en validación:", valid.groupby("Status")["score"].mean().round(1).to_dict())
