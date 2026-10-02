# Módulo 1 · Flujo completo de la sesión 9: discretizar (2 a 6 cortes) -> elegir por IV -> mapa WoE -> logística
# Datos reales: IBM Telco Customer Churn (7,043 clientes). Evento = el cliente abandona la compañía.
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "telco_churn_ibm.csv"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL)
df["Churn_Value"] = (df["Churn"] == "Yes").astype(int)
tgt = "Churn_Value"

# 1) Partición ANTES de calcular IV/WoE: los mapas se aprenden solo con el 70 %
train, valid = train_test_split(df, train_size=0.7, random_state=0, stratify=df[tgt])
train, valid = train.copy(), valid.copy()
print(f"train {train.shape} | valid {valid.shape} | tasa de fuga train = {train[tgt].mean():.3f}\n")

def iv_woe(d, var):
    t = d.groupby(var, observed=True)[tgt].agg(total="count", eventos="sum")
    t["no_eventos"] = t["total"] - t["eventos"]
    pe, pn = t["eventos"] / t["eventos"].sum(), t["no_eventos"] / t["no_eventos"].sum()
    woe = np.log(pn / pe)                           # WoE = ln(%no evento / %evento)
    return float(((pn - pe) * woe).sum()), woe

# 2) Discretizar cada continua con 2..6 cortes por cuantiles y quedarse con el número de mayor IV
mejores = {}
for var in ["MonthlyCharges", "tenure"]:
    res = []
    for k in range(2, 7):
        cortes = np.unique(np.quantile(train[var], np.linspace(0, 1, k + 1)))
        cortes[0], cortes[-1] = -np.inf, np.inf           # los extremos se "conglomeran" en los bines de las orillas
        iv, _ = iv_woe(train.assign(b=pd.cut(train[var], cortes)), "b")
        res.append((k, iv, cortes))
    print(f"{var:<15}", "  ".join(f"k={k}: IV={iv:.4f}" for k, iv, _ in res))
    k, iv, cortes = max(res, key=lambda r: r[1])
    mejores[var] = cortes
    print(f"{'':<15} -> se elige k = {k} (IV = {iv:.4f})")

# 3) Mapa WoE aprendido en train y aplicado a ambos conjuntos
cols_woe = []
for var in ["MonthlyCharges", "tenure", "Contract"]:
    if var in mejores:
        bt, bv = pd.cut(train[var], mejores[var]), pd.cut(valid[var], mejores[var])
    else:
        bt, bv = train[var], valid[var]
    iv, woe = iv_woe(train.assign(b=bt), "b")
    train[f"w_{var}"], valid[f"w_{var}"] = bt.map(woe).astype(float), bv.map(woe).astype(float)
    cols_woe.append(f"w_{var}")
    print(f"\nWoE de {var} (IV = {iv:.4f}):\n", woe.round(4).to_string(), sep="")

# 4) Regresión logística sobre las variables WoE
m = LogisticRegression().fit(train[cols_woe], train[tgt])
p = m.predict_proba(valid[cols_woe])[:, 1]
print("\nCoeficientes:", m.coef_.round(3), "| intercepto:", m.intercept_.round(3))
print(f"Validación -> AUC = {roc_auc_score(valid[tgt], p):.3f} | ACC = {accuracy_score(valid[tgt], m.predict(valid[cols_woe])):.3f}")
print("Matriz de confusión:\n", confusion_matrix(valid[tgt], m.predict(valid[cols_woe])))
