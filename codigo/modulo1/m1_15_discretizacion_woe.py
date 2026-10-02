# Módulo 1 · Flujo completo de la sesión 9: discretizar (2 a 6 cortes) -> elegir por IV -> mapa WoE -> logística
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

rng = np.random.default_rng(11)
n = 4000
df = pd.DataFrame({
    "Monthly_Charge": rng.uniform(18, 120, n).round(2),
    "Tenure": rng.integers(0, 73, n),
    "Contract": rng.choice(["Month-to-Month", "One Year", "Two Year"], n, p=[0.5, 0.25, 0.25]),
})
logit = (-1.2 + 0.02 * (df["Monthly_Charge"] - 65) - 0.04 * (df["Tenure"] - 36)
         + df["Contract"].map({"Month-to-Month": 1.0, "One Year": -0.6, "Two Year": -1.5}))
df["Churn_Value"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
tgt = "Churn_Value"

# 1) Partición ANTES de calcular IV/WoE: los mapas se aprenden solo con el 70 %
train, valid = train_test_split(df, train_size=0.7, random_state=0, stratify=df[tgt])
print(f"train {train.shape} | valid {valid.shape} | tasa de fuga train = {train[tgt].mean():.3f}\n")

def iv_woe(d, var):
    t = d.groupby(var, observed=True)[tgt].agg(total="count", eventos="sum")
    t["no_eventos"] = t["total"] - t["eventos"]
    pe, pn = t["eventos"] / t["eventos"].sum(), t["no_eventos"] / t["no_eventos"].sum()
    woe = np.log(pn / pe)                           # WoE = ln(%no evento / %evento)
    return float(((pn - pe) * woe).sum()), woe

# 2) Discretizar cada continua con 2..6 cortes por cuantiles y quedarse con el de mayor IV
mejores = {}
for var in ["Monthly_Charge", "Tenure"]:
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
for var in ["Monthly_Charge", "Tenure", "Contract"]:
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
