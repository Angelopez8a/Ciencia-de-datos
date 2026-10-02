# Módulo 2 · Métricas para elegir el modelo campeón y monitorearlo (KS, Gini, Lift, PSI), con datos reales
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (max_error, mean_absolute_error, mean_squared_error, r2_score,
                             median_absolute_error, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, balanced_accuracy_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATOS = Path(__file__).resolve().parents[1] / "datos"
RD = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv"
def leer(nombre, url, **kw):
    return pd.read_csv(DATOS / nombre if (DATOS / nombre).exists() else url, **kw)

# ---------------- Regresión: estatura del hijo según la de sus padres y su sexo (Galton) ----------------
g = leer("galton_familias.csv", f"{RD}/HistData/GaltonFamilies.csv", index_col=0)
X = np.c_[g["father"] * 2.54, g["mother"] * 2.54, (g["gender"] == "male").astype(int)]
y = g["childHeight"].to_numpy() * 2.54
Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=0)
pred = LinearRegression().fit(Xt, yt).predict(Xv)
print(f"REGRESIÓN (estatura del hijo en cm; {len(yv)} personas en el conjunto de prueba)")
print(f"  Max Error = {max_error(yv, pred):.2f}")
print(f"  MAE       = {mean_absolute_error(yv, pred):.2f}   (desviación estándar de y = {yv.std():.2f})")
print(f"  MSE       = {mean_squared_error(yv, pred):.2f}")
print(f"  RMSE      = {np.sqrt(mean_squared_error(yv, pred)):.2f}")
print(f"  MedAE     = {median_absolute_error(yv, pred):.2f}")
print(f"  R²        = {r2_score(yv, pred):.3f}")

# ---------------- Clasificación: abandono de clientes de una telefónica (IBM) ----------------
t = leer("telco_churn_ibm.csv",
         "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")
t["TotalCharges"] = pd.to_numeric(t["TotalCharges"], errors="coerce").fillna(0)   # 11 clientes sin cargos aún
y = (t["Churn"] == "Yes").astype(int)
X = t.drop(columns=["customerID", "Churn"])
numericas = ["tenure", "MonthlyCharges", "TotalCharges"]
categoricas = [c for c in X.columns if c not in numericas]
modelo = make_pipeline(
    make_column_transformer((StandardScaler(), numericas), (OneHotEncoder(drop="if_binary"), categoricas)),
    LogisticRegression(max_iter=2000))
Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
p = modelo.fit(Xt, yt).predict_proba(Xv)[:, 1]
yhat = (p >= 0.5).astype(int)
print(f"\nCLASIFICACIÓN (regresión logística; {len(yv):,} clientes de prueba; corte 0.5)")
print("  Matriz [[TN, FP], [FN, TP]] =", confusion_matrix(yv, yhat).tolist())
print(f"  Accuracy={accuracy_score(yv, yhat):.3f}  Precision={precision_score(yv, yhat):.3f}  "
      f"Recall={recall_score(yv, yhat):.3f}  F1={f1_score(yv, yhat):.3f}  "
      f"BalancedAcc={balanced_accuracy_score(yv, yhat):.3f}")
auc = roc_auc_score(yv, p)
print(f"  AUC (C-statistic) = {auc:.3f}  ->  Gini = 2·AUC − 1 = {2 * auc - 1:.3f}"
      f"   (AUC en entrenamiento = {roc_auc_score(yt, modelo.predict_proba(Xt)[:, 1]):.3f})")
print(f"  KS = máx. separación entre distribuciones de eventos y no eventos = "
      f"{ks_2samp(p[yv == 1], p[yv == 0]).statistic:.3f}")

# ---------------- Lift por decil ----------------
d = pd.DataFrame({"score": p, "evento": yv.to_numpy()}).sort_values("score", ascending=False, kind="stable")
d["decil"] = pd.qcut(np.arange(len(d)), 10, labels=range(1, 11))
por_decil = d.groupby("decil", observed=True)["evento"]
lift = por_decil.mean() / d["evento"].mean()
captura = por_decil.sum().cumsum() / d["evento"].sum()
print(f"\nLIFT por decil (1 = 10 % con mayor score; tasa natural = {d['evento'].mean():.3f}):")
print("  ", lift.round(2).tolist())
print("  Abandonos capturados, % acumulado:", (captura * 100).round(1).tolist())

# ---------------- PSI / índice de desviación ----------------
# PIB per cápita de 142 países (Gapminder, dólares ajustados por inflación) en 6 bandas fijas:
# menos de 1,000; 1,000-2,000; 2,000-5,000; 5,000-10,000; 10,000-20,000; 20,000 o más
gm = leer("gapminder.csv", f"{RD}/gapminder/gapminder.csv", index_col=0)
bandas = [1000, 2000, 5000, 10000, 20000]
def distribucion(anio):
    s = gm.loc[gm["year"] == anio, "gdpPercap"]
    return np.bincount(np.searchsorted(bandas, s, side="right"), minlength=len(bandas) + 1) / len(s)
def psi(esperado, actual):
    return float(np.sum((actual - esperado) * np.log(actual / esperado)))
estado = lambda v: "estable" if v < 0.1 else "cambio menor" if v <= 0.25 else "cambio MAYOR"

anios = sorted(gm["year"].unique())
print("\nPSI del PIB per cápita de 142 países")
print(f"  {'Año':<6}{'contra 1952 (referencia)':<29}contra el periodo anterior")
for anterior, anio in zip(anios[:-1], anios[1:]):
    v0, v1 = psi(distribucion(1952), distribucion(anio)), psi(distribucion(anterior), distribucion(anio))
    print(f"  {anio:<6}{v0:<7.3f}{estado(v0):<22}{v1:<7.3f}{estado(v1)}")
