# Módulo 1 · Limpieza de la TAD: ausentes, imputación, KS, varianza, extremos y correlación
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import ks_2samp

rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "ingreso": rng.lognormal(10, 0.5, n),
    "edad": rng.integers(18, 70, n).astype(float),
    "saldo": rng.normal(5000, 1500, n),
    "constante": np.r_[np.ones(n - 3), [2, 2, 2]],     # casi sin varianza
})
df["gasto"] = df["ingreso"] * 0.3 + rng.normal(0, 500, n)   # muy correlacionada con ingreso
df.loc[rng.choice(n, 100, replace=False), "edad"] = np.nan    # 10% nulos
df.loc[rng.choice(n, 450, replace=False), "saldo"] = np.nan   # 45% nulos

# 1) Porcentaje de valores ausentes
miss = (1 - df.describe().T["count"] / len(df)) * 100
print("% de ausentes:\n", miss.round(1).to_string(), "\n")

# 2) Imputación con la mediana ("entrenar" = calcular la mediana; "transform" = rellenar)
X = df.copy()
im = SimpleImputer(strategy="median")
X[X.columns] = im.fit_transform(X)

# 3) ¿La imputación alteró la distribución? KS entre original (sin nulos) e imputada
ks = pd.DataFrame([(c, ks_2samp(df[c].dropna(), X[c]).statistic) for c in X.columns],
                  columns=["variable", "ks"])
print(ks.round(4).to_string(index=False))
rotas = ks.loc[ks["ks"] > 0.1, "variable"].tolist()
print("Variables 'rotas' (KS > 0.1) -> se descartan:", rotas, "\n")
X = X.drop(columns=rotas)

# 4) Varianza: quitar variables casi constantes
vt = VarianceThreshold(threshold=0.1).fit(X)
sin_var = [c for c, ok in zip(X.columns, vt.get_support()) if not ok]
print("Sin varianza:", sin_var)
X = X.drop(columns=sin_var)

# 5) Extremos: marcar registros fuera de los percentiles 1 y 99
lim = X.describe(percentiles=[.01, .99]).T[["1%", "99%"]]
ext = pd.concat([(X[c] < li) | (X[c] > ls) for c, (li, ls) in lim.iterrows()], axis=1).any(axis=1)
print(f"Registros con algún extremo: {ext.mean() * 100:.1f}%")
X = X[~ext]
print("Dimensión final:", X.shape, "\n")

# 6) Análisis bivariado: correlación (multicolinealidad)
print(X.corr().round(2))
