# Módulo 1 · Limpieza de la TAD: ausentes, imputación, KS, varianza, extremos y correlación
# Datos reales: mediciones diarias de calidad del aire en Nueva York, de mayo a septiembre de 1973
# (ozono en ppb, radiación solar, viento en mph y temperatura máxima en °F). Hay días sin medición.
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import ks_2samp

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "calidad_aire_nueva_york_1973.csv"
URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv"
df = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL, index_col=0)
df = df[["Ozone", "Solar.R", "Wind", "Temp"]]          # Month y Day forman la fecha: no son predictoras
print("Días medidos:", len(df))

# 1) Porcentaje de valores ausentes (fórmula del notebook)
miss = (1 - df.describe().T["count"] / len(df)) * 100
print("\n% de ausentes:\n", miss.round(1).to_string(), sep="")

# 2) Imputación con la mediana ("fit" calcula la mediana; "transform" rellena)
X = df.copy()
im = SimpleImputer(strategy="median")
X[X.columns] = im.fit_transform(X)
print("\nMedianas usadas:", {c: round(float(v), 1) for c, v in zip(X.columns, im.statistics_)})

# 3) ¿La imputación alteró la distribución? KS entre la variable original (sin nulos) y la imputada
ks = pd.DataFrame([(c, ks_2samp(df[c].dropna(), X[c]).statistic) for c in X.columns], columns=["variable", "ks"])
print("\n", ks.round(4).to_string(index=False), sep="")
rotas = ks.loc[ks["ks"] > 0.1, "variable"].tolist()
print("Variables 'rotas' (KS > 0.1) -> se descartan:", rotas)
print("Antes de descartarla, correlación de Ozone con las demás:", df.corr()["Ozone"].drop("Ozone").round(2).to_dict())
X = X.drop(columns=rotas)

# 4) Varianza: quitar variables casi constantes
vt = VarianceThreshold(threshold=0.1).fit(X)
print("\nVarianzas:", X.var(ddof=0).round(2).to_dict())
print("Sin varianza (umbral 0.1):", [c for c, ok in zip(X.columns, vt.get_support()) if not ok])

# 5) Extremos: marcar registros fuera de los percentiles 1 y 99 en alguna variable
lim = X.describe(percentiles=[.01, .99]).T[["1%", "99%"]]
ext = pd.concat([(X[c] < li) | (X[c] > ls) for c, (li, ls) in lim.iterrows()], axis=1).any(axis=1)
print(f"\nDías con algún extremo: {ext.sum()} ({ext.mean() * 100:.1f}%)")
X = X[~ext]
print("Dimensión final:", X.shape)

# 6) Análisis bivariado: correlación (multicolinealidad)
print("\n", X.corr().round(2).to_string(), sep="")
