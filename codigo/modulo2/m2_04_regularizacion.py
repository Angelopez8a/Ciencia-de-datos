# Módulo 2 · Regresión lineal vs LASSO (L1) vs Ridge (L2) vs ElasticNet vs LARS
# Datos reales: 442 pacientes con diabetes, publicados por Efron, Hastie, Johnstone y Tibshirani (2004) en el
# artículo que presentó el método LARS. Diez variables medidas al inicio: edad (age), sexo (sex), índice de masa
# corporal (bmi), presión arterial media (bp) y seis análisis de sangre (s1 a s6; por ejemplo, s3 es el colesterol
# HDL). La variable y mide el avance de la enfermedad un año después.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoLars

X, y = load_diabetes(return_X_y=True, as_frame=True)
X = StandardScaler().fit_transform(X)
nombres = load_diabetes().feature_names

modelos = {
    "Lineal":            LinearRegression(),
    "LASSO a=5":         Lasso(alpha=5),
    "Ridge a=50":        Ridge(alpha=50),
    "ElasticNet a=5,l1=0.5": ElasticNet(alpha=5, l1_ratio=0.5),
    "LassoLars a=5":     LassoLars(alpha=5),
}
print(f"{'variable':<8}" + "".join(f"{k:>23}" for k in modelos))
coefs = {k: m.fit(X, y).coef_ for k, m in modelos.items()}
for i, v in enumerate(nombres):
    print(f"{v:<8}" + "".join(f"{coefs[k][i]:>23.2f}" for k in modelos))
print(f"{'# ceros':<8}" + "".join(f"{int(np.sum(np.isclose(c, 0))):>23}" for c in coefs.values()))

print("\nCamino de LASSO: coeficientes en cero conforme sube alpha")
for a in [0.1, 1, 5, 10, 20, 40]:
    c = Lasso(alpha=a).fit(X, y).coef_
    print(f"  alpha = {a:>4}: variables que sobreviven = {int(np.sum(~np.isclose(c, 0)))}")
