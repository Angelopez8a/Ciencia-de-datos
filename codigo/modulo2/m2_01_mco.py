# Módulo 2 · Mínimos cuadrados ordinarios (MCO) "a mano" vs scikit-learn
# Datos reales: estaturas de 934 hijos y de sus padres en 205 familias, registradas por Francis Galton (1886).
# Con estos datos Galton describió la "regresión hacia la media", el fenómeno que dio nombre a la técnica.
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "galton_familias.csv"
URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/HistData/GaltonFamilies.csv"
g = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL, index_col=0)
CM = 2.54                                        # las estaturas vienen en pulgadas
X = (g["midparentHeight"] * CM).to_numpy()       # estatura promedio de los padres (Galton multiplicó la de la madre por 1.08)
y = (g["childHeight"] * CM).to_numpy()           # estatura del hijo o de la hija

# Fórmulas cerradas: β1 = Cov(X, y) / Var(X),  β0 = ȳ − β1·x̄
b1 = np.sum((X - X.mean()) * (y - y.mean())) / np.sum((X - X.mean()) ** 2)
b0 = y.mean() - b1 * X.mean()
print(f"A mano  : β0 = {b0:.4f}, β1 = {b1:.4f}")

lr = LinearRegression().fit(X.reshape(-1, 1), y)
print(f"sklearn : β0 = {lr.intercept_:.4f}, β1 = {lr.coef_[0]:.4f}")

residuales = y - (b0 + b1 * X)
print("Suma de residuales:", round(float(residuales.sum()), 6) + 0.0, "| primeros 5:", residuales[:5].round(2))
print(f"Interpretación: por cada centímetro adicional en la estatura promedio de los padres, "
      f"la del hijo aumenta {b1:.3f} cm en promedio")

# Regresión hacia la media: padres 10 cm más altos que el promedio tienen hijos solo ~6 cm más altos
print(f"Padres 10 cm sobre la media ({X.mean() + 10:.1f} cm) -> hijo esperado {b0 + b1 * (X.mean() + 10):.1f} cm, "
      f"es decir, {b1 * 10:.1f} cm sobre la media de los hijos ({y.mean():.1f} cm)")

# Regresión múltiple: estatura = β0 + β1·padre + β2·madre + β3·hombre
Xm = np.c_[g["father"] * CM, g["mother"] * CM, (g["gender"] == "male").astype(int)]
m = LinearRegression().fit(Xm, y)
print(f"\nMúltiple: estatura = {m.intercept_:.2f} {m.coef_[0]:+.3f}·padre {m.coef_[1]:+.3f}·madre "
      f"{m.coef_[2]:+.2f}·hombre   (R² = {m.score(Xm, y):.3f})")
