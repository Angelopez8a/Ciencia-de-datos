# Módulo 2 · Mínimos cuadrados ordinarios (MCO) "a mano" vs scikit-learn
import numpy as np
from sklearn.linear_model import LinearRegression

# Precio de autos (miles) vs antigüedad (años)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([30, 27, 25, 22, 20, 18, 15, 14], dtype=float)

# Fórmulas cerradas: β1 = Cov(X, y) / Var(X),  β0 = ȳ − β1·x̄
b1 = np.sum((X - X.mean()) * (y - y.mean())) / np.sum((X - X.mean()) ** 2)
b0 = y.mean() - b1 * X.mean()
print(f"A mano  : β0 = {b0:.4f}, β1 = {b1:.4f}")

lr = LinearRegression().fit(X.reshape(-1, 1), y)
print(f"sklearn : β0 = {lr.intercept_:.4f}, β1 = {lr.coef_[0]:.4f}")

residuales = y - (b0 + b1 * X)
print("Residuales:", residuales.round(3), "| suma =", round(residuales.sum(), 10))
print(f"Interpretación: cada año extra de antigüedad, el precio baja {abs(b1):.3f} mil en promedio")

# Regresión múltiple: precio = β0 + β1*millas + β2*antigüedad
rng = np.random.default_rng(0)
millas = rng.uniform(5, 150, 200)
antig = rng.uniform(0, 15, 200)
precio = 40 - 0.08 * millas - 1.2 * antig + rng.normal(0, 1.5, 200)
m = LinearRegression().fit(np.c_[millas, antig], precio)
print(f"\nMúltiple: precio = {m.intercept_:.2f} {m.coef_[0]:+.3f}*millas {m.coef_[1]:+.3f}*antigüedad")
