# Módulo 2 · Validar los 4 supuestos de la regresión lineal
import numpy as np
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

rng = np.random.default_rng(1)
x = rng.uniform(0, 10, 300)

def diagnostico(nombre, y):
    modelo = sm.OLS(y, sm.add_constant(x)).fit()
    r = modelo.resid
    _, p_bp, _, _ = het_breuschpagan(r, sm.add_constant(x))
    _, p_sw = stats.shapiro(r)
    print(f"{nombre}")
    print(f"  R² = {modelo.rsquared:.3f}")
    print(f"  Homocedasticidad (Breusch-Pagan) p = {p_bp:.4f} -> {'OK' if p_bp > 0.05 else 'HETEROCEDASTICIDAD'}")
    print(f"  Independencia (Durbin-Watson) = {durbin_watson(r):.3f}  (≈2 sin autocorrelación)")
    print(f"  Normalidad (Shapiro-Wilk) p = {p_sw:.4f} -> {'OK' if p_sw > 0.05 else 'NO normal'}")

# Caso 1: todo se cumple
diagnostico("Caso 1: y = 3 + 2x + ruido constante", 3 + 2 * x + rng.normal(0, 1, 300))
# Caso 2: varianza que crece con x (forma de cono)
diagnostico("Caso 2: ruido proporcional a x", 3 + 2 * x + rng.normal(0, 1, 300) * x)
# Caso 2 corregido con log(y)
y2 = np.exp(0.5 + 0.2 * x + rng.normal(0, 0.3, 300))
diagnostico("Caso 3: y exponencial (sin transformar)", y2)
diagnostico("Caso 3 corregido: log(y)", np.log(y2))
