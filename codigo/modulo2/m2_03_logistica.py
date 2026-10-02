# Módulo 2 · Regresión logística: momios, logit y probabilidad
import numpy as np
from sklearn.linear_model import LogisticRegression

# Horas de estudio vs aprobar (1) / reprobar (0)
horas = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]).reshape(-1, 1)
aprueba = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1])

modelo = LogisticRegression(C=1e6).fit(horas, aprueba)     # C grande ≈ sin regularización
b0, b1 = modelo.intercept_[0], modelo.coef_[0, 0]
print(f"ln(p/(1-p)) = {b0:.3f} + {b1:.3f}·horas")
print(f"Por cada hora extra, los momios se multiplican por e^{b1:.3f} = {np.exp(b1):.2f}")

for h in [1, 3, 5]:
    logit = b0 + b1 * h
    p = 1 / (1 + np.exp(-logit))                            # sigmoide
    print(f"  {h} h -> logit = {logit:6.3f}, momios = {np.exp(logit):6.3f}, P(aprobar) = {p:.3f}"
          f" -> clase {int(p >= 0.5)}")

print("predict_proba(3 h) de sklearn:", modelo.predict_proba([[3]]).round(3))
print(f"Frontera de decisión (p = 0.5) en horas = {-b0 / b1:.2f}")
