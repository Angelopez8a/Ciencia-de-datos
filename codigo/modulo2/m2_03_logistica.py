# Módulo 2 · Regresión logística: momios, logit y probabilidad
# Datos reales: los 23 vuelos del transbordador espacial anteriores al accidente del Challenger
# (28 de enero de 1986), con la temperatura al despegar (°F) y cuántas de las seis juntas tóricas
# (O-rings) de los cohetes resultaron dañadas. La noche anterior al lanzamiento se discutió si el frío
# aumentaba el riesgo; el pronóstico para la hora del despegue era de 31 °F (−0.6 °C).
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "challenger_juntas.csv"
URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/openintro/orings.csv"
d = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL)
d["falla"] = (d["damaged"] > 0).astype(int)        # 1 = al menos una junta dañada en ese vuelo
celsius = lambda f: (f - 32) / 1.8

print(f"Vuelos: {len(d)} | con daño: {d['falla'].sum()} | temperaturas de {d['temperature'].min()} a "
      f"{d['temperature'].max()} °F")
print("Temperaturas de los vuelos con daño:", sorted(d.loc[d["falla"] == 1, "temperature"].tolist()))

X, y = d[["temperature"]], d["falla"]
modelo = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)   # C grande ≈ sin regularización
b0, b1 = modelo.intercept_[0], modelo.coef_[0, 0]
print(f"\nln(p/(1-p)) = {b0:.4f} {b1:+.4f}·temperatura")
print(f"Factor de momios por cada grado Fahrenheit adicional: e^({b1:.4f}) = {np.exp(b1):.3f}")
print(f"Factor de momios por cada 10 °F MENOS: e^({-10 * b1:.3f}) = {np.exp(-10 * b1):.1f}")

for t in [75, 65, 53, 31]:
    logit = b0 + b1 * t
    p = 1 / (1 + np.exp(-logit))                    # sigmoide
    print(f"  {t} °F ({celsius(t):5.1f} °C) -> logit = {logit:6.3f}, momios = {np.exp(logit):9.3f}, "
          f"P(daño) = {p:.4f} -> clase {int(p >= 0.5)}")

sin_dano, dano = modelo.predict_proba(pd.DataFrame({"temperature": [31]}))[0]
print(f"predict_proba de sklearn a 31 °F: P(sin daño) = {sin_dano:.4f}, P(daño) = {dano:.4f}")
print(f"Frontera de decisión (p = 0.5): {-b0 / b1:.1f} °F ({celsius(-b0 / b1):.1f} °C)")
