# Módulo 1 · Variables categóricas: nulos como categoría, normalización y variables dummy
import numpy as np
import pandas as pd

# --- Ejemplo de la diapositiva ---
frutas = pd.DataFrame({"id": [1, 2, 3, 4], "fruit": ["banana", "apple", "banana", "apple"]})
print(pd.get_dummies(frutas, columns=["fruit"], prefix="data", dtype=int).to_string(index=False))
print("\nCon drop_first=True (evita multicolinealidad / trampa de la dummy):")
print(pd.get_dummies(frutas, columns=["fruit"], drop_first=True, dtype=int).to_string(index=False))

# --- Nulos como categoría + normalización (agrupar categorías raras) ---
ciudad = pd.Series(["CDMX"] * 50 + ["Monterrey"] * 30 + ["Juárez"] * 15
                   + ["Tijuana"] * 2 + ["Mérida"] * 1 + [np.nan] * 2, name="ciudad")
ciudad = ciudad.fillna("Sin categoría")              # "la falta de información también es información"

def normalizar(s, umbral=0.03, otros="Otros"):
    freq = s.value_counts(normalize=True)
    raras = freq[freq < umbral].index                 # categorías con < 3% de frecuencia
    return s.where(~s.isin(raras), otros)

tabla = pd.DataFrame({"antes": ciudad.value_counts(normalize=True),
                      "después": normalizar(ciudad).value_counts(normalize=True)})
tabla = tabla.sort_values("antes", ascending=False)
print("\nFrecuencias relativas:\n", tabla.round(2).fillna("-").to_string())
