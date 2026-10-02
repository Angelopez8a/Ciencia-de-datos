# Módulo 1 · Variables categóricas: nulos como categoría, normalización y variables dummy
# Primero el ejemplo de la diapositiva; después, datos reales: 4,454 solicitudes de crédito (curso de la UPC).
from pathlib import Path
import pandas as pd

# --- Ejemplo de la diapositiva ---
frutas = pd.DataFrame({"id": [1, 2, 3, 4], "fruit": ["banana", "apple", "banana", "apple"]})
print(pd.get_dummies(frutas, columns=["fruit"], prefix="data", dtype=int).to_string(index=False))
print("\nCon drop_first=True (evita la multicolinealidad o 'trampa de la dummy'):")
print(pd.get_dummies(frutas, columns=["fruit"], drop_first=True, dtype=int).to_string(index=False))

# --- Datos reales ---
ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "credito_upc.csv"
URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/modeldata/credit_data.csv"
df = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL, index_col=0)

def normalizar(s, umbral=0.03, otros="Otros"):
    s = s.fillna("Sin categoría")                    # "la falta de información también es información"
    freq = s.value_counts(normalize=True)
    raras = freq[freq < umbral].index                 # categorías con menos de 3 % de los registros
    return s.where(~s.isin(raras), otros)

for var in ["Home", "Marital"]:                       # tipo de vivienda y estado civil
    tabla = pd.DataFrame({"antes": df[var].fillna("Sin categoría").value_counts(normalize=True),
                          "después": normalizar(df[var]).value_counts(normalize=True)})
    tabla = tabla.sort_values("antes", ascending=False)
    print(f"\nFrecuencias relativas de {var} (nulos originales: {df[var].isna().sum()}):")
    print(tabla.round(4).fillna("-").to_string())

# Dummies de las variables ya normalizadas: cuántas columnas se crean
norm = pd.DataFrame({v: normalizar(df[v]) for v in ["Home", "Marital", "Job", "Records"]})
print("\nCategorías tras normalizar:", norm.nunique().to_dict())
print("Columnas dummy sin drop_first:", pd.get_dummies(norm, dtype=int).shape[1],
      "| con drop_first:", pd.get_dummies(norm, drop_first=True, dtype=int).shape[1])
