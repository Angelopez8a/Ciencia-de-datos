# Módulo 1 · Primeros pasos con pandas: la inspección con la que empiezan todos los notebooks
# Datos reales de Gapminder: esperanza de vida, población y PIB per cápita de 142 países (1952-2007).
from pathlib import Path
import pandas as pd

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "gapminder.csv"
URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/gapminder/gapminder.csv"
df = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL, index_col=0)   # la primera columna es solo un número de fila

print("1) Dimensión (filas, columnas):", df.shape)
print("\n2) Tipos de dato:\n", df.dtypes.to_string(), sep="")
print("\n3) Registros por continente:\n", df["continent"].value_counts().to_string(), sep="")
print("\n4) Número de países distintos:", df["country"].nunique())

# describe() con percentiles propios (el notebook de Ecobici usa np.arange(0, 1.01, 0.1))
print("\n5) Resumen de la esperanza de vida (años), todos los periodos:")
print(df["lifeExp"].describe(percentiles=[0.25, 0.5, 0.75]).round(2).to_string())

# Filtrar con condiciones (máscaras booleanas)
print("\n6) Países con esperanza de vida menor a 43 años en 2007:")
print(df[(df["year"] == 2007) & (df["lifeExp"] < 43)][["country", "continent", "lifeExp"]].to_string(index=False))

# De formato largo (una fila por país y año) a formato ancho (una columna por año)
ancho = df[df["country"].isin(["Mexico", "Brazil", "Japan"])].pivot_table(
    index="country", columns="year", values="lifeExp")
print("\n7) pivot_table (formato ancho), años seleccionados:\n", ancho[[1952, 1977, 2007]].round(1).to_string(), sep="")

# groupby: resumen por grupo
print("\n8) Esperanza de vida en 2007 por continente:")
print(df[df["year"] == 2007].groupby("continent")["lifeExp"].agg(["count", "mean", "std", "min", "max"]).round(2).to_string())
