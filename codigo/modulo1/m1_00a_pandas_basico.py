# Módulo 1 · Primeros pasos con pandas: lo mínimo que usan los notebooks antes de modelar
# Es la misma "inspección" con la que empieza la sesión 8 (Ecobici): dtypes, value_counts, describe, pivot_table
import numpy as np
import pandas as pd

# Formato "largo" (como estaciones.csv): una fila por estación y periodo
df = pd.DataFrame({
    "afluencia": [120, 135, 150, 90, 80, 85, 405, 405, 405],
    "id_estacion": ["E01", "E01", "E01", "E02", "E02", "E02", "E03", "E03", "E03"],
    "t": [1, 2, 3, 1, 2, 3, 1, 2, 3],
})

print("1) Dimensión (filas, columnas):", df.shape)
print("\n2) Tipos de dato:\n", df.dtypes.to_string(), sep="")
print("\n3) ¿Cuántos registros tiene cada estación?\n", df["id_estacion"].value_counts().to_string(), sep="")
print("\n4) Número de estaciones distintas:", df["id_estacion"].nunique())

# describe() con percentiles propios (el notebook usa np.arange(0, 1.01, 0.1))
print("\n5) Resumen de la afluencia:")
print(df["afluencia"].describe(percentiles=[0.25, 0.5, 0.75]).round(2).to_string())

# Filtrar con condiciones (máscaras booleanas)
print("\n6) Registros con afluencia > 100:")
print(df[df["afluencia"] > 100].to_string(index=False))

# De formato largo a formato ancho: una fila por estación, una columna por periodo
ancho = df.pivot_table(index="id_estacion", columns="t", values="afluencia", aggfunc="sum")
print("\n7) pivot_table (formato ancho):\n", ancho.to_string(), sep="")

# La estación E03 siempre vale 405: en Ecobici ese valor mínimo se repite en estaciones sin movimiento
print("\n8) Desviación estándar por estación (0 = serie constante):")
print(df.groupby("id_estacion")["afluencia"].std(ddof=0).round(2).to_string())
