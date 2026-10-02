# Módulo 1 · OLTP -> OLAP: construir "cubos" con pandas (idea de la sesión 10)
# Datos reales: 6,433 viajes de taxi en la ciudad de Nueva York durante marzo de 2019
# (registros de la NYC Taxi & Limousine Commission). Cada fila es una transacción: un viaje.
from pathlib import Path
import pandas as pd

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "taxis_nyc_2019_03.csv"
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"
oltp = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL, parse_dates=["pickup", "dropoff"])
print("OLTP (registros):", oltp.shape)
print(oltp[["pickup", "pickup_borough", "pickup_zone", "distance", "fare", "tip"]].head(3).to_string(index=False), "\n")

# Dimensiones derivadas de la fecha y de la ubicación
dias = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
oltp["hora"] = oltp["pickup"].dt.hour
oltp["dia"] = pd.Categorical(oltp["pickup"].dt.dayofweek.map(dict(enumerate(dias))), categories=dias, ordered=True)
oltp["franja"] = pd.cut(oltp["hora"], [-1, 5, 11, 17, 23], labels=["0-5 h", "6-11 h", "12-17 h", "18-23 h"])
oltp["borough"] = oltp["pickup_borough"].fillna("Sin dato")   # 26 viajes sin zona de origen registrada

# --- Cubo 1: dimensión tiempo (día de la semana) ---
print("Cubo tiempo (día de la semana):\n", oltp.groupby("dia", observed=True)["fare"].agg(["count", "mean", "max"]).round(2).to_string(), sep="")

# --- Roll-up (subir de nivel: zona -> borough) y drill-down (bajar: borough -> zona) ---
print("\nRoll-up por borough:\n", oltp.groupby("borough")["fare"].agg(["count", "mean"]).round(2).to_string(), sep="")
print("\nDrill-down en Queens: las 3 zonas de origen con más viajes:")
queens = oltp[oltp["borough"] == "Queens"].groupby("pickup_zone")[["fare", "distance"]].agg(["count", "mean"])
print(queens.sort_values(("fare", "count"), ascending=False).head(3).round(2).to_string())

# --- Slice (fijar una dimensión) y dice (subcubo con varias condiciones) ---
slice_m = oltp[oltp["borough"] == "Manhattan"]
dice = oltp[oltp["borough"].isin(["Manhattan", "Brooklyn"]) & oltp["hora"].between(7, 9)]
print(f"\nSlice Manhattan: {len(slice_m)} viajes | Dice (Manhattan o Brooklyn, 7 a 9 h): {len(dice)} viajes, "
      f"tarifa media = {dice['fare'].mean():.2f}")

# --- Pivot: rotar el cubo (franja horaria en filas, forma de pago en columnas) ---
print("\nPivot: propina media por franja horaria y forma de pago:")
print(oltp.pivot_table(index="franja", columns="payment", values="tip", aggfunc="mean", observed=True).round(2).to_string())
