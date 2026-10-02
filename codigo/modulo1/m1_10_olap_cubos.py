# Módulo 1 · OLTP -> OLAP: construir "cubos" con pandas (idea de la sesión 10, contaminantes CDMX)
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)
fechas = pd.date_range("2021-01-01", "2021-12-31 23:00", freq="h")
estaciones = {"ACO": "Zona 1", "MER": "Zona 3", "UIZ": "Zona 2"}

# Tabla transaccional (OLTP): una fila por estación-hora (registro "crudo")
oltp = pd.DataFrame([(e, f) for e in estaciones for f in fechas], columns=["id_station", "date"])
hora, mes = oltp["date"].dt.hour, oltp["date"].dt.month
diurno = 30 * np.sin((hora - 9) / 24 * 2 * np.pi).clip(0)       # pico de ozono al mediodía
temporada = 12 * np.exp(-((mes - 4.5) ** 2) / 4)                 # temporada alta: mar-may
por_zona = oltp["id_station"].map({"ACO": -4, "MER": 0, "UIZ": 5})
oltp["O3"] = (18 + diurno + temporada + por_zona + rng.normal(0, 5, len(oltp))).round(1)
print("OLTP (registros):", oltp.shape)
print(oltp.head(3).to_string(index=False), "\n")

# Dimensiones derivadas de la fecha y de la ubicación
oltp["hrs"] = oltp["date"].dt.hour
oltp["mes"] = oltp["date"].dt.month
oltp["trimestre"] = "Q" + oltp["date"].dt.quarter.astype(str)
oltp["zona"] = oltp["id_station"].map(estaciones)

# --- Cubo 1: dimensión tiempo (varias granularidades apiladas) ---
lst = []
for d in ["hrs", "mes", "trimestre"]:
    aux = oltp.groupby(d)["O3"].agg(["mean", "max"]).reset_index().rename(columns={d: "tiempo"})
    aux.insert(0, "dimension", d)
    lst.append(aux)
cubo_t = pd.concat(lst, ignore_index=True)
print("Cubo tiempo (trimestres):\n", cubo_t[cubo_t["dimension"] == "trimestre"].round(2).to_string(index=False))

# --- Roll-up (subir de nivel) y drill-down (bajar de nivel) ---
print("\nRoll-up por zona:\n", oltp.groupby("zona")["O3"].mean().round(2).to_string())
print("\nDrill-down zona -> estación -> trimestre:")
print(oltp.pivot_table(index=["zona", "id_station"], columns="trimestre",
                       values="O3", aggfunc="mean").round(2).to_string())

# --- Slice (fijar una dimensión) y dice (subcubo con varias condiciones) ---
slice_q1 = oltp[oltp["trimestre"] == "Q1"]
dice = oltp[(oltp["trimestre"].isin(["Q2", "Q3"])) & (oltp["hrs"].between(12, 16))]
print(f"\nSlice Q1: {len(slice_q1)} filas | Dice (Q2-Q3, 12-16 h): {len(dice)} filas, "
      f"O3 medio = {dice['O3'].mean():.2f}")
