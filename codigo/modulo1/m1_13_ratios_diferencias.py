# Módulo 1 · Ingeniería de variables con razones y diferencias (sesión 9, churn)
# Datos reales de IBM: los 11 clientes recién llegados tienen tenure = 0 meses, así que cualquier razón
# que divida entre tenure se enfrenta a una división entre cero. El notebook la evita con np.where.
from pathlib import Path
import numpy as np
import pandas as pd

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "telco_churn_ibm.csv"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")   # en el CSV viene como texto
print("Clientes con tenure = 0:", (df["tenure"] == 0).sum(), "| TotalCharges vacíos:", df["TotalCharges"].isna().sum())

# Razón sin protección: dividir entre 0 produce inf (no NaN) y "rompe" estadísticas y modelos
sin_proteger = df["MonthlyCharges"] / df["tenure"]
print("Sin protección, valores infinitos:", int(np.isinf(sin_proteger).sum()))

# Patrón del notebook: np.where(denominador != 0, razón, NaN)
# (np.where evalúa ambas ramas; errstate solo silencia el aviso de NumPy, el resultado ya no tiene inf)
with np.errstate(divide="ignore", invalid="ignore"):
    df["ratio_cargo_tenure"] = np.where(df["tenure"] != 0, df["MonthlyCharges"] / df["tenure"], np.nan)
    df["prom_mensual"] = np.where(df["tenure"] != 0, df["TotalCharges"] / df["tenure"], np.nan)
df["dif_cargo"] = df["MonthlyCharges"] - df["prom_mensual"]   # cargo actual menos el promedio histórico

cols = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "ratio_cargo_tenure", "prom_mensual", "dif_cargo"]
print("\n", df.loc[[0, 1, 2, 488], cols].round(2).to_string(index=False), sep="")   # 488 es un cliente con tenure = 0
print("\n¿Quedó algún inf?", bool(np.isinf(df.select_dtypes("number")).any().any()))
print("Nulos creados (se imputan después, en la limpieza):", int(df["prom_mensual"].isna().sum()))

# ¿Aporta algo la diferencia? Tasa de fuga según si el cargo actual supera al promedio histórico
df["sube_cargo"] = np.where(df["dif_cargo"] > 0, "cargo actual mayor", "cargo actual menor o igual")
print("\n", df.dropna(subset=["dif_cargo"]).groupby("sube_cargo")["Churn"]
      .apply(lambda s: (s == "Yes").mean()).round(4).to_string(), sep="")
