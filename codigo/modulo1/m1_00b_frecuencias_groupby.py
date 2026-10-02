# Módulo 1 · Tablas de frecuencias, groupby, np.where, merge y concat
# Datos reales: IBM Telco Customer Churn, 7,043 clientes de una compañía telefónica (el caso de la sesión 9).
# La tabla FA / FR / FAA / FRA es la que imprime la función freq() del notebook de churn.
from pathlib import Path
import numpy as np
import pandas as pd

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "telco_churn_ibm.csv"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
clientes = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL)
clientes["Churn_Value"] = (clientes["Churn"] == "Yes").astype(int)   # 1 = el cliente se fue
print("Clientes:", clientes.shape[0], "| tasa de fuga global:", round(clientes["Churn_Value"].mean(), 4))

# 1) Tabla de frecuencias: absoluta, relativa y sus acumuladas
def freq(df, var):
    t = df[var].value_counts().to_frame("FA")
    t["FR"] = t["FA"] / t["FA"].sum()
    t["FAA"] = t["FA"].cumsum()
    t["FRA"] = t["FR"].cumsum()
    return t

print("\nTabla de frecuencias de Contract:\n", freq(clientes, "Contract").round(4).to_string(), sep="")

# 2) groupby: clientes, tasa de fuga (media del 0/1) y cargo mensual promedio por tipo de contrato
res = clientes.groupby("Contract").agg(clientes=("customerID", "count"),
                                       tasa_fuga=("Churn_Value", "mean"),
                                       cargo_medio=("MonthlyCharges", "mean"))
print("\nTasa de fuga por contrato:\n", res.round(4).to_string(), sep="")

# 3) np.where: variable nueva con una condición (vectorizada, sin ciclos)
clientes["cargo_alto"] = np.where(clientes["MonthlyCharges"] > 70, 1, 0)
print("\nTasa de fuga según cargo_alto (cargo mensual > 70):")
print(clientes.groupby("cargo_alto")["Churn_Value"].agg(["count", "mean"]).round(4).to_string())

# 4) merge: pegar otra tabla por la llave (customerID). Esta tabla solo tiene a quienes contratan internet.
internet = clientes.loc[clientes["InternetService"] != "No", ["customerID", "InternetService"]]
internet = internet.rename(columns={"InternetService": "tipo_internet"})
base = clientes[["customerID", "Contract", "Churn_Value"]]
inner = base.merge(internet, on="customerID", how="inner")
left = base.merge(internet, on="customerID", how="left")
print(f"\nmerge inner: {inner.shape} | merge left: {left.shape} | "
      f"clientes sin tipo_internet tras el left: {left['tipo_internet'].isna().sum()}")
left["tipo_internet"] = left["tipo_internet"].fillna("Sin internet")   # la falta de información también es información
print(left["tipo_internet"].value_counts().to_string())

# 5) concat: apilar tablas con las mismas columnas (así se apilan las anclas en la sesión 8)
cols = ["customerID", "tenure", "Contract"]
nuevos = clientes.loc[clientes["tenure"] == 0, cols]       # clientes recién llegados
veteranos = clientes.loc[clientes["tenure"] == 72, cols]   # clientes con 72 meses, el máximo de la base
apilada = pd.concat([nuevos, veteranos], ignore_index=True)
print(f"\nconcat: {nuevos.shape} + {veteranos.shape} -> {apilada.shape}")
print(apilada.groupby("tenure")["Contract"].value_counts().to_string())
