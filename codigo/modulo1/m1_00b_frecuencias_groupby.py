# Módulo 1 · Tablas de frecuencias, groupby, merge, concat y np.where
# La tabla FA / FR / FAA / FRA es la que imprime la función freq() de la sesión 9 (churn)
import numpy as np
import pandas as pd

clientes = pd.DataFrame({
    "Customer_ID": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
    "Contract": ["Month-to-Month", "One Year", "Month-to-Month", "Two Year",
                 "Month-to-Month", np.nan, "One Year", "Month-to-Month"],
    "Monthly_Charge": [70.0, 55.5, 99.9, 20.0, 89.0, 45.0, 60.0, 101.0],
    "Churn_Value": [1, 0, 1, 0, 1, 0, 0, 0],
})

# 1) "La falta de información también es información": el nulo se vuelve una categoría
clientes["Contract"] = clientes["Contract"].fillna("Sin categoría")

# 2) Tabla de frecuencias: absoluta, relativa y sus acumuladas
def freq(df, var):
    t = df[var].value_counts().to_frame("FA")
    t["FR"] = t["FA"] / t["FA"].sum()
    t["FAA"] = t["FA"].cumsum()
    t["FRA"] = t["FR"].cumsum()
    return t

print("Tabla de frecuencias de Contract:\n", freq(clientes, "Contract").round(3).to_string(), sep="")

# 3) groupby: tasa de fuga (media del 0/1) y cargo promedio por tipo de contrato
res = clientes.groupby("Contract").agg(clientes=("Customer_ID", "count"),
                                       tasa_fuga=("Churn_Value", "mean"),
                                       cargo_medio=("Monthly_Charge", "mean"))
print("\nTasa de fuga por contrato:\n", res.round(3).to_string(), sep="")

# 4) np.where: variable nueva con una condición (vectorizado, sin for)
clientes["cargo_alto"] = np.where(clientes["Monthly_Charge"] > 80, 1, 0)
print("\nCargo alto (> 80):", clientes["cargo_alto"].tolist())

# 5) merge: pegar otra tabla por la llave (unidad muestral)
demograficos = pd.DataFrame({"Customer_ID": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C9"],
                             "Age": [25, 41, 33, 67, 29, 52, 38, 45]})
inner = clientes.merge(demograficos, on="Customer_ID", how="inner")
left = clientes.merge(demograficos, on="Customer_ID", how="left")
print(f"\nmerge inner: {inner.shape} | merge left: {left.shape} | "
      f"clientes sin edad tras el left: {left['Age'].isna().sum()} ({left.loc[left['Age'].isna(), 'Customer_ID'].tolist()})")

# 6) concat: apilar tablas con las mismas columnas (así se apilan las anclas en la sesión 8)
a = pd.DataFrame({"id": ["E1", "E2"], "ancla": [20, 20]})
b = pd.DataFrame({"id": ["E1", "E2"], "ancla": [21, 21]})
print("\nconcat:\n", pd.concat([a, b], ignore_index=True).to_string(index=False), sep="")
