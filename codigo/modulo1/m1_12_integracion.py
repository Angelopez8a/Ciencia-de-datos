# Módulo 1 · Integración de datos (paso 2 del proceso): varias fuentes -> una tabla por unidad muestral
# En la sesión 9 la información de los clientes llegó repartida en varios archivos de Excel. Para reproducir
# esa situación con datos reales, el CSV público de IBM se separa en tres tablas temáticas que comparten la
# llave customerID y, como ocurrió en el curso, también alguna columna repetida (tenure).
from functools import reduce
from pathlib import Path
import pandas as pd

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "telco_churn_ibm.csv"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
telco = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL)

um = ["customerID"]
demograficos = telco[um + ["gender", "SeniorCitizen", "Partner", "Dependents"]]
servicios = telco[um + ["tenure", "PhoneService", "InternetService", "TechSupport"]]
cuenta = telco[um + ["tenure", "Contract", "PaymentMethod", "MonthlyCharges", "Churn"]]

# 0) Antes de unir: la llave debe ser ÚNICA en cada tabla (si no, el merge duplica filas)
for nombre, t in [("demograficos", demograficos), ("servicios", servicios), ("cuenta", cuenta)]:
    print(f"{nombre:<13} filas={len(t):,}  llaves únicas={t['customerID'].nunique():,}  "
          f"duplicadas={t['customerID'].duplicated().sum()}")

# 1) merge que descarta las columnas repetidas en lugar de crear tenure_x / tenure_y
def merge_sin_duplicados(izq, der):
    repetidas = [c for c in der.columns if c in izq.columns and c not in um]
    return izq.merge(der.drop(columns=repetidas), on=um, how="inner", validate="one_to_one")

df = reduce(merge_sin_duplicados, [demograficos, servicios, cuenta])
print("\nTabla integrada:", df.shape)
print(df.head(3).T.to_string())

# 2) Lo que produce un merge directo, sin quitar las repetidas: columnas con sufijos _x / _y
directo = demograficos.merge(servicios, on=um).merge(cuenta, on=um)
print("\nColumnas con merge directo:", [c for c in directo.columns if c.endswith(("_x", "_y"))])

# 3) Si por error un cliente apareciera dos veces, validate="one_to_one" lo detecta
cuenta_con_error = pd.concat([cuenta, cuenta.head(1)])
print("Filas si no se valida:", len(demograficos.merge(cuenta_con_error, on=um)))
try:
    demograficos.merge(cuenta_con_error, on=um, validate="one_to_one")
except pd.errors.MergeError as error:
    print("MergeError:", str(error).splitlines()[0])

# 4) Separar la llave y la variable objetivo; el resto son predictoras candidatas
tgt = ["Churn"]
predictoras = [c for c in df.columns if c not in um + tgt]
print("\nPredictoras candidatas:", predictoras)
