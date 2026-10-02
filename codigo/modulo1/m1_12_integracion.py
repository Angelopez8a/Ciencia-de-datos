# Módulo 1 · Integración de datos (paso 2 del proceso): varias fuentes -> una tabla por unidad muestral
# Idea de la sesión 9 (churn): 5 archivos de Excel unidos por Customer_ID con reduce + merge
from functools import reduce
import pandas as pd

um = ["Customer_ID"]

servicios = pd.DataFrame({"Customer_ID": ["A1", "A2", "A3", "A4"], "Count": 1, "Quarter": "Q3",
                          "Tenure_in_Months": [1, 8, 18, 25], "Contract": ["Month", "Month", "Year", "Two"]})
demograficos = pd.DataFrame({"Customer_ID": ["A1", "A2", "A3", "A4"], "Count": 1,
                             "Age": [78, 74, 71, 30], "Married": ["No", "Yes", "No", "Yes"]})
estatus = pd.DataFrame({"Customer_ID": ["A1", "A2", "A3", "A4"], "Count": 1, "Quarter": "Q3",
                        "Customer_Status": ["Churned", "Churned", "Stayed", "Joined"],
                        "Churn_Label": ["Yes", "Yes", "No", "No"], "Churn_Score": [91, 69, 30, 25],
                        "Churn_Reason": ["Competitor", "Price", None, None], "Churn_Value": [1, 1, 0, 0]})

# 0) Antes de unir: la llave debe ser ÚNICA en cada tabla (si no, el merge duplica filas)
for nombre, t in [("servicios", servicios), ("demograficos", demograficos), ("estatus", estatus)]:
    print(f"{nombre:<13} filas={len(t)}  llaves únicas={t['Customer_ID'].nunique()}  duplicadas={t['Customer_ID'].duplicated().sum()}")

# 1) merge que descarta las columnas repetidas (Count, Quarter) en lugar de crear _x / _y
def merge_sin_duplicados(izq, der):
    repetidas = [c for c in der.columns if c in izq.columns and c not in um]
    return izq.merge(der.drop(columns=repetidas), on=um, how="inner", validate="one_to_one")

df = reduce(merge_sin_duplicados, [servicios, demograficos, estatus])
print("\nTabla integrada:", df.shape)
print(df.to_string(index=False))

# 2) Lo que pasa con un merge "ingenuo": columnas Count_x, Count_y...
ingenuo = servicios.merge(demograficos, on=um).merge(estatus, on=um)
print("\nColumnas con merge ingenuo:", [c for c in ingenuo.columns if c.endswith(("_x", "_y"))])

# 3) Separar llave, objetivo y variables que "dan la respuesta" (data leakage)
tgt = ["Churn_Value"]
var_fuera = ["Churn_Label", "Churn_Score", "Churn_Reason"]       # se calcularon DESPUÉS de la fuga
predictoras = [c for c in df.columns if c not in um + tgt + var_fuera]
print("\nPredictoras candidatas:", predictoras)
print("Ojo: Customer_Status sigue en la lista y contiene 'Churned' -> también da la respuesta.")
