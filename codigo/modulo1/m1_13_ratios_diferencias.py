# Módulo 1 · Ingeniería de variables con razones y diferencias (sesión 9, churn)
# El notebook crea ratio_m_charge_age, ratio_ref_tenure, dif_tot_month... protegiendo la división entre cero
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "Monthly_Charge": [70.0, 20.0, 99.0, 55.0],
    "Total_Charges": [70.0, 480.0, 1980.0, 0.0],
    "Age": [25, 40, 33, 60],
    "Number_of_Referrals": [0, 2, 1, 3],
    "Tenure_in_Months": [1, 24, 20, 0],          # un cliente recién llegado: 0 meses
})

# Razón sin protección: la división entre 0 produce inf (no NaN) y "rompe" estadísticas y modelos
sin_proteger = df["Number_of_Referrals"] / df["Tenure_in_Months"]
print("Sin protección :", sin_proteger.tolist())

# Patrón del notebook: np.where(denominador != 0, razón, NaN)
# (np.where evalúa ambas ramas, por eso numpy puede avisar de la división; el resultado ya no tiene inf)
with np.errstate(divide="ignore", invalid="ignore"):
    df["ratio_ref_tenure"] = np.where(df["Tenure_in_Months"] != 0,
                                      df["Number_of_Referrals"] / df["Tenure_in_Months"], np.nan)
df["ratio_m_charge_age"] = np.where(df["Age"] != 0, df["Monthly_Charge"] / df["Age"], np.nan)
df["dif_tot_month"] = df["Total_Charges"] - df["Monthly_Charge"]     # diferencias: no hay riesgo

print("\n", df.round(3).to_string(index=False), sep="")
print("\n¿Quedó algún inf?", bool(np.isinf(df.select_dtypes("number")).any().any()))
print("Nulos creados (se imputan después, en la limpieza):", int(df["ratio_ref_tenure"].isna().sum()))
