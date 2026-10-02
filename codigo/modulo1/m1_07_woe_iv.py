# Módulo 1 · Credit scoring con datos reales: discretización, WoE (Weight of Evidence) e IV (Information Value)
# WoE_i = ln(%NoEvento_i / %Evento_i)      IV = Σ (%NoEvento_i - %Evento_i) * WoE_i
# Datos: 4,454 solicitudes de crédito del curso de minería de datos de la UPC. Evento = "bad" (crédito incumplido).
from pathlib import Path
import numpy as np
import pandas as pd

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "credito_upc.csv"
URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/modeldata/credit_data.csv"
df = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL, index_col=0)
df["evento"] = (df["Status"] == "bad").astype(int)
print(f"Solicitudes: {len(df):,} | tasa de evento (malos): {df['evento'].mean():.3f}\n")

def bines(x, k=5):
    """Continuas: k cortes por cuantiles; los nulos forman su propio bin. Discretas: nulos -> 'Sin categoría'."""
    if pd.api.types.is_numeric_dtype(x):
        b = pd.qcut(x, q=k, duplicates="drop")
        return b.cat.add_categories("Sin dato").fillna("Sin dato") if x.isna().any() else b
    return x.fillna("Sin categoría")

def tabla_woe(df, var, tgt="evento"):
    t = df.groupby(bines(df[var]), observed=True)[tgt].agg(total="count", eventos="sum")
    t["no_eventos"] = t["total"] - t["eventos"]
    t["%evento"] = t["eventos"] / t["eventos"].sum()            # distribución de los eventos
    t["%no_evento"] = t["no_eventos"] / t["no_eventos"].sum()
    with np.errstate(divide="ignore"):
        t["WoE"] = np.log(t["%no_evento"] / t["%evento"])        # logaritmo natural
    t["IV_i"] = (t["%no_evento"] - t["%evento"]) * t["WoE"]
    return t, t["IV_i"].sum()

def poder(iv):
    return ("No ayuda" if iv < 0.02 else "Bajo" if iv < 0.1 else "Regular" if iv < 0.3 else
            "Fuerte" if iv < 0.5 else "Sobrepredictiva (¡revisar!)")

for var in ["Seniority", "Records"]:              # antigüedad en el empleo (años) y antecedentes negativos
    t, iv = tabla_woe(df, var)
    print(f"Variable: {var}")
    print(t[["total", "eventos", "%evento", "%no_evento", "WoE", "IV_i"]].round(4).to_string())
    print(f"IV = {iv:.4f} -> poder predictivo: {poder(iv)}\n")

variables = ["Seniority", "Home", "Time", "Age", "Marital", "Records", "Job",
             "Expenses", "Income", "Assets", "Debt", "Amount", "Price"]
ranking = pd.Series({v: tabla_woe(df, v)[1] for v in variables}).sort_values(ascending=False)
print("IV de todas las variables (continuas en 5 cortes por cuantiles):")
print(pd.DataFrame({"IV": ranking.round(4), "poder": ranking.map(poder)}).to_string())

t, _ = tabla_woe(df, "Job")
print("\n¿Por qué Job tiene IV infinito?\n", t[["total", "eventos", "no_eventos", "WoE"]].round(3).to_string(), sep="")
