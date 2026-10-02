# Módulo 1 · Credit scoring: discretización, WoE (Weight of Evidence) e IV (Information Value)
# WoE_i = ln(%NoEvento_i / %Evento_i)      IV = Σ (%NoEvento_i - %Evento_i) * WoE_i
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
n = 5000
edad = rng.integers(18, 75, n)
ingreso = rng.lognormal(9.8, 0.6, n)
# Probabilidad de incumplimiento (evento = 1): mayor en jóvenes y en ingresos bajos
logit = 1.5 - 0.05 * edad - 0.00004 * ingreso
evento = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame({"edad": edad, "ingreso": ingreso, "evento": evento})
print(f"Tasa de evento (malos): {df['evento'].mean():.3f}\n")

def tabla_woe(df, var, tgt="evento", bins=5):
    d = df[[var, tgt]].copy()
    d["bin"] = pd.qcut(d[var], q=bins)                       # 1) discretizar en cuantiles
    t = d.groupby("bin", observed=True)[tgt].agg(total="count", eventos="sum")
    t["no_eventos"] = t["total"] - t["eventos"]
    t["%evento"] = t["eventos"] / t["eventos"].sum()          # 2) distribución de eventos
    t["%no_evento"] = t["no_eventos"] / t["no_eventos"].sum()
    t["WoE"] = np.log(t["%no_evento"] / t["%evento"])         # 3) logaritmo natural
    t["IV_i"] = (t["%no_evento"] - t["%evento"]) * t["WoE"]
    return t, t["IV_i"].sum()

def poder(iv):
    return ("No ayuda" if iv < 0.02 else "Bajo" if iv < 0.1 else
            "Regular" if iv < 0.3 else "Fuerte" if iv < 0.5 else "Sobrepredictiva (¡sospechosa!)")

for var in ["edad", "ingreso"]:
    t, iv = tabla_woe(df, var)
    print(f"Variable: {var}")
    print(t[["total", "eventos", "%evento", "%no_evento", "WoE", "IV_i"]].round(4).to_string())
    print(f"IV = {iv:.4f} -> poder predictivo: {poder(iv)}\n")
