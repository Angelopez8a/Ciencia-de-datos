# Módulo 2 · Validar los cuatro supuestos de la regresión lineal con datos reales
# Linealidad (prueba RESET de Ramsey), homocedasticidad (Breusch-Pagan), independencia (Durbin-Watson)
# y normalidad de los residuales (Shapiro-Wilk). En las pruebas, p > 0.05 significa "no hay evidencia de falla".
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import durbin_watson

DATOS = Path(__file__).resolve().parents[1] / "datos"
RD = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv"
def leer(nombre, url, **kw):
    return pd.read_csv(DATOS / nombre if (DATOS / nombre).exists() else url, **kw)

def diagnostico(nombre, y, X):
    modelo = sm.OLS(np.asarray(y, float), sm.add_constant(np.asarray(X, float))).fit()
    r = modelo.resid
    p_reset = linear_reset(modelo, power=2, use_f=True).pvalue
    _, p_bp, _, _ = het_breuschpagan(r, modelo.model.exog)
    _, p_sw = stats.shapiro(r)
    ok = lambda p: "OK" if p > 0.05 else "FALLA"
    print(f"{nombre} (n = {int(modelo.nobs)}, R² = {modelo.rsquared:.3f})")
    print(f"  Linealidad RESET p = {p_reset:.4f} {ok(p_reset):<5} | Homocedasticidad BP p = {p_bp:.4f} {ok(p_bp)}")
    print(f"  Independencia DW = {durbin_watson(r):.3f} (≈ 2 sin autocorrelación) | Normalidad SW p = {p_sw:.4f} {ok(p_sw)}")

# Caso 1: Galton. Estatura del hijo según la de sus padres y su sexo (centímetros)
g = leer("galton_familias.csv", f"{RD}/HistData/GaltonFamilies.csv", index_col=0)
diagnostico("Caso 1 · Galton: hijo ~ padres + sexo", g["childHeight"] * 2.54,
            np.c_[g["midparentHeight"] * 2.54, (g["gender"] == "male").astype(int)])

# Caso 2: Gapminder 2007. Esperanza de vida según el PIB per cápita, sin transformar y con logaritmo
gm = leer("gapminder.csv", f"{RD}/gapminder/gapminder.csv", index_col=0)
d = gm[gm["year"] == 2007]
diagnostico("Caso 2 · Gapminder 2007: esperanza de vida ~ PIB", d["lifeExp"], d["gdpPercap"])
diagnostico("Caso 2 corregido: esperanza de vida ~ log(PIB)", d["lifeExp"], np.log(d["gdpPercap"]))

# Caso 3: churn de IBM. Cargos totales según los meses de antigüedad
t = leer("telco_churn_ibm.csv", "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")
t["TotalCharges"] = pd.to_numeric(t["TotalCharges"], errors="coerce")
t = t.dropna(subset=["TotalCharges"]).sample(5000, random_state=0)   # Shapiro-Wilk admite hasta 5,000 datos
diagnostico("Caso 3 · Churn: cargos totales ~ antigüedad", t["TotalCharges"], t["tenure"])

# Caso 4: concentración mensual de CO₂ en Mauna Loa, Hawái (1958-2001), según el tiempo
co2 = sm.datasets.co2.load_pandas().data["co2"].interpolate().resample("MS").mean()
diagnostico("Caso 4 · CO₂ de Mauna Loa: ppm ~ mes", co2.values, np.arange(len(co2)))
