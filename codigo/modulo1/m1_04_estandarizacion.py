# Módulo 1 · Estandarización: z-score (StandardScaler) vs Min-Max (MinMaxScaler)
# Datos reales de Gapminder, año 2007: esperanza de vida (años) y PIB per cápita (dólares, ajustado por inflación).
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.set_printoptions(suppress=True)

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "gapminder.csv"
URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/gapminder/gapminder.csv"
gm = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL, index_col=0)
paises = ["Bolivia", "Brazil", "Mexico", "Chile", "United States"]
df = gm[(gm["year"] == 2007) & gm["country"].isin(paises)].set_index("country")[["lifeExp", "gdpPercap"]].round(1)

ss = StandardScaler()            # (x - media) / desviación estándar
mm = MinMaxScaler()              # (x - min) / (max - min)
z = pd.DataFrame(ss.fit_transform(df), index=df.index, columns=["z_lifeExp", "z_gdp"])
m = pd.DataFrame(mm.fit_transform(df), index=df.index, columns=["mm_lifeExp", "mm_gdp"])
print(pd.concat([df, z.round(3), m.round(3)], axis=1).to_string())

print("\nMedias aprendidas :", ss.mean_.round(2))
print("Desv. aprendidas  :", ss.scale_.round(2))
print("Media z  :", z.mean().round(6).tolist(), "| Desv z:", z.std(ddof=0).round(6).tolist())

# ¿Por qué estandarizar? Distancia entre México y Chile con y sin escalar
d_orig = (df.loc["Mexico"] - df.loc["Chile"]) ** 2
d_z = (z.loc["Mexico"] - z.loc["Chile"]) ** 2
print(f"\nDistancia México-Chile sin escalar: {np.sqrt(d_orig.sum()):.1f}  "
      f"(el PIB aporta {d_orig['gdpPercap'] / d_orig.sum():.4%} de la distancia al cuadrado)")
print(f"Distancia México-Chile con z-score: {np.sqrt(d_z.sum()):.3f}  "
      f"(el PIB aporta {d_z['z_gdp'] / d_z.sum():.2%})")
