# Módulo 1 · Estandarización: z-score (StandardScaler) vs Min-Max (MinMaxScaler)
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.DataFrame({"edad": [20, 30, 40, 50, 60],
                   "ingreso": [10000, 20000, 30000, 40000, 100000]})

ss = StandardScaler()            # (x - media) / desviación estándar
mm = MinMaxScaler()              # (x - min) / (max - min)

z = pd.DataFrame(ss.fit_transform(df), columns=["z_edad", "z_ingreso"])
m = pd.DataFrame(mm.fit_transform(df), columns=["mm_edad", "mm_ingreso"])
print(pd.concat([df, z.round(3), m.round(3)], axis=1).to_string(index=False))

print("\nMedias aprendidas :", ss.mean_)
print("Desv. aprendidas  :", ss.scale_.round(2))
print("Media z  :", z.mean().round(6).tolist(), "| Desv z:", z.std(ddof=0).round(6).tolist())
print("Min-Max  : min =", m.min().tolist(), "max =", m.max().tolist())
# Min-Max conserva proporciones: 40,000 es el doble de 20,000 (respecto al mínimo)
