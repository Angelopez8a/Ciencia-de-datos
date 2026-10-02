# Módulo 3 · Ciclo completo de clustering: preparar -> estandarizar -> reducir -> agrupar -> evaluar -> perfilar
# Datos reales: gasto anual de 440 clientes de un distribuidor mayorista de Portugal en seis categorías de producto
# (UCI, Cardoso, 2014). El canal de cada cliente (hotel/restaurante/café o minorista) NO se usa para agrupar:
# se guarda para comprobar al final si los segmentos tienen sentido.
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "mayorista_lisboa.csv"
URL = "https://raw.githubusercontent.com/udacity/machine-learning/master/projects/customer_segments/customers.csv"
clientes = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL)
canal = clientes.pop("Channel").map({1: "Hotel/Rest./Café", 2: "Minorista"}).rename("canal")
clientes = clientes.drop(columns="Region")
print("1) Datos:", clientes.shape, "| gasto medio vs mediano en frescos:",
      f"{clientes['Fresh'].mean():,.0f} vs {clientes['Fresh'].median():,.0f} (distribución sesgada)")

X = StandardScaler().fit_transform(np.log(clientes))    # 2) logaritmo (reduce el sesgo) y misma escala
print("2) Logaritmo del gasto y estandarización: cada categoría con media 0 y desviación 1")
pca = PCA(n_components=0.9).fit(X)                       # 3) conservar al menos 90 % de la varianza
Xp = pca.transform(X)
print(f"3) PCA: {Xp.shape[1]} componentes explican {pca.explained_variance_ratio_.sum():.1%}")

sil = {k: silhouette_score(Xp, KMeans(k, n_init=10, random_state=0).fit_predict(Xp)) for k in range(2, 7)}
k = max(sil, key=sil.get)
print("4-5) Silueta por k:", {kk: round(float(v), 3) for kk, v in sil.items()}, "-> k elegido =", k)

clientes["segmento"] = KMeans(k, n_init=10, random_state=0).fit_predict(Xp)
print("6) Perfil de los segmentos (mediana del gasto anual):")
print(clientes.groupby("segmento").agg(n=("Fresh", "size"), frescos=("Fresh", "median"), lacteos=("Milk", "median"),
                                       abarrotes=("Grocery", "median"), congelados=("Frozen", "median"),
                                       limpieza_papel=("Detergents_Paper", "median"),
                                       delicatessen=("Delicatessen", "median")).round(0).astype(int).to_string())
print("7) Validación con el canal real (no se usó para agrupar):")
print(pd.crosstab(clientes["segmento"], canal).to_string())
