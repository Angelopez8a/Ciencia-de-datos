# Módulo 3 · DBSCAN: sismos reales cerca de Fiyi y, para comparar formas, dos "lunas" sintéticas
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1) Datos reales: 1,000 sismos de magnitud mayor a 4.0 registrados cerca de Fiyi desde 1964 (Universidad de Harvard)
ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "sismos_fiji.csv"
URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/quakes.csv"
q = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL, index_col=0)
X = q[["long", "lat"]].to_numpy()                 # longitud y latitud en grados (1 grado ≈ 100 km)

# Gráfica de k-distancias: distancia al 10.º vecino (kneighbors cuenta al propio punto, igual que min_samples)
dist, _ = NearestNeighbors(n_neighbors=10).fit(X).kneighbors(X)
k_dist = np.sort(dist[:, -1])
print("Distancia al 10.º vecino, percentiles 50/90/95/98/99 (grados):", np.percentile(k_dist, [50, 90, 95, 98, 99]).round(2))

for eps in [0.75, 1.5, 2.0]:
    et = DBSCAN(eps=eps, min_samples=10).fit_predict(X)
    tam = sorted(np.bincount(et[et >= 0]).tolist(), reverse=True)
    print(f"eps = {eps:<4} -> clusters = {len(tam)}, ruido = {int((et == -1).sum()):>3}, tamaños = {tam}")

q["dbscan"] = DBSCAN(eps=2.0, min_samples=10).fit_predict(X)
q["kmeans"] = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
print("\nPerfil de los grupos de DBSCAN (eps = 2.0):")
print(q.groupby("dbscan").agg(sismos=("mag", "size"), longitud=("long", "mean"), latitud=("lat", "mean"),
                              profundidad_km=("depth", "mean"), magnitud=("mag", "mean")).round(1).to_string())
sin_ruido = q["dbscan"] >= 0
print(f"Coincidencia con K-means (k = 2) sin contar el ruido: ARI = "
      f"{adjusted_rand_score(q.loc[sin_ruido, 'dbscan'], q.loc[sin_ruido, 'kmeans']):.3f}")

# 2) Datos sintéticos: dos "lunas" entrelazadas más 3 atípicos, para ver qué pasa con formas no esféricas
Xs, ys = make_moons(n_samples=400, noise=0.07, random_state=0)
Xs = StandardScaler().fit_transform(np.vstack([Xs, [[2.5, 1.5], [-1.5, -0.8], [0.5, 1.6]]]))
ys = np.r_[ys, [-1, -1, -1]]
print()
for nombre, et in [("K-means (k = 2)", KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(Xs)),
                   ("DBSCAN (eps = 0.25)", DBSCAN(eps=0.25, min_samples=5).fit_predict(Xs))]:
    print(f"Lunas · {nombre:<20} ARI contra la forma real = {adjusted_rand_score(ys, et):.3f} | "
          f"ruido = {int((et == -1).sum())}")
