# Módulo 3 · DBSCAN vs K-means en formas no esféricas (lunas) + elección de eps con k-distancias
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

X, y = make_moons(n_samples=400, noise=0.07, random_state=0)
X = np.vstack([X, [[2.5, 1.5], [-1.5, -0.8], [0.5, 1.6]]])      # 3 puntos atípicos
y = np.r_[y, [-1, -1, -1]]
X = StandardScaler().fit_transform(X)

# Gráfica de k-distancias: distancia al 5º vecino de cada punto (min_samples = 5)
dist, _ = NearestNeighbors(n_neighbors=5).fit(X).kneighbors(X)
k_dist = np.sort(dist[:, -1])
print("Percentiles de la 5-distancia (el 'codo' sugiere eps):",
      np.percentile(k_dist, [50, 90, 95, 99]).round(3))

for eps in [0.1, 0.25, 0.5]:
    db = DBSCAN(eps=eps, min_samples=5).fit(X)
    et = db.labels_
    n_cl = len(set(et)) - (1 if -1 in et else 0)
    print(f"eps={eps:<5} clusters={n_cl}  ruido={int((et == -1).sum()):>3}  "
          f"núcleo={len(db.core_sample_indices_):>3}  ARI vs real={adjusted_rand_score(y, et):.3f}")

km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
print(f"K-means k=2           ARI vs real={adjusted_rand_score(y, km.labels_):.3f}  (no sabe de ruido)")
