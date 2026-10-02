# Módulo 3 · K-means: método del codo (inercia) y coeficiente de silueta para elegir k
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.0, random_state=7)
X = StandardScaler().fit_transform(X)

print(f"{'k':>2}{'Inercia (WCSS)':>16}{'Silueta':>10}{'Davies-B':>10}{'Calinski-H':>12}")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    et = km.labels_
    print(f"{k:>2}{km.inertia_:>16.1f}{silhouette_score(X, et):>10.3f}"
          f"{davies_bouldin_score(X, et):>10.3f}{calinski_harabasz_score(X, et):>12.1f}")

km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
print("\nCentroides (k=4):\n", km.cluster_centers_.round(2))
print("Tamaño de cada cluster:", [int((km.labels_ == c).sum()) for c in range(4)])
