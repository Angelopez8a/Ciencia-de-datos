# Módulo 3 · K-means: método del codo (inercia) y coeficiente de silueta para elegir k
# Datos reales: 272 erupciones del géiser Old Faithful (Parque Nacional de Yellowstone): duración de la erupción
# y tiempo de espera hasta la siguiente, ambos en minutos (Azzalini y Bowman, 1990).
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "old_faithful.csv"
URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/faithful.csv"
geiser = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL, index_col=0)
escala = StandardScaler().fit(geiser)
X = escala.transform(geiser)                  # duración y espera en la misma escala

print(f"{'k':>2}{'Inercia (WCSS)':>16}{'Silueta':>10}{'Davies-B':>10}{'Calinski-H':>12}")
print(f"{1:>2}{KMeans(n_clusters=1, n_init=10, random_state=0).fit(X).inertia_:>16.1f}"
      f"{'-':>10}{'-':>10}{'-':>12}   (todas las erupciones en un solo grupo)")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    et = km.labels_
    print(f"{k:>2}{km.inertia_:>16.1f}{silhouette_score(X, et):>10.3f}"
          f"{davies_bouldin_score(X, et):>10.3f}{calinski_harabasz_score(X, et):>12.1f}")

km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
centros = pd.DataFrame(escala.inverse_transform(km.cluster_centers_), columns=["duración (min)", "espera (min)"])
centros["erupciones"] = [int((km.labels_ == c).sum()) for c in range(2)]
print("\nCentroides con k = 2, en minutos:\n", centros.round(2).to_string())
