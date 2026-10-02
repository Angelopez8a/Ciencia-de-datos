# Módulo 3 · Clustering jerárquico aglomerativo: matriz de enlace (linkage) y corte del dendrograma
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

puntos = np.array([[1, 1], [1.5, 1], [5, 5], [5, 5.5], [9, 1], [9, 1.8]])
nombres = ["A", "B", "C", "D", "E", "F"]

for metodo in ["single", "complete", "average", "ward"]:
    Z = linkage(puntos, method=metodo)
    print(f"Enlace '{metodo}':")
    for i, (a, b, dist, n) in enumerate(Z):
        print(f"  paso {i + 1}: une {int(a)} y {int(b)} a distancia {dist:.3f} -> nuevo grupo con {int(n)} puntos")

Z = linkage(puntos, method="ward")
etiquetas = fcluster(Z, t=3, criterion="maxclust")          # "cortar" el árbol en 3 grupos
print("\nCorte en 3 clusters (ward):", dict(zip(nombres, etiquetas.tolist())))

ac = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(puntos)
print("sklearn AgglomerativeClustering:", dict(zip(nombres, ac.labels_.tolist())))
