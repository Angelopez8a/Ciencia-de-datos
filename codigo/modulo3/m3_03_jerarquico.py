# Módulo 3 · Clustering jerárquico aglomerativo: matriz de enlace (linkage) y corte del dendrograma
# Datos reales: ubicación de seis ciudades de México (latitud y longitud del centro de cada ciudad, en grados).
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

ciudades = {"Ciudad de México": (19.43, -99.13), "Toluca": (19.29, -99.65), "Monterrey": (25.67, -100.31),
            "Saltillo": (25.42, -101.00), "Mérida": (20.97, -89.62), "Cancún": (21.16, -86.85)}
nombres = list(ciudades)
lat, lon = np.radians(np.array(list(ciudades.values()))).T

# Proyección plana en kilómetros (equirectangular): suficiente para distancias de unos cientos de km
R = 6371
puntos = np.c_[R * lon * np.cos(lat.mean()), R * lat]
d = lambda a, b: np.linalg.norm(puntos[nombres.index(a)] - puntos[nombres.index(b)])
print(f"Distancias en línea recta: Ciudad de México-Toluca = {d('Ciudad de México', 'Toluca'):.0f} km, "
      f"Monterrey-Saltillo = {d('Monterrey', 'Saltillo'):.0f} km, Mérida-Cancún = {d('Mérida', 'Cancún'):.0f} km")

for metodo in ["single", "complete", "average", "ward"]:
    Z = linkage(puntos, method=metodo)
    print(f"\nEnlace '{metodo}':")
    grupos = {i: [n] for i, n in enumerate(nombres)}      # scipy numera los grupos nuevos desde 6
    for i, (a, b, dist, n) in enumerate(Z):
        grupos[len(nombres) + i] = grupos[int(a)] + grupos[int(b)]
        altura = f"a {dist:,.0f} km" if metodo != "ward" else f"a una altura de {dist:,.0f}"   # en ward no es una distancia
        print(f"  paso {i + 1}: une {' + '.join(grupos[int(a)])} con {' + '.join(grupos[int(b)])} {altura}")

Z = linkage(puntos, method="ward")
etiquetas = fcluster(Z, t=3, criterion="maxclust")          # "cortar" el árbol en 3 grupos
print("\nCorte en 3 clusters (ward):", dict(zip(nombres, etiquetas.tolist())))
ac = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(puntos)
print("sklearn AgglomerativeClustering:", dict(zip(nombres, ac.labels_.tolist())))
