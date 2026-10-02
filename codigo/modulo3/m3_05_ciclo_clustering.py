# Módulo 3 · Ciclo completo de clustering: preparar -> estandarizar -> reducir -> agrupar -> evaluar -> perfilar
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

rng = np.random.default_rng(5)
def grupo(n, gasto, visitas, antig, desc):
    return pd.DataFrame({"gasto_mensual": rng.normal(gasto, gasto * 0.15, n),
                         "visitas_mes": rng.poisson(visitas, n),
                         "antiguedad_meses": rng.normal(antig, 6, n).clip(1),
                         "uso_descuentos": rng.beta(*desc, n)})
clientes = pd.concat([grupo(300, 800, 2, 40, (2, 8)), grupo(200, 3500, 8, 60, (2, 5)),
                      grupo(250, 1200, 12, 10, (8, 2))], ignore_index=True)
print("1) Datos:", clientes.shape)

X = StandardScaler().fit_transform(clientes)               # 2) misma escala
pca = PCA(n_components=0.9).fit(X)                          # 3) quedarnos con 90% de varianza
Xp = pca.transform(X)
print(f"3) PCA: {Xp.shape[1]} componentes explican {pca.explained_variance_ratio_.sum():.1%}")

sil = {k: silhouette_score(Xp, KMeans(k, n_init=10, random_state=0).fit_predict(Xp)) for k in range(2, 7)}
k = max(sil, key=sil.get)
print("4-5) Silueta por k:", {kk: round(v, 3) for kk, v in sil.items()}, "-> k elegido =", k)

clientes["segmento"] = KMeans(k, n_init=10, random_state=0).fit_predict(Xp)
perfil = clientes.groupby("segmento").agg(n=("gasto_mensual", "size"), gasto=("gasto_mensual", "mean"),
                                          visitas=("visitas_mes", "mean"),
                                          antiguedad=("antiguedad_meses", "mean"),
                                          descuentos=("uso_descuentos", "mean")).round(2)
print("6) Perfilamiento de segmentos (medias):\n", perfil.to_string())
