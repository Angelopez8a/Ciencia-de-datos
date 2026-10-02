# Módulo 1 · Reducción de dimensiones: PCA (para modelar) vs t-SNE (para visualizar)
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X, y = load_wine(return_X_y=True)                     # 178 vinos, 13 variables, 3 clases
X_ss = StandardScaler().fit_transform(X)              # ¡PCA usa covarianza: estandarizar antes!

pca = PCA().fit(X_ss)
var = pca.explained_variance_ratio_
print("Varianza explicada por componente:", (var * 100).round(1))
print("Varianza acumulada               :", (np.cumsum(var) * 100).round(1))
k = int(np.argmax(np.cumsum(var) >= 0.80) + 1)
print(f"Componentes necesarios para >= 80% de la varianza: {k} (de 13)")

X_pca = PCA(n_components=2).fit_transform(X_ss)
print("\nPrimeras 3 filas en 2 componentes principales:\n", X_pca[:3].round(3))

tsne = TSNE(n_components=2, perplexity=30, random_state=0)
X_tsne = tsne.fit_transform(X_ss)
print("\nt-SNE -> forma:", X_tsne.shape, "| divergencia KL final:", round(tsne.kl_divergence_, 3))
print("t-SNE NO tiene .transform() para datos nuevos:", hasattr(tsne, "transform"))
