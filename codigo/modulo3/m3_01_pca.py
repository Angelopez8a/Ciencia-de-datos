# Módulo 3 · PCA paso a paso: covarianza -> eigenvectores -> proyección (y comparación con sklearn)
import numpy as np
from sklearn.decomposition import PCA

np.set_printoptions(precision=4, suppress=True)
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
              [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]])

Xc = X - X.mean(axis=0)                      # 1) centrar (en la práctica: estandarizar)
C = np.cov(Xc, rowvar=False)                 # 2) matriz de covarianza
valores, vectores = np.linalg.eigh(C)        # 3) eigen-descomposición
orden = np.argsort(valores)[::-1]
valores, vectores = valores[orden], vectores[:, orden]
print("Matriz de covarianza:\n", C)
print("Eigenvalores (varianza de cada componente):", valores)
print("Proporción de varianza explicada:", valores / valores.sum())
Z = Xc @ vectores[:, :1]                     # 4) proyectar en la 1a componente
print("Datos proyectados en PC1:", Z.ravel())

pca = PCA(n_components=2).fit(X)
print("\nsklearn explained_variance_ratio_:", pca.explained_variance_ratio_)
print("sklearn PC1 (signo puede variar):", pca.transform(X)[:, 0])
