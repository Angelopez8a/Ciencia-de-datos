# Módulo 3 · PCA paso a paso: covarianza -> eigenvectores -> proyección (y comparación con sklearn)
# Datos reales: 150 flores de iris de tres especies (Anderson, 1935; Fisher, 1936), medidas en centímetros.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=4, suppress=True)
iris = load_iris()
X = iris.data[:, 2:4]                        # largo y ancho del pétalo
print("Correlación entre el largo y el ancho del pétalo:", round(float(np.corrcoef(X, rowvar=False)[0, 1]), 3))

Xc = X - X.mean(axis=0)                      # 1) centrar (ambas en cm: no hace falta estandarizar)
C = np.cov(Xc, rowvar=False)                 # 2) matriz de covarianza
valores, vectores = np.linalg.eigh(C)        # 3) eigen-descomposición
orden = np.argsort(valores)[::-1]
valores, vectores = valores[orden], vectores[:, orden]
print("Matriz de covarianza:\n", C)
print("Eigenvalores (varianza de cada componente):", valores)
print("Proporción de varianza explicada:", valores / valores.sum())
print("Pesos (loadings) de PC1 [largo, ancho]:", vectores[:, 0])
Z = Xc @ vectores[:, :1]                     # 4) proyectar en la primera componente
print("Primeras 5 flores proyectadas en PC1:", Z[:5].ravel())

pca = PCA(n_components=2).fit(X)
print("\nsklearn explained_variance_ratio_:", pca.explained_variance_ratio_)
print("sklearn PC1, primeras 5 flores (el signo puede variar):", pca.transform(X)[:5, 0])

# Con las cuatro medidas (sépalo y pétalo), estandarizadas
pca4 = PCA().fit(StandardScaler().fit_transform(iris.data))
print("\nCuatro medidas estandarizadas")
print("  Eigenvalores:", pca4.explained_variance_)
print("  Varianza explicada:", pca4.explained_variance_ratio_, "| acumulada:", pca4.explained_variance_ratio_.cumsum())
