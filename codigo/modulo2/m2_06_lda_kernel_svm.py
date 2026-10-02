# Módulo 2 · Análisis discriminante (LDA/QDA), truco del kernel y Máquinas de Vector Soporte
import numpy as np
from sklearn.datasets import load_iris, make_circles
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge

# --- LDA vs QDA ---
X, y = load_iris(return_X_y=True)
Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
for m in [LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis()]:
    print(f"{type(m).__name__:<32} accuracy = {m.fit(Xt, yt).score(Xv, yv):.3f}")

# --- Truco del kernel: k(x, y) = (x·y)² es igual a φ(x)·φ(y) sin calcular φ ---
x, z = np.array([1., 2., 3.]), np.array([4., 5., 6.])
phi = lambda v: np.array([v[i] * v[j] for i in range(3) for j in range(3)])   # 9 dimensiones
print(f"\nφ(x)·φ(z) en 9 dimensiones = {phi(x) @ phi(z):.0f}")
print(f"(x·z)² en 3 dimensiones    = {(x @ z) ** 2:.0f}   <- mismo resultado, mucho menos cómputo")

# --- Datos NO linealmente separables: círculos concéntricos ---
X, y = make_circles(n_samples=400, factor=0.4, noise=0.08, random_state=0)
Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=0)
for kernel in ["linear", "poly", "rbf"]:
    svm = SVC(kernel=kernel, C=1.0, degree=2).fit(Xt, yt)
    print(f"SVM kernel={kernel:<6} accuracy = {svm.score(Xv, yv):.3f} | vectores soporte = {svm.n_support_.sum()}")

kr = KernelRidge(kernel="rbf", alpha=1.0).fit(Xt, yt)
print(f"Kernel Ridge (rbf) accuracy = {np.mean((kr.predict(Xv) > 0.5) == yv):.3f}")

# --- Pérdida de bisagra (hinge): max(0, 1 − y·f(x)), con y ∈ {−1, 1} ---
for y_real, fx in [(1, 2.0), (1, 0.5), (1, -1.0), (-1, -3.0)]:
    print(f"y = {y_real:>2}, f(x) = {fx:>4} -> hinge = {max(0, 1 - y_real * fx):.1f}")
