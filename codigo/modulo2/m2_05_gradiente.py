# Módulo 2 · Gradiente descendente sobre y = x² − 2x − 3 (ejemplo del documento del módulo)
# Derivada: y' = 2x − 2   ->   mínimo en x = 1 (y = −4)
f = lambda x: x ** 2 - 2 * x - 3
df = lambda x: 2 * x - 2

def gradiente_descendente(x0=3.0, lr=0.1, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(1, max_iter + 1):
        paso = lr * df(x)               # tamaño del paso = gradiente * tasa de aprendizaje
        x = x - paso                    # nuevos parámetros = anteriores − paso
        if i <= 5:
            print(f"  iter {i}: x = {x:.5f}, y = {f(x):.5f}, gradiente = {df(x):.5f}")
        if abs(df(x)) < tol:            # criterio de paro: gradiente ≈ 0
            return x, i
    return x, max_iter

for lr in [0.1, 0.5, 0.9]:
    print(f"lr = {lr}")
    x, it = gradiente_descendente(lr=lr)
    print(f"  -> converge a x = {x:.5f} en {it} iteraciones\n")

# Gradiente descendente para una regresión lineal (batch, SGD y mini-batch)
import numpy as np
rng = np.random.default_rng(0)
X = rng.uniform(0, 10, 1000); y = 4 + 3 * X + rng.normal(0, 1, 1000)

def entrenar(tam_lote, lr=0.01, epocas=50):
    b0 = b1 = 0.0
    entrenar.actualizaciones = 0
    for _ in range(epocas):
        idx = rng.permutation(len(X))
        for k in range(0, len(X), tam_lote):
            j = idx[k:k + tam_lote]
            err = (b0 + b1 * X[j]) - y[j]
            b0 -= lr * 2 * err.mean()
            b1 -= lr * 2 * (err * X[j]).mean()
            entrenar.actualizaciones += 1
    return b0, b1

for nombre, lote in [("Batch (todos)", 1000), ("Mini-batch (32)", 32), ("SGD (1)", 1)]:
    b0, b1 = entrenar(lote, lr=0.01 if lote > 1 else 0.001)
    print(f"{nombre:<16}: β0 = {b0:.3f}, β1 = {b1:.3f}  (reales: 4, 3) "
          f"| {entrenar.actualizaciones:,} actualizaciones en 50 épocas")
