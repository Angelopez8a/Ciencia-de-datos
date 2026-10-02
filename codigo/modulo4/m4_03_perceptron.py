# Módulo 4 · Perceptrón de una sola capa con la REGLA DELTA (compuerta lógica AND)
# Salida: 1 si la suma ponderada > 0 (umbral), -1 en otro caso.
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, -1, -1, 1])          # AND con etiquetas {-1, 1}

w = np.zeros(2)
b = 0.0
lr = 0.1

def predecir(x):
    return 1 if (w @ x + b) > 0 else -1

for epoca in range(1, 11):
    errores = 0
    for xi, yi in zip(X, y):
        y_pred = predecir(xi)
        if y_pred != yi:
            # Regla delta: ajusta los pesos en proporción al error
            w += lr * (yi - y_pred) * xi
            b += lr * (yi - y_pred)
            errores += 1
    print(f"Época {epoca}: errores = {errores}, w = {w.round(2)}, b = {round(b, 2)}")
    if errores == 0:
        break

print("Predicciones finales:", [predecir(xi) for xi in X])
