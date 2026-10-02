# Módulo 4 · Red 2-3-1: propagación hacia adelante, pérdida y UN paso de backpropagation
# Todo con NumPy para ver exactamente qué pasa dentro de la red.
import numpy as np

np.set_printoptions(precision=4, suppress=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Entrada (2 variables) y etiqueta real (clase 1)
x = np.array([1.0, 0.5])
y = 1.0

# Pesos "aleatorios" fijos para que el resultado sea reproducible
W1 = np.array([[ 0.2, -0.4],    # neurona oculta 1: w11, w12
               [ 0.7,  0.1],    # neurona oculta 2: w21, w22
               [-0.3,  0.5]])   # neurona oculta 3: w31, w32
b1 = np.array([0.0, 0.0, 0.0])
W2 = np.array([0.6, -0.1, 0.4])  # w'1, w'2, w'3
b2 = 0.0

# ---------- 1) FORWARD ----------
z1 = W1 @ x + b1          # suma ponderada de cada neurona oculta
a1 = sigmoid(z1)          # activación (aquí entra la NO linealidad)
z2 = W2 @ a1 + b2         # neurona de salida
y_hat = sigmoid(z2)       # probabilidad de clase 1
print("z1 (sumas ponderadas ocultas):", z1)
print("a1 (activaciones ocultas):    ", a1)
print(f"z2 = {z2:.4f}  ->  y_hat = {y_hat:.4f}")

# ---------- 2) PÉRDIDA (entropía cruzada binaria) ----------
loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
print(f"Pérdida (binary cross-entropy) = {loss:.4f}")

# ---------- 3) BACKPROPAGATION (regla de la cadena) ----------
dz2 = y_hat - y                    # dL/dz2 para sigmoide + BCE
dW2 = dz2 * a1                     # dL/dW2
db2 = dz2
da1 = dz2 * W2                     # el error "viaja hacia atrás"
dz1 = da1 * a1 * (1 - a1)          # derivada de la sigmoide: s(1-s)
dW1 = np.outer(dz1, x)
db1 = dz1
print("dL/dW2:", dW2)
print("dL/dW1:\n", dW1)

# ---------- 4) ACTUALIZACIÓN con gradiente descendente ----------
lr = 0.5                           # learning rate (tamaño del "salto")
W2 -= lr * dW2;  b2 -= lr * db2
W1 -= lr * dW1;  b1 -= lr * db1

y_hat_nuevo = sigmoid(W2 @ sigmoid(W1 @ x + b1) + b2)
loss_nuevo = -np.log(y_hat_nuevo)
print(f"Después de 1 paso: y_hat = {y_hat_nuevo:.4f}, pérdida = {loss_nuevo:.4f}")
