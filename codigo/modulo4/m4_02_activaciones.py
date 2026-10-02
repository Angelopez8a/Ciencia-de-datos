# Módulo 4 · Funciones de activación y sus derivadas
import numpy as np

np.set_printoptions(precision=4, suppress=True)

def sigmoid(z):  return 1 / (1 + np.exp(-z))
def tanh(z):     return np.tanh(z)
def relu(z):     return np.maximum(0, z)
def softmax(z):
    e = np.exp(z - np.max(z))      # restar el máximo evita desbordamiento
    return e / e.sum()

z = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
print("z       :", z)
print("sigmoid :", sigmoid(z), "-> rango (0, 1)")
print("tanh    :", tanh(z),    "-> rango (-1, 1)")
print("ReLU    :", relu(z),    "-> rango [0, inf)")

# Derivadas: clave para entender el gradiente que se desvanece
d_sig  = sigmoid(z) * (1 - sigmoid(z))
d_tanh = 1 - np.tanh(z) ** 2
d_relu = (z > 0).astype(float)
print("\nderivada sigmoid:", d_sig, " (máx 0.25 en z=0)")
print("derivada tanh   :", d_tanh, " (máx 1 en z=0)")
print("derivada ReLU   :", d_relu, " (constante = 1 si z>0)")

# Softmax: convierte "logits" en probabilidades que suman 1
logits = np.array([2.0, 1.0, 0.1])
p = softmax(logits)
print("\nlogits :", logits)
print("softmax:", p, " suma =", p.sum().round(4))
print("clase predicha:", np.argmax(p))
