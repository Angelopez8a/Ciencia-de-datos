# Módulo 4 · Convolución 2D, convolución RGB, ReLU, pooling y flattening (NumPy)
import numpy as np

def conv2d(imagen, kernel, stride=1, padding=0):
    """Convolución 'válida' como la usan las CNN (técnicamente correlación cruzada)."""
    if padding > 0:
        imagen = np.pad(imagen, padding)          # agrega orilla de ceros
    k = kernel.shape[0]
    salida_n = (imagen.shape[0] - k) // stride + 1
    salida = np.zeros((salida_n, salida_n))
    for i in range(salida_n):
        for j in range(salida_n):
            region = imagen[i*stride:i*stride+k, j*stride:j*stride+k]
            salida[i, j] = np.sum(region * kernel)   # multiplicar elemento a elemento y sumar
    return salida

# --- Ejemplo de la diapositiva: imagen 5x5 y kernel 3x3 ---
imagen = np.array([[1, 1, 1, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1],
                   [0, 0, 1, 1, 0],
                   [0, 1, 1, 0, 0]])
kernel = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])
mapa = conv2d(imagen, kernel)
print("Mapa de características (5x5 * 3x3, stride 1, sin padding):")
print(mapa.astype(int))

print("\nCon padding=1 ('same'), la salida conserva el tamaño 5x5:")
print(conv2d(imagen, kernel, padding=1).astype(int))

# --- Convolución sobre imagen RGB (3 canales): se suma cada canal + el sesgo ---
R = np.array([[0, 0, 0], [0, 156, 155], [0, 153, 154]])
G = np.array([[0, 0, 0], [0, 167, 166], [0, 164, 165]])
B = np.array([[0, 0, 0], [0, 163, 162], [0, 160, 161]])
kR = np.array([[-1, -1, 1], [0, 1, -1], [0, 1, 1]])
kG = np.array([[1, 0, 0], [1, -1, -1], [1, 0, -1]])
kB = np.array([[0, 1, 1], [0, 1, 0], [1, -1, 1]])
sR, sG, sB = np.sum(R*kR), np.sum(G*kG), np.sum(B*kB)
sesgo = 1
print(f"\nRGB: canal R = {sR}, canal G = {sG}, canal B = {sB}, sesgo = {sesgo}")
print(f"Valor de salida = {sR} + ({sG}) + {sB} + {sesgo} = {sR + sG + sB + sesgo}")

# --- ReLU sobre un mapa con negativos ---
fm = np.array([[ 3, -2], [-1, 5]])
print("\nReLU([[3,-2],[-1,5]]) =", np.maximum(0, fm).tolist())

# --- Pooling 2x2 con stride 2 (ejemplo de la diapositiva) ---
X = np.array([[2, 2, 7, 3],
              [9, 4, 6, 1],
              [8, 5, 2, 4],
              [3, 1, 2, 6]])
bloques = X.reshape(2, 2, 2, 2).swapaxes(1, 2)   # 4 bloques de 2x2
print("\nMax pooling 2x2:\n", bloques.max(axis=(2, 3)))
print("Average pooling 2x2:\n", bloques.mean(axis=(2, 3)))

# --- Flattening: la matriz se vuelve un vector (fila por fila) ---
print("\nFlatten del max pooling:", bloques.max(axis=(2, 3)).flatten())

# --- Tamaño de salida: O = (W - K + 2P) / S + 1 ---
def tam_salida(W, K, P=0, S=1):
    return (W - K + 2 * P) // S + 1
print("\nTamaños de salida:")
print(" 28x28, kernel 3, P=0, S=1 ->", tam_salida(28, 3))
print(" 28x28, kernel 3, P=1, S=1 ->", tam_salida(28, 3, P=1))
print(" 32x32, kernel 5, P=0, S=1 ->", tam_salida(32, 5))
print(" 224x224, kernel 7, P=3, S=2 ->", tam_salida(224, 7, P=3, S=2))
