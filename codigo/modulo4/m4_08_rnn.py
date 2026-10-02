# Módulo 4 · RNN simple "a mano": la MISMA función (mismos pesos) en cada paso de tiempo
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Secuencia de 3 pasos, cada entrada con 2 variables
X = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])

W_xh = np.array([[0.5, -0.3],     # entrada -> oculto (2 unidades ocultas)
                 [0.8,  0.2]])
W_hh = np.array([[0.1,  0.4],     # oculto(t-1) -> oculto(t)  ← esto es la "memoria"
                 [-0.2, 0.3]])
b_h = np.zeros(2)
W_hy = np.array([1.0, -1.0])      # oculto -> salida

h = np.zeros(2)                   # estado oculto inicial h0
for t, x_t in enumerate(X, start=1):
    h = np.tanh(W_xh @ x_t + W_hh @ h + b_h)   # h_t = tanh(Wxh·x_t + Whh·h_{t-1} + b)
    y_t = W_hy @ h
    print(f"t={t}: x_t={x_t}, h_t={h}, y_t={y_t:.4f}")

# ¿Por qué se desvanece / explota el gradiente? BPTT multiplica factores paso a paso
print("\nFactor^pasos (lo que le pasa al gradiente al retroceder en el tiempo):")
for factor in [0.5, 0.9, 1.0, 1.1, 1.5]:
    print(f"  {factor}^50 = {factor ** 50:.3e}")
