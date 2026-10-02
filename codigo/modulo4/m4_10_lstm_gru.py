# Módulo 4 · Un paso de una celda LSTM y de una celda GRU (notación de las diapositivas)
import numpy as np

np.set_printoptions(precision=4, suppress=True)
sig = lambda z: 1 / (1 + np.exp(-z))

x_t = np.array([1.0, 0.5])        # entrada en el tiempo t
a_prev = np.array([0.2, -0.1])    # a_{t-1} (estado oculto anterior)
c_prev = np.array([0.5, 0.3])     # c_{t-1} (estado de celda anterior)
concat = np.concatenate([a_prev, x_t])     # [a_{t-1}, x_t]

rng = np.random.default_rng(1)
Wc, Wu, Wf, Wo = (rng.normal(scale=0.5, size=(2, 4)) for _ in range(4))
bc = bu = bf = bo = np.zeros(2)

# ----------------- LSTM -----------------
c_tilde = np.tanh(Wc @ concat + bc)        # candidato a memoria
G_u = sig(Wu @ concat + bu)                # puerta de actualización (entrada)
G_f = sig(Wf @ concat + bf)                # puerta de olvido
G_o = sig(Wo @ concat + bo)                # puerta de salida
c_t = G_u * c_tilde + G_f * c_prev         # nueva memoria
a_t = G_o * np.tanh(c_t)                   # nuevo estado oculto
print("LSTM")
print(" c~_t =", c_tilde, " G_u =", G_u, " G_f =", G_f, " G_o =", G_o)
print(" c_t  =", c_t, "  a_t =", a_t)

# ----------------- GRU -----------------
Wr = rng.normal(scale=0.5, size=(2, 4))
G_r = sig(Wr @ concat)                                   # puerta de reinicio
G_u2 = sig(Wu @ concat)                                  # puerta de actualización
c_tilde2 = np.tanh(Wc @ np.concatenate([G_r * c_prev, x_t]))
c_t2 = G_u2 * c_tilde2 + (1 - G_u2) * c_prev             # mezcla: nuevo vs. viejo
a_t2 = c_t2                                              # en GRU a_t = c_t
print("\nGRU")
print(" G_r =", G_r, " G_u =", G_u2, " c~_t =", c_tilde2)
print(" c_t = a_t =", c_t2)
