# Módulo 4 · One-hot vs Word Embeddings y similitud del coseno
import numpy as np

def coseno(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

# --- One-hot con V = {Have, a, good, great, day} ---
V = ["Have", "a", "good", "great", "day"]
one_hot = {w: np.eye(len(V))[i] for i, w in enumerate(V)}
print("one-hot 'good' :", one_hot["good"])
print("one-hot 'great':", one_hot["great"])
print("cos(good, great) =", coseno(one_hot["good"], one_hot["great"]))
print("cos(day, Have)   =", coseno(one_hot["day"], one_hot["Have"]))
print("-> con one-hot TODAS las palabras son igual de distintas (ángulo de 90°)\n")

# --- Embedding de 4 dimensiones (valores de la diapositiva) ---
emb = {"cat": np.array([1.2, -0.1, 4.3, 3.2]),
       "mat": np.array([0.4, 2.5, -0.9, 0.5]),
       "on":  np.array([2.1, 0.3, 0.1, 0.4])}
for a, b in [("cat", "mat"), ("cat", "on"), ("mat", "on")]:
    print(f"cos({a}, {b}) = {coseno(emb[a], emb[b]): .4f}")

# --- Analogía estilo Word2Vec con vectores de juguete [realeza, masculino, femenino] ---
w2v = {"rey":    np.array([0.95, 0.90, 0.05]),
       "reina":  np.array([0.95, 0.05, 0.90]),
       "hombre": np.array([0.10, 0.92, 0.04]),
       "mujer":  np.array([0.10, 0.03, 0.91]),
       "manzana": np.array([0.01, 0.10, 0.12])}
objetivo = w2v["rey"] - w2v["hombre"] + w2v["mujer"]
ranking = sorted(((coseno(objetivo, v), w) for w, v in w2v.items()
                  if w not in ("rey", "hombre", "mujer")), reverse=True)
print("\nrey - hombre + mujer ≈", [(w, round(float(s), 4)) for s, w in ranking])

# --- Lo que hace por dentro la capa Embedding: buscar una FILA de una matriz ---
vocab_size, dim = 11, 4                   # índices 0..10 -> input_dim = 11
rng = np.random.default_rng(0)
E = rng.normal(size=(vocab_size, dim)).round(2)   # pesos entrenables (inician aleatorios)
oracion = [3, 7, 1]                        # palabras ya convertidas a enteros
print("\nMatriz E:", E.shape, "-> parámetros =", E.size)
print("Embedding de la oración [3, 7, 1]:\n", E[oracion])
