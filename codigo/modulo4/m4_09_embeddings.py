# Módulo 4 · One-hot vs Word Embeddings y similitud del coseno
from pathlib import Path
import numpy as np
import pandas as pd

def coseno(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# --- One-hot con V = {Have, a, good, great, day} (ejemplo de la diapositiva) ---
V = ["Have", "a", "good", "great", "day"]
one_hot = {w: np.eye(len(V))[i] for i, w in enumerate(V)}
print("one-hot 'good' :", one_hot["good"])
print("one-hot 'great':", one_hot["great"])
print("cos(good, great) =", coseno(one_hot["good"], one_hot["great"]))
print("cos(day, Have)   =", coseno(one_hot["day"], one_hot["Have"]))
print("-> con one-hot TODAS las palabras son igual de distintas (ángulo de 90°)\n")

# --- Embedding de 4 dimensiones (valores ilustrativos de la diapositiva) ---
emb = {"cat": np.array([1.2, -0.1, 4.3, 3.2]),
       "mat": np.array([0.4, 2.5, -0.9, 0.5]),
       "on":  np.array([2.1, 0.3, 0.1, 0.4])}
for a, b in [("cat", "mat"), ("cat", "on"), ("mat", "on")]:
    print(f"cos({a}, {b}) = {coseno(emb[a], emb[b]): .4f}")

# --- Vectores reales: GloVe (Stanford NLP), 50 dimensiones, entrenados con Wikipedia 2014 y Gigaword 5 ---
# El archivo contiene 39 palabras (en inglés) copiadas sin cambios de los 400,000 vectores de glove-wiki-gigaword-50.
ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "glove_6B_50d_subconjunto.csv"
if ARCHIVO.exists():
    glove = pd.read_csv(ARCHIVO, index_col="palabra")
else:                                                # descarga el modelo completo (69 MB) y toma las mismas palabras
    import gzip, urllib.request
    URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
    PALABRAS = {"king", "man", "woman", "queen", "daughter", "prince", "throne", "princess", "walked", "walk", "swim",
                "swam", "raced", "rowed", "swimmers", "swims", "paris", "france", "spain", "aires", "buenos", "madrid",
                "rome", "santiago", "good", "great", "excellent", "bad", "cat", "dog", "horse", "mouse", "mat", "on",
                "apple", "banana", "orange", "grape", "computer"}
    with gzip.open(urllib.request.urlopen(URL), "rt", encoding="utf-8") as f:
        next(f)                                      # la primera línea es "400000 50"
        filas = {p[0]: p[1:] for p in (linea.split() for linea in f) if p[0] in PALABRAS}
    glove = pd.DataFrame.from_dict(filas, orient="index", dtype=float)
print(f"\nGloVe: {glove.shape[0]} palabras x {glove.shape[1]} dimensiones | 'king' empieza con",
      glove.loc["king"].values[:4], "...")
for a, b in [("cat", "dog"), ("cat", "mat"), ("good", "great"), ("good", "excellent"), ("good", "bad"),
             ("apple", "banana"), ("apple", "computer")]:
    print(f"cos({a}, {b}) = {coseno(glove.loc[a].values, glove.loc[b].values):.4f}")

def analogia(a, b, c, n=5):
    """Palabras más cercanas (coseno) al vector a - b + c, sin contar a, b ni c."""
    objetivo = (glove.loc[a] - glove.loc[b] + glove.loc[c]).values
    sim = {w: coseno(v, objetivo) for w, v in zip(glove.index, glove.values) if w not in (a, b, c)}
    return [(w, round(s, 3)) for w, s in sorted(sim.items(), key=lambda kv: -kv[1])[:n]]

print()
for a, b, c in [("king", "man", "woman"), ("walked", "walk", "swim"), ("paris", "france", "spain")]:
    print(f"{a} - {b} + {c} ≈ {analogia(a, b, c)}")

# --- Lo que hace por dentro la capa Embedding: buscar una FILA de una matriz ---
vocab_size, dim = 11, 4                   # índices 0..10 -> input_dim = 11
rng = np.random.default_rng(0)
E = rng.normal(size=(vocab_size, dim)).round(2)   # pesos entrenables (inician aleatorios)
oracion = [3, 7, 1]                        # palabras ya convertidas a enteros
print("\nMatriz E:", E.shape, "-> parámetros =", E.size)
print("Embedding de la oración [3, 7, 1]:\n", E[oracion])
