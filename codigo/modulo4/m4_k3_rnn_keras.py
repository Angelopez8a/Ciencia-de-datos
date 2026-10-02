# Módulo 4 · Embedding + SimpleRNN / LSTM / GRU con Keras (clasificación de sentimiento)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(42)

# ---------- Parte A: comparar cuántos parámetros tiene cada celda ----------
VOCAB, DIM, LARGO = 10000, 16, 100
for Celda in (layers.SimpleRNN, layers.LSTM, layers.GRU):
    m = keras.Sequential([
        keras.Input(shape=(LARGO,)),                 # 100 enteros (índices de palabras)
        layers.Embedding(input_dim=VOCAB, output_dim=DIM),
        Celda(32),                                   # muchos-a-uno: solo la última salida
        layers.Dense(1, activation="sigmoid"),
    ])
    capas = [(l.name, l.count_params()) for l in m.layers]
    print(f"{Celda.__name__:<10} -> {capas}  total = {m.count_params():,}")

# ---------- Parte B: mini ejemplo de sentimiento (1 = positivo, 0 = negativo) ----------
textos = ["me encanta esta película", "excelente actuación y gran historia",
          "muy buena la recomiendo", "una obra maestra increíble",
          "me gustó muchísimo", "gran película muy divertida",
          "pésima no la recomiendo", "muy aburrida y lenta",
          "la peor película del año", "terrible actuación mala historia",
          "no me gustó nada", "aburrida y sin sentido"]
etiquetas = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

vectorizador = layers.TextVectorization(max_tokens=100, output_sequence_length=6)
vectorizador.adapt(textos)                           # equivale al Tokenizer: palabra -> entero
X = vectorizador(np.array(textos))
print("\n'me encanta esta película' ->", X[0].numpy(), "(0 = relleno/padding)")
vocab_size = len(vectorizador.get_vocabulary())

modelo = keras.Sequential([
    keras.Input(shape=(6,)),
    layers.Embedding(input_dim=vocab_size, output_dim=8),
    layers.LSTM(8),
    layers.Dense(1, activation="sigmoid"),
])
modelo.compile(optimizer=keras.optimizers.Adam(0.01), loss="binary_crossentropy",
               metrics=["accuracy"])
modelo.fit(X, etiquetas, epochs=60, verbose=0)
print("Accuracy en entrenamiento:", round(modelo.evaluate(X, etiquetas, verbose=0)[1], 3))

nuevas = np.array(["muy buena historia", "pésima y aburrida"])
for frase, p in zip(nuevas, modelo.predict(vectorizador(nuevas), verbose=0).ravel()):
    print(f"'{frase}' -> P(positivo) = {p:.3f}")
