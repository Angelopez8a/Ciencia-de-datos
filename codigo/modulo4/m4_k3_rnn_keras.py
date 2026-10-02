# Módulo 4 · Embedding + SimpleRNN / LSTM / GRU con Keras (clasificación de sentimiento con reseñas de IMDB)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import keras
from keras import layers

keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()      # mismos resultados en cada ejecución

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

# ---------- Parte B: sentimiento en reseñas reales de películas (IMDB) ----------
# 50,000 reseñas de IMDB (Maas et al., 2011): 25,000 para entrenar y 25,000 para probar, mitad positivas
# (1) y mitad negativas (0). Keras las entrega ya convertidas en enteros: cada palabra es su lugar en el
# ranking de frecuencia, desplazado 3 posiciones (0 = relleno, 1 = inicio de reseña, 2 = palabra fuera del vocabulario).
(x_tr, y_tr), (x_te, y_te) = keras.datasets.imdb.load_data(num_words=VOCAB)
palabra = {i + 3: w for w, i in keras.datasets.imdb.get_word_index().items()}
texto = lambda seq, n: " ".join(palabra.get(i, "?") for i in seq[1:n + 1])
print(f"\nReseñas: {len(x_tr):,} de entrenamiento y {len(x_te):,} de prueba | positivas: {y_tr.mean():.0%}")
print("Inicio de la 1.ª reseña de entrenamiento:", texto(x_tr[0], 12), "...")
print("  como enteros:", x_tr[0][:13])

LARGO = 200
x_tr = keras.utils.pad_sequences(x_tr, maxlen=LARGO)         # recorta o rellena cada reseña a 200 palabras
x_te_pad = keras.utils.pad_sequences(x_te, maxlen=LARGO)
print("Tensor de entrada:", x_tr.shape)

modelo = keras.Sequential([
    keras.Input(shape=(LARGO,)),
    layers.Embedding(input_dim=VOCAB, output_dim=32),
    layers.LSTM(32),
    layers.Dense(1, activation="sigmoid"),
])
modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
hist = modelo.fit(x_tr, y_tr, epochs=3, batch_size=128, validation_split=0.2, verbose=0)
for e, (a, va) in enumerate(zip(hist.history["accuracy"], hist.history["val_accuracy"]), start=1):
    print(f"Época {e}: accuracy entrenamiento = {a:.3f} | validación = {va:.3f}")
print(f"Accuracy en las {len(x_te):,} reseñas de prueba: {modelo.evaluate(x_te_pad, y_te, verbose=0)[1]:.3f}")

for k in (1, 3, 5):                                          # tres reseñas de prueba, con su etiqueta real
    p = modelo.predict(x_te_pad[k:k + 1], verbose=0)[0, 0]
    print(f"'{texto(x_te[k], 12)} ...'\n   -> P(positiva) = {p:.3f} | real: {'positiva' if y_te[k] else 'negativa'}")
