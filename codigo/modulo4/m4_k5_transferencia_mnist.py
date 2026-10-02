# Módulo 4 · Un experimento real de transfer learning con MNIST
# Datos reales: 70,000 dígitos escritos a mano (MNIST; LeCun, Cortes y Burges), 60,000 de entrenamiento y 10,000 de prueba.
# Tarea origen: distinguir los dígitos 0 a 4, con todos sus ejemplos (unos 30,000).
# Tarea destino: distinguir los dígitos 5 a 9 con solo 500 imágenes de entrenamiento.
# Se compara entrenar la red destino desde cero contra reutilizar la base convolucional aprendida en la tarea
# origen, ya sea congelada (extracción de características) o reentrenándola (ajuste fino).
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import keras
from keras import layers

keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()      # mismos resultados en cada ejecución

(x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
x_tr, x_te = (x_tr / 255.0)[..., None], (x_te / 255.0)[..., None]   # píxeles en [0, 1], 1 canal
origen = y_tr < 5
destino = np.flatnonzero(y_tr >= 5)[:500]                         # solo 500 imágenes de 5 a 9
prueba = y_te >= 5
print(f"Origen: {origen.sum():,} imágenes (0-4) | destino: {len(destino)} imágenes (5-9) | prueba: {prueba.sum():,}")

def base_convolucional():
    return keras.Sequential([keras.Input(shape=(28, 28, 1)),
                             layers.Conv2D(32, (3, 3), activation="relu"), layers.MaxPooling2D((2, 2)),
                             layers.Conv2D(64, (3, 3), activation="relu"), layers.MaxPooling2D((2, 2)),
                             layers.Flatten()], name="base")

# 1) Entrenar la base en la tarea origen (una época)
base = base_convolucional()
modelo_origen = keras.Sequential([base, layers.Dense(5, activation="softmax")])
modelo_origen.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
modelo_origen.fit(x_tr[origen], y_tr[origen], epochs=1, batch_size=128, verbose=0)
print(f"Tarea origen, accuracy en prueba (0-4): {modelo_origen.evaluate(x_te[~prueba], y_te[~prueba], verbose=0)[1]:.3f}")

# 2) Tarea destino: la misma arquitectura entrenada de tres maneras
pesos_origen = base.get_weights()
def entrenar_destino(pesos=None, congelar=False):
    b = base_convolucional()
    if pesos is not None:
        b.set_weights(pesos)                                       # reutiliza lo aprendido en 0-4
    b.trainable = not congelar
    modelo = keras.Sequential([b, layers.Dense(5, activation="softmax")])
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = modelo.fit(x_tr[destino], y_tr[destino] - 5, epochs=15, batch_size=32, verbose=0,
                      validation_data=(x_te[prueba], y_te[prueba] - 5))
    return hist.history["val_accuracy"]

desde_cero = entrenar_destino()
congelada = entrenar_destino(pesos_origen, congelar=True)        # extracción de características
ajuste_fino = entrenar_destino(pesos_origen, congelar=False)     # fine-tuning de toda la base
print("\nAccuracy en las imágenes de prueba de 5 a 9, por época:")
print(f"{'época':>5}{'desde cero':>13}{'base congelada':>17}{'ajuste fino':>14}")
for e, fila in enumerate(zip(desde_cero, congelada, ajuste_fino), start=1):
    print(f"{e:>5}" + "".join(f"{v:>{w}.3f}" for v, w in zip(fila, (13, 17, 14))))
