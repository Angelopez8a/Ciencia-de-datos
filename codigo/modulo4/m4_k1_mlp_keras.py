# Módulo 4 · Red feedforward (MLP) con Keras para clasificar dígitos 8x8 (10 clases)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # oculta avisos de TensorFlow
import math
import keras
from keras import layers
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

keras.utils.set_random_seed(42)                    # reproducibilidad

X, y = load_digits(return_X_y=True)                # 1797 imágenes de 8x8 = 64 píxeles
X = X / 16.0                                       # escalar píxeles a [0, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = keras.Sequential([
    keras.Input(shape=(64,)),                      # capa de entrada: 64 variables (sin pesos)
    layers.Dense(32, activation="relu"),           # capa oculta
    layers.Dropout(0.2),                           # regularización
    layers.Dense(10, activation="softmax"),        # salida: 10 probabilidades que suman 1
])
modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",   # etiquetas enteras 0..9
               metrics=["accuracy"])
modelo.summary()

hist = modelo.fit(X_train, y_train, epochs=30, batch_size=32,
                  validation_split=0.1, verbose=0)
n_ent = int(len(X_train) * 0.9)                    # 90% para entrenar, 10% validación
print(f"Muestras de entrenamiento: {n_ent} -> batches por época = ceil({n_ent}/32) = {math.ceil(n_ent / 32)}")
print(f"Accuracy entrenamiento (última época): {hist.history['accuracy'][-1]:.3f}")
print(f"Accuracy validación   (última época): {hist.history['val_accuracy'][-1]:.3f}")
loss, acc = modelo.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy en prueba: {acc:.3f}")

probs = modelo.predict(X_test[:1], verbose=0)[0]
print("Probabilidades del 1er dígito de prueba:", probs.round(3))
print("Predicción:", probs.argmax(), "| Real:", y_test[0])
