# Módulo 4 · Red convolucional (CNN) con Keras
# Parte A: arquitectura clásica tipo MNIST (28x28x1) solo para estudiar el summary.
# Parte B: la misma idea entrenada con los dígitos 8x8 de scikit-learn.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
from keras import layers
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

keras.utils.set_random_seed(42)

# ---------- Parte A: summary de una CNN para MNIST ----------
cnn_mnist = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),                       # alto, ancho, canales
    layers.Conv2D(32, (3, 3), activation="relu"),         # 28-3+1 = 26 -> (26, 26, 32)
    layers.MaxPooling2D((2, 2)),                          # 26/2 = 13   -> (13, 13, 32)
    layers.Conv2D(64, (3, 3), activation="relu"),         # 13-3+1 = 11 -> (11, 11, 64)
    layers.MaxPooling2D((2, 2)),                          # 11//2 = 5   -> (5, 5, 64)
    layers.Flatten(),                                     # 5*5*64 = 1600
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
], name="cnn_mnist")
cnn_mnist.summary()

# ---------- Parte B: entrenar con dígitos 8x8 ----------
X, y = load_digits(return_X_y=True)
X = (X / 16.0).reshape(-1, 8, 8, 1)                       # tensor (N, alto, ancho, canales)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cnn = keras.Sequential([
    keras.Input(shape=(8, 8, 1)),
    layers.Conv2D(16, (3, 3), padding="same", activation="relu"),   # padding same -> 8x8
    layers.MaxPooling2D((2, 2)),                                     # -> 4x4
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),                                     # -> 2x2
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("Tensor de entrada:", X_train.shape)
cnn.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
loss, acc = cnn.evaluate(X_test, y_test, verbose=0)
print(f"CNN dígitos 8x8 -> accuracy en prueba: {acc:.3f}  (parámetros: {cnn.count_params()})")
