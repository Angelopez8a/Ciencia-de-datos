# Módulo 4 · Transfer learning con VGG16 (Oxford) preentrenada en ImageNet
# 1) Extracción de características: congelar la base y entrenar solo la "cabeza" nueva.
# 2) Fine-tuning: descongelar el último bloque y re-entrenar con learning rate MUY bajo.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
from keras import layers
from keras.applications import VGG16

PESOS = "imagenet"     # descarga ~58 MB la primera vez (usa None para probar sin internet)

base = VGG16(weights=PESOS, include_top=False,       # sin las capas densas originales
             input_shape=(224, 224, 3))
base.trainable = False                               # congelar TODO lo aprendido
print("Salida de la base convolucional:", base.output.shape)

modelo = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),
    base,
    layers.GlobalAveragePooling2D(),                  # (7, 7, 512) -> 512
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(3, activation="softmax"),            # nuestra tarea: 3 clases
])
modelo.compile(optimizer=keras.optimizers.Adam(1e-3),
               loss="categorical_crossentropy", metrics=["accuracy"])

def resumen(m, titulo):
    entrenables = sum(keras.ops.size(w) for w in m.trainable_weights)
    congelados = sum(keras.ops.size(w) for w in m.non_trainable_weights)
    print(f"{titulo}: entrenables = {int(entrenables):,} | congelados = {int(congelados):,}")

resumen(modelo, "Fase 1 (base congelada)")
# modelo.fit(train_ds, validation_data=val_ds, epochs=10)

# ----- Fase 2: fine-tuning del último bloque (block5) -----
base.trainable = True
for capa in base.layers:
    capa.trainable = capa.name.startswith("block5")
modelo.compile(optimizer=keras.optimizers.Adam(1e-5),   # LR 100 veces menor
               loss="categorical_crossentropy", metrics=["accuracy"])
resumen(modelo, "Fase 2 (fine-tuning block5)")
# modelo.fit(train_ds, validation_data=val_ds, epochs=5)
