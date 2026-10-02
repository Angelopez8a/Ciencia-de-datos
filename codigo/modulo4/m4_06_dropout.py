# Módulo 4 · Dropout ("inverted dropout", el que usa Keras)
import numpy as np

rng = np.random.default_rng(seed=7)
p = 0.5                                  # tasa de abandono (probabilidad de APAGAR)
activaciones = np.array([0.8, 0.1, 0.5, 0.9, 0.3, 0.6, 0.2, 0.7])

for paso in range(1, 4):
    mascara = (rng.random(activaciones.shape) >= p).astype(float)   # 1 = sigue viva
    salida = activaciones * mascara / (1 - p)                        # se re-escala por 1/(1-p)
    print(f"Paso {paso}: máscara = {mascara.astype(int)}  -> neuronas activas: {int(mascara.sum())}/8")
    print(f"         salida  = {salida.round(2)}")

print("\nEn PREDICCIÓN (model.predict) dropout se desactiva: se usan TODAS las neuronas.")
print("Neuronas esperadas activas por paso en una capa de 8 con p=0.5:", 8 * (1 - p))
