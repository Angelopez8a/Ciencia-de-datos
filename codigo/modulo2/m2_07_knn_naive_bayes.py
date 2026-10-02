# Módulo 2 · K-vecinos más cercanos (distancias) y Bayes ingenuo (ejemplo del coche robado)
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# ---------------- KNN ----------------
a, b = np.array([1, 2]), np.array([4, 6])
print("Manhattan  :", np.sum(np.abs(a - b)))                 # |1-4| + |2-6| = 7
print("Euclidiana :", np.sqrt(np.sum((a - b) ** 2)))          # √(9+16) = 5
print("Minkowski p=3:", round(np.sum(np.abs(a - b) ** 3) ** (1 / 3), 4))

X = np.array([[1, 1], [1, 2], [2, 1], [6, 6], [7, 6], [6, 7]])
y = np.array(["A", "A", "A", "B", "B", "B"])
nuevo = np.array([3, 3])
d = np.sqrt(((X - nuevo) ** 2).sum(axis=1))                  # 1) distancias
orden = np.argsort(d)                                         # 2) ordenar
k = 3
print(f"\nDistancias a {nuevo}: {d.round(2)}")
print(f"{k} vecinos más cercanos: {y[orden[:k]]} -> clase mayoritaria: "
      f"{pd.Series(y[orden[:k]]).mode()[0]}")                   # 3) votar
print("sklearn:", KNeighborsClassifier(n_neighbors=3).fit(X, y).predict([nuevo]))

# ---------------- Bayes ingenuo ----------------
autos = pd.DataFrame({
    "Color":  ["Rojo", "Rojo", "Rojo", "Amarillo", "Amarillo", "Amarillo", "Amarillo", "Amarillo", "Rojo", "Rojo"],
    "Tipo":   ["Deportivo", "Deportivo", "Deportivo", "Deportivo", "SUV", "SUV", "SUV", "SUV", "SUV", "Deportivo"],
    "Origen": ["Nacional", "Nacional", "Nacional", "Importado", "Importado", "Importado", "Importado", "Nacional", "Importado", "Importado"],
    "Robado": ["Sí", "No", "Sí", "No", "Sí", "No", "Sí", "No", "No", "Sí"],
})

def naive_bayes(nuevo, laplace=0):
    resultados = {}
    for clase, grupo in autos.groupby("Robado"):
        prob = len(grupo) / len(autos)                        # P(y)  (a priori)
        detalle = [f"P({clase})={prob:.2f}"]
        for var, valor in nuevo.items():
            k = autos[var].nunique()
            p = ((grupo[var] == valor).sum() + laplace) / (len(grupo) + laplace * k)
            prob *= p                                         # Π P(x_i | y)  (independencia)
            detalle.append(f"P({valor}|{clase})={p:.2f}")
        resultados[clase] = prob
        print("   ", " · ".join(detalle), f"= {prob:.4f}")
    return max(resultados, key=resultados.get)

for caso in [{"Color": "Rojo", "Tipo": "Deportivo", "Origen": "Importado"},
             {"Color": "Amarillo", "Tipo": "SUV", "Origen": "Nacional"}]:
    print(f"\n{caso}")
    print("  -> ¿Robado?", naive_bayes(caso))

print("\nFrecuencia cero: P(Verde|Sí) sin suavizar = 0/5 = 0  -> anula todo el producto")
print("Con suavizado de Laplace (+1):", round((0 + 1) / (5 + 1 * 3), 3), "(suponiendo 3 colores posibles)")
