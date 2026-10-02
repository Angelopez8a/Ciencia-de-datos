# Módulo 2 · K vecinos más cercanos (KNN) y Bayes ingenuo
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------- KNN ----------------
# Datos reales: 150 flores de iris de tres especies (Anderson, 1935; Fisher, 1936).
# Se usan el largo y el ancho del pétalo, en centímetros.
iris = load_iris(as_frame=True)
petalo = iris.data[["petal length (cm)", "petal width (cm)"]].to_numpy()
especie = iris.target_names[iris.target]

a, b = petalo[0], petalo[50]                          # flor 1 (setosa) y flor 51 (versicolor)
print(f"Flor 1 {a} ({especie[0]}) y flor 51 {b} ({especie[50]})")
print("Manhattan    :", round(float(np.sum(np.abs(a - b))), 4))              # |1.4 − 4.7| + |0.2 − 1.4|
print("Euclidiana   :", round(float(np.sqrt(np.sum((a - b) ** 2))), 4))      # √(3.3² + 1.2²)
print("Minkowski p=3:", round(float(np.sum(np.abs(a - b) ** 3) ** (1 / 3)), 4))

nueva = np.array([5.1, 1.6])                          # pétalo de 5.1 × 1.6 cm, donde se traslapan dos especies
d = np.sqrt(((petalo - nueva) ** 2).sum(axis=1))      # 1) distancia a cada una de las 150 flores
orden = np.argsort(d, kind="stable")                  # 2) ordenar de menor a mayor
k = 5
print(f"\nLos {k} vecinos más cercanos a un pétalo de {nueva[0]} × {nueva[1]} cm:")
for i in orden[:k]:
    print(f"   flor {i + 1:>3}: {petalo[i]}  distancia = {d[i]:.3f}  {especie[i]}")
votos = pd.Series(especie[orden[:k]]).value_counts()  # 3) votar
print("Votos:", votos.to_dict(), "-> clase asignada:", votos.idxmax())
print("sklearn:", KNeighborsClassifier(n_neighbors=k).fit(petalo, especie).predict([nueva])[0])

# La escala importa. Datos reales: 178 vinos de tres cultivares de una misma región de Italia, con 13 mediciones
# químicas. La prolina va de 278 a 1,680 y el tono (hue) de 0.48 a 1.71: sin estandarizar, la prolina domina la distancia.
X, y = load_wine(return_X_y=True)
Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
sin_escala = KNeighborsClassifier(5).fit(Xt, yt).score(Xv, yv)
con_escala = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(Xt, yt).score(Xv, yv)
print(f"\nVinos, k = 5 -> accuracy sin estandarizar = {sin_escala:.3f} | estandarizando = {con_escala:.3f}")

# ---------------- Bayes ingenuo ----------------
def naive_bayes(datos, objetivo, nuevo, laplace=0):
    """Calcula P(y) · Π P(x_i | y) para cada clase y devuelve la clase con el producto mayor."""
    print("    clase: P(y) · " + " · ".join(f"P({valor} | y)" for valor in nuevo.values()))
    resultados = {}
    for clase, grupo in datos.groupby(objetivo):
        factores = [len(grupo) / len(datos)]          # P(y), probabilidad a priori
        for var, valor in nuevo.items():
            k = datos[var].nunique()
            factores.append(((grupo[var] == valor).sum() + laplace) / (len(grupo) + laplace * k))
        resultados[clase] = float(np.prod(factores))  # supuesto ingenuo: características independientes
        print(f"    {clase:>5}: " + " · ".join(f"{f:.3f}" for f in factores) + f" = {resultados[clase]:.5f}")
    total = sum(resultados.values())
    print("    Al normalizar:", {c: round(v / total, 3) for c, v in resultados.items()})
    return max(resultados, key=resultados.get)

# 1) Tabla de ejemplo del curso: diez autos y si fueron robados (datos ilustrativos)
autos = pd.DataFrame({
    "Color":  ["Rojo", "Rojo", "Rojo", "Amarillo", "Amarillo", "Amarillo", "Amarillo", "Amarillo", "Rojo", "Rojo"],
    "Tipo":   ["Deportivo", "Deportivo", "Deportivo", "Deportivo", "SUV", "SUV", "SUV", "SUV", "SUV", "Deportivo"],
    "Origen": ["Nacional", "Nacional", "Nacional", "Importado", "Importado", "Importado", "Importado", "Nacional", "Importado", "Importado"],
    "Robado": ["Sí", "No", "Sí", "No", "Sí", "No", "Sí", "No", "No", "Sí"],
})
for caso in [{"Color": "Rojo", "Tipo": "Deportivo", "Origen": "Importado"},
             {"Color": "Amarillo", "Tipo": "SUV", "Origen": "Nacional"}]:
    print(f"\n{caso}")
    print("  -> ¿Robado?", naive_bayes(autos, "Robado", caso))

print("\nFrecuencia cero: ningún auto robado es verde, así que P(Verde | Sí) = 0/5 = 0 y anula todo el producto")
print("Con suavizado de Laplace (+1):", round((0 + 1) / (5 + 1 * 3), 3), "(con 3 colores posibles)")

# 2) Datos reales: 7,043 clientes de una compañía telefónica (IBM). ¿Abandonará la compañía (Churn) un cliente
#    según su tipo de contrato, su servicio de internet y su forma de pago?
ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "telco_churn_ibm.csv"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
telco = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL)
for cliente in [{"Contract": "Month-to-month", "InternetService": "Fiber optic", "PaymentMethod": "Electronic check"},
                {"Contract": "Two year", "InternetService": "DSL", "PaymentMethod": "Credit card (automatic)"}]:
    print(f"\n{cliente}")
    print("  -> ¿Abandona?", naive_bayes(telco, "Churn", cliente))
    iguales = telco[(telco[list(cliente)] == pd.Series(cliente)).all(axis=1)]
    print(f"  En los datos: de los {len(iguales):,} clientes con esa combinación, abandonó el "
          f"{(iguales['Churn'] == 'Yes').mean():.1%}")
