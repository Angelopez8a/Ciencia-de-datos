# Módulo 2 · Árboles de decisión: Gini, entropía y elección del mejor corte
# Tabla de ejemplo "Diabetic" del curso: diez pacientes (datos ilustrativos, no provienen de un estudio)
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.DataFrame({
    "Calorias": ["High", "Low", "Moderate", "High", "High", "Low", "Moderate", "High", "High", "Low"],
    "Azucar":   ["High", "Moderate", "High", "Moderate", "High", "Low", "Moderate", "Moderate", "High", "Low"],
    "Actividad": ["Moderate", "High", "Low", "High", "Low", "High", "Moderate", "Low", "High", "Moderate"],
    "Diabetico": ["Yes", "No", "Yes", "No", "Yes", "No", "No", "Yes", "Yes", "No"],
})

def gini(s):
    p = s.value_counts(normalize=True)
    return float(np.sum(p * (1 - p)))          # Σ p_k (1 - p_k)

def entropia(s):
    p = s.value_counts(normalize=True)
    return float(-np.sum(p * np.log2(p)))

print(f"Nodo raíz: Gini = {gini(df['Diabetico']):.3f}, Entropía = {entropia(df['Diabetico']):.3f}\n")
for var in ["Calorias", "Azucar", "Actividad"]:
    g_pond = 0
    print(f"Corte por {var}:")
    for valor, grupo in df.groupby(var):
        g = gini(grupo["Diabetico"])
        g_pond += len(grupo) / len(df) * g
        print(f"   {valor:<9} n={len(grupo)}  { {k: int(v) for k, v in grupo['Diabetico'].value_counts().items()} }  Gini={g:.3f}")
    print(f"   Gini ponderado = {g_pond:.3f}  |  Ganancia = {gini(df['Diabetico']) - g_pond:.3f}\n")

X = pd.get_dummies(df.drop(columns="Diabetico"), dtype=int)
arbol = DecisionTreeClassifier(criterion="gini", max_depth=2, random_state=0)
arbol.fit(X, df["Diabetico"])
print(export_text(arbol, feature_names=list(X.columns)))
