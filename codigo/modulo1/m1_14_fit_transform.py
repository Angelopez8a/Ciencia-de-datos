# Módulo 1 · fit vs transform: "aprender" solo del entrenamiento y aplicarlo a validación
# Mismo principio que la sesión 9: "se toma el 70% para entrenar (o generar WoE) y el 30% para validar"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(7)
n = 1000
X = pd.DataFrame({"cargo": rng.normal(65, 30, n), "antiguedad": rng.integers(0, 72, n).astype(float)})
logit = -1 + 0.03 * (X["cargo"] - 65) - 0.05 * (X["antiguedad"] - 36)
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
X.loc[rng.choice(n, 100, replace=False), "cargo"] = np.nan          # 10 % de nulos

Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("Entrenamiento:", Xt.shape, "| Validación:", Xv.shape)
print(f"Tasa de evento -> train {yt.mean():.3f} | valid {yv.mean():.3f}  (stratify la conserva)\n")

# 1) fit SOLO con entrenamiento: aquí se "aprende" la mediana, la media y la desviación
im = SimpleImputer(strategy="median").fit(Xt)
ss = StandardScaler().fit(im.transform(Xt))
print("Mediana aprendida (train):", im.statistics_.round(3))
print("Media aprendida   (train):", ss.mean_.round(3))
print("Desv. aprendida   (train):", ss.scale_.round(3))

# 2) transform en ambos con lo aprendido en train (validación NUNCA se usa para calcular nada)
Zt = ss.transform(im.transform(Xt))
Zv = ss.transform(im.transform(Xv))
print("\nMedia de Z en train:", Zt.mean(axis=0).round(3), "<- exactamente 0")
print("Media de Z en valid:", Zv.mean(axis=0).round(3), "<- cerca de 0, no exacta (y así debe ser)")

# 3) Lo mismo en una sola pieza: Pipeline = imputar -> estandarizar -> modelo
pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression())
pipe.fit(Xt, yt)
print(f"\nPipeline -> AUC train {roc_auc_score(yt, pipe.predict_proba(Xt)[:, 1]):.3f} | "
      f"AUC valid {roc_auc_score(yv, pipe.predict_proba(Xv)[:, 1]):.3f}")
print("Coeficientes (escala estandarizada):", pipe[-1].coef_.round(3), "| intercepto:", pipe[-1].intercept_.round(3))
