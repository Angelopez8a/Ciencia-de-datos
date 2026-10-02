# Módulo 1 · fit vs transform: "aprender" solo del entrenamiento y aplicarlo a la validación
# Mismo principio que la sesión 9: "se toma el 70 % para entrenar (o generar WoE) y el 30 % para validar".
# Datos reales: IBM Telco Customer Churn (7,043 clientes). TotalCharges tiene 11 valores vacíos.
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "telco_churn_ibm.csv"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL)
X = df[["MonthlyCharges", "tenure"]].assign(TotalCharges=pd.to_numeric(df["TotalCharges"], errors="coerce"))
y = (df["Churn"] == "Yes").astype(int)

Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("Entrenamiento:", Xt.shape, "| Validación:", Xv.shape)
print(f"Tasa de fuga -> train {yt.mean():.4f} | valid {yv.mean():.4f}  (stratify la conserva)")
print("Nulos de TotalCharges -> train", int(Xt["TotalCharges"].isna().sum()), "| valid", int(Xv["TotalCharges"].isna().sum()), "\n")

# 1) fit SOLO con entrenamiento: aquí se "aprenden" la mediana, la media y la desviación
im = SimpleImputer(strategy="median").fit(Xt)
ss = StandardScaler().fit(im.transform(Xt))
print("Mediana aprendida (train):", im.statistics_.round(2))
print("Media aprendida   (train):", ss.mean_.round(2))
print("Desv. aprendida   (train):", ss.scale_.round(2))

# 2) transform en ambos conjuntos con lo aprendido en train (la validación nunca se usa para calcular nada)
Zt = ss.transform(im.transform(Xt))
Zv = ss.transform(im.transform(Xv))
print("\nMedia de Z en train:", Zt.mean(axis=0).round(3), "<- exactamente 0")
print("Media de Z en valid:", Zv.mean(axis=0).round(3), "<- cercana a 0, pero no exacta (así debe ser)")

# 3) Lo mismo en una sola pieza: Pipeline = imputar -> estandarizar -> modelo
pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression())
pipe.fit(Xt, yt)
print(f"\nPipeline -> AUC train {roc_auc_score(yt, pipe.predict_proba(Xt)[:, 1]):.3f} | "
      f"AUC valid {roc_auc_score(yv, pipe.predict_proba(Xv)[:, 1]):.3f}")
print("Coeficientes (escala estandarizada):", pipe[-1].coef_.round(3), "| intercepto:", pipe[-1].intercept_.round(3))
