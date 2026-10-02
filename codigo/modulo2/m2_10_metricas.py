# Módulo 2 · Métricas para elegir el modelo campeón y monitorearlo (KS, Gini, Lift, PSI)
import numpy as np
import pandas as pd
from sklearn.metrics import (max_error, mean_absolute_error, mean_squared_error, r2_score,
                             median_absolute_error, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, balanced_accuracy_score,
                             roc_auc_score)
from scipy.stats import ks_2samp

# ---------------- Regresión ----------------
y_real = np.array([3.0, 5.0, 2.5, 7.0, 4.0])
y_pred = np.array([2.5, 5.0, 3.0, 8.0, 3.5])
print("REGRESIÓN")
print(f"  Max Error = {max_error(y_real, y_pred):.3f}")
print(f"  MAE       = {mean_absolute_error(y_real, y_pred):.3f}")
print(f"  MSE       = {mean_squared_error(y_real, y_pred):.3f}")
print(f"  RMSE      = {np.sqrt(mean_squared_error(y_real, y_pred)):.3f}")
print(f"  MedAE     = {median_absolute_error(y_real, y_pred):.3f}")
print(f"  R²        = {r2_score(y_real, y_pred):.3f}")

# ---------------- Clasificación ----------------
y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
p = np.array([0.9, 0.8, 0.6, 0.3, 0.7, 0.4, 0.3, 0.2, 0.1, 0.05])
yhat = (p >= 0.5).astype(int)
print("\nCLASIFICACIÓN (corte 0.5)")
print("  Matriz [[TN, FP], [FN, TP]] =", confusion_matrix(y, yhat).tolist())
print(f"  Accuracy={accuracy_score(y, yhat):.3f}  Precision={precision_score(y, yhat):.3f}  "
      f"Recall={recall_score(y, yhat):.3f}  F1={f1_score(y, yhat):.3f}  "
      f"BalancedAcc={balanced_accuracy_score(y, yhat):.3f}")
auc = roc_auc_score(y, p)
print(f"  AUC (C-statistic) = {auc:.3f}  ->  Gini = 2·AUC − 1 = {2 * auc - 1:.3f}")
print(f"  KS = máx. separación entre distribuciones de eventos y no eventos = "
      f"{ks_2samp(p[y == 1], p[y == 0]).statistic:.3f}")

# ---------------- Lift ----------------
rng = np.random.default_rng(0)
n = 10000
score = rng.random(n)
evento = (rng.random(n) < 0.05 + 0.4 * score ** 3).astype(int)
d = pd.DataFrame({"score": score, "evento": evento}).sort_values("score", ascending=False)
d["decil"] = np.repeat(np.arange(1, 11), n // 10)
lift = d.groupby("decil")["evento"].mean() / d["evento"].mean()
print("\nLIFT por decil (1 = 10% con mayor score):", lift.round(2).tolist())

# ---------------- PSI / Índice de desviación ----------------
esperado = np.array([0.10, 0.20, 0.30, 0.25, 0.15])   # % por bin cuando se construyó el modelo
actual = np.array([0.08, 0.18, 0.28, 0.28, 0.18])     # % por bin en el mes actual
psi = np.sum((actual - esperado) * np.log(actual / esperado))
estado = "estable" if psi < 0.1 else "cambio menor" if psi <= 0.25 else "cambio MAYOR: recalibrar"
print(f"\nPSI = {psi:.4f} -> {estado}")
