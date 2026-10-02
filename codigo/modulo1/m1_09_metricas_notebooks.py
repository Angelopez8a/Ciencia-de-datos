# Módulo 1 · Interpretar las matrices de confusión que salieron en los notebooks del curso
# Formato de sklearn: [[TN, FP],
#                      [FN, TP]]   (filas = real, columnas = predicción)
import numpy as np

matrices = {
    "Sesión 8 · Ecobici (alta afluencia)": np.array([[58359, 2093], [4592, 8521]]),
    "Sesión 9 · Churn (fuga)":             np.array([[1186, 128], [223, 269]]),
}

for nombre, cm in matrices.items():
    (tn, fp), (fn, tp) = cm
    acc = (tp + tn) / cm.sum()
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)                 # sensibilidad / TPR
    esp = tn / (tn + fp)                 # especificidad / TNR
    f1 = 2 * prec * rec / (prec + rec)
    print(nombre)
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  "
          f"Especificidad={esp:.3f}  F1={f1:.3f}  BalancedAcc={(rec + esp) / 2:.3f}\n")
