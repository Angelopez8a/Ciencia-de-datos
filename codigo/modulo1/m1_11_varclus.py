# Módulo 1 · Multicolinealidad con VarClusHi: agrupar variables y quedarse con 1 por cluster
# RS_Ratio = (1 - RS_Own) / (1 - RS_NC)   -> entre MÁS BAJO, mejor representa a su cluster
import numpy as np
import pandas as pd
from varclushi import VarClusHi

rng = np.random.default_rng(0)
n = 500
base_a, base_b = rng.normal(size=n), rng.normal(size=n)
X = pd.DataFrame({
    "v_sum":  base_a * 10 + rng.normal(0, 1, n),     # familia "volumen"
    "v_mean": base_a * 0.5 + rng.normal(0, 0.05, n),
    "v_max":  base_a * 12 + rng.normal(0, 3, n),
    "v_inc":  base_b + rng.normal(0, 0.2, n),        # familia "tendencia"
    "v_racha": base_b * 2 + rng.normal(0, 1, n),
})

vc = VarClusHi(df=X, feat_list=X.columns.tolist())
vc.varclus()
rs = vc.rsquare.sort_values(by=["Cluster", "RS_Ratio"]).reset_index(drop=True)
rs["id"] = rs.groupby("Cluster").cumcount() + 1
print(rs.round(4).to_string(index=False))

seleccion = rs.loc[rs["id"] == 1, "Variable"].tolist()
print("\nVariables seleccionadas (id == 1 de cada cluster):", seleccion)

# Comprobación manual con la primera fila de la tabla del notebook de la sesión 8
rs_own, rs_nc = 0.978437, 0.082052
print(f"\nNotebook: v_max_afluencia -> (1-{rs_own})/(1-{rs_nc}) = {(1 - rs_own) / (1 - rs_nc):.6f}")
