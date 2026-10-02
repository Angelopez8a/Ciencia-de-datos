# Módulo 1 · Multicolinealidad con VarClusHi: agrupar variables y quedarse con una por cluster
# RS_Ratio = (1 - RS_Own) / (1 - RS_NC)   -> entre MÁS BAJO, mejor representa a su cluster
# Datos reales: las 10 medidas "promedio" de 569 tumores de mama (Breast Cancer Wisconsin, en scikit-learn),
# calculadas a partir de imágenes digitalizadas de biopsias. Radio, perímetro y área miden casi lo mismo.
from sklearn.datasets import load_breast_cancer
from varclushi import VarClusHi

X = load_breast_cancer(as_frame=True).data.iloc[:, :10]
X.columns = [c.replace("mean ", "").replace(" ", "_") for c in X.columns]
print("Variables:", X.columns.tolist())
print("Correlaciones radio-perímetro-área:", X[["radius", "perimeter", "area"]].corr().round(3).values[0, 1:].tolist(), "\n")

vc = VarClusHi(df=X, feat_list=X.columns.tolist())
vc.varclus()
rs = vc.rsquare.sort_values(by=["Cluster", "RS_Ratio"]).reset_index(drop=True)
rs["id"] = rs.groupby("Cluster").cumcount() + 1
print(rs.round(4).to_string(index=False))

seleccion = rs.loc[rs["id"] == 1, "Variable"].tolist()
print("\nVariables seleccionadas (id == 1 de cada cluster):", seleccion)

# Comprobación manual con la primera fila de la tabla del notebook de la sesión 8 (Ecobici)
rs_own, rs_nc = 0.978437, 0.082052
print(f"\nNotebook: v_max_afluencia -> (1-{rs_own})/(1-{rs_nc}) = {(1 - rs_own) / (1 - rs_nc):.6f}")
