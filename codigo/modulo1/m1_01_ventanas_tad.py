# Módulo 1 · Ventanas de tiempo y construcción de la TAD (Tabla Analítica de Datos)
# Idea de la sesión 8 (Ecobici): con lo observado en [ancla-vobs+1, ancla] predecir t = ancla + vdes
import pandas as pd

# Afluencia de 2 estaciones durante 8 periodos (datos de juguete)
df = pd.DataFrame({
    "id_estacion": ["E1"] * 8 + ["E2"] * 8,
    "t": list(range(1, 9)) * 2,
    "afluencia": [10, 12, 15, 11, 18, 20, 19, 25,
                  50, 48, 47, 52, 40, 38, 41, 37],
})

um = ["id_estacion"]          # unidad muestral
vobs, vdes = 3, 1             # ventana de observación (3 periodos) y de desempeño (1 periodo)
t_min, t_max = df["t"].min(), df["t"].max()
anclai, anclaf = t_min + vobs - 1, t_max - vdes
print(f"Anclas posibles: de {anclai} a {anclaf}")

def sum_inc(l):               # ¿cuántas veces subió la serie?
    return sum(int(y > x) for x, y in zip(l, l[1:]))

lst_X, lst_y = [], []
for ancla in range(anclai, anclaf + 1):
    # ---- X: SOLO información del pasado (ventana de observación) ----
    obs = df[(df["t"] > ancla - vobs) & (df["t"] <= ancla)]
    aux = obs.pivot_table(index=um, values="afluencia",
                          aggfunc=["sum", "min", "max", "mean", sum_inc])
    aux.columns = [f"v_{a}_{b}" for a, b in aux.columns]
    aux.insert(0, "ancla", ancla)
    lst_X.append(aux.reset_index())
    # ---- y: el valor en el FUTURO (ventana de desempeño) ----
    fut = df[df["t"] == ancla + vdes][um + ["afluencia"]].rename(columns={"afluencia": "y"})
    fut.insert(1, "ancla", ancla)
    lst_y.append(fut)

X = pd.concat(lst_X, ignore_index=True)
y = pd.concat(lst_y, ignore_index=True)
tad = X.merge(y, on=um + ["ancla"], how="inner")
tad["y2"] = (tad["y"] > 30).astype(int)       # versión clasificación: ¿alta afluencia?
print(tad.to_string(index=False))
