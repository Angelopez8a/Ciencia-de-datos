# Módulo 1 · Ventanas de tiempo y construcción de la TAD (Tabla Analítica de Datos)
# Es la misma lógica del notebook de Ecobici (sesión 8), aplicada a datos reales de Gapminder:
# la esperanza de vida de 142 países, medida cada cinco años entre 1952 y 2007.
from pathlib import Path
import pandas as pd

ARCHIVO = Path(__file__).resolve().parents[1] / "datos" / "gapminder.csv"
URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/gapminder/gapminder.csv"
gm = pd.read_csv(ARCHIVO if ARCHIVO.exists() else URL)

# Formato largo: una fila por país y periodo; t = 1 (1952), 2 (1957), ..., 12 (2007)
df = gm[["country", "year", "lifeExp"]].copy()
df["t"] = (df["year"] - 1952) // 5 + 1

um = ["country"]              # unidad muestral: el país
vobs, vdes = 3, 1             # se observan 3 periodos (15 años) para predecir el periodo siguiente (5 años)
t_min, t_max = df["t"].min(), df["t"].max()
anclai, anclaf = t_min + vobs - 1, t_max - vdes
print(f"Periodos de {t_min} a {t_max}; anclas posibles: de {anclai} a {anclaf}")

def sum_inc(l):               # número de veces que la serie sube
    return sum(int(y > x) for x, y in zip(l, l[1:]))

lst_X, lst_y = [], []
for ancla in range(anclai, anclaf + 1):
    # ---- X: solo información del pasado (ventana de observación) ----
    obs = df[(df["t"] > ancla - vobs) & (df["t"] <= ancla)].sort_values("t")
    aux = obs.pivot_table(index=um, values="lifeExp", aggfunc=["min", "max", "mean", "last", sum_inc])
    aux.columns = ["v_min", "v_max", "v_mean", "v_ultimo", "v_sum_inc"]
    aux.insert(0, "ancla", ancla)
    lst_X.append(aux.reset_index())
    # ---- y: el valor en el futuro (ventana de desempeño) ----
    fut = df[df["t"] == ancla + vdes][um + ["lifeExp"]].rename(columns={"lifeExp": "y"})
    fut.insert(1, "ancla", ancla)
    lst_y.append(fut)

X = pd.concat(lst_X, ignore_index=True)
y = pd.concat(lst_y, ignore_index=True)
tad = X.merge(y, on=um + ["ancla"], how="inner")
tad["y2"] = (tad["y"] < tad["v_ultimo"]).astype(int)   # versión de clasificación: ¿la esperanza de vida baja?
tad.insert(2, "anio", 1952 + 5 * (tad["ancla"] - 1))    # año que corresponde a cada ancla

print(f"TAD: {tad.shape[0]:,} renglones ({tad['country'].nunique()} países x {tad['ancla'].nunique()} anclas)")
print(f"Renglones con y2 = 1 (la esperanza de vida bajó): {tad['y2'].sum()} ({tad['y2'].mean():.1%})\n")
cols = ["country", "ancla", "anio", "v_min", "v_max", "v_mean", "v_ultimo", "v_sum_inc", "y", "y2"]
ejemplo = tad.loc[tad["country"].isin(["Mexico", "Botswana"]), cols].sort_values(["country", "ancla"])
print(ejemplo.round(2).to_string(index=False))
