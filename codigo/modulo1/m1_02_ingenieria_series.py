# Módulo 1 · Ingeniería de variables sobre una serie (funciones de la sesión 8, "ejercicio del Excel")
import numpy as np

def sum_inc(l):  return sum(int(y > x) for x, y in zip(l, l[1:]))          # nº de subidas
def sum_dec(l):  return sum(int(y < x) for x, y in zip(l, l[1:]))          # nº de bajadas
def media_inc(l): return np.mean([int(y > x) for x, y in zip(l, l[1:])])   # % de subidas
def media_dec(l): return np.mean([int(y < x) for x, y in zip(l, l[1:])])
def deltas(l):    return [float(y - x) for x, y in zip(l, l[1:])]           # cambios absolutos
def pct_deltas(l): return [float((y - x) / x) for x, y in zip(l, l[1:])]    # cambios relativos
def rachas(l, f):  return [len(s) for s in "".join(str(int(f(x, y))) for x, y in zip(l, l[1:])).split("0")]

lst = [1065, 1068, 1266, 1492, 1051, 1009, 1340, 1485]

print("sum_inc        :", sum_inc(lst))
print("sum_dec        :", sum_dec(lst))
print("media_inc      :", media_inc(lst))
print("media_dec      :", media_dec(lst))
print("delta_min      :", min(deltas(lst)))
print("delta_max      :", max(deltas(lst)))
print("delta_mean     :", np.mean(deltas(lst)))
print("delta_desv     :", np.std(deltas(lst)))
print("pct_delta_min  :", min(pct_deltas(lst)))
print("pct_delta_max  :", max(pct_deltas(lst)))
print("pct_delta_mean :", np.mean(pct_deltas(lst)))
print("pct_delta_desv :", np.std(pct_deltas(lst)))
print("max_racha_inc  :", max(rachas(lst, lambda x, y: y > x)))
print("max_racha_dec  :", max(rachas(lst, lambda x, y: y < x)))
print("media_racha_inc:", np.mean(rachas(lst, lambda x, y: y > x)))
print("media_racha_dec:", np.mean(rachas(lst, lambda x, y: y < x)))
