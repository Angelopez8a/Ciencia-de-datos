# Módulo 1 · Interpretar la regresión logística de la sesión 8 (Ecobici, ¿alta afluencia?)
# Coeficientes e intercepto reales que imprimió el notebook (variables estandarizadas, z-score)
import numpy as np

variables = ["ss_v_max_afluencia", "ss_v_pct_delta_desv_afluencia",
             "ss_v_sum_inc_afluencia", "ss_v_sum_dec_afluencia"]
beta = np.array([3.24427681, -0.1219525, 0.47388188, 0.48031202])
alpha = -3.22891895

def p_alta(z):
    return 1 / (1 + np.exp(-(alpha + z @ beta)))     # sigmoide del logit

# 1) Razón de momios: subir 1 desviación estándar en una variable multiplica los momios por e^beta
print("Variable                         beta    e^beta")
for v, b in zip(variables, beta):
    print(f"{v:<31} {b:>6.3f}  {np.exp(b):>7.3f}")

# 2) Una estación "promedio" (todas las z = 0): solo cuenta el intercepto
z0 = np.zeros(4)
print(f"\nEstación promedio      -> logit = {alpha:.3f} -> P(alta) = {p_alta(z0):.4f}")

# 3) Misma estación pero con un máximo reciente 1 desviación arriba del promedio
z1 = np.array([1.0, 0, 0, 0])
print(f"v_max a +1 desviación  -> logit = {alpha + beta[0]:.3f} -> P(alta) = {p_alta(z1):.4f}")
print(f"Momios: {np.exp(alpha):.4f} -> {np.exp(alpha + beta[0]):.4f}  (x{np.exp(beta[0]):.2f})")

# 4) La primera estación de la TAD final del notebook (Est00000, ancla 20)
z = np.array([1.012736, 1.409126, 0.113030, 0.327499])
print(f"\nEst00000, ancla 20     -> P(alta) = {p_alta(z):.4f} -> clase {int(p_alta(z) >= 0.5)} (real y2 = 1, y = 308,424)")

# 5) El corte 0.5 es una decisión: ¿qué valor de v_max (en z) hace P = 0.5 con las demás en 0?
print(f"Con las demás z = 0, P(alta) = 0.5 cuando z_max = -alpha/beta = {-alpha / beta[0]:.3f} desviaciones")
