# Módulo 1 · De la regresión logística a la tarjeta de puntos (scorecard)
# Factor = PDO / ln(2)        Offset = Score - Factor * ln(odds)
# Score total = Offset + Factor * ln(momios bueno:malo)
# Repartido entre las n variables (logística que modela P(malo) con variables WoE):
#   Puntos = (-WoE * beta - alpha / n) * Factor + Offset / n
import numpy as np

PDO = 20            # Points to Double the Odds: cada 20 puntos los momios bueno:malo se duplican
SCORE_REF = 600     # score de referencia...
ODDS_REF = 50       # ...que corresponde a momios 50:1 (50 buenos por cada malo)

factor = PDO / np.log(2)
offset = SCORE_REF - factor * np.log(ODDS_REF)
print(f"Factor = {PDO}/ln(2) = {factor:.4f}")
print(f"Offset = {SCORE_REF} - {factor:.4f}*ln({ODDS_REF}) = {offset:.4f}\n")

# Supongamos que la regresión logística (evento = malo, entrenada con variables WoE) dio:
alpha = -1.39                                   # intercepto ≈ ln(20% malos / 80% buenos)
betas = {"edad": -0.95, "ingreso": -0.80}       # negativos: más WoE (más "bueno") -> menos riesgo
n = len(betas)
woe = {"edad":    {"<25": -0.60, "25-30": -0.10, ">=30": 0.45},
       "ingreso": {"<10k": -0.70, "10k-25k": 0.05, ">=25k": 0.65}}

def puntos(var, atributo):
    return (-woe[var][atributo] * betas[var] - alpha / n) * factor + offset / n

print("SCORECARD")
for var, mapa in woe.items():
    for atributo, w in mapa.items():
        print(f"  {var:<8} {atributo:<8} WoE={w:>5.2f}  ->  {round(puntos(var, atributo)):>4} puntos")

# Calificar a un cliente: 28 años, ingreso 26,500
cliente = {"edad": "25-30", "ingreso": ">=25k"}
score = sum(puntos(v, a) for v, a in cliente.items())
print(f"\nCliente {cliente} -> score = {score:.1f}")

# Comprobación: el score es Offset + Factor * ln(momios bueno:malo del cliente)
logit_malo = alpha + sum(betas[v] * woe[v][a] for v, a in cliente.items())
p_malo = 1 / (1 + np.exp(-logit_malo))
print(f"P(malo) = {p_malo:.4f} | momios bueno:malo = {np.exp(-logit_malo):.2f}"
      f" | Offset + Factor*ln(momios) = {offset + factor * (-logit_malo):.1f}")
print(f"+{PDO} puntos (PDO) => momios x2: {np.exp((score + PDO - offset) / factor):.2f}")
