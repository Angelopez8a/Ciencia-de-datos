# Módulo 4 · Optimizadores implementados desde cero sobre f(θ) = θ²  (mínimo en θ = 0)
# Gradiente: f'(θ) = 2θ.  Todos empiezan en θ0 = 5.
import numpy as np

grad = lambda t: 2 * t
theta0 = 5.0

def sgd(pasos, lr=0.1):
    t = theta0
    for _ in range(pasos):
        t -= lr * grad(t)                 # θ ← θ − α·∇f
    return t

def momentum(pasos, lr=0.1, beta=0.9):
    t, v = theta0, 0.0
    for _ in range(pasos):
        v = beta * v + grad(t)            # "inercia": recuerda la dirección anterior
        t -= lr * v
    return t

def adagrad(pasos, lr=0.5, eps=1e-8):
    t, G = theta0, 0.0
    for _ in range(pasos):
        g = grad(t)
        G += g ** 2                       # acumula TODOS los gradientes al cuadrado
        t -= lr * g / (np.sqrt(G) + eps)  # el paso efectivo solo puede encogerse
    return t

def rmsprop(pasos, lr=0.1, rho=0.9, eps=1e-8):
    t, Eg2 = theta0, 0.0
    for _ in range(pasos):
        g = grad(t)
        Eg2 = rho * Eg2 + (1 - rho) * g ** 2   # media móvil EXPONENCIAL (olvida lo viejo)
        t -= lr * g / (np.sqrt(Eg2) + eps)
    return t

def adadelta(pasos, rho=0.95, eps=1e-6):
    t, Eg2, Edx2 = theta0, 0.0, 0.0
    for _ in range(pasos):
        g = grad(t)
        Eg2 = rho * Eg2 + (1 - rho) * g ** 2
        dx = -np.sqrt(Edx2 + eps) / np.sqrt(Eg2 + eps) * g   # ¡sin learning rate!
        Edx2 = rho * Edx2 + (1 - rho) * dx ** 2
        t += dx
    return t

def adam(pasos, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
    t, m, v = theta0, 0.0, 0.0
    for k in range(1, pasos + 1):
        g = grad(t)
        m = b1 * m + (1 - b1) * g         # 1er momento: promedio de gradientes (≈ momentum)
        v = b2 * v + (1 - b2) * g ** 2    # 2º momento: promedio de gradientes² (≈ RMSProp)
        m_hat = m / (1 - b1 ** k)         # corrección de sesgo (al inicio m y v valen ~0)
        v_hat = v / (1 - b2 ** k)
        t -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return t

optimizadores = [("SGD (lr=0.1)", sgd), ("Momentum (lr=0.1, β=0.9)", momentum),
                 ("Adagrad (lr=0.5)", adagrad), ("RMSProp (lr=0.1)", rmsprop),
                 ("Adadelta (sin lr)", adadelta), ("Adam (lr=0.1)", adam)]

print(f"θ después de N pasos (inicio θ0 = {theta0}, mínimo θ* = 0)")
print(f"{'Optimizador':<26}{'N=10':>11}{'N=50':>11}{'N=200':>11}")
for nombre, f in optimizadores:
    print(f"{nombre:<26}" + "".join(f"{f(n):>11.4f}" for n in (10, 50, 200)))

print("\nEfecto del learning rate en SGD (50 pasos):")
for lr in [0.01, 0.1, 0.5, 0.9, 1.1]:
    print(f"  lr = {lr:<4} -> θ final = {sgd(50, lr): .4e}")
