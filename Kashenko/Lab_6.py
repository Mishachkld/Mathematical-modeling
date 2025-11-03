import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
L = 10
T = 1.0
Nx = 100
dx = L / Nx

# Для устойчивости dt <= 0.5 * dx^2 / Dmax
Dmax = 25
dt = 0.5 * dx**2 / Dmax * 0.9
Nt = int(T / dt)

print(f"dx = {dx:.4f}, dt = {dt:.6f}, Nt = {Nt}")

x = np.linspace(0, L, Nx + 1)

# Коэффициенты
D = 25 - 2 * x
f = x

# Начальное условие
U0 = x**2 * (10 - x)
U0[0] = 0
U0[-1] = 0

# ---------- ЯВНАЯ СХЕМА ----------
U_explicit = U0.copy()
Ue = [U_explicit.copy()]

for n in range(Nt):
    U_new = U_explicit.copy()
    for i in range(1, Nx):
        U_new[i] = U_explicit[i] + dt * (
            D[i] * (U_explicit[i+1] - 2*U_explicit[i] + U_explicit[i-1]) / dx**2
            - f[i] * U_explicit[i]
            + 5
        )
    U_new[0] = 0
    U_new[-1] = 0
    U_explicit = U_new
    if n % (Nt // 5) == 0:
        Ue.append(U_explicit.copy())

# ---------- НЕЯВНАЯ СХЕМА (МЕТОД ПРОГОНКИ) ----------
U_implicit = U0.copy()
Ui = [U_implicit.copy()]

for n in range(Nt):
    A = np.zeros(Nx + 1)
    B = np.zeros(Nx + 1)
    C = np.zeros(Nx + 1)
    F = np.zeros(Nx + 1)

    for i in range(1, Nx):
        A[i] = -dt * D[i] / dx**2
        B[i] = 1 + 2 * dt * D[i] / dx**2 + dt * f[i]
        C[i] = -dt * D[i] / dx**2
        F[i] = U_implicit[i] + dt * 5

    # Граничные условия
    B[0], B[-1] = 1, 1
    F[0], F[-1] = 0, 0

    # Метод прогонки
    alpha = np.zeros(Nx + 1)
    beta = np.zeros(Nx + 1)

    for i in range(1, Nx + 1):
        denom = B[i-1] + A[i-1]*alpha[i-1]
        alpha[i] = -C[i-1] / denom
        beta[i] = (F[i-1] - A[i-1]*beta[i-1]) / denom

    U_new = np.zeros(Nx + 1)
    for i in reversed(range(0, Nx)):
        U_new[i] = alpha[i+1]*U_new[i+1] + beta[i+1]

    U_implicit = U_new
    if n % (Nt // 5) == 0:
        Ui.append(U_implicit.copy())

# ---------- ВИЗУАЛИЗАЦИЯ ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

for u in Ue:
    axs[0].plot(x, u)
axs[0].set_title("Явная схема (временные слои)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("U(x,t)")

for u in Ui:
    axs[1].plot(x, u)
axs[1].set_title("Неявная схема (временные слои)")
axs[1].set_xlabel("x")
axs[1].set_ylabel("U(x,t)")

plt.tight_layout()
plt.show()
