import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# -----------------------------
# Модель Лотки–Вольтерры (конкуренция)
# -----------------------------
def competition_model(N, t, r1, r2, K1, K2, a12, a21):
    N1, N2 = N
    dN1 = r1 * N1 * (K1 - N1 - a12 * N2) / K1
    dN2 = r2 * N2 * (K2 - N2 - a21 * N1) / K2
    return [dN1, dN2]


# -------------------------------------------------------
# Построение фазового портрета и изоклин
# -------------------------------------------------------
def plot_phase_portrait(ax, r1, r2, K1, K2, a12, a21):

    # Сетка
    N1 = np.linspace(0, K1 * 1.1, 20)
    N2 = np.linspace(0, K2 * 1.1, 20)
    N1_grid, N2_grid = np.meshgrid(N1, N2)

    # Векторное поле
    dN1 = np.zeros_like(N1_grid)
    dN2 = np.zeros_like(N2_grid)

    for i in range(len(N1)):
        for j in range(len(N2)):
            d = competition_model([N1_grid[i, j], N2_grid[i, j]], 0, r1, r2, K1, K2, a12, a21)
            dN1[i, j], dN2[i, j] = d

    # Нормировка стрелок
    L = np.sqrt(dN1 ** 2 + dN2 ** 2)

    # Избегаем деления на ноль
    dN1_norm = np.zeros_like(dN1)
    dN2_norm = np.zeros_like(dN2)

    mask = L > 1e-12  # только там, где длина вектора не нулевая

    dN1_norm[mask] = dN1[mask] / L[mask]
    dN2_norm[mask] = dN2[mask] / L[mask]

    ax.quiver(N1_grid, N2_grid, dN1_norm, dN2_norm, alpha=0.5)

    # Изоклины
    N1_iso = np.linspace(0, K1, 400)
    N2_iso1 = (K1 - N1_iso) / a12           # dN1/dt = 0
    N2_iso2 = (K2 - a21 * N1_iso)           # dN2/dt = 0

    ax.plot(N1_iso, N2_iso1, label='dN1/dt = 0')
    ax.plot(N1_iso, N2_iso2, label='dN2/dt = 0')

    # Особые точки
    eq_points = []

    eq_points.append((0, 0))
    eq_points.append((K1, 0))
    eq_points.append((0, K2))

    denom = 1 - a12 * a21
    if abs(denom) > 1e-9:
        N1_star = (K1 - a12 * K2) / denom
        N2_star = (K2 - a21 * K1) / denom
        if N1_star > 0 and N2_star > 0:
            eq_points.append((N1_star, N2_star))

    for x, y in eq_points:
        ax.plot(x, y, 'ro')

    # Траектории
    t = np.linspace(0, 50, 2000)
    initial_conditions = [
        [10, 10],
        [150, 10],
        [10, 150],
        [100, 20],
        [30, 120],
    ]

    for N0 in initial_conditions:
        sol = odeint(competition_model, N0, t, args=(r1, r2, K1, K2, a12, a21))
        ax.plot(sol[:, 0], sol[:, 1], linewidth=1)

    ax.set_xlabel('Вид 1')
    ax.set_ylabel('Вид 2')
    ax.set_title('Фазовый портрет')
    ax.set_xlim(0, K1 * 1.1)
    ax.set_ylim(0, K2 * 1.1)
    ax.legend()
    ax.grid(alpha=0.3)


# -------------------------------------------------------
# Построение временных рядов
# -------------------------------------------------------
def plot_time_series(ax, r1, r2, K1, K2, a12, a21):
    t = np.linspace(0, 50, 2000)

    initial_conditions = [
        [10, 20],
    ]

    for N0 in initial_conditions:
        sol = odeint(competition_model, N0, t, args=(r1, r2, K1, K2, a12, a21))
        ax.plot(t, sol[:, 0], label=f'Популяция вида 1 (N_1)')
        ax.plot(t, sol[:, 1], '--', label=f'Популяция вида (N_2)')

    ax.set_xlabel('Время')
    ax.set_ylabel('Численность')
    ax.set_title('Временные ряды')
    ax.legend()
    ax.grid(alpha=0.3)


# ------------------------
# ПАРАМЕТРЫ ВАШЕГО ВАРИАНТА
# ------------------------
r1 = 2
r2 = 2
K1 = 200
K2 = 200
a12 = 0.5
a21 = 0.5

# ------------------------
# Построение графиков
# ------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

plot_phase_portrait(axes[0], r1, r2, K1, K2, a12, a21)
plot_time_series(axes[1], r1, r2, K1, K2, a12, a21)

plt.tight_layout()
plt.show()
