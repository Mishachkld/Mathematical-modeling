import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Модель Лотки-Вольтерры
def competition_model(N, t, r1, r2, K1, K2, alpha12, alpha21):
    N1, N2 = N
    dN1_dt = r1 * N1 * (K1 - N1 - alpha12 * N2) / K1
    dN2_dt = r2 * N2 * (K2 - N2 - alpha21 * N1) / K2
    return [dN1_dt, dN2_dt]


# Фазовый портрет
def plot_phase_portrait(ax, r1, r2, K1, K2, alpha12, alpha21, title):
    points = []  # Особых точки
    point_phase = []
    points.append((0, 0))
    point_phase.append('Неустойчивый узел')
    points.append((K1, 0))
    point_phase.append('Устойчивый узел' if K2 < alpha21 * K1 else 'Седло')
    points.append((0, K2))
    point_phase.append('Устойчивый узел' if K1 < alpha12 * K2 else 'Седло')
    denom = 1 - alpha12 * alpha21
    if abs(denom) > 1e-10:
        N1_star = (K1 - alpha12 * K2) / denom
        N2_star = (K2 - alpha21 * K1) / denom
        if N1_star > 0 and N2_star > 0:
            points.append((N1_star, N2_star))
            point_phase.append('Устойчивый узел' if K1 > (
                        alpha21 * (K1 ** 2) + alpha12 * (K2 ** 2) - alpha12 * alpha21 * K1 * K2) / K2 else 'Седло')

    # Создание сетки для фазового портрета
    N1 = np.linspace(0, K1 * 1.2, 20)
    N2 = np.linspace(0, K2 * 1.2, 20)
    N1_grid, N2_grid = np.meshgrid(N1, N2)

    # Вычисление скоростей
    dN1_dt = np.zeros_like(N1_grid)
    dN2_dt = np.zeros_like(N2_grid)

    for i in range(len(N1)):
        for j in range(len(N2)):
            dN = competition_model([N1_grid[i, j], N2_grid[i, j]], 0,
                                   r1, r2, K1, K2, alpha12, alpha21)
            dN1_dt[i, j] = dN[0]
            dN2_dt[i, j] = dN[1]

    # Нормализация стрелок
    magnitude = np.sqrt(dN1_dt ** 2 + dN2_dt ** 2)
    dN1_dt_norm = np.zeros_like(dN1_dt)
    dN2_dt_norm = np.zeros_like(dN2_dt)

    mask = magnitude > 1e-12

    dN1_dt_norm[mask] = dN1_dt[mask] / magnitude[mask]
    dN2_dt_norm[mask] = dN2_dt[mask] / magnitude[mask]

    # Построение векторного поля
    ax.quiver(N1_grid, N2_grid, dN1_dt_norm, dN2_dt_norm,
              color='black', alpha=0.6, scale=30)

    # Изоклины
    N1_iso = np.linspace(0, K1 * 1.2, 100)
    # dN1/dt = 0: N2 = (K1 - N1)/alpha12
    N2_iso1 = (K1 - N1_iso) / alpha12
    # dN2/dt = 0: N2 = K2 - alpha21*N1
    N2_iso2 = K2 - alpha21 * N1_iso

    ax.plot(N1_iso, N2_iso1, 'blue', linewidth=1)
    ax.plot(N1_iso, N2_iso2, 'blue', linewidth=1)
    ax.axvline(x=0, ymin=0, ymax=K2 * 1.2, color='blue', linewidth=1)
    ax.plot([0, K1 * 1.2], [0, 0], 'blue', linewidth=1)

    # Особые точки
    color = ['yellow', 'green', 'blue', 'purple']
    for i, (x, y) in enumerate(points):
        if x >= 0 and y >= 0 and x <= K1 * 1.2 and y <= K2 * 1.2:
            ax.plot(x, y, 'o', markersize=8, color=color[i], label=point_phase[i])

    # TODO: Траектории из разных начальных условий
    initial_conditions = [
        [20, 20], [10, 10],
        [50, 5], [5, 80], [10, 20]
    ]

    t = np.linspace(0, 20, 1000)
    for N0 in initial_conditions:
        solution = odeint(competition_model, N0, t,
                          args=(r1, r2, K1, K2, alpha12, alpha21))
        ax.plot(solution[:, 0], solution[:, 1], 'red', alpha=0.7, linewidth=1)
        ax.plot(solution[0, 0], solution[0, 1], 'ko', markersize=3)

    ax.set_xlabel('N_1 (Популяция вида 1)')
    ax.set_ylabel('N_2 (Популяция вида 2)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, K1 * 1.2)
    ax.set_ylim(-0.05, K2 * 1.2)


def plot_time_series(ax, r1, r2, K1, K2, alpha12, alpha21, title):
    """Построение временных рядов"""
    t = np.linspace(0, 20, 1000)

    # TODO: Разные начальные условия
    initial_conditions = [[10, 10], [20, 20]]
    colors = ['red', 'blue']

    for i, N0 in enumerate(initial_conditions):
        solution = odeint(competition_model, N0, t,
                          args=(r1, r2, K1, K2, alpha12, alpha21))

        ax.plot(t, solution[:, 0], color=colors[0],
                linestyle=['-.', '-'][i])
        ax.plot(t, solution[:, 1], color=colors[1],
                linestyle=['--', '-'][i])

    ax.set_xlabel('Время')
    ax.set_ylabel('Численность')
    ax.set_title(title)
    if ax.get_legend_handles_labels()[1]:
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


params = {
    "Заданные условия": {
        "r1": 2.0, "r2": 2.0,
        "K1": 200.0, "K2": 200.0,
        "alpha12": 0.5, "alpha21": 0.5
    }
}
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

regime = "Заданные условия"
param = params[regime]

plot_phase_portrait(axes[0],
                   param["r1"], param["r2"],
                   param["K1"], param["K2"],
                   param["alpha12"], param["alpha21"],
                   f"Фазовый портрет: {regime}")

plot_time_series(axes[1],
                param["r1"], param["r2"],
                param["K1"], param["K2"],
                param["alpha12"], param["alpha21"],
                f"Временные ряды: {regime}")