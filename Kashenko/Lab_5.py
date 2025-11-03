
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# 1. Задание параметров задачи
# -----------------------------
x_min, x_max = 0.0, 10.0
t_min, t_max = 0.0, 100.0

# Скорость и правая часть
V = lambda x: 11.0 - x
f = lambda x: 6.5

# Начальное и граничное условия
U0 = lambda x: max(0.0, (x - 5.0) * (10.0 - x))          # начальное U(0,x)
g = lambda t: t * (t - 20.0) * (t - 50.0) / 20000.0      # граничное U(t,0)

# --------------------------------------------
# 2. Аналитическое решение (метод характеристик)
# --------------------------------------------
def analytic_solution(t, x):
    # x0 — координата, из которой характеристика пришла в (t,x)
    x0 = 11.0 - (11.0 - x) * np.exp(t)
    if x0 >= 0.0:
        # характеристика из области начального условия
        return U0(x0) + 6.5 * t
    else:
        # характеристика пересекла левую границу
        s = t + np.log(1.0 - x / 11.0)
        return g(s) + 6.5 * (t - s)

# --------------------------------------------
# 3. Явная схема (upwind, первый порядок)
# --------------------------------------------
def explicit_upwind(dx, dt):
    nx = int((x_max - x_min) / dx) + 1
    x = np.linspace(x_min, x_max, nx)
    nt = int(np.ceil((t_max - t_min) / dt)) + 1
    dt = (t_max - t_min) / (nt - 1)

    U = np.array([U0(xx) for xx in x])

    for n in range(1, nt):
        t = (n - 1) * dt
        U_new = U.copy()
        # граничное условие на x=0
        U_new[0] = g(t + dt)
        # внутренние узлы
        for i in range(1, nx):
            Vi = V(x[i])
            U_new[i] = U[i] - dt / dx * Vi * (U[i] - U[i - 1]) + dt * f(x[i])
        U = U_new

    return x, U

# --------------------------------------------
# 4. Неявная схема (upwind, первый порядок)
# --------------------------------------------
def implicit_upwind(dx, dt):
    nx = int((x_max - x_min) / dx) + 1
    x = np.linspace(x_min, x_max, nx)
    nt = int(np.ceil((t_max - t_min) / dt)) + 1
    dt = (t_max - t_min) / (nt - 1)

    U = np.array([U0(xx) for xx in x])

    for n in range(1, nt):
        t = (n - 1) * dt
        U_new = np.zeros_like(U)
        U_new[0] = g(t + dt)
        for i in range(1, nx):
            Vi = V(x[i])
            alpha = dt / dx * Vi
            U_new[i] = (U[i] + dt * f(x[i]) + alpha * U_new[i - 1]) / (1.0 + alpha)
        U = U_new

    return x, U

# --------------------------------------------
# 5. Тестируем 5 вариантов шагов
# --------------------------------------------
dx_list = [1.0, 0.5, 0.2, 0.1, 0.05]
results = []

for dx in dx_list:
    dt = 0.9 * dx / 11.0  # шаг по времени (условие КФЛ)
    x_e, U_e = explicit_upwind(dx, dt)
    x_i, U_i = implicit_upwind(dx, dt)
    U_a = np.array([analytic_solution(t_max, xi) for xi in x_e])

    # Среднеквадратичная ошибка (СКО)
    rms_e = np.sqrt(np.mean((U_e - U_a) ** 2))
    rms_i = np.sqrt(np.mean((U_i - U_a) ** 2))

    results.append({
        "dx": dx,
        "dt": dt,
        "nx": len(x_e),
        "nt": int(np.ceil((t_max - t_min) / dt)) + 1,
        "rms_explicit": rms_e,
        "rms_implicit": rms_i
    })

    # Графики
    plt.figure(figsize=(8, 4))
    plt.plot(x_e, U_a, label='Analytic', linewidth=1.5)
    plt.plot(x_e, U_e, '--', label='Explicit upwind')
    plt.plot(x_i, U_i, ':', label='Implicit upwind')
    plt.xlabel('x')
    plt.ylabel('U(t=100, x)')
    plt.title(f'dx={dx:.3f}, dt={dt:.5f}, nx={len(x_e)}, nt={int(np.ceil((t_max - t_min)/dt))+1}')
    plt.legend()
    plt.grid(True)
    plt.show()

# --------------------------------------------
# 6. Таблица ошибок
# --------------------------------------------
df = pd.DataFrame(results)
print("\nРезультаты сравнения схем:")
print(df.to_string(index=False))
