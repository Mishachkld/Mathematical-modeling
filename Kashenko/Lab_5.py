# %%
import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
x_min, x_max = 0.0, 10.0
t_min, t_max = 0.0, 10.0

V = lambda x: 11 - x
f = lambda x: 6.5
u_init = lambda x: np.maximum(0, (x - 5) * (10 - x))
u_left = lambda t: t * (t - 20) * (t - 50) / 20000


# Явная схема
def explicit(N_x, N_t):
    h = (x_max - x_min) / N_x
    tau = (t_max - t_min) / N_t
    x = np.linspace(x_min, x_max, N_x + 1)
    t = np.linspace(t_min, t_max, N_t + 1)
    U = np.zeros((N_t + 1, N_x + 1))
    U[0, :] = u_init(x)
    for n in range(N_t):
        U[n + 1, 0] = u_left(t[n + 1])
        for i in range(1, N_x + 1):
            U[n + 1, i] = U[n, i] - (tau / h) * V(x[i]) * (U[n, i] - U[n, i - 1]) + tau * f(x[i])
    return x, t, U.T

def U_analytic(x, t):
    x0 = 11.0 - (11.0 - x) * np.exp(t)
    if x0 >= 0.0:
        return u_init(x0) + 6.5 * t
    else:
        s = t + np.log(1.0 - x / 11.0)
        return u_left(s) + 6.5 * (t - s)

# Неявная схема
def implicit(N_x, N_t):
    h = (x_max - x_min) / N_x
    tau = (t_max - t_min) / N_t
    x = np.linspace(x_min, x_max, N_x + 1)
    t = np.linspace(t_min, t_max, N_t + 1)
    U = np.zeros((N_t + 1, N_x + 1))
    U[0, :] = u_init(x)
    for n in range(N_t):
        U[n + 1, 0] = u_left(t[n + 1])
        for i in range(1, N_x + 1):
            sigma = tau / h * V(x[i])
            U[n + 1, i] = (U[n, i] + tau * f(x[i]) + sigma * U[n + 1, i - 1]) / (1 + sigma)
    return x, t, U.T


# Аналитическое решение по методу характеристик
def analytical(x, t):
    z = np.exp(t) * (11 - x)
    if 1 <= z <= 6:
        return 6.5 * t
    elif 6 <= z <= 11:
        return 6.5 * t + (z - 1) * (6 - z)
    else:
        return 6.5 * t + t * ((t*t - 70 * t - 12000) / 20_000)


def analytic(N_x, N_t):
    x = np.linspace(x_min, x_max, N_x + 1)
    t = np.linspace(t_min, t_max, N_t + 1)
    U = np.zeros((N_x + 1, N_t + 1))
    for i, x_i in enumerate(x):
        for j, t_j in enumerate(t):
            U[i][j] = U_analytic(x_i, t_j)
    return x, t, U

# Основная функция сравнения
def calculate(N_x, N_t):
    print(f"Сетка {N_x}x{N_t}")
    x_ex, t_ex, u_ex = explicit(N_x, N_t)
    x_impl, t_impl, u_impl = implicit(N_x, N_t)
    x_al, t_al, u_al = analytic(N_x, N_t)

    # Последний временной слой (t = t_max)
    plt.figure(figsize=(10, 6))
    plt.plot(x_ex, u_ex[:, -1], label="Явная схема", lw=2)
    plt.plot(x_impl, u_impl[:, -1], label="Неявная схема", lw=2)
    plt.plot(x_al, u_al[:, -1], label="Аналитическое решение", lw=2, linestyle="--")
    plt.xlabel("x")
    plt.ylabel("U(x, t_max)")
    plt.title(f"Сравнение решений при t = {t_max}, N_x={N_x}, N_t={N_t}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Отклонение между решениями (явная - неявная): {np.var(u_impl - u_ex)}")
    print(f"Отклонение между решениями (явная - аналитическое): {np.var(u_ex - u_al)}")
    print(f"Отклонение между решениями (неявная - аналитическое): {np.var(u_al - u_impl)}")
    print("-" * 60)


calculate(7, 700)

calculate(10, 1000)
calculate(30, 3000)
calculate(50, 5000)
calculate(100, 10000)

