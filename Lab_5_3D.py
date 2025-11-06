import numpy as np
import matplotlib.pyplot as plt

x_min, x_max = 0.0, 10.0
t_min, t_max = 0.0, 100.0
V = lambda x: 11 - x
f = lambda x: 6.5
u_init = lambda x: np.maximum(0, (x - 5) * (10 - x))
u_left = lambda t: t * (t - 20) * (t - 50) / 20000


def explicit_corner(N_x, N_t):
    h = (x_max - x_min) / N_x
    tau = (t_max - t_min) / N_t
    x = np.linspace(x_min, x_max, N_x+1)
    t = np.linspace(t_min, t_max, N_t+1)
    U = np.zeros((N_t+1, N_x+1))
    U[0, :] = u_init(x)
    for n in range(N_t):
        U[n+1, 0] = u_left(t[n+1])
        for i in range(1, N_x+1):
            U[n+1, i] = U[n, i] - (tau / h) * V(x[i]) * (U[n, i] - U[n, i-1]) + tau * f(x[i])
    return x, t, U.T

def implicit_corner(N_x, N_t):
    h = (x_max - x_min) / N_x
    tau = (t_max - t_min) / N_t
    x = np.linspace(x_min, x_max, N_x+1)
    t = np.linspace(t_min, t_max, N_t+1)
    U = np.zeros((N_t+1, N_x+1))
    U[0, :] = u_init(x)
    for n in range(N_t):
        U[n+1, 0] = u_left(t[n+1])
        for i in range(1, N_x+1):
            sigma = tau / h * V(x[i])
            U[n+1, i] = (U[n, i] + tau * f(x[i]) + sigma * U[n+1, i-1]) / (1 + sigma)
    return x, t, U.T

def U_analytic(x, t):
    z = np.exp(t) * (11 - x)
    if 1 <= z <= 6:
        return 6.5 * t
    elif 6 <= z <= 11:
        return 6.5 * t + (z - 1) * (6 - z)
    else:
        return 6.5 * t + t * ((t*t - 70 * t - 12000) / 20_000)

def U_analytic(x, t):
    # x0 — координата, из которой характеристика пришла в (t,x)
    x0 = 11.0 - (11.0 - x) * np.exp(t)
    if x0 >= 0.0:
        # характеристика из области начального условия
        return u_init(x0) + 6.5 * t
    else:
        # характеристика пересекла левую границу
        s = t + np.log(1.0 - x / 11.0)
        return u_left(s) + 6.5 * (t - s)

def analytic(N_x, N_t):
    x = np.linspace(x_min, x_max, N_x+1)
    t = np.linspace(t_min, t_max, N_t+1)
    U = np.zeros((N_x+1, N_t+1))
    for i, x_i in enumerate(x):
        for j, t_j in enumerate(t):
            U[i][j] = U_analytic(x_i, t_j)
    return x, t, U

# %%
def solve(N_x, N_t):
    print(f"Сетка {N_x}x{N_t}")
    x_ex, t_ex, u_ex = explicit_corner(N_x, N_t)
    x_impl, t_impl, u_impl = implicit_corner(N_x, N_t)
    x_al, t_al, u_al = analytic(N_x, N_t)
    X_ex, T_ex = np.meshgrid(x_ex, t_ex)
    X_impl, T_impl = np.meshgrid(x_impl, t_impl)
    X_al, T_al = np.meshgrid(x_al, t_al)
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X_ex, T_ex, u_ex.T, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('U(x, t)' )
    ax1.set_title("Явная схема")
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X_impl, T_impl, u_impl.T, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('U(x, t)' )
    ax2.set_title("Неявная схема")
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X_al, T_al, u_al.T, cmap='viridis')
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    ax3.set_zlabel('U(x, t)')
    ax3.set_title("Аналитическое решение")
    plt.show()
    print(f"Отклонение между явной и неявной схемами: {np.var(u_impl - u_ex)}")
    print(f"Отклонение между явной схемой и аналитическим решением: {np.var(u_ex - u_al)}")
    print(f"Отклонение между неявной схемой и аналитическим решением: {np.var(u_al - u_impl)}")

# %%
solve(6, 600)

# %%
solve(10, 1000)

# %%
solve(25, 2500)

# %%
solve(50, 5000)

# %%
solve(100, 10000)

# %%
