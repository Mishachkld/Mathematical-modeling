# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d

# ---------------------------------------------------------
# УСЛОВИЯ ВАРИАНТА 24
# ---------------------------------------------------------
L = 10.0
maxT = 1.0


def D(x):
    return 25 - 2 * x  # по условию лабораторной


def f(x):
    return x  # по условию лабораторной


def initial_condition(x):
    return x ** 2 * (10 - x)  # по условию лабораторной


# ---------------------------------------------------------
# ЯВНАЯ СХЕМА
# ---------------------------------------------------------
def explicit_scheme(h, tau):
    Nx = int(L / h) + 1
    Nt = int(maxT / tau) + 1

    x = np.linspace(0, L, Nx)
    t = np.linspace(0, maxT, Nt)
    U = np.zeros((Nt, Nx))

    U[0, :] = initial_condition(x)

    for n in range(0, Nt - 1):
        for i in range(1, Nx - 1):
            d2u = (U[n, i + 1] - 2 * U[n, i] + U[n, i - 1]) / h ** 2
            U[n + 1, i] = U[n, i] + tau * (D(x[i]) * d2u - f(x[i]) * U[n, i] + 5)

        # граничные условия
        U[n + 1, 0] = 0
        U[n + 1, -1] = 0

    return x, t, U


# ---------------------------------------------------------
# НЕЯВНАЯ СХЕМА
# ---------------------------------------------------------
def implicit_scheme(h, tau):
    Nx = int(L / h) + 1
    Nt = int(maxT / tau) + 1

    x = np.linspace(0, L, Nx)
    t = np.linspace(0, maxT, Nt)
    U = np.zeros((Nt, Nx))

    U[0, :] = initial_condition(x)

    alpha = np.zeros(Nx)
    beta = np.zeros(Nx)
    gamma = np.zeros(Nx)

    for i in range(1, Nx - 1):
        alpha[i] = -tau * D(x[i]) / h ** 2
        beta[i] = 1 + 2 * tau * D(x[i]) / h ** 2 + tau * f(x[i])
        gamma[i] = -tau * D(x[i]) / h ** 2

    beta[0] = 1
    beta[-1] = 1

    diagonals = [alpha[1:], beta, gamma[:-1]]
    A = diags(diagonals, [-1, 0, 1], format='csc')

    for n in range(0, Nt - 1):
        b = U[n, :].copy()
        b[1:-1] += 5 * tau
        b[0] = 0
        b[-1] = 0

        U[n + 1, :] = spsolve(A, b)

    return x, t, U


# ---------------------------------------------------------
# Оценка ошибки по пространству
# ---------------------------------------------------------
def compute_error_h(u_real, u_calc, x_real, x_calc):
    interp_func = interp1d(
        x_real, u_real, kind='cubic',
        bounds_error=False, fill_value='extrapolate'
    )
    u_real_interp = interp_func(x_calc)
    return np.max(np.abs(u_real_interp - u_calc))


# ---------------------------------------------------------
# Оценка ошибки по времени
# ---------------------------------------------------------
def compute_error_t(u_real, u_calc, t_real, t_calc):
    interp_func = interp1d(
        t_real, u_real, axis=0, kind='cubic',
        bounds_error=False, fill_value='extrapolate'
    )
    u_real_interp = interp_func(t_calc)
    return np.max(np.abs(u_real_interp - u_calc))


# ---------------------------------------------------------
# Визуализация (ТОЛЬКО 2D)
# ---------------------------------------------------------
def solve(h, tau):
    x_ex, t_ex, u_ex = explicit_scheme(h, tau)
    x_im, t_im, u_im = implicit_scheme(h, tau)

    plt.figure(figsize=(14, 6))

    # явная схема — финальное решение
    plt.subplot(1, 2, 1)
    # plt.plot(x_ex, u_ex[-1, :], label="U(x, T)")
    plt.plot(x_ex, u_ex[0, :], label="U")
    plt.title("Явная схема")
    plt.xlabel("x")
    plt.ylabel("U")
    plt.grid()
    plt.legend()

    # неявная схема — финальное решение
    plt.subplot(1, 2, 2)
    # plt.plot(x_im, u_im[-1, :], label="U(x, T)")
    plt.plot(x_im, u_im[0, :], label="U")
    plt.title("Неявная схема")
    plt.xlabel("x")
    plt.ylabel("U")
    plt.grid()
    plt.legend()

    plt.show()

    return x_ex, t_ex, u_ex, u_im



x_r, t_r, u_ex_r, u_im_r = solve(0.05, 0.00001)
# %%
x_1, _, u_ex1, u_im1 = solve(0.7, 0.00001)

# %%
x_2, _, u_ex2, u_im2 = solve(0.5, 0.00001)

# %%
x_3, _, u_ex3, u_im3 = solve(0.25, 0.00001)

# %%
x_4, _, u_ex4, u_im4 = solve(0.125, 0.00001)

#
# def compute_error_h(u_real, u_calc, x_real, x_calc, error_type='max'):
#     interp_func = interp1d(x_real, u_real, kind='cubic',
#                            bounds_error=False, fill_value='extrapolate')
#     u_real_interp = interp_func(x_calc)
#
#     diff = u_real_interp - u_calc
#
#     error = np.max(np.abs(diff))
#
#     return error
#
#
# # %%
# er1 = compute_error_h(u_ex_r, u_ex1, x_r, x_1)
# er2 = compute_error_h(u_ex_r, u_ex2, x_r, x_2)
# er3 = compute_error_h(u_ex_r, u_ex3, x_r, x_3)
# er4 = compute_error_h(u_ex_r, u_ex4, x_r, x_4)
#
# er5 = compute_error_h(u_im_r, u_im1, x_r, x_1)
# er6 = compute_error_h(u_im_r, u_im2, x_r, x_2)
# er7 = compute_error_h(u_im_r, u_im3, x_r, x_3)
# er8 = compute_error_h(u_im_r, u_im4, x_r, x_4)
#
# # %%
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot([0.7, 0.5, 0.25, 0.125], [er1, er2, er3, er4])
# plt.title("Явная схема")
# plt.xlabel("h")
# plt.ylabel("error")
# plt.subplot(1, 2, 2)
# plt.plot([0.7, 0.5, 0.25, 0.125], [er5, er6, er7, er8])
# plt.title("Неявная схема")
# plt.xlabel("h")
# plt.ylabel("error")
# plt.show()
#
# # %%
# _, t_1, u_et_1, u_it_1 = solve(0.05, 0.00005)
#
# # %%
# _, t_2, u_et_2, u_it_2 = solve(0.05, 0.00004)
#
# # %%
# _, t_3, u_et_3, u_it_3 = solve(0.05, 0.00003)
#
# # %%
# _, t_4, u_et_4, u_it_4 = solve(0.05, 0.00002)
#
#
# # %%
# def compute_error_t(u_real, u_calc, t_real, t_calc, error_type='max'):
#     interp_func = interp1d(t_real, u_real, axis=0, kind='cubic',
#                            bounds_error=False, fill_value='extrapolate')
#     u_real_interp = interp_func(t_calc)
#
#     diff = u_real_interp - u_calc
#
#     error = np.max(np.abs(diff))
#
#     return error
#
#
# # %%
# er1 = compute_error_t(u_ex_r, u_et_1, t_r, t_1)
# er2 = compute_error_t(u_ex_r, u_et_2, t_r, t_2)
# er3 = compute_error_t(u_ex_r, u_et_3, t_r, t_3)
# er4 = compute_error_t(u_ex_r, u_et_4, t_r, t_4)
#
# er5 = compute_error_t(u_im_r, u_it_1, t_r, t_1)
# er6 = compute_error_t(u_im_r, u_it_2, t_r, t_2)
# er7 = compute_error_t(u_im_r, u_it_3, t_r, t_3)
# er8 = compute_error_t(u_im_r, u_it_4, t_r, t_4)
# # %%
# plt.figure(figsize=(15, 10))
# plt.subplot(1, 2, 1)
# plt.plot([0.00005, 0.00004, 0.00003, 0.00002], [er1, er2, er3, er4])
# plt.title("Явная схема")
# plt.xlabel("t")
# plt.ylabel("error")
# plt.subplot(1, 2, 2)
# plt.plot([0.00005, 0.00004, 0.00003, 0.00002], [er5, er6, er7, er8])
# plt.title("Неявная схема")
# plt.xlabel("t")
# plt.ylabel("error")
# plt.show()