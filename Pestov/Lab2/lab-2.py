import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Производная функции
def f_deriv(y):
    return -2 * y / (1 + y ** 2) ** 2


# Функция
def n(y):
    return 1 / (1 + y ** 2)


# система
def scheme(k, z, p):
    x, y, q = z
    ny = n(y)
    dqdt = f_deriv(y) / ny
    dxdt = p / ny ** 2
    dydt = q / ny ** 2
    return [dxdt, dydt, dqdt]


# интервал интегрирования
start = 0
stop = 30
t_span = (start, stop)
#
t_eval = np.linspace(start, stop, 10_000)
#
p_values = [0.99, 0.75, 0.6, 0.5]

plt.figure(figsize=(8, 5))

for p in p_values:
    q0 = np.sqrt(n(0) ** 2 - p ** 2)
    # solve_ivp решает ЗК 
    solution = solve_ivp(
        scheme,
        t_span,
        [0, 0, q0], # начальные условия
        t_eval=t_eval,
        args=(p,)) # передаем константу в ур-е
    plt.plot(solution.y[0], solution.y[1], label=f"p={p}")

plt.axhline(0, color="black", linewidth=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
