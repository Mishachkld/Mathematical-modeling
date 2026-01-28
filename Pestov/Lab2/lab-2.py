import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f(y):
    return 1 / (1 + y**2)

def f_deriv(y):
    return -2*y / (1 + y**2)**2

def system(z, p):
    x, y, q = z
    ny = f(y)
    dxdt = p / ny**2
    dydt = q / ny**2
    dqdt = f_deriv(y) / ny
    return [dxdt, dydt, dqdt]

t_span = (0, 30)
t_eval = np.linspace(*t_span, 5000)

plt.figure(figsize=(8, 5))

for p in [0.9, 0.8, 0.7, 0.6]:
    q0 = np.sqrt(f(0) ** 2 - p ** 2)
    sol = solve_ivp(system, t_span, [0, 0, q0],
                    t_eval=t_eval, args=(p,))
    plt.plot(sol.y[0], sol.y[1], label=f"p = {p}")

plt.axhline(0, color="black", linewidth=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Rays in waveguide")
plt.legend()
plt.grid()
plt.show()


