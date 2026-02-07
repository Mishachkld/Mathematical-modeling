import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

n0 = 1.5
alpha = 1.2
p0 = 0.3

def n(y):
    return n0 * np.exp(-alpha * y)

def n_inv(p):
    return -np.log(p / n0) / alpha

def T_of_p(p):
    integrand = lambda y: n(y)**2 / np.sqrt(n(y)**2 - p**2)
    val, _ = quad(integrand, 0, n_inv(p))
    return 2 * val

def ray_x(y, p):
    return p / np.sqrt(n(y)**2 - p**2)

p_rays = np.linspace(p0, 1.2, 5)

plt.figure(figsize=(6, 4))

for p in p_rays:
    y_top = n_inv(p)
    y = np.linspace(0, y_top, 300)
    x = np.zeros_like(y)

    for i in range(1, len(y)):
        dy = y[i] - y[i - 1]
        x[i] = x[i - 1] + ray_x(y[i], p) * dy

    x_full = np.concatenate([x, 2 * x[-1] - x[::-1]])
    y_full = np.concatenate([y, y[::-1]])

    plt.plot(x_full, y_full, label=f"p={p:.2f}, T={T_of_p(p):.2f}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Траектории лучей")
plt.legend()
plt.grid()
plt.show()

def X_of_p(p):
    integrand = lambda y: 1 / np.sqrt(n(y)**2 - p**2)
    val, _ = quad(integrand, 0, n_inv(p))
    return 2 * p * val

p_vals = np.linspace(p0, n0 * 0.95, 50)
X_vals = np.array([X_of_p(p) for p in p_vals])

plt.figure(figsize=(6, 4))
plt.plot(p_vals, X_vals)
plt.xlabel("p")
plt.ylabel("X(p)")
plt.title("Прямая задача")
plt.grid()
plt.show()

X_interp = interp1d(p_vals, X_vals, kind="cubic", fill_value="extrapolate")

def n_inv_rec(r):
    integrand = lambda p: X_interp(p) / np.sqrt(p**2 - r**2)
    val, _ = quad(integrand, r, n0 * 0.95)
    return val / np.pi

r_vals = np.linspace(p0, n0 * 0.95, 40)
y_rec = np.array([n_inv_rec(r) for r in r_vals])

y_true = np.linspace(0, max(y_rec), 200)
n_true = n(y_true)

plt.figure(figsize=(6, 4))
plt.plot(n_true, y_true, label="Исходная n(y)")
plt.plot(r_vals, y_rec, label="Восстановленная n(y)")
plt.xlabel("n")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.title("Сравнение n(y)")
plt.show()





