import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

n0 = 1.5
a = 1.2
p0 = 0.3


def n(y):
    return n0 * np.exp(-a * y)


def n_inv(p):
    return -np.log(p / n0) / a


def T_of_p(p):
    integr = lambda y: n(y) ** 2 / np.sqrt(n(y) ** 2 - p ** 2)
    val, error = quad(integr, 0, n_inv(p)) # считаем интеграл
    return 2 * val

# прямая задача
def X_of_p(p):
    integrand = lambda y: 1 / np.sqrt(n(y) ** 2 - p ** 2)
    val, _ = quad(integrand, 0, n_inv(p))
    return 2 * p * val


def ray_x(y, p):
    return p / np.sqrt(n(y) ** 2 - p ** 2)


# возвращает восстановленное значение y = n^-1(r)
def n_inv_rec(r):
    integrand = lambda p: X_interp(p) / np.sqrt(p ** 2 - r ** 2)
    val, _ = quad(integrand, r, n0 * 0.95)
    return val / np.pi


p_rays = np.linspace(p0, 1.4, 5)

plt.figure(figsize=(6, 4))

for p in p_rays:
    y_max = n_inv(p)  # точка поворота
    y = np.linspace(0, y_max, 300, endpoint=False)
    x = np.zeros_like(y)

    for i in range(1, len(y)):
        dy = y[i] - y[i - 1]
        x[i] = x[i - 1] + ray_x(y[i], p) * dy  # численное интегрирование (Эйлером)

    x_full = np.concatenate([x, 2 * x[-1] - x[::-1]])
    y_full = np.concatenate([y, y[::-1]])

    plt.plot(x_full, y_full, label=f" T={T_of_p(p):.2f} p={p:.2f}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Лучи")
plt.legend()
plt.grid()
plt.show()

p_vals = np.linspace(p0, n0 * 0.95, 50)
X_vals = np.array([X_of_p(p) for p in p_vals])

plt.figure(figsize=(6, 4))
plt.plot(p_vals, X_vals)
plt.xlabel("p")
plt.ylabel("X(p)")
plt.title("Прямая задача")
plt.grid()
plt.show()

X_interp = interp1d(p_vals, X_vals,
                    kind="cubic",  # кубическая интерполяция (устойчиво вычисления интеграла)
                    fill_value="extrapolate")  # позволяет оценивать значения функций за пределами исходных точек

r_vals = np.linspace(p0, n0 * 0.95, 40)
y_rec = np.array([n_inv_rec(r) for r in r_vals]) # мы знаем как далеко идут лучи X(p) и поэтому востанавливаем
                                                 # на какой высоте показательно преломления равен r

# проверяем корректность метода
y_true = np.linspace(0, max(y_rec), 200)
n_true = n(y_true)

plt.figure(figsize=(6, 4))
plt.plot(r_vals, y_rec, label="Восстановленная n(y)")
plt.plot(n_true, y_true, label="Начальная n(y)")
plt.xlabel("n")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.title("Compare n(y)")
plt.show()
