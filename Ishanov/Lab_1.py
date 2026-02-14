import numpy as np


def f(y):
    phi, w = y

    N = g * np.cos(phi) + R * w * w
    tr = mu * N * np.sign(w)

    phi_dd = -(g / R) * np.sin(phi) - tr / R
    return np.array([w, phi_dd])


# Метод Рунге–Кутты 3-го порядка
def rk3_step(y, h):
    k1 = f(y)
    k2 = f(y + h * k1 / 2)
    k3 = f(y - h * k1 + 2 * h * k2)
    return y + (h / 6) * (k1 + 4 * k2 + k3)


def simulate(phi0, w0, dt=1e-4, tmax=5.0):
    t = 0.0
    y = np.array([phi0, w0])

    while t < tmax and y[0] < np.pi / 2:
        y = rk3_step(y, dt)
        t += dt

        if y[1] < 0:
            break

    return y[0]


def reaches_vertical(v0):
    phi_end = simulate(0, v0 / R)
    return phi_end >= np.pi / 2


def find_min_v0(vmin=0, vmax=40, eps=1e-3):
    while vmax - vmin > eps:
        vmid = (vmin + vmax) / 2
        if reaches_vertical(vmid):
            vmax = vmid
        else:
            vmin = vmid
    return (vmin + vmax) / 2


g = 9.81

R = 9.0
mu = 0.1

v0_num = find_min_v0()
print("Минимальное V_0 ~", v0_num)
