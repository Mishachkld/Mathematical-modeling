import numpy as np
import matplotlib.pyplot as plt


U0 = 10.0
R0 = 0.05  # эквивалентный радиус для внешнего условия шара, м

L = 1.0    # размер квадратной области 1x1 м
h_coarse = 0.01
h_fine = 0.002

EPS = 1e-5
MAX_ITER_COARSE = 10000
MAX_ITER_FINE = 14000


def make_axis(length, h):
    nodes = int(round(length / h)) + 1
    return np.linspace(-length / 2.0, length / 2.0, nodes)


def set_outer_boundary_analytic(u, x, y):
    xx, yy = np.meshgrid(x, y, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    rr = np.maximum(rr, 1e-12)
    ub = U0 * R0 / rr
    u[0, :] = ub[0, :]
    u[-1, :] = ub[-1, :]
    u[:, 0] = ub[:, 0]
    u[:, -1] = ub[:, -1]


def mark_rect(mask, values, x, y, x_min, x_max, y_min, y_max, value):
    xx, yy = np.meshgrid(x, y, indexing="ij")
    area = (xx >= x_min) & (xx <= x_max) & (yy >= y_min) & (yy <= y_max)
    mask[area] = True
    values[area] = value


def mark_antenna(mask, values, x, y, start, length, angle_deg, value):
    h_local = x[1] - x[0]
    angle = np.deg2rad(angle_deg)
    points = int(np.ceil(length / (h_local / 2.0))) + 1

    for k in range(points):
        s = min(k * (h_local / 2.0), length)
        xp = start[0] + s * np.cos(angle)
        yp = start[1] + s * np.sin(angle)
        i = int(np.argmin(np.abs(x - xp)))
        j = int(np.argmin(np.abs(y - yp)))
        mask[i, j] = True
        values[i, j] = value


def build_internal_geometry(x, y):
    shape = (x.size, y.size)
    fixed_mask = np.zeros(shape, dtype=bool)
    fixed_values = np.zeros(shape, dtype=float)

    half = 0.05  # половина стороны 10 см
    panel = 0.10

    # Центральный кубсат.
    mark_rect(fixed_mask, fixed_values, x, y, -half, half, -half, half, U0)

    # 4 солнечные батареи (крест вокруг кубсата).
    mark_rect(fixed_mask, fixed_values, x, y, -half, half, half, half + panel, U0)            # верх
    mark_rect(fixed_mask, fixed_values, x, y, -half, half, -half - panel, -half, U0)          # низ
    mark_rect(fixed_mask, fixed_values, x, y, -half - panel, -half, -half, half, U0)          # лево
    mark_rect(fixed_mask, fixed_values, x, y, half, half + panel, -half, half, U0)            # право

    # Антенна 14 см под углом 30 град., старт в углу между верхней и правой батареями.
    antenna_start = (half, half)
    mark_antenna(fixed_mask, fixed_values, x, y, antenna_start, 0.14, 30.0, U0)

    return fixed_mask, fixed_values


def solve_laplace(u, fixed_mask, fixed_values, eps, max_iter):
    u = u.copy()
    u[fixed_mask] = fixed_values[fixed_mask]

    for it in range(max_iter):
        u_old = u.copy()

        u[1:-1, 1:-1] = 0.25 * (
            u_old[2:, 1:-1] + u_old[:-2, 1:-1] + u_old[1:-1, 2:] + u_old[1:-1, :-2]
        )

        u[fixed_mask] = fixed_values[fixed_mask]
        delta = np.max(np.abs(u - u_old))
        if delta < eps:
            return u, it + 1, delta

    return u, max_iter, delta


def bilinear_sample(u_coarse, x_coarse, y_coarse, xf, yf):
    xf = np.clip(xf, x_coarse[0], x_coarse[-1])
    yf = np.clip(yf, y_coarse[0], y_coarse[-1])

    dx = x_coarse[1] - x_coarse[0]
    dy = y_coarse[1] - y_coarse[0]

    tx = (xf - x_coarse[0]) / dx
    ty = (yf - y_coarse[0]) / dy

    i0 = np.floor(tx).astype(int)
    j0 = np.floor(ty).astype(int)
    i1 = np.clip(i0 + 1, 0, len(x_coarse) - 1)
    j1 = np.clip(j0 + 1, 0, len(y_coarse) - 1)
    i0 = np.clip(i0, 0, len(x_coarse) - 1)
    j0 = np.clip(j0, 0, len(y_coarse) - 1)

    ax = tx - i0
    ay = ty - j0

    f00 = u_coarse[i0, j0]
    f10 = u_coarse[i1, j0]
    f01 = u_coarse[i0, j1]
    f11 = u_coarse[i1, j1]

    return (
        (1.0 - ax) * (1.0 - ay) * f00
        + ax * (1.0 - ay) * f10
        + (1.0 - ax) * ay * f01
        + ax * ay * f11
    )


def interpolate_to_fine(u_coarse, x_coarse, y_coarse, x_fine, y_fine):
    xx_f, yy_f = np.meshgrid(x_fine, y_fine, indexing="ij")
    return bilinear_sample(u_coarse, x_coarse, y_coarse, xx_f, yy_f)


def main():
    x_c = make_axis(L, h_coarse)
    y_c = make_axis(L, h_coarse)
    x_f = make_axis(L, h_fine)
    y_f = make_axis(L, h_fine)

    u_coarse = np.zeros((x_c.size, y_c.size), dtype=float)
    set_outer_boundary_analytic(u_coarse, x_c, y_c)
    fixed_mask_c, fixed_vals_c = build_internal_geometry(x_c, y_c)
    u_coarse[fixed_mask_c] = fixed_vals_c[fixed_mask_c]

    u_coarse, it_c, err_c = solve_laplace(
        u_coarse, fixed_mask_c, fixed_vals_c, eps=EPS, max_iter=MAX_ITER_COARSE
    )

    # Начальное поле мелкой сетки + ее внешняя граница из крупной сетки.
    u_fine = interpolate_to_fine(u_coarse, x_c, y_c, x_f, y_f)
    fixed_mask_f, fixed_vals_f = build_internal_geometry(x_f, y_f)

    boundary_mask_f = np.zeros_like(u_fine, dtype=bool)
    boundary_mask_f[0, :] = True
    boundary_mask_f[-1, :] = True
    boundary_mask_f[:, 0] = True
    boundary_mask_f[:, -1] = True

    fixed_mask_total_f = fixed_mask_f | boundary_mask_f
    fixed_vals_total_f = fixed_vals_f.copy()
    fixed_vals_total_f[boundary_mask_f] = u_fine[boundary_mask_f]

    u_fine[fixed_mask_total_f] = fixed_vals_total_f[fixed_mask_total_f]
    u_fine, it_f, err_f = solve_laplace(
        u_fine, fixed_mask_total_f, fixed_vals_total_f, eps=EPS, max_iter=MAX_ITER_FINE
    )

    print(f"Крупная сетка: итерации = {it_c}, финальная невязка = {err_c:.3e}")
    print(f"Мелкая сетка:  итерации = {it_f}, финальная невязка = {err_f:.3e}")

    plt.figure(figsize=(8, 7))
    im = plt.imshow(
        u_fine.T,
        origin="lower",
        cmap="jet",
        extent=[x_f[0], x_f[-1], y_f[0], y_f[-1]],
        aspect="equal",
    )
    plt.title("Распределение потенциала")
    plt.xlabel("x, м")
    plt.ylabel("y, м")
    plt.colorbar(im, label="Потенциал, В")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
