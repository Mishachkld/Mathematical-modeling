import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

# ---- 1. Определяем невыпуклый 9-угольник ----
# Координаты можно менять (пример невыпуклой фигуры)
polygon_points = np.array([
    [0, 0], [2, 0], [3, 1], [2.5, 2], [3, 3],
    [1.5, 2.5], [1, 4], [0, 3], [-0.5, 1.5]
])
polygon = Polygon(polygon_points)


# ---- 2. Функция для интегрирования ----
def f(x, y):
    return x ** 3 + y ** 2


# ---- 3. Метод Монте-Карло ----
def monte_carlo_area_and_integral(polygon, func, n_points):
    minx, miny, maxx, maxy = polygon.bounds
    xs = np.random.uniform(minx, maxx, n_points)
    ys = np.random.uniform(miny, maxy, n_points)
    points = np.vstack((xs, ys)).T

    inside = np.array([polygon.contains(Point(x, y)) for x, y in points])
    inside_points = points[inside]

    area_box = (maxx - minx) * (maxy - miny)
    area_est = area_box * np.sum(inside) / n_points
    integral_est = area_box * np.mean(func(xs[inside], ys[inside])) * np.sum(inside) / n_points

    return area_est, integral_est, points, inside


# ---- 4. Проверим для разных N ----
N_values = [1000, 2500, 5000, 10_000, 50_000]
areas, integrals = [], []

for N in N_values:
    area, integral, points, inside = monte_carlo_area_and_integral(polygon, f, N)
    areas.append(area)
    integrals.append(integral)
    print(f"N={N:6d} | Площадь ≈ {area:.4f} | Интеграл ≈ {integral:.4f}")

# ---- 5. Визуализация ----
plt.figure(figsize=(8, 6))
plt.plot(*polygon.exterior.xy, 'k-', linewidth=2, label="9-угольник")
plt.scatter(points[:, 0], points[:, 1], s=3, c=np.where(inside, 'green', 'red'), alpha=0.5)
plt.title("Метод Монте-Карло: точки внутри/вне многоугольника")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis('equal')
plt.show()

# ---- 6. Графики зависимости результатов от числа точек ----
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(N_values, areas, marker='o')
plt.title("Оценка площади")
plt.xlabel("N")
plt.ylabel("Площадь")

plt.subplot(1, 2, 2)
plt.plot(N_values, integrals, marker='o', color='orange')
plt.title("Оценка интеграла")
plt.xlabel("N")
plt.ylabel("Интеграл")
plt.tight_layout()
plt.show()
