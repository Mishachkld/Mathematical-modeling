from random import random, choice

import numpy as np
import matplotlib.pyplot as plt

# Матрицы и векторы
A1 = np.array([[-0.497, 1.444],
               [ 0.080,-0.276]])
b1 = np.array([48.320, -61.883])

A2 = np.array([[-0.637,-0.620],
               [ 0.268,-0.356]])
b2 = np.array([-13.780, 51.595])

A3 = np.array([[0.171,-0.880],
               [0.664,-0.036]])
b3 = np.array([-34.169, 27.239])

transforms = [(A1,b1), (A2,b2), (A3,b3)]

def ifs_attractor(n_points=200000):
    x = np.zeros(2)
    points = []
    for i in range(n_points):
        #На каждом шаге случайно выбирается одно из трёх отображений — и получается новая точка.
        A, b = transforms[np.random.randint(3)]
        x = A @ x + b
        if i > 100:  # пропускаем первые итерации
            points.append(x)
    # После большого числа итераций формируется устойчивое множество точек (=аттрактор)
    return np.array(points)


def fractal_dimension(Z):
    assert len(Z.shape) == 2

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k * k))[0])

    Z = (Z > 0)
    p = min(Z.shape)
    n = 2 ** np.floor(np.log2(p))
    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, int(size)) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

### n_particles - колличество частиц
def random_walk_stick(grid_size=201, n_particles=500, stick_radius=1):
    grid = np.zeros((grid_size, grid_size), dtype=int)
    center = grid_size // 2
    grid[center, center] = 1  # начальная "заданная точка"

    max_r = 1  # максимальный радиус скопления

    for i in range(n_particles):
        # Старт частицы — на окружности вокруг центра
        # Каждая частица начинает на окружности чуть дальше от уже образованного кластера:
        angle = 2 * np.pi * random()
        r_start = max_r + 5
        x = int(center + r_start * np.cos(angle))
        y = int(center + r_start * np.sin(angle))

        # Пока частица не прилипла
        while True:
            # Вероятность горизонтального перехода в 2 раза больше вертикального
            p = random()
            if p < 0.5:
                dx, dy = choice([(1, 0), (-1, 0)])  # горизонталь
            else:
                dx, dy = choice([(0, 1), (0, -1)])  # вертикаль

            x += dx
            y += dy

            # Проверка выхода за границы
            if not (1 <= x < grid_size - 1 and 1 <= y < grid_size - 1):
                break  # "улетела"

            # Проверка прилипания
            # Если частица оказывается рядом (в квадрате 3×3) с уже прилипшими точками — она “приклеивается”:
            if np.any(grid[x - 1:x + 2, y - 1:y + 2]):
                grid[x, y] = 1
                r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
                if r > max_r:
                    max_r = r
                break

    return grid


grid = random_walk_stick(grid_size=201, n_particles=800) # фрактальное разветвлённое образование, похожее на снежинку.
plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap='plasma', origin='lower')
plt.title("Случайное блуждание с прилипанием")
plt.axis('off')
plt.show()


fd = fractal_dimension(grid)
print("Фрактальная (метрическая) размерность:", fd)
#Аттрактор
points = ifs_attractor(300000)
plt.figure(figsize=(7,7))
plt.scatter(points[:,0], points[:,1], s=0.2, color='black')
plt.title("Аттрактор")# Получили трихлиственное отображение (трехлиствиник)
plt.axis('equal')
plt.axis('off')
plt.show()