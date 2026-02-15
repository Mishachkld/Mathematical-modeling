import numpy as np
import matplotlib.pyplot as plt

# Параметры
L = 1.0
h = 0.01
N = int(L / h)
U0 = 10.0
R0 = 0.05

U = np.zeros((N, N))

cx, cy = N // 2, N // 2

# ---- КУБСАТ 10x10 см ----
cube_size = int(0.10 / h)
half = cube_size // 2

U[cx-half:cx+half, cy-half:cy+half] = U0

# ---- АНТЕННА 10 см под 45 градусов ----
antenna_len = int(0.10 / h)  # 10 см в узлах

antenna_points = []
for k in range(antenna_len):
    i = cx + k
    j = cy + k
    if 0 <= i < N and 0 <= j < N:
        U[i, j] = U0
        antenna_points.append((i, j))

# ---- ВНЕШНЯЯ ГРАНИЦА ----
for i in range(N):
    for j in [0, N-1]:
        r = np.sqrt((i-cx)**2 + (j-cy)**2) * h
        if r != 0:
            U[i, j] = U0 * R0 / r

for j in range(N):
    for i in [0, N-1]:
        r = np.sqrt((i-cx)**2 + (j-cy)**2) * h
        if r != 0:
            U[i, j] = U0 * R0 / r

eps = 1e-4
max_iter = 10000

for it in range(max_iter):
    U_old = U.copy()

    for i in range(1, N-1):
        for j in range(1, N-1):

            # пропуск кубсата
            if cx-half <= i < cx+half and cy-half <= j < cy+half:
                continue

            # пропуск антенны
            if (i, j) in antenna_points:
                continue

            U[i, j] = 0.25 * (
                U_old[i+1, j] +
                U_old[i-1, j] +
                U_old[i, j+1] +
                U_old[i, j-1]
            )

    if np.max(np.abs(U - U_old)) < eps:
        print("Сошлось за итераций:", it)
        break

# ---- ВИЗУАЛИЗАЦИЯ ----
plt.imshow(U, origin='lower', cmap='jet')
plt.colorbar(label='Потенциал (В)')
plt.title('Потенциал с антенной 45° (см)')
plt.show()
