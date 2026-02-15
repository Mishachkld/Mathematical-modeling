import numpy as np
import matplotlib.pyplot as plt

# Параметры
L = 1.0          # размер области (м)
h = 0.01         # шаг сетки (м)
N = int(L / h)   # количество узлов
U0 = 10.0        # потенциал кубсата
R0 = 0.05        # радиус шара (м)

U = np.zeros((N, N))

# Координаты центра
cx, cy = N // 2, N // 2

# ---- ВНУТРЕННЕЕ ГРАНИЧНОЕ УСЛОВИЕ (кубсат 10x10 см) ----
cube_size = int(0.10 / h)  # 10 см
half = cube_size // 2

U[cx-half:cx+half, cy-half:cy+half] = U0

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

            if cx-half <= i < cx+half and cy-half <= j < cy+half:
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

# ---- Визуализация ----
plt.imshow(U, origin='lower', cmap='jet')
plt.colorbar(label='Потенциал (В)')
plt.title('Распределение потенциала')
plt.show()
