import numpy as np
import matplotlib.pyplot as plt

# условие Куранта:
# c * tau < h,

c = 1.0 # скорость волны
l = 1.0
T = 1.5 # время наблюдения

Nx = 100 # разрешение сетки
Ny = 100

h = l / (Nx - 1) # шаг пространственный
tau = 0.4 * h / c # временной шаг
Nt = int(T / tau) # число временных шагов


# f (две отдельные области)
def initial_function(x, y):
    f = np.zeros_like(x)
    f[((x-0.3)**2 + (y-0.3)**2) < 0.05**2] = 1.0
    f[((x-0.7)**2 + (y-0.6)**2) < 0.06**2] = 1.0
    return f


# Граничные условия Неймана
def apply_neumann(u):
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]
    return u


# Прямая задача
def solve(f):
    u_prev = f.copy()
    u_curr = f.copy()
    data = []

    for n in range(Nt):
        u_next = np.zeros_like(u_curr)

        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                u_next[i, j] = (
                    2*u_curr[i, j] - u_prev[i, j]
                    + (c*tau/h)**2 * (
                        u_curr[i+1, j] + u_curr[i-1, j]
                        + u_curr[i, j+1] + u_curr[i, j-1]
                        - 4*u_curr[i, j]
                    )
                )

        u_next = apply_neumann(u_next)

        data.append(u_next.copy())
        u_prev, u_curr = u_curr, u_next

    return data


# Обратная задача (обращение времени)
def solve_inverse(boundary_data):
    v_next = np.zeros((Nx, Ny))
    v_curr = np.zeros((Nx, Ny))

    for n in reversed(range(Nt)):
        v_prev = np.zeros_like(v_curr)

        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                v_prev[i, j] = (
                    2*v_curr[i, j] - v_next[i, j]
                    + (c * tau / h) ** 2
                    * (v_curr[i+1, j] + v_curr[i-1, j]
                        + v_curr[i, j+1] + v_curr[i, j-1]
                        - 4*v_curr[i, j])
                )

        # Подставляем измерения на границе
        v_prev[0, :] = boundary_data[n][0, :]
        v_prev[-1, :] = boundary_data[n][-1, :]
        v_prev[:, 0] = boundary_data[n][:, 0]
        v_prev[:, -1] = boundary_data[n][:, -1]

        v_next, v_curr = v_curr, v_prev

    return v_curr


x = np.linspace(0, l, Nx)
y = np.linspace(0, l, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

f = initial_function(X, Y)

data_forward_task = solve(f)
data_restored = solve_inverse(data_forward_task)


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Исходная f(x,y)")
plt.imshow(f, extent=[0,l,0,l])
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Восстановленная v(x,y,0)")
plt.imshow(data_restored, extent=[0, l, 0, l], vmax=1, vmin=0)
plt.colorbar()

plt.show()
