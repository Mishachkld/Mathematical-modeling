import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# -------------------------------
# Параметры модели
# -------------------------------
r1 = 2
r2 = 2
K1 = 200
K2 = 200
a12 = 0.5
a21 = 0.5

# -------------------------------
# Система дифференциальных уравнений
# -------------------------------
def dNdt(N, t):
    N1, N2 = N
    dN1 = r1 * N1 * ( (K1 - N1 - a12*N2) / K1 )
    dN2 = r2 * N2 * ( (K2 - N2 - a21*N1) / K2 )
    return [dN1, dN2]

# -------------------------------
# Численное решение
# -------------------------------
t = np.linspace(0, 50, 2000)

# Пример нескольких стартовых точек
initial_conditions = [
    (10, 10),
    (20, 20),
    (20, 100),
    (80, 30),
    (100, 80)
]

initial_conditions_2 = [
    (10, 10),
    (20, 20),
]

solutions = [odeint(dNdt, ic, t) for ic in initial_conditions]
solutions_2 = [odeint(dNdt, ic, t) for ic in initial_conditions_2]

# -------------------------------
# Функции для изоклин
# -------------------------------
def isocline_N1(N2):
    return K1 - a12*N2

def isocline_N2(N1):
    return K2 - a21*N1

# -------------------------------
# Особые точки
# -------------------------------
eq1 = (0, 0)
eq2 = (K1, 0)
eq3 = (0, K2)

# Точка совместного равновесия
N1_eq = (K1 - a12*K2) / (1 - a12*a21)
N2_eq = (K2 - a21*K1) / (1 - a12*a21)
eq4 = (N1_eq, N2_eq)

# -------------------------------
# Построение графиков N(t)
# -------------------------------
plt.figure(figsize=(10,5))
for sol in solutions_2:
    plt.plot(t, sol[:,0], label="N1(t)")
    plt.plot(t, sol[:,1], label="N2(t)")

plt.xlabel("t")
plt.ylabel("Популяции")
plt.title("Зависимости N1(t), N2(t)")
plt.grid()
plt.legend(["N1(t)", "N2(t)"])
plt.show()

# -------------------------------
# Фазовый портрет
# -------------------------------
plt.figure(figsize=(8,8))

# Изоклины
N = np.linspace(0, 200, 400)
plt.plot(isocline_N1(N), N, 'r', label="Изоклина dN1/dt=0")
plt.plot(N, isocline_N2(N), 'b', label="Изоклина dN2/dt=0")

# Траектории
for sol in solutions:
    plt.plot(sol[:,0], sol[:,1])

# Векторное поле
X, Y = np.meshgrid(np.linspace(0,200,20), np.linspace(0,200,20))
dX = r1*X*( (K1 - X - a12*Y)/K1 )
dY = r2*Y*( (K2 - Y - a21*X)/K2 )
plt.quiver(X, Y, dX, dY)

# Особые точки
plt.scatter([eq1[0], eq2[0], eq3[0], eq4[0]],
            [eq1[1], eq2[1], eq3[1], eq4[1]],
            c=["k","k","k","magenta"], s=80)

plt.text(eq4[0]+3, eq4[1], "т. равновесия", color="magenta")

plt.xlabel("N1")
plt.ylabel("N2")
plt.title("Фазовая диаграмма с изоклинами и особыми точками")
plt.grid()
plt.legend()
plt.show()
