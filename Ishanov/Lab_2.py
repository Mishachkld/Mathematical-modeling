import numpy as np

def f(m):
    return -k * m

# шаг метода Рунге–Кутты
def rk4_step(m, h):
    k1 = f(m)
    k2 = f(m + h * k1 / 2)
    k3 = f(m + h * k2 / 2)
    k4 = f(m + h * k3)
    return m + h * (k1 + 2*k2 + 2*k3 + k4) / 6


life_period = 1590
k = np.log(2) / life_period
t_start = 0
t_end = 200
step = 1
m = 1
t = t_start

while t < t_end:
    m = rk4_step(m, step)
    t += step

procent = (1 - m) * 100


print(f"Процент распавшейся массы: {procent:.2f}%", )