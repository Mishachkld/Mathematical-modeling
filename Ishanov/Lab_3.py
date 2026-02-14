import math

k = math.log(4) / 4  # коэффициент

def f(m):
    return k * m  # правая часть ОДУ

def rk4_step(m, h):
    k1 = f(m)
    k2 = f(m + h * k1 / 2)
    k3 = f(m + h * k2 / 2)
    k4 = f(m + h * k3)
    return m + h * (k1 + 2*k2 + 2*k3 + k4) / 6

def rk4_revers_loop(start_values, t_end, h):
    t = start_values[0]
    m = start_values[1]
    steps = [(t, m)]

    while t > t_end:
        if t + h < t_end:
            h = t_end - t

        m = rk4_step(m, h)
        t += h

        steps.append((t, m))

    return steps

time_step = -0.1 # идем от граниченого условия до 0
t_end = 0.0

t_start = 7   ## нужно менять вместе с m_start, поэтому снизу использован кортеж
m_start = 2   ## нужно менять вместе с t_start =, поэтому снизу использован кортеж
first_start_m_and_t = (t_start, m_start)
second_start_m_and_t = (3, 0.5)

rk_steps_m7 = rk4_revers_loop(first_start_m_and_t, t_end, time_step)
rk_steps_m3 = rk4_revers_loop(second_start_m_and_t, t_end, time_step)
print("Первоначальное количество фермента m(0) =", rk_steps_m7[-1][1])
print("Первоначальное количество фермента (другие нач. условия) m(0) =", rk_steps_m3[-1][1])
