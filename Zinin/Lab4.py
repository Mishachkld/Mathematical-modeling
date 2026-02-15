import math
import numpy as np


# Константы задачи
K_B = 1.380649e-23          # Постоянная Больцмана, Дж/К
M_PROTON = 1.67262192369e-27  # Масса протона, кг

T = 5000.0                  # Температура, K
ALPHA_DEG = 10.0            # Угол раствора коллиматора, градусы
N_CONCENTRATION = 1.0       # Концентрация n (по условию)
V1 = 2_000.0                # Нижняя граница скорости, м/с (2 км/с)
V2 = 5_000.0                # Верхняя граница скорости, м/с (5 км/с)


def maxwell_speed_pdf(v: np.ndarray | float, m: float, t: float) -> np.ndarray | float:
    """Плотность распределения Максвелла по модулю скорости."""
    coef = 4.0 * math.pi * (m / (2.0 * math.pi * K_B * t)) ** 1.5
    return coef * (v ** 2) * np.exp(-m * (v ** 2) / (2.0 * K_B * t))


def flux_numeric(v1: float, v2: float, alpha_deg: float, n: float, m: float, t: float) -> float:
    """
    Численный расчет:
    W = 2*pi * ∫_{mu_a}^{1} ∫_{v1}^{v2} n * f(v) * v * mu dv dmu
    """
    alpha = math.radians(alpha_deg)
    mu_a = math.cos(alpha)

    v_grid = np.linspace(v1, v2, 200_001)
    f_v = maxwell_speed_pdf(v_grid, m, t)
    integral_v = np.trapezoid(f_v * v_grid, v_grid)

    integral_mu = 0.5 * (1.0 - mu_a ** 2)
    return 2.0 * math.pi * n * integral_mu * integral_v


def flux_analytic(v1: float, v2: float, alpha_deg: float, n: float, m: float, t: float) -> float:
    """
    Аналитически:
    W = pi * n * sin^2(alpha) * C * I
    C = 4*pi*(m/(2*pi*kT))^(3/2)
    I = ∫ v^3 exp(-a v^2) dv, a = m/(2kT)
      = (1/(2a^2)) * [(a*v1^2 + 1)e^{-a v1^2} - (a*v2^2 + 1)e^{-a v2^2}]
    """
    alpha = math.radians(alpha_deg)
    sin2 = math.sin(alpha) ** 2

    a = m / (2.0 * K_B * t)
    c = 4.0 * math.pi * (m / (2.0 * math.pi * K_B * t)) ** 1.5

    term1 = (a * v1 ** 2 + 1.0) * math.exp(-a * v1 ** 2)
    term2 = (a * v2 ** 2 + 1.0) * math.exp(-a * v2 ** 2)
    integral_v = (term1 - term2) / (2.0 * a ** 2)

    return math.pi * n * sin2 * c * integral_v


def main() -> None:
    w_num = flux_numeric(V1, V2, ALPHA_DEG, N_CONCENTRATION, M_PROTON, T)
    w_ana = flux_analytic(V1, V2, ALPHA_DEG, N_CONCENTRATION, M_PROTON, T)

    print("Лабораторная работа №4: поток частиц в детектор")
    print(f"T = {T:.1f} K")
    print(f"alpha = {ALPHA_DEG:.1f} deg")
    print(f"v1 = {V1/1000:.1f} km/s, v2 = {V2/1000:.1f} km/s")
    print(f"n = {N_CONCENTRATION:g} 1/m^3")
    print(f"m (proton) = {M_PROTON:.6e} kg")
    print(f"k = {K_B:.6e} J/K")
    print()
    print(f"W (numeric)  = {w_num:.6e} 1/(m^2*s)")
    print(f"W (analytic) = {w_ana:.6e} 1/(m^2*s)")
    print(f"relative diff = {abs(w_num - w_ana) / abs(w_ana):.3e}")


if __name__ == "__main__":
    main()
