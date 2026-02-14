from math import sin, cos, pow

x1, y1, x2, y2 = 0, 0, 3, 3

def f(t):
    return (1 - cos(t)) / (t - sin(t)) - y2 / x2

def df(t):
    return (t * sin(t) + 2 * cos(t) - 2) / (t - sin(t)) ** 2

# метод Ньютона
def newton(f, df):
    x0 = (x1 + x2) / 2
    xn = f(x0)
    xn1 = xn - f(xn) / df(xn)
    while abs(xn1 - xn) > pow(10, -3):
        xn = xn1
        xn1 = xn - f(xn) / df(xn)
    return xn1

t = newton(f, df)
C1 = 2 * y2 / (1 - cos(t))
print('t =', round(t, 3))
print('C1 =', round(C1, 3))
print('C1/2 =', round(C1, 3) / 2)
