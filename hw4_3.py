import numpy as np
import sympy as sp

# ----------- (1) 精確值計算（符號積分） -----------
x, y = sp.symbols('x y')
f_sym = 2*y*sp.sin(x) + sp.cos(x)**2
inner = sp.integrate(f_sym, (y, sp.sin(x), sp.cos(x)))
exact_value = sp.integrate(inner, (x, 0, sp.pi/4)).evalf()

# ----------- (2) Simpson's Rule -----------
def f(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

def simpsons_double_integral(f, ax, bx, ay_func, by_func, nx, ny):
    x = np.linspace(ax, bx, nx + 1)
    hx = (bx - ax) / nx
    total = 0.0

    for i in range(nx + 1):
        xi = x[i]
        wx = 1 if i == 0 or i == nx else (4 if i % 2 == 1 else 2)
        ay = ay_func(xi)
        by = by_func(xi)
        y = np.linspace(ay, by, ny + 1)
        hy = (by - ay) / ny

        inner_sum = 0.0
        for j in range(ny + 1):
            yj = y[j]
            wy = 1 if j == 0 or j == ny else (4 if j % 2 == 1 else 2)
            inner_sum += wy * f(xi, yj)
        total += wx * inner_sum * hy / 3.0
    return hx / 3.0 * total

simpsons_result = simpsons_double_integral(f, 0, np.pi/4, np.sin, np.cos, 4, 4)

# ----------- (3) Gaussian Quadrature -----------
gauss_nodes = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
gauss_weights = np.array([5/9, 8/9, 5/9])

def gaussian_double_integral(f, ax, bx, ay_func, by_func, nx, ny):
    total = 0.0
    for i in range(nx):
        xi = 0.5 * (bx - ax) * gauss_nodes[i] + 0.5 * (bx + ax)
        wi = gauss_weights[i]
        ay = ay_func(xi)
        by = by_func(xi)
        for j in range(ny):
            yj = 0.5 * (by - ay) * gauss_nodes[j] + 0.5 * (by + ay)
            wj = gauss_weights[j]
            total += wi * wj * f(xi, yj) * 0.25 * (bx - ax) * (by - ay)
    return total

gaussian_result = gaussian_double_integral(f, 0, np.pi/4, np.sin, np.cos, 3, 3)

# ----------- (4) 結果輸出 -----------
print(f"精確值 (Exact value)                       = {exact_value:.7f}")
print(f"Simpson's Rule (n=4, m=4) 結果            = {simpsons_result:.7f}")
print(f"Gaussian Quadrature (n=3, m=3) 結果       = {gaussian_result:.7f}")
print()
print(f"Simpson's Rule 誤差                        = {abs(simpsons_result - exact_value):.7f}")
print(f"Gaussian Quadrature 誤差                   = {abs(gaussian_result - exact_value):.7f}")
