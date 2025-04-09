import numpy as np
from scipy.integrate import quad

# 複合辛普森法則
def composite_simpson(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])  # 逐點計算
    result = (h / 3) * (y[0] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2]) + y[n])
    return result

# a. 積分 \int_0^1 x^(-1/4) \sin x dx
def g_a(t):
    return t**(-7/4) * np.sin(1/t)  # 確保指數為 -7/4

# b. 積分 \int_1^\infty x^(-4) \sin x dx
def g_b(t):
    if t == 0:
        return 0.0  # 處理 t=0 的極限
    return t**2 * np.sin(1/t)

# 參數
n = 4
a_a, b_a = 1.0, 100.0  # 積分 a 的截斷區間 [1, 100]
a_b, b_b = 0.0, 1.0    # 積分 b 的區間 [0, 1]

# 計算
simpson_a = composite_simpson(g_a, a_a, b_a, n)
simpson_b = composite_simpson(g_b, a_b, b_b, n)

# 真實值
def f_a(x):
    return x**(-1/4) * np.sin(x)
def f_b(x):
    return x**(-4) * np.sin(x)
true_a, _ = quad(f_a, 0, 1)
true_b, _ = quad(f_b, 1, np.inf)

# 輸出
print("Composite Simpson's Rule (n=4):")
print(f"a. \int_0^1 x^(-1/4) \sin x dx ≈ {simpson_a:.8f}, True value: {true_a:.8f}")
print(f"b. \int_1^∞ x^(-4) \sin x dx ≈ {simpson_b:.8f}, True value: {true_b:.8f}")
print(f"Error a: {abs(simpson_a - true_a):.8f}")
print(f"Error b: {abs(simpson_b - true_b):.8f}")