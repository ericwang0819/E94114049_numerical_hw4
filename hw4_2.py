# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:52:07 2025

@author: ericd
"""

import numpy as np
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

# 定義被積函數
def g(x):
    return x**2 * np.log(x)

# 高斯正交計算
def gauss_quadrature(a, b, n):
    # 獲取勒讓德節點和權重
    nodes, weights = leggauss(n)
    # 變換到 [a, b]
    t = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    w = 0.5 * (b - a) * weights
    # 計算積分
    return sum(w * g(t))

# 真實值計算
def exact_integral(a, b):
    def integrand(x):
        return x**2 * np.log(x)
    result, _ = quad(integrand, a, b)
    return result

# 參數
a, b = 1.0, 1.5

# 計算 n=3 和 n=4 的結果
result_n3 = gauss_quadrature(a, b, 3)
result_n4 = gauss_quadrature(a, b, 4)
exact = exact_integral(a, b)

# 輸出
print("Gaussian Quadrature Results:")
print(f"n=3: {result_n3:.8f}, Error: {abs(result_n3 - exact):.8f}")
print(f"n=4: {result_n4:.8f}, Error: {abs(result_n4 - exact):.8f}")
print(f"Exact: {exact:.8f}")