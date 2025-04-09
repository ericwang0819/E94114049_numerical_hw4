# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:49:45 2025

@author: ericd
"""

import numpy as np
from scipy.integrate import quad

# 被積函數
def integrand(t):
    return np.exp(t) * np.sin(4 * t)

# 參數設定
lower_bound = 1.0
upper_bound = 2.0
step_size = 0.1
num_intervals = int((upper_bound - lower_bound) / step_size)

# 生成積分點
points = np.linspace(lower_bound, upper_bound, num_intervals + 1)
func_vals = integrand(points)

# a. 梯形法則
def trapezoid_method(vals, dx, n):
    total = dx * (vals[0] + vals[-1] + 2 * sum(vals[1:-1])) / 2
    return total

# b. 辛普森法則
def simpson_method(vals, dx, n):
    if n % 2 == 1:
        raise ValueError("Simpson's rule requires even number of intervals")
    odd_terms = 4 * sum(vals[1:-1:2])
    even_terms = 2 * sum(vals[2:-2:2])
    total = (dx / 3) * (vals[0] + odd_terms + even_terms + vals[-1])
    return total

# c. 中點法則
def midpoint_method(func, grid, dx, n):
    mids = grid[:-1] + dx / 2
    total = dx * sum(func(mids))
    return total

# 計算各方法結果
trap_result = trapezoid_method(func_vals, step_size, num_intervals)
simp_result = simpson_method(func_vals, step_size, num_intervals)
mid_result = midpoint_method(integrand, points, step_size, num_intervals)

# 驗證用的真值
exact_result, error = quad(integrand, lower_bound, upper_bound)

# 格式化輸出
print("Numerical Integration Results:")
print(f"Trapezoidal Method: {trap_result:.6f}")
print(f"Simpson Method:    {simp_result:.6f}")
print(f"Midpoint Method:   {mid_result:.6f}")
print(f"Exact Result:      {exact_result:.6f}")