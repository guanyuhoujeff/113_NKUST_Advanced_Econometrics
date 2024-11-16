#!/usr/bin/env python
# coding: utf-8

# In[6]:


# !pip install sympy


# # 無限制式非線性最適化問題
# ## Unconstrained NLP (nonlinear programming)。
# ## 問題敘述
# 
# 考慮以下目標函數：
# 
# $$
# f(x_1, x_2, x_3) = (x_1)^2 + x_1(1 - x_2) + (x_2)^2 - x_2x_3 + (x_3)^2 + x_3
# $$
# 
# 目標是找到使 $f(x_1, x_2, x_3)$ 最小化的點 $x = [x_1, x_2, x_3]$。
# 

# In[4]:


import numpy as np
from sympy import symbols, diff, solve, Matrix

# 定義變數
x1, x2, x3 = symbols('x1 x2 x3')

# 定義目標函數
f = (x1)**2 + x1*(1 - x2) + (x2)**2 - x2*x3 + (x3)**2 + x3

# 計算梯度向量
grad_f = Matrix([diff(f, x1), diff(f, x2), diff(f, x3)])
print("Gradient Vector:")
print(grad_f)

# 設梯度為0，求解臨界點
critical_points = solve(grad_f, [x1, x2, x3])
print("\nCritical Points:")
print(critical_points)

# 計算 Hessian 矩陣
hessian = Matrix([
    [diff(grad_f[i], var) for var in [x1, x2, x3]] 
    for i in range(3)
])
print("\nHessian Matrix:")
print(hessian)

# 檢查 Hessian 是否正定
eigenvalues = hessian.eigenvals()
print("\nEigenvalues of Hessian:")
print(eigenvalues)

# 判斷正定性
is_positive_definite = all(ev > 0 for ev in eigenvalues.keys())
print("\nIs the Hessian positive definite?")
print(is_positive_definite)


# # 題目描述：用 Quasi-Newton 演算法解無限制式非線性最適化問題
# 
# ## 問題敘述
# 
# 考慮以下目標函數：
# 
# $$
# f(x_1, x_2, x_3) = (x_1)^2 + x_1(1 - x_2) + (x_2)^2 - x_2x_3 + (x_3)^2 + x_3
# $$
# 
# 目標是找到使 $\nabla f = 0$ 的點，即滿足 **一階必要條件（FOC: First Order Condition）** 的解。
# 
# ---
# 

# In[5]:


import numpy as np
from scipy.optimize import minimize

# 定義目標函數 f(x1, x2, x3)
def objective(x):
    x1, x2, x3 = x
    return (x1)**2 + x1*(1 - x2) + (x2)**2 - x2*x3 + (x3)**2 + x3

# 定義梯度向量（Gradient）
def gradient(x):
    x1, x2, x3 = x
    grad = np.array([
        2*x1 + 1 - x2,  # ∂f/∂x1
        -x1 + 2*x2 - x3, # ∂f/∂x2
        -x2 + 2*x3 + 1   # ∂f/∂x3
    ])
    return grad

# 初始猜測點
x0 = np.array([0.0, 0.0, 0.0])  # 可以改變初始值來檢查收斂情況

# 使用Quasi-Newton方法（BFGS）
result = minimize(objective, x0, method='BFGS', jac=gradient, options={'disp': True})

# 結果
print("Optimal Solution (x):", result.x)
print("Optimal Function Value (f):", result.fun)
print("Success:", result.success)


# In[ ]:





# 目標函數的形式為：
# 
# $$
# f(x_1, x_2, x_3) = (x_1)^2 + x_1(1 - x_2) + (x_2)^2 - x_2x_3 + (x_3)^2 + x_3
# $$
# 
# ### 梯度向量
# 目標函數的梯度向量為：
# $$
# \nabla f = 
# \begin{bmatrix}
# 2x_1 + 1 - x_2 \\
# -x_1 + 2x_2 - x_3 \\
# -x_2 + 2x_3 + 1
# \end{bmatrix}
# $$
# 
# 在初始點 $x_0 = [0, 0, 0]$，梯度為：
# $$
# \nabla f(0, 0, 0) = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}
# $$
# 
# ### Hessian 矩陣
# Hessian 矩陣定義如下：
# $$
# H = 
# \begin{bmatrix}
# 2  & -1 &  0 \\
# -1 &  2 & -1 \\
# 0  & -1 &  2
# \end{bmatrix}
# $$
# 
# ### 梯度下降更新公式
# 根據問題描述，使用以下公式更新點：
# $$
# x_1 = x_0 - H^{-1} \nabla f(0, 0, 0)
# $$
# 
# 初始點為：
# $$
# x_0 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
# $$
# 
# ---
# 
# ## Hessian 的正定性檢查
# 
# ### 特徵值計算
# Hessian 矩陣的特徵值為：
# $$
# \lambda_1 = 3.414, \quad \lambda_2 = 0.586, \quad \lambda_3 = 2
# $$
# 
# 由於所有特徵值均大於零，因此 Hessian 矩陣是正定的，證明此問題有唯一的極小值。
# 
# ---
# 

# In[7]:


import numpy as np

# Define the Hessian matrix
H = np.array([
    [2, -1,  0],
    [-1, 2, -1],
    [0, -1,  2]
])

# Define the gradient vector at x0 = [0, 0, 0]
grad_f = np.array([2 * 0 + 1 - 0, -0 + 2 * 0 - 0, -0 + 2 * 0 + 1])  # Gradient at (0, 0, 0)
print("Gradient Vector at x0:", grad_f)

# Compute the inverse of the Hessian matrix
H_inv = np.linalg.inv(H)
print("Inverse of Hessian Matrix:\n", H_inv)

# Initial point
x0 = np.array([0, 0, 0])

# Update rule: x1 = x0 - H_inv * grad_f
x1 = x0 - np.dot(H_inv, grad_f)
print("Next Point (x1):", x1)

# Check if the solution is a minimum
eigenvalues = np.linalg.eigvals(H)
is_positive_definite = all(eigenvalues > 0)
print("Eigenvalues of Hessian:", eigenvalues)
print("Is Hessian Positive Definite?", is_positive_definite)


# In[ ]:




