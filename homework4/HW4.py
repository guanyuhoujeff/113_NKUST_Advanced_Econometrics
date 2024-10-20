#!/usr/bin/env python
# coding: utf-8


# Q1. Use R or python to print  the following table:  
# 1+1=; 1+2=; 1+3=;
# 2+1=; 2+2=; 2+3=; 
# 3+1=; 3+2=; 3+3=; 
print("Q1. Use R or python to print  the following table:  ")
print("1+1=; 1+2=; 1+3=;")
print("2+1=; 2+2=; 2+3=; ")
print("3+1=; 3+2=; 3+3=; ")
print("\n\n\n解法:")

for i in range(1, 4):  # 外層循環：1 到 3
    for j in range(1, 4):  # 內層循環：1 到 3
        print(f"{i}+{j}={i+j}", end="; ")  # 打印加法表達式，不換行
    print()  # 每完成一行後換行

print("=========================================================================\n\n")

# Q2. For equation x=x^(1/3)-5, 
# use newton method
# to solve x

print("Q2. For equation x=x^(1/3)-5, ")
print("use newton method")
print("to solve x ")
print("\n\n\n解法:")

def f(x):
    return x**(1/3) - 5 - x

def f_prime(x):
    return 1/(3*x**(2/3)) - 1

def newton_method(initial_guess, tolerance=1e-7, max_iterations=100):
    x = initial_guess
    for i in range(max_iterations):
        fx = f(x)
        if abs(fx) < tolerance:
            return x
        x = x - fx / f_prime(x)
    return x  # 如果達到最大迭代次數仍未收斂,返回最後的值

# 使用牛頓法求解
initial_guess = 4.0  # 初始猜測值
solution = newton_method(initial_guess)

print(f"方程 x=x^(1/3)-5 的解約為: {solution:.7f}")
print(f"驗證: {solution:.7f}^(1/3) - 5 = {solution**(1/3) - 5:.7f}")


print("=========================================================================\n\n")


