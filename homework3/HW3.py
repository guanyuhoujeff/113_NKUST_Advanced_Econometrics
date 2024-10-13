#!/usr/bin/env python
# coding: utf-8
# !pip install scipy yfinance numpy matplotlib


import yfinance as yf
import numpy as np
import pandas as pd


# 1.台積電(tsmc)2019年的股價報酬率的基本統計分析
# 取得股票資料
code = "2330.TW" 
stock_price_data = yf.download(
    code, 
    # period="1y",
    start="2019-01-01",
    end="2020-01-01"
)  # Download data for 1 year
stock_price_data['return'] = stock_price_data['Close'].diff() / stock_price_data['Close'].shift()
stock_price_data["date"] = stock_price_data.index.values
stock_price_data.head()


print("1.台積電(tsmc)2019年的股價報酬率的基本統計分析: ")
print(stock_price_data.describe())
print("\n\n================================================================")


# 2.台積電的股價報酬率列聯表:
# 以正負報酬與上半年及下半年當分類的二個因子.

# Q1: 使用 np.where() 來根據報酬率分類
stock_price_data['Q1'] = np.where(stock_price_data['return'] > 0, 'Positive', 'Negative')

# Q2: 分成上半年(1-6月)和下半年(7-12月)
stock_price_data['Q2'] = np.where(stock_price_data['date'].dt.month <= 6, 'First Half', 'Second Half')

# 創建列聯表
contingency_table = pd.crosstab(stock_price_data['Q1'], stock_price_data['Q2'])

print("2.台積電的股價報酬率列聯表 ")
print(contingency_table)
print("\n\n================================================================")

# 3. plot the histogram of tsmc

import matplotlib.pyplot as plt

print("3. plot the histogram of tsmc ")
plt.figure(figsize=(8,6))
plt.hist(stock_price_data['return'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Histogram of TSMC Stock Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
print("\n\n================================================================")

# 4. Find integrate (exp(1/2*t2), t=-1..0) 
import numpy as np
from scipy import integrate

# 定義被積分函數
def integrand(t):
    return np.exp(0.5 * t**2)

# 計算積分
result, error = integrate.quad(integrand, -1, 0)
print("4. Find integrate (exp(1/2*t2), t=-1..0) ")
print(f"積分結果: {result:.6f}")
print(f"估計誤差: {error:.6e}")
print("\n\n================================================================")

# 5. find the MGF function of standard normal Z i.e., int( exp(xt) *dnorm)==>get: exp1/2*t2 and plot its MGF :exp(1/2*t2) as t=-1…1
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def integrand(x, t):
    # exp(xt) * dnorm(x)
    return np.exp(x*t) * (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)

# 計算不同t值的MGF
t_values = np.linspace(-1, 1, 100)
mgf_values = []

for t in t_values:
    # 積分區間取 [-10, 10]，因為標準常態分配在這個範圍外的機率很小
    result, _ = integrate.quad(integrand, -10, 10, args=(t,))
    mgf_values.append(result)

# 理論MGF值
theoretical_mgf = np.exp(t_values**2/2)

# 繪圖
print("5. find the MGF function of standard normal Z i.e., \n int( exp(xt) *dnorm)==>get: exp1/2*t2 and plot its MGF :exp(1/2*t2) as t=-1…1 ")
plt.figure(figsize=(10, 6))
plt.plot(t_values, mgf_values, 'b-', label='Numerical Integration')
plt.plot(t_values, theoretical_mgf, 'r--', label='Theoretical: exp(t²/2)')
plt.xlabel('t')
plt.ylabel('MGF(t)')
plt.title('Moment Generating Function of Standard Normal Distribution')
plt.legend()
plt.grid(True)
plt.show()

# 驗證數值解和理論解的差異
print("數值解和理論解的最大絕對誤差:", np.max(np.abs(np.array(mgf_values) - theoretical_mgf)))
print("\n\n================================================================")

# 6. Find the mean and variance given the data you found above using normal likelihood MLE and solving with  optim or maxlik.

import numpy as np
from scipy import optimize

# 生成一些模擬數據
np.random.seed(42)
true_mean = 0
true_std = 1
n_samples = 1000
data = np.random.normal(true_mean, true_std, n_samples)

def negative_log_likelihood(params, data):
    """計算負對數似然函數"""
    mu, sigma = params
    if sigma <= 0:  # 確保標準差為正
        return np.inf
    
    return -np.sum((-0.5 * np.log(2 * np.pi) - 
                    np.log(sigma) - 
                    (data - mu)**2 / (2 * sigma**2)))

# 初始猜測值
initial_guess = [0.1, 1.1]  # [mu, sigma]

# 使用scipy.optimize來最小化負對數似然函數
result = optimize.minimize(negative_log_likelihood, 
                         initial_guess,
                         args=(data,),
                         method='Nelder-Mead')

# 獲取估計結果
estimated_mean, estimated_std = result.x

print("6. Find the mean and variance given the data you found above \n using normal likelihood MLE and solving with  optim or maxlik. ")

print(f"真實均值: {true_mean:.4f}")
print(f"估計均值: {estimated_mean:.4f}")
print(f"真實標準差: {true_std:.4f}")
print(f"估計標準差: {estimated_std:.4f}")
print("\n最佳化結果:")
print(f"收斂狀態: {result.success}")
print(f"迭代次數: {result.nit}")
print("\n\n================================================================")





