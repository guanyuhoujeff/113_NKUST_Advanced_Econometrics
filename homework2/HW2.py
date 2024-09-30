import yfinance as yf
from scipy.stats import norm
import numpy as np

# 1: data=[2 5 7 9 10], find E(data) and Var(data)
data = np.array([2, 5, 7, 9, 10])

# Calculate mean and variance
mean_data = np.mean(data)
var_data = np.var(data)

print("1: data=[2 5 7 9 10], find E(data) and Var(data)")
print()
print("Mean of data:", mean_data)
print("Variance of data:", var_data)
print("==================================================================")



# 2. Download TAIEX stock index daily prices and compute returns¶
ticker = "^TWII"  # TAIEX stock index ticker
print("start download TWII data : ")
taiex = yf.download(ticker, period="1y")  # Download data for 1 year
taiex['Returns'] = taiex['Adj Close'].pct_change().dropna()
# Define RR (Returns)
RR = taiex['Returns'].dropna()

# (1)Compute m=E(RR) and v=Var(RR), assuming normal distribution.
mean_RR = RR.mean()
var_RR = RR.var()
print()
print()
print("2.1  Compute m=E(RR) and v=Var(RR), assuming normal distribution.")
print("Mean of RR:", mean_RR)
print("Variance of RR:", var_RR)
print("==================================================================")
print()
print()


# (2) Compute the probability that RR > 0 assuming normal distribution
prob = 1 - norm.cdf(0, loc=mean_RR, scale=np.sqrt(var_RR))
print("2.2  Compute the probability that RR > 0 assuming normal distribution")
print("Probability that RR > 0:", prob)
print("==================================================================")