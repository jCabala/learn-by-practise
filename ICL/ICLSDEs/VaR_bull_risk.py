import numpy as np
import math
import statistics
from scipy.stats import norm
import matplotlib.pyplot as plt

# Stock data
S0 = 100
Sig = 0.2
r = 0.01
miu = 0.05

# Options strikes and maturities
k1 = 110
k2 = 90
T = 5

# Number of scenarios, confidence level, risk horizon
n = 1000000
confidence = 0.95
h = 1

# Option prices at time 0
d1c = (math.log(S0 / k1) + (r + 0.5 * Sig**2) * T) / (Sig * T**0.5)
d1p = (math.log(S0 / k2) + (r + 0.5 * Sig**2) * T) / (Sig * T**0.5)

c0 = S0 * norm.cdf(d1c) - k1 * math.exp(-r * T) * norm.cdf(d1c - Sig * T**0.5)
p0 = -S0 * norm.cdf(-d1p) + k2 * math.exp(-r * T) * norm.cdf(-d1p + Sig * T**0.5)

v0 = c0 - p0

# computing the prices at time h
T = T - h

Zt = np.zeros(n)
St = np.zeros(n)

from random import seed
seed(1)

# Generate Gaussian random values and the stock scenarios at time h
Zt = np.random.randn(n)
St = S0 * np.exp((miu - 0.5 * Sig**2) * h) * np.exp(Zt * Sig * (h**0.5))

# Generating call and put scenarios at time h
d1cnew = (np.log(St / k1) + (r + 0.5 * Sig**2) * T) / (Sig * (T**0.5))
d1pnew = (np.log(St / k2) + (r + 0.5 * Sig**2) * T) / (Sig * (T**0.5))

ct = St * norm.cdf(d1cnew) - k1 * np.exp(-r * T) * norm.cdf(d1cnew - Sig * (T**0.5))
pt = -St * norm.cdf(-d1pnew) + k2 * np.exp(-r * T) * norm.cdf(-d1pnew + Sig * (T**0.5))

# Final portfolio value in all scenarios
vt = ct - pt

# Loss scenarios
vvar = v0 - vt
vvar = np.sort(vvar)

# Extract VaR at the right confidence level
ivar = int(confidence * n)
var = vvar[ivar]

# Calculating ES
ESv = statistics.mean(vvar[int(math.floor(confidence * n)):])

print("VaR:", var)
print("ES:", ESv)

# Plotting loss histogram
plt.hist(vvar, bins=100)
plt.show()
