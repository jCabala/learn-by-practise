from pathlib import Path
import numpy as np
import math
import statistics
from scipy.stats import norm
import matplotlib.pyplot as plt

# Extracted vvar calculation into a reusable function
def compute_vvar(S0, Sig, r, miu, k1, k2, T, n=1000000, confidence=0.95, h=1, rng_seed=1):
	"""Compute loss scenarios and return VaR, ES and intermediate arrays.

	Returns a dict with keys: 'var', 'ES', 'vvar', 'v0', 'vt'.
	var: Value at Risk at the given confidence level
	ES: Expected Shortfall at the given confidence level
	vvar: array of loss scenarios
	v0: initial portfolio value
    vt: portfolio value at risk horizon
	"""
	# Option prices at time 0
	d1c = (math.log(S0 / k1) + (r + 0.5 * Sig**2) * T) / (Sig * T**0.5)
	d1p = (math.log(S0 / k2) + (r + 0.5 * Sig**2) * T) / (Sig * T**0.5)

	c0 = S0 * norm.cdf(d1c) - k1 * math.exp(-r * T) * norm.cdf(d1c - Sig * T**0.5)
	p0 = -S0 * norm.cdf(-d1p) + k2 * math.exp(-r * T) * norm.cdf(-d1p + Sig * T**0.5)

	# BEAR risk reversal: long put, short call
	# So portfolio value = put - call
	v0 = p0 - c0

	# computing the prices at time h
	T_remain = T - h

	# Generate Gaussian random values and the stock scenarios at time h
	np.random.seed(rng_seed)
	Zt = np.random.randn(n)
	St = S0 * np.exp((miu - 0.5 * Sig**2) * h) * np.exp(Zt * Sig * (h**0.5))

	# Generating call and put scenarios at time h
	d1cnew = (np.log(St / k1) + (r + 0.5 * Sig**2) * T_remain) / (Sig * (T_remain**0.5))
	d1pnew = (np.log(St / k2) + (r + 0.5 * Sig**2) * T_remain) / (Sig * (T_remain**0.5))

	ct = St * norm.cdf(d1cnew) - k1 * np.exp(-r * T_remain) * norm.cdf(d1cnew - Sig * (T_remain**0.5))
	pt = -St * norm.cdf(-d1pnew) + k2 * np.exp(-r * T_remain) * norm.cdf(-d1pnew + Sig * (T_remain**0.5))

	# BEAR risk reversal portfolio value at time h
	vt = pt - ct

	# Loss scenarios
	vvar = v0 - vt
	vvar = np.sort(vvar)

	# Extract VaR at the right confidence level
	ivar = int(confidence * n)
	var = vvar[ivar]

	# Calculating ES
	ESv = statistics.mean(vvar[int(math.floor(confidence * n)):])

	return {"var": var, "ES": ESv, "vvar": vvar, "v0": v0, "vt": vt}

# Initial parameters
# Stock data
S0 = 100
sigs = [math.trunc(i * 0.2 * 100) / 100 for i in range(1, 30)]  # volatilities to test
r = 0.05
miu = 0.1

# Options strikes and maturities
k1 = 105  # call strike
k2 = 95   # put strike
T = 10

# Number of scenarios, confidence level, risk horizon
n = 100000
confidence = 0.95
h = 1

vars = []
ess = []

dir = "./plots/"
if not Path("./plots/").exists():
    Path("./plots/").mkdir()
else:
    # Clear existing plots
    for file in Path(dir).glob("*.png"):
        file.unlink()

for sig in sigs:
    res = compute_vvar(S0, sig, r, miu, k1, k2, T, n=n, confidence=confidence, h=h, rng_seed=1)
    print("Sigma =", sig)
    print("VaR:", res["var"])
    print("ES:", res["ES"]) 
    print()

    vars.append(res["var"])
    ess.append(res["ES"])

    # Save histogram of loss scenarios
    plt.title(f"Histogram of Loss Scenarios (Volatility={sig})")
    plt.hist(res["vvar"], bins=100)

    # Add ES and VaR lines
    plt.axvline(res["var"], color='r', linestyle='dashed', linewidth=1, label='VaR')
    plt.axvline(res["ES"], color='g', linestyle='dashed', linewidth=1, label='ES')
    plt.legend()

    filename = f"vvar_histogram_{sig}.png"
    plt.savefig(Path.joinpath(Path(dir), filename))
    plt.close()

# Plot VaR and ES against volatility
plt.plot(sigs, vars, label="VaR", marker='o')
plt.plot(sigs, ess, label="ES", marker='o')
plt.xlabel("Volatility (Sigma)")
plt.ylabel("Risk Measure")
plt.title("VaR and ES vs Volatility for BEAR Risk Reversal")
plt.legend()
plt.grid()
plt.savefig(Path.joinpath(Path("./plots/"), "var_es_vs_volatility.png"))
plt.close()

# Tail depth calculation
tail_depths = [es - var for var, es in zip(vars, ess)]
plt.plot(sigs, tail_depths, label="Tail Depth", marker='o', color='purple')
plt.xlabel("Volatility (Sigma)")
plt.ylabel("Tail Depth (VaR - ES)")
plt.title("Tail Depth vs Volatility for BEAR Risk Reversal")
plt.legend()
plt.grid()
plt.savefig(Path.joinpath(Path("./plots/"), "tail_depth_vs_volatility.png"))
plt.close()