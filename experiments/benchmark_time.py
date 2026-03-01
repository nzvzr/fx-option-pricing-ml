import time
import numpy as np
from src.bs_fx import garman_kohlhagen_call
from src.monte_carlo import monte_carlo_call


S = 1.1
K = 1.1
T = 1
rd = 0.02
rf = 0.01
sigma = 0.03

# Black–Scholes timing
start = time.time()
for _ in range(10000):
    garman_kohlhagen_call(S, K, T, rd, rf, sigma)
bs_time = time.time() - start

# Monte Carlo timing
start = time.time()
for _ in range(100):
    monte_carlo_call(S, K, T, rd, rf, sigma, 100000)
mc_time = time.time() - start

print("BS time:", bs_time)
print("Monte Carlo time:", mc_time)