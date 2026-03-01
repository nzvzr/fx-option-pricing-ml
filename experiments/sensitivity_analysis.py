import numpy as np
import matplotlib.pyplot as plt
from src.bs_fx import garman_kohlhagen_call


S_values = np.linspace(0.8, 1.3, 100)
K = 1.1
T = 1
rd = 0.02
rf = 0.01
sigma = 0.03

prices = [garman_kohlhagen_call(S, K, T, rd, rf, sigma)
          for S in S_values]

plt.plot(S_values, prices)
plt.title("Sensitivity to Spot FX")
plt.xlabel("Spot")
plt.ylabel("Call Price")
plt.show()