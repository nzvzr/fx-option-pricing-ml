import numpy as np
from scipy.stats import norm

def _d1_d2(S, K, T, rd, rf, sigma):
    eps = 1e-12
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    T = np.maximum(np.asarray(T, float), eps)
    sigma = np.maximum(np.asarray(sigma, float), eps)
    vol = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma**2) * T) / vol
    d2 = d1 - vol
    return d1, d2

def garman_kohlhagen_price(S, K, T, rd, rf, sigma, option="call"):
    d1, d2 = _d1_d2(S, K, T, rd, rf, sigma)
    df_d = np.exp(-rd * T)
    df_f = np.exp(-rf * T)
    if option.lower() == "call":
        return S * df_f * norm.cdf(d1) - K * df_d * norm.cdf(d2)
    elif option.lower() == "put":
        return K * df_d * norm.cdf(-d2) - S * df_f * norm.cdf(-d1)
    else:
        raise ValueError("option must be 'call' or 'put'")

def delta(S, K, T, rd, rf, sigma, option="call"):
    d1, _ = _d1_d2(S, K, T, rd, rf, sigma)
    df_f = np.exp(-rf * T)
    if option.lower() == "call":
        return df_f * norm.cdf(d1)
    return -df_f * norm.cdf(-d1)

def vega(S, K, T, rd, rf, sigma):
    d1, _ = _d1_d2(S, K, T, rd, rf, sigma)
    df_f = np.exp(-rf * T)
    return S * df_f * norm.pdf(d1) * np.sqrt(T)