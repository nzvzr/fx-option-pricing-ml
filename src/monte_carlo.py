import numpy as np

def mc_fx_price(S, K, T, rd, rf, sigma, option="call", n_paths=200000, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S * np.exp((rd - rf - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

    payoff = np.maximum(ST - K, 0.0) if option.lower()=="call" else np.maximum(K - ST, 0.0)
    disc = np.exp(-rd*T)

    price = disc * payoff.mean()
    se = disc * payoff.std(ddof=1) / np.sqrt(n_paths)
    ci = (price - 1.96*se, price + 1.96*se)

    return price, ci, ST