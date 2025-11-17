import numpy as np

def almgren_chriss_schedule(Q, T, N, sigma, eta, gamma, risk_aversion=1e-6):
    dt = T / N
    alpha = risk_aversion

    k = alpha * sigma * sigma * dt / eta + 1e-12
    weights = np.exp(-k * np.arange(N))
    weights = weights / weights.sum()
    return Q * weights
