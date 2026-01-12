import numpy as np
from scipy.stats import norm

class BlackScholesModel:
    #Black-Scholes Model for European Option Pricing and Greeks
    def __init__(self, S0, K, T, r, sigma):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    #as explained in 'concepts'
    def _d1(self):
        return (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    # as explained in 'concepts'
    def _d2(self):
        return self._d1() - self.sigma * np.sqrt(self.T)
    #Call price (C)
    def call_price(self):
        d1 = self._d1()
        d2 = self._d2()
        return self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
    #dC/dS
    def delta_call(self):
        return norm.cdf(self._d1())
    # d^2(C)/d(S^2) = d(delta)/dS
    def gamma(self):
        return norm.pdf(self._d1()) / (self.S0 * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S0 * norm.pdf(self._d1()) * np.sqrt(self.T)
    #Calculates IV using Newton-Raphson method
    @staticmethod
    def implied_volatility(market_price, S0, K, T, r, tol=1e-5, max_iter=100):
        sigma = 0.5
        for i in range(max_iter):
            bs = BlackScholesModel(S0, K, T, r, sigma)
            price = bs.call_price()
            diff = market_price - price
            vega = bs.vega()
            if abs(diff) < tol:
                return sigma
            if vega == 0:
                break
            sigma = sigma + diff / vega
        return None
#Generating GBM paths using the exact solution (vectorized)."""
def generate_gbm_paths(S0, mu, sigma, T, dt, n_sims):
    N_steps = int(T / dt)
    Z = np.random.normal(0, 1, (N_steps, n_sims))
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion
    log_paths = np.cumsum(log_returns, axis=0)
    log_paths = np.vstack([np.zeros((1, n_sims)), log_paths])
    return S0 * np.exp(log_paths)

