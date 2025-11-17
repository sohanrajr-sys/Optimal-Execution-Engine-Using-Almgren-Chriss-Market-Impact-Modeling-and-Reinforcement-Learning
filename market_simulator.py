import math
import numpy as np

class MarketSimulator:
    """
    Simulates market mid-prices (GBM) + OU short-term signal.
    Includes temporary & permanent market impact.
    """

    def __init__(self,
                 init_price=100.0,
                 mu=0.0,
                 sigma=0.001,
                 ou_theta=5.0,
                 ou_sigma=0.002,
                 dt=1/390,
                 temporary_impact=1e-4,
                 permanent_impact=1e-5,
                 spread=0.01,
                 seed=None):

        self.init_price = init_price
        self.mu = mu
        self.sigma = sigma
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.dt = dt
        self.tmp = temporary_impact
        self.perm = permanent_impact
        self.spread = spread
        self.rng = np.random.RandomState(seed)

    def simulate_path(self, T_steps, actions=None):
        prices = np.zeros(T_steps + 1)
        ou = np.zeros(T_steps + 1)

        prices[0] = self.init_price

        for t in range(T_steps):
            # OU signal
            ou[t+1] = ou[t] + self.ou_theta * (-ou[t]) * self.dt + \
                      self.ou_sigma * math.sqrt(self.dt) * self.rng.randn()

            # GBM mid-price
            dW = math.sqrt(self.dt) * self.rng.randn()
            prices[t+1] = prices[t] * (1 + self.mu * self.dt + self.sigma * dW)

        result = {"prices": prices, "ou": ou}

        if actions is None:
            return result

        exec_prices = np.zeros(T_steps)
        cash_flow, shares, cum_volume = 0.0, 0.0, 0.0

        for t in range(T_steps):
            v = actions[t]

            # impacts
            perm_shift = self.perm * v
            tmp_shift = self.tmp * v
            half_spread = 0.5 * self.spread * (1 + 0.1 * self.rng.randn())

            exec_price = prices[t] + perm_shift + tmp_shift + half_spread

            cash_flow -= exec_price * v
            shares += v
            cum_volume += v
            exec_prices[t] = exec_price

        final_value = cash_flow + shares * prices[-1]
        realized_cost = -final_value

        result.update({
            "exec_prices": exec_prices,
            "final_value": final_value,
            "realized_cost": realized_cost
        })

        return result
