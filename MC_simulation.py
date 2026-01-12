import numpy as np
import matplotlib.pyplot as plt
from pricing import BlackScholesModel, generate_gbm_paths


def run_simulation():
    S0 = 100.0;  # spot price
    K = 100.0;  # strike price
    T = 1.0;  # time to expiration
    r = 0.05;  # risk-free rate
    sigma = 0.20  # volatility  (dev standard)
    n_sims = 50000;  # number of simulations
    dt = 1 / 252  # infinitesimal time

    # (1) - Analytical price calculated with the resolution of the PDE
    bs = BlackScholesModel(S0, K, T, r, sigma)
    analytical_price = bs.call_price()

    # Generation of all the MC possible paths
    paths = generate_gbm_paths(S0, r, sigma, T, dt, n_sims)

    # Mean of all the paths
    mean_path = np.mean(paths, axis=1)

    # (2) - MC price taking the mean of the payoffs
    S_T = paths[-1]
    payoffs = np.maximum(S_T - K, 0)
    mc_price = np.mean(payoffs) * np.exp(-r * T)

    # Console output
    print(f"(1) analytical price : {analytical_price:.4f}")
    print(f"(2) MC price:    {mc_price:.4f}")
    print(f"Convergence error: {abs(analytical_price - mc_price):.4f}")

    # Plotting the graph
    N_steps = paths.shape[0]
    time_index = np.linspace(0, T * 252, N_steps)
    plt.figure(figsize=(10, 6))
    plt.plot(time_index, paths[:, :15], alpha=0.3, color='gray')
    plt.plot(time_index, mean_path,
             label='Expected value of the Asset',
             color='red',
             linewidth=3)
    plt.xlabel("Trading Days")
    plt.ylabel("Asset Price")
    plt.axhline(y=K, color='k', linestyle='--', label='Strike Price (K)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    run_simulation()