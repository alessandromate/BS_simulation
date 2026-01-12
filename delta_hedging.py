import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pricing import BlackScholesModel, generate_gbm_paths

def delta_hedging_simulation(S0=100.0, K_ratio=1.0, T_years=0.5, r=0.045, sigma=0.35):
    #Dynamically simulates the delta hedging process for a sold European Call option
    K = S0 * K_ratio
    HEDGES_PER_DAY = 100                     ###
    dt = 1 / (252 * HEDGES_PER_DAY)
    T = T_years
    N_steps = int(T / dt)
    paths = generate_gbm_paths(S0, r, sigma, T, dt, n_sims=1)
    S_path = paths[:, 0]
    time_points = np.linspace(0, T, N_steps + 1)
    bs_t0 = BlackScholesModel(S0, K, T, r, sigma)
    C_0 = bs_t0.call_price()
    Delta_0 = bs_t0.delta_call()
    initial_capital = 0                      ###
    Cash_t = C_0 - Delta_0 * S0
    shares_held = Delta_0
    cash_account = initial_capital + C_0 - (Delta_0 * S0)
    shares_held = Delta_0
    Pi_0 = cash_account + (shares_held * S0) - C_0
    portfolio_values = [Pi_0]
    delta_values = [Delta_0]
    # loop for rehedging
    for i in range(1, N_steps + 1):
        t_curr = time_points[i]
        S_curr = S_path[i]
        time_to_maturity = T - t_curr
        cash_account += cash_account * r * dt
        #closing and liquidating the positions
        if time_to_maturity <= dt / 2:
            option_payoff = np.maximum(S_curr - K, 0)
            cash_account += shares_held * S_curr
            shares_held = 0
            Pi_curr = cash_account - option_payoff
            Delta_curr = 0.0
        else:
            bs_curr = BlackScholesModel(S_curr, K, time_to_maturity, r, sigma)
            Delta_curr = bs_curr.delta_call()
            C_curr = bs_curr.call_price()
            shares_change = Delta_curr - shares_held
            cash_account -= shares_change * S_curr
            shares_held = Delta_curr
            Pi_curr = cash_account + shares_held * S_curr - C_curr
        portfolio_values.append(Pi_curr)
        delta_values.append(Delta_curr)
    # Console output
    Pi_T = portfolio_values[-1]
    expected_Pi_T = Pi_0 * np.exp(r * T)
    print(f"1. Final Value Actual (Pi_T): {Pi_T:.6f}")
    print(f"2. Final Value Expected (Pi_0 * exp(rT)): {expected_Pi_T:.6f}")
    print(f"3. Error (Pi_T - Pi_0 * exp(rT)):   {Pi_T - expected_Pi_T:.6f}")
    # Plotting
    df = pd.DataFrame({
        'Time': time_points,
        'Portfolio_Value': portfolio_values,
        'Stock_Price': S_path,
        'Delta_Value': delta_values
    })
    expected_growth = Pi_0 * np.exp(r * df['Time'])
    plt.figure(figsize=(12, 8))
    # portfolio value graph
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df['Time'], df['Portfolio_Value'], label=r'Actual Portfolio Value ($\Pi_t$)', color='blue', linewidth=2)
    ax1.plot(df['Time'], expected_growth, label=rf'Expected Risk-Free Growth ($\Pi_0 e^{{rt}}$)',linestyle='--', color='red')
    ax1.set_title(rf'Dynamic Delta Hedging: P&L - K={K:.2f}, $\sigma$={sigma * 100:.0f}%')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    #  Stock price and Delta graph
    ax2 = plt.subplot(2, 1, 2)
    color_S = 'gray'
    ax2.plot(df['Time'], df['Stock_Price'], label='Stock Price (S)', color=color_S, alpha=0.7)
    ax2.axhline(K, color='k', linestyle=':', label='Strike Price (K)')
    ax2.set_ylabel('Stock Price ($)', color=color_S)
    ax2.tick_params(axis='y', labelcolor=color_S)
    ax2.set_xlabel('Time (Years)')
    ax2_D = ax2.twinx()
    color_D = 'green'
    ax2_D.plot(df['Time'], df['Delta_Value'], label=r'Hedge Delta ($\Delta$)', color=color_D, linestyle='-', linewidth=2)
    ax2_D.set_ylabel(r'Hedge Delta ($\Delta$)', color=color_D)
    ax2_D.tick_params(axis='y', labelcolor=color_D)
    ax2_D.set_ylim(0, 1)
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_D.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    delta_hedging_simulation(S0=100.0, K_ratio=1.0, T_years=0.5, r=0.045, sigma=0.35)

