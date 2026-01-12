import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from pricing import BlackScholesModel

def plot_surface(ticker="MSFT", risk_free_rate=0.045):
    stock = yf.Ticker(ticker)
    try:
        S0 = stock.history(period="1d")['Close'].iloc[-1]
    except IndexError:
        print(f"Error: Ticker not found {ticker}.")
        return
    expirations = stock.options
    options_data = []
    for expr_date in expirations[:4]:
        calls = stock.option_chain(expr_date).calls
        T = (date.fromisoformat(expr_date) - date.today()).days / 365.0
        # Filtering (default: near the money options (ATM))
        calls = calls[(calls['strike'] > S0 * 0.9) & (calls['strike'] < S0 * 1.1)]
        for _, row in calls.iterrows():
            K = row['strike']
            market_price = (row['bid'] + row['ask']) / 2
            if market_price <= 0: continue
            # IV Calculation
            iv = BlackScholesModel.implied_volatility(market_price, S0, K, T, risk_free_rate)
            if iv and iv < 2.0:
                options_data.append({'Strike': K, 'T': T, 'IV': iv})
    if not options_data:
        print("Valid option data not found for the surface.")
        return
    df = pd.DataFrame(options_data)

    #Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(df['Strike'], df['T'], df['IV'], cmap='viridis', linewidth=0.2)
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Time to Maturity (T)')
    ax.set_zlabel('Implied Volatility (IV)')
    ax.set_title(f'Volatility Surface - {ticker}')
    plt.show()

if __name__ == "__main__":
    plot_surface()