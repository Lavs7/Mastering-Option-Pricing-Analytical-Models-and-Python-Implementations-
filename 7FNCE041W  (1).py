#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import norm


# In[3]:


# Parameters for the option
S0 = 100  # Initial stock price
K = 105   # Strike price
T = 1     # Time to maturity (in years)
r = 0.03  # Risk-free rate
sigma = 0.25  # Volatility


# In[5]:


def bsm_call_price(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


# In[7]:


# Calculate BSM price
bsm_price = bsm_call_price(100, 105, 1, 0.03, 0.25)
print(f"BSM Call Option Price: ${bsm_price:.4f}")


# In[9]:


import numpy as np


# In[11]:


# Parameters for the option
S0 = 100  # Initial stock price
K = 105   # Strike price
T = 1     # Time to maturity (in years)
r = 0.03  # Risk-free rate
sigma = 0.25  # Volatility
num_simulations = 1000000 # num_simulations


# In[13]:


def monte_carlo_call_price(S0, K, T, r, sigma, num_simulations):
    Z = np.random.standard_normal(num_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price


# In[15]:


# Calculate Monte Carlo price
np.random.seed(42)  # for reproducibility
mc_price = monte_carlo_call_price(100, 105, 1, 0.03, 0.25, 1000000)
print(f"Monte Carlo Call Option Price: ${mc_price:.4f}")


# In[17]:


bsm_price = bsm_call_price(S0, K, T, r, sigma)
mc_price = monte_carlo_call_price(S0, K, T, r, sigma, num_simulations)

print(f"BSM Call Option Price: {bsm_price:.4f}")
print(f"Monte Carlo Call Option Price: {mc_price:.4f}")


# In[19]:


print(f"BSM Call Option Price: ${bsm_price:.4f}")
print(f"Monte Carlo Call Option Price: ${mc_price:.4f}")
print(f"Absolute Difference: ${abs(bsm_price - mc_price):.4f}")
print(f"Relative Difference: {abs(bsm_price - mc_price) / bsm_price * 100:.4f}%")


# In[21]:


import numpy as np
from scipy.stats import norm


# In[23]:


# Step 1: Define the Black-Scholes formula for a European call option
def black_scholes_call(S, K, r, T, sigma):
    """
    Calculate the theoretical price of a European call option using Black-Scholes formula.
    S: Current stock price
    K: Strike price
    r: Risk-free rate
    T: Time to maturity (in years)
    sigma: Volatility
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


# In[25]:


# Step 2: Define Vega (the derivative of the option price with respect to volatility)
def vega(S, K, r, T, sigma):
    """
    Calculate Vega (sensitivity of the option price to changes in volatility).
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)


# In[27]:


# Step 3: Implement the Newton-Raphson method to estimate implied volatility
def implied_volatility(S, K, r, T, market_price, initial_guess=0.2, tolerance=1e-6, max_iterations=100):
    """
    Estimate the implied volatility using the Newton-Raphson method.
    S: Current stock price
    K: Strike price
    r: Risk-free rate
    T: Time to maturity (in years)
    market_price: Observed market price of the option
    initial_guess: Initial guess for the volatility (default is 0.2)
    tolerance: Tolerance for the solution (default is 1e-6)
    max_iterations: Maximum number of iterations (default is 100)
    """
    sigma = initial_guess
    for i in range(max_iterations):
        theoretical_price = black_scholes_call(S, K, r, T, sigma)
        vega_value = vega(S, K, r, T, sigma)
        
        # Update sigma using Newton-Raphson formula
        price_diff = theoretical_price - market_price
        if abs(price_diff) < tolerance:
            return sigma
        
        sigma -= price_diff / vega_value
    
    # If the loop completes without finding a solution
    raise ValueError("Implied volatility did not converge after maximum iterations")


# In[29]:


# Step 4: Example Usage
# Parameters
S = 100  # Current stock price
K = 100  # Strike price
r = 0.05  # Risk-free rate (5%)
T = 1     # Time to maturity (1 year)
market_price = 10  # Observed market price of the call option

# Estimate implied volatility
try:
    iv = implied_volatility(S, K, r, T, market_price)
    print(f"Estimated Implied Volatility: {iv:.4f}")
except ValueError as e:
    print(e)


# In[31]:


pip install yfinance


# In[33]:


import yfinance as yf
import pandas as pd


# In[39]:


# Step 1: Download Tesla data
ticker = "TSLA"
tsla = yf.Ticker(ticker)


# In[41]:


# Step 2: Get Tesla's spot price
spot_price = tsla.history(period="1d")["Close"].iloc[-1]
print(f"Spot Price (S): ${spot_price:.2f}")


# In[43]:


# Step 3: Fetch options data
expiry_date = "2025-01-17"
options = tsla.option_chain(date=expiry_date)


# In[45]:


# Extract relevant put options
puts = options.puts
strike_140 = puts[puts['strike'] == 140]
strike_340 = puts[puts['strike'] == 340]


# In[47]:


put_price_140 = strike_140['lastPrice'].values[0]
put_price_340 = strike_340['lastPrice'].values[0]


# In[49]:


print(f"Put Price for K=140: ${put_price_140:.2f}")
print(f"Put Price for K=340: ${put_price_340:.2f}")


# In[51]:


import numpy as np
from scipy.stats import norm


# In[53]:


def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# In[55]:


def vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)


# In[57]:


def implied_volatility_put(P, S, K, T, r, tol=1e-5, max_iter=100):
    sigma = 0.5  # Initial guess
    for i in range(max_iter):
        price = black_scholes_put(S, K, T, r, sigma)
        diff = P - price
        if abs(diff) < tol:
            return sigma
        sigma = sigma + diff / vega(S, K, T, r, sigma)
    return sigma


# In[59]:


# Parameters
S = 416.5908
T = 1.1
r = 0.05


# In[61]:


# Case 1: K = $140
K1 = 140
P1 = 2.50
iv1 = implied_volatility_put(P1, S, K1, T, r)
print(f"Implied Volatility (K=$140): {iv1:.4f}")


# In[63]:


# Case 2: K = $340
K2 = 340
P2 = 25.00
iv2 = implied_volatility_put(P2, S, K2, T, r)
print(f"Implied Volatility (K=$340): {iv2:.4f}")


# In[65]:


pip install yfinance


# In[67]:


import numpy as np

# Extract closing prices from the data
prices = [419.12, 419.77, 419.55, 418.29, 416.625, 416.37, 416.0928, 416.7399, 416.88, 417.13, 415.17, 414.83, 414.72, 414.27, 413.5389, 413.5, 412.9001, 411.6921, 411.94, 413.44, 413.795, 413.13, 412.6715, 411.126, 410.9722, 410.98, 410.145, 408.2916, 407.2299, 404.08, 406.7987, 407.75]


# In[69]:


# Calculate daily returns
returns = np.diff(np.log(prices))


# In[71]:


# Calculate annualized volatility
annual_volatility = np.std(returns) * np.sqrt(252)  # Assuming 252 trading days in a year


# In[73]:


print(f"Annual Historical Volatility: {annual_volatility:.4f}")


# In[75]:


import numpy as np

# Step 1: Define the parameters
S0 = 100         # Initial stock price
K = 100          # Strike price
r = 0.05         # Risk-free rate (5%)
sigma = 0.20     # Volatility (20%)
T = 1            # Time to maturity (1 year)
n = 3            # Number of steps (3 periods)
dt = T / n       # Length of each period


# In[77]:


# Step 2: Calculate the up factor (u), down factor (d), and risk-neutral probability (p)
u = np.exp(sigma * np.sqrt(dt))  # Up factor
d = 1 / u                        # Down factor
p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability


# In[79]:


# Step 3: Create the binomial tree for stock prices
def generate_stock_tree(S0, u, d, n):
    stock_tree = np.zeros((n+1, n+1))  # Initialize a 2D array for the stock tree
    stock_tree[0, 0] = S0  # Set the initial stock price

    # Fill the tree with the stock prices
    for i in range(1, n+1):
        for j in range(i+1):
            stock_tree[j, i] = S0 * (u**(i-j)) * (d**j)
    return stock_tree


# In[81]:


# Step 4: Calculate the option payoff at maturity (time = 3)
def calculate_payoffs(stock_tree, K):
    n = len(stock_tree) - 1
    payoff_tree = np.maximum(stock_tree[:, n] - K, 0)  # Payoff at maturity for American call
    return payoff_tree


# In[83]:


# Step 5: Perform backward induction to calculate option values
def backward_induction(stock_tree, payoffs, u, d, p, r, dt):
    n = len(stock_tree) - 1
    # Discount factor for backward induction
    discount_factor = np.exp(-r * dt)

    # Initialize the option value tree (with the same shape as stock_tree)
    option_tree = np.zeros_like(stock_tree)

    # Set the option values at maturity (final column)
    option_tree[:, n] = payoffs

    # Perform backward induction
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            # Risk-neutral expected value for option at node (i,j)
            expected_value = p * option_tree[j, i+1] + (1 - p) * option_tree[j+1, i+1]
            # American option value: max between early exercise and holding the option
            option_tree[j, i] = np.maximum(stock_tree[j, i] - K, discount_factor * expected_value)

    return option_tree


# In[85]:


# Step 6: Calculate the final option value at the root (time 0)
stock_tree = generate_stock_tree(S0, u, d, n)  # Generate the stock price tree
payoffs = calculate_payoffs(stock_tree, K)     # Calculate the payoffs at maturity
option_tree = backward_induction(stock_tree, payoffs, u, d, p, r, dt)  # Perform backward induction


# In[87]:


# Step 7: Output the final results
print("Stock Price Tree:")
print(stock_tree)

print("\nOption Value Tree:")
print(option_tree)

# The option value at the root (time 0) is the value of the American call option
print(f"\nThe value of the American call option is: {option_tree[0, 0]:.2f}")


# In[91]:


import numpy as np
from scipy.stats import norm


# In[93]:


def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# In[95]:


def vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)


# In[97]:


def implied_volatility_put(P, S, K, T, r, tol=1e-5, max_iter=100):
    sigma = 0.5  # Initial guess
    for i in range(max_iter):
        price = black_scholes_put(S, K, T, r, sigma)
        diff = P - price
        if abs(diff) < tol:
            return sigma
        sigma = sigma + diff / vega(S, K, T, r, sigma)
    return sigma


# In[99]:


# Parameters
S = 424.77
T = 1.1
r = 0.05


# In[101]:


# Case 1: K = $140
K1 = 140
P1 = 2.50
iv1 = implied_volatility_put(P1, S, K1, T, r)
print(f"Implied Volatility (K=$140): {iv1:.4f}")


# In[103]:


# Case 2: K = $340
K2 = 340
P2 = 25.00
iv2 = implied_volatility_put(P2, S, K2, T, r)
print(f"Implied Volatility (K=$340): {iv2:.4f}")


# In[105]:


from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np


# In[107]:


# Input parameters
S = 424.77  # Spot price
K1, K2 = 140, 340  # Strike prices
P1, P2 = 274.17, 91.25  # Put option prices
r = 0.05  # Risk-free rate
T = 2  # Time to expiry in years


# In[109]:


# Black-Scholes model for put option price
def put_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# In[111]:


# Objective function to minimize (difference between model and market price)
def objective(sigma, S, K, T, r, P_market):
    P_model = put_price(S, K, T, r, sigma)
    return (P_model - P_market) ** 2


# In[113]:


# Solve for implied volatility
def implied_volatility(S, K, T, r, P_market):
    result = minimize(objective, 0.2, args=(S, K, T, r, P_market), bounds=[(0.001, 5)])
    return result.x[0]


# In[115]:


# Calculate implied volatilities for both strike prices
iv1 = implied_volatility(S, K1, T, r, P1)
iv2 = implied_volatility(S, K2, T, r, P2)

iv1, iv2


# In[117]:


pip install yfinance


# In[119]:


import yfinance as yf
import numpy as np


# In[121]:


# Download Tesla's historical stock data for the past year
tesla_data = yf.download("TSLA", start="2022-12-01", end="2023-12-01")


# In[123]:


# Calculate daily log returns
tesla_data['Log Returns'] = np.log(tesla_data['Adj Close'] / tesla_data['Adj Close'].shift(1))


# In[125]:


# Calculate standard deviation of daily returns (excluding NaN values)
daily_volatility = tesla_data['Log Returns'].std()


# In[127]:


# Annualize the daily volatility
annual_volatility = daily_volatility * np.sqrt(252)


# In[129]:


print(f"Annual Historical Volatility: {annual_volatility}")


# In[ ]:




