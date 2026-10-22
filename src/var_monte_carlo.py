import numpy as np
import pandas as pd


from portfolio import get_varcovar
from var_historical import historical_var_es

def t_samples(N : int, k : int, rng : np.random.default_rng, nu : float):
    """ Producing an N,k np array of samples from a students t distribution , normalised to give sd = 1 """
    Z = rng.standard_normal((N,k))
    chi = rng.chisquare(nu, N)
    turbulence = np.sqrt(nu/chi)[:, None]
    return Z * np.sqrt( (nu - 2)/ nu) * turbulence

def normal_samples(N : int, k : int, rng : np.random.default_rng):
    """ Produces an N,k np array of independent normal samples """
    return rng.standard_normal((N, k))

def simulate_returns(r : pd.DataFrame, var_covar : pd.DataFrame, weights : dict, N : int = 10000, seed : int = 42, sample_strategy = normal_samples)-> np.ndarray:

    """ Simulating N rows of returns data, one for each equity in the weights portfolio"""

    rng = np.random.default_rng(seed)
    tickers = list(weights.keys())
    r = r[tickers] # ensuring ordering is preserved
    assert list(var_covar.index) == tickers

    L = np.linalg.cholesky(var_covar)
    Z = sample_strategy(N, len(weights), rng)
    mu = r.mean().to_numpy()

    simulated_returns = mu + Z @ L.T
    return simulated_returns

def simulate_portfolio(r : pd.DataFrame, var_covar : pd.DataFrame, weights : dict, N : int = 10000, seed : int = 42, sample_strategy = normal_samples) -> np.ndarray:
    """ Simulating N rows of returns data, with each equity weighted according to weights """
    simulated_returns = simulate_returns(r, var_covar, weights, N, seed, sample_strategy=sample_strategy)
    tickers = list(weights.keys())

    w = np.array([weights[t] for t in tickers])
    simulated_portfolio_returns = simulated_returns @ w

    return simulated_portfolio_returns

def mc_var_es(data : pd.DataFrame, weights, CL, N = 1_000_000, seed = 42, sample_strategy = normal_samples):
    vcov = get_varcovar(data, list(weights))
    sims = simulate_portfolio(data, vcov, weights, N=N, seed=seed, sample_strategy = sample_strategy)
    return historical_var_es(sims, CL)

