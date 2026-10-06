""" Build portfolio dataclass and get all attributes """

import pandas as pd
import numpy as np

from dataclasses import dataclass

from config import DATA_DIR, WEIGHTS, TICKERS


def load_returns() -> pd.DataFrame:
    prices = pd.read_csv(DATA_DIR / "prices.csv", index_col = "Date", parse_dates= True)

    returns  = prices.pct_change().dropna()
    return returns[TICKERS]

def equities_returns(weights = WEIGHTS) -> pd.DataFrame:
    prices = pd.read_csv(DATA_DIR / "prices.csv", index_col = "Date", parse_dates= True)
    returns  = prices.pct_change().dropna()
    returns = returns[list(weights.keys())]
    return returns

def portfolio_returns(returns: pd.DataFrame = None) -> pd.Series:
    if returns is None:
        returns = load_returns()
    w = pd.Series(WEIGHTS)
    return returns[w.index] @ w

def get_varcovar(r : pd.DataFrame, tickers : list) -> pd.DataFrame:
    return r[tickers].cov()

def get_sigma_p(varcovar : pd.DataFrame, weights : dict) -> float:
    """ return the portfolio returns standard deviation """
    w = pd.Series(weights)

    sigma_p = np.sqrt(w.T @ varcovar @ w)
    return sigma_p

def get_mu_p(r : pd.DataFrame, weights : dict) -> float:
    """ return portfolio mean daily returns """
    w = pd.Series(weights)
    mu_p =  r[w.index].mean() @ w
    return mu_p


@dataclass(frozen=True)
class Portfolio:
    tickers : list[str] ## Portfolio tickers only
    returns : pd.DataFrame ## Per equity daily returns data
    portfolio_returns : pd.Series ## Weighted daily portfolio returns
    weights : np.ndarray ## Portfolio equity weights
    cov : np.ndarray ## Variance-Covariance matrix
    mu : np.ndarray ## Average faily returns per equity

    sigma_p : float ## Average portfolio std dev
    mu_p : float ## Average portfolio daily returns

    @classmethod
    def from_returns(cls, returns, weights = WEIGHTS):
        
        tickers_ = list(weights)
        weights_ = np.array([weights[t] for t in tickers_])    

        equity_returns_ = returns[tickers_]
        portfolio_returns_ = portfolio_returns(equity_returns_)

        cov_ = get_varcovar(equity_returns_, tickers = tickers_).to_numpy()
        mu_ = equity_returns_.mean().to_numpy()

        mu_p_ = float(get_mu_p(equity_returns_, weights))
        
        sigma_p_ = float(np.sqrt(weights_ @ cov_ @ weights_))

        return cls(tickers= tickers_, returns = equity_returns_, portfolio_returns = portfolio_returns_,
                   weights = weights_, cov = cov_, mu = mu_, sigma_p = sigma_p_, mu_p = mu_p_)
