
""" Historical-simulation VaR and Expected Shortfall for the portfolio"""
import pandas as pd
import numpy as np

def historical_var_es(r :pd.Series | np.ndarray, confidence : float) -> tuple[float, float]:

    """ Return "(VaR, ES) as positive decimal losses at given confidence level """
    r = np.asarray(r) # extending to numpy native
    threshold = np.quantile(r, 1 - confidence)
    var = -threshold #converting to a positive loss
    es = -r[r <= threshold].mean() #average of the losses that are beyond threshold

    return var, es

def historical_from_returns(r, weights, confidence):
    """ Wrapper to match signature of other var,es calculation methods """
    series = r[list(weights)] @ np.array([weights[t] for t in weights])
    return historical_var_es(series, confidence)


