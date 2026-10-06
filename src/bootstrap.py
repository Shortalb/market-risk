import numpy as np
import pandas as pd
from collections.abc import Callable, Iterator

def resample_indices(n : int, m : int, rng, block = 1) -> Iterator[np.ndarray]:
    """ Yield m arrays of n row positions 
    block 1 = iid bootstrap (single rows)
    block > 1 = moving-block bootstrap - not yet implemented """

    if block == 1:
        for _ in range(m):
            yield rng.integers(0, n ,n)
    else:
        k = int(np.ceil(n / block))
        for _ in range(m):
            starts = rng.integers(0, n - block + 1, k)
            yield np.concatenate([np.arange(s , s + block) for s in starts])[:n]
            

def bootstrap(data, statistic : Callable, m : int = 2000, seed : int = 42,
              block : int = 1) -> np.ndarray:
    """ Bootstrap distr. of "statistic". Shape (m,) or (m ,k) """
    rng = np.random.default_rng(seed)

    return np.array([statistic(data.iloc[i]) for i in resample_indices(len(data), m , rng, block)])



def percentile_ci(dist : np.ndarray, level : float  = 0.95):
    alpha = (1 - level)/2
    return np.quantile(dist, [alpha, 1-alpha], axis  = 0)

def simple_bootstrap_ci(data : pd.Series, statistic, m = 2000, seed = 42, output_CL = 0.95):
    """ Generic bootstrap functional """
    rng  = np.random.default_rng(seed) # same seed every time to reproduce readme values
    n = len(data)
    estimates = np.array([statistic(data.iloc[rng.integers(0, n, n)], param_CL) for _ in range(m)])
    alpha = (1 - output_CL)/2
    lo, hi = np.quantile(estimates, [alpha, 1-alpha], axis = 0)

    return np.asarray(statistic(data, param_CL)), lo, hi



