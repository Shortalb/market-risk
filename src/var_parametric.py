"""" Producing a parametric VaR estimation using the standard assumption of a normal distribution in returns """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from functools import partial

from portfolio import load_returns, get_varcovar, get_sigma_p, get_mu_p
from bootstrap import bootstrap, percentile_ci
from config import WEIGHTS, CONFIDENCE_LEVELS




def parametric_var_es(r : pd.DataFrame, weights : dict, CL : float)-> tuple[float]:
    var_covar = get_varcovar(r, list(weights.keys()))
    sigma_p = get_sigma_p(var_covar, weights)
    mu_p = get_mu_p(r, weights)

    z = norm.ppf(CL)

    VaR_param = sigma_p * z - mu_p
    ES_param = sigma_p * norm.pdf(z)/ (1- CL) - mu_p

    return VaR_param, ES_param


if __name__ == "__main__":

    data = load_returns() ## unweighted returns for all data incl. benchmark

    for c in CONFIDENCE_LEVELS:
         
        VaR, ES = parametric_var_es(data, WEIGHTS, c)
        statistic = partial(parametric_var_es, weights = WEIGHTS, CL = c)
        est_params = bootstrap(data, statistic = statistic )
        param_ci = percentile_ci(est_params)

        print(f"Confidence Levels - {c*100} %\nVaR : {VaR:.3%} \nES : {ES:.3%}")
        print(param_ci)


