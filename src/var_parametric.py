"""" Producing a parametric VaR estimation using the standard assumption of a normal distribution in returns """

import pandas as pd
from scipy.stats import norm

from portfolio import get_varcovar, get_sigma_p, get_mu_p




def parametric_var_es(r : pd.DataFrame, weights : dict, CL : float)-> tuple[float]:
    var_covar = get_varcovar(r, list(weights.keys()))
    sigma_p = get_sigma_p(var_covar, weights)
    mu_p = get_mu_p(r, weights)

    z = norm.ppf(CL)

    VaR_param = sigma_p * z - mu_p
    ES_param = sigma_p * norm.pdf(z)/ (1- CL) - mu_p

    return VaR_param, ES_param




