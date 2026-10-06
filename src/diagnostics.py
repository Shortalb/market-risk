""" Analysing properties of the returns data """
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import shapiro, jarque_bera 

from portfolio import portfolio_returns


def summary_statistics(data : pd.Series) -> None:

    mu = data.mean()
    sigma = data.std()

    kurtosis = data.kurtosis()
    skewness = data.skew()

    jb_test = jarque_bera(data, nan_policy = "raise")
    sw_test = shapiro(data, nan_policy = "raise")


    results = {
        "Mean " : mu,
        "Standard Dev. " : sigma,
        "Skewness " : skewness,
        "Kurtosis " : kurtosis,
        "Jarque-Bera test " : jb_test,
        "Shapiro-Wilks test " : sw_test 
    }

    for key, value in results.items():
        print(key + str(value) + "\n")
    plt.hist(data, bins = 80)
    plt.show()

def get_acf(data : pd.Series, min_lag : int  = 1, max_lag : int = 51):
    return [data.autocorr(lag = k) for k in range(min_lag, max_lag)]



if __name__ == "__main__":
    data = portfolio_returns()
    print(data.kurtosis())
    summary_statistics(data)

