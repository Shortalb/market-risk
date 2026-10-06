
""" Historical-simulation VaR and Expected Shortfall for the portfolio"""
import pandas as pd
import numpy as np
from functools import partial

from config import DATA_DIR, WEIGHTS, CONFIDENCE_LEVELS, PORTFOLIO_VALUE
from bootstrap import percentile_ci, bootstrap

prices = pd.read_csv(DATA_DIR / "prices.csv", index_col = "Date", parse_dates= True)
returns = prices.pct_change().dropna()

weights = pd.Series(WEIGHTS)
portfolio = returns[weights.index] @ weights

def historical_var_es(r :pd.Series | np.ndarray, confidence : float) -> tuple[float, float]:

    """ Return "(VaR, ES) as positive decimal losses at given confidence level """
    r = np.asarray(r) # extending to numpy native
    threshold = np.quantile(r, 1 - confidence)
    var = -threshold #converting to a positive loss
    es = -r[r <= threshold].mean() #average of the losses that are beyond threshold

    return var, es


if __name__ == "__main__":

    rows = []

    for c in CONFIDENCE_LEVELS:
        var, es = historical_var_es(portfolio, c)
        breaches = (portfolio <= -var).sum()
        statistic = partial(historical_var_es, confidence = c)
        lo, hi = percentile_ci(bootstrap(portfolio, statistic))


        rows.append({
            "confidence": f"{c:.0%}",
            "VaR" : var,
            "VaR_Lo" : lo[0],
            "VaR_Hi" : hi[0],
            "ES" : es,
            "ES_Lo" : lo[1],
            "ES_Hi" : hi[1],
            "VaR_eur" : var*PORTFOLIO_VALUE,
            "ES_eur" : es*PORTFOLIO_VALUE,
            "breach_rate": breaches/len(portfolio),
            "ES_over_VaR" : es / var,
        })



    results = pd.DataFrame(rows).set_index("confidence")
    print(results.head())
    print(f"Sample: {len(portfolio)} days, "
        f"{portfolio.index.min().date()} to {portfolio.index.max().date()}\n")
    print(results[["VaR", "VaR_Lo", "VaR_Hi", "ES", "ES_Lo", "ES_Hi", "breach_rate"]].map(lambda x: f"{x:.3%}"))
    print()
    print(results[["VaR_eur", "ES_eur"]].map(lambda x: f"{x:,.0f}"))
    print(results[["ES_over_VaR"]].round(3))
    print("\nWorst 5 days in sample:")
    print((portfolio.nsmallest(5) * 100).round(2).to_string())

    """ Checking for volatility clustering """

    for name, n in {"full": len(portfolio), "1000d": 1000, "500d": 500, "250d": 250}.items():
        sub = portfolio.iloc[-n:]
        print(name, f"{historical_var_es(sub, 0.99)[0]:.3%}")
