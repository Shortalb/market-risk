
""" Historical-simulation VaR and Expected Shortfall for the portfolio"""
import pandas as pd
from config import DATA_DIR, WEIGHTS, CONFIDENCE_LEVELS, PORTFOLIO_VALUE

prices = pd.read_csv(DATA_DIR / "prices.csv", index_col = "Date", parse_dates= True)
returns = prices.pct_change().dropna()

weights = pd.Series(WEIGHTS)
portfolio = returns[weights.index] @ weights

def historical_var_es(r :pd.Series, confidence : float) -> tuple[float, float]:

    """ Return "(VaR, ES) as positive decimal losses at given confidence level """

    threshold = r.quantile(1 - confidence)
    var = -threshold #converting to a positive loss
    es = -r[r <= threshold].mean() #average of the losses that are beyond threshold

    return var, es

rows = []

for c in CONFIDENCE_LEVELS:
    var, es = historical_var_es(portfolio, c)
    breaches = (portfolio <= -var).sum()

    rows.append({
        "confidence": f"{c:.0%}",
        "VaR" : var,
        "ES" : es,
        "VaR_eur" : var*PORTFOLIO_VALUE,
        "ES_eur" : var*PORTFOLIO_VALUE,
        "breach_rate": breaches/len(portfolio),
        "ES_over_VaR" : es / var,
    })

results = pd.DataFrame(rows).set_index("confidence")
print(f"Sample: {len(portfolio)} days, "
      f"{portfolio.index.min().date()} to {portfolio.index.max().date()}\n")
print(results[["VaR", "ES", "breach_rate"]].map(lambda x: f"{x:.3%}"))
print()
print(results[["VaR_eur", "ES_eur"]].map(lambda x: f"{x:,.0f}"))
print(results[["ES_over_VaR"]].round(3))
print("\nWorst 5 days in sample:")
print((portfolio.nsmallest(5) * 100).round(2).to_string())