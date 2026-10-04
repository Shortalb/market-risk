from pathlib import Path
import pandas as pd

from config import WEIGHTS, DATA_DIR

prices = pd.read_csv(DATA_DIR / "prices.csv", index_col = "Date", parse_dates= True)
print("missing prices:", prices.isna().sum().sum()) # should be 0

# Daily Simple returns
returns = prices.pct_change().dropna()

print(returns.shape)
print(returns.describe().T[["mean", "std", "min", "max"]])
print(returns.abs().max().sort_values(ascending=False).head()) # Biggest single day moves

weights = pd.Series(WEIGHTS)

portfolio = returns[weights.index] @ weights

""" Note on above : portfolio matrix calculation
    is assuming that each day the weights are rebalanced
    to adjust for changes in price """


print(portfolio.describe())
print(portfolio.nsmallest(5)) ## Five worst days