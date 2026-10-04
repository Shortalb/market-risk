from pathlib import Path
import yfinance as yf

from config import TICKERS, START, END, DATA_DIR


DATA_DIR.mkdir(exist_ok = True)

prices = yf.download(TICKERS, start = START, end = END, auto_adjust = True)["Close"]

print(prices.shape)
print(prices.apply(lambda s : s.first_valid_index()).sort_values())
print(prices.isna().sum())

prices.to_csv(DATA_DIR / "prices.csv")
