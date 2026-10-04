import yfinance as yf


print("yfinance", yf.__version__)


df = yf.download(["AAPL", "SPY"], start="2007-01-01", auto_adjust=True)["Close"]

print(df.shape)
print(df.index.min(), df.index.max())
print(df.isna().sum())
print(df.tail(3))