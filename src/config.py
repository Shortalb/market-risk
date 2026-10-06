from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

START = "2007-01-01"
END = "2026-10-01"          # exclusive, so the last row is 30 Sep 2026

EQUITIES = ["AAPL", "MSFT", "AMZN", "JPM", "GS", "XOM", "CVX",
            "JNJ", "PFE", "PG", "KO", "CAT", "NEE", "WMT"]
WEIGHTS = {**{t: 0.05 for t in EQUITIES}, "TLT": 0.20, "GLD": 0.10}

PORTFOLIO_TICKERS = list(WEIGHTS)

BENCHMARK = "SPY"

PORTFOLIO_TICKERS = list(WEIGHTS.keys())

TICKERS = list(WEIGHTS) + [BENCHMARK]

assert abs(sum(WEIGHTS.values()) - 1) < 1e-9, "weights must sum to 1"

CONFIDENCE_LEVELS = [0.95, 0.99]
PORTFOLIO_VALUE = 1_000_000 #Arbitrary, for expressin VaR in currency
