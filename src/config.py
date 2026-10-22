from pathlib import Path
import pandas as pd
import numpy as np
import subprocess, datetime, platform

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

START = "2007-01-01"
END = "2026-10-01"          # exclusive, so the last row is 30 Sep 2026

EQUITIES = ["AAPL", "MSFT", "AMZN", "JPM", "GS", "XOM", "CVX",
            "JNJ", "PFE", "PG", "KO", "CAT", "NEE", "WMT"]
WEIGHTS = {**{t: 0.05 for t in EQUITIES}, "TLT": 0.20, "GLD": 0.10}

BENCHMARK = "SPY"

PORTFOLIO_TICKERS = list(WEIGHTS.keys())

TICKERS = list(WEIGHTS) + [BENCHMARK]

assert abs(sum(WEIGHTS.values()) - 1) < 1e-9, "weights must sum to 1"

CONFIDENCE_LEVELS = [0.95, 0.99]
PORTFOLIO_VALUE = 1_000_000 #Arbitrary, for expressin VaR in currency

WINDOWS = {
    "full":    lambda r: r,
    "2,500d":  lambda r: r.iloc[-2500:],
    "250d":    lambda r: r.iloc[-250:],
    "2008":    lambda r: r.loc["2008"],
    "2020":    lambda r: r.loc["2020"],
    "2009-19": lambda r: r.loc["2009":"2019"],
}

STRATEGY_ORDER = ["Historical", "Parametric", "Monte Carlo (Normal)", "Monte Carlo (t)"]

BLOCKS = [1,2,5,10,25,60,100]
""" Parameters fitted from the data """
NU_FREE = 2.83
NU_PINNED = 8.21
MAX_LAG = 126

""" Hyper parameters """
MC_N_ESTIMATES = 1_000_000 # Used for MC point estimates
MC_N_INTERVALS = 100_000 # Used for estimation of Confidence Intervals which rely on MC draws
BOOTSTRAP_M = 500 # Number of resamples per interval
SEED = 42 # RNG seed



def metadata(portfolio):
    """ Recording run metadata """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        sha = "unknown"
    return {
        "generated_at":   datetime.datetime.now().isoformat(timespec="seconds"),
        "git_sha":        sha,
        "sample_start":   str(portfolio.returns.index.min().date()),
        "sample_end":     str(portfolio.returns.index.max().date()),
        "n_observations": len(portfolio.portfolio_returns),
        "weights":        WEIGHTS,
        "confidence_levels": CONFIDENCE_LEVELS,
        "nu":             NU,
        "mc_N_estimates": MC_N_ESTIMATES,
        "mc_N_intervals": MC_N_INTERVALS,
        "bootstrap_m":    BOOTSTRAP_M,
        "seed":           SEED,
        "python":         platform.python_version(),
        "pandas":         pd.__version__,
        "numpy":          np.__version__,
    }
