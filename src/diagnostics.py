""" Analysing properties of the returns data """
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import numpy as np
from scipy.stats import jarque_bera, t




def summary_statistics(data : pd.Series) -> None:

    mu = data.mean()
    sigma = data.std()

    kurtosis = data.kurtosis()
    skewness = data.skew()
    
    resampled_skewness = data.drop(data.abs().nlargest(10).index).skew()
    q1, med, q3 = np.quantile(data, [0.25, 0.5, 0.75])
    bowley_skewness = (q3 + q1 - 2*med)/(q3-q1)
    jb_test = jarque_bera(data, nan_policy = "raise")

    autocorr = data.autocorr()
    sq_autocorr = data.pow(2).autocorr()
    results = {
        "Daily Mean" : mu,
        "Annualised Mean" : mu * 252,
        "Standard Dev." : sigma,
        "Annualised Standard Dev." : sigma * np.sqrt(252),
        "Skewness" : skewness,
        "Resampled skewness" : resampled_skewness,
        "Bowley Skewness" : bowley_skewness,
        "Kurtosis" : kurtosis,
        "Jarque-Bera test statistic" : jb_test[0],
        "Jarque-Bera test p-value" : jb_test[1],   
        "Autocorrelation (k=1)" : autocorr,
        "Autocorr. returns**2 (k=1)" : sq_autocorr
            }

    return results

def get_acf(data : pd.Series, min_lag : int  = 1, max_lag : int = 51, lag_step : int = 1):
    return [data.autocorr(lag = k) for k in range(min_lag, max_lag, lag_step)]

def plot_acf(acf, n, min_lag=1, max_lag=51, lag_step=1, bartlett=True):
    """ACF of squared returns with naive and Bartlett significance bands."""
    acf = np.asarray(acf)
    lags = np.arange(min_lag, max_lag, lag_step)
    _, ax = plt.subplots(figsize=(9, 4.2))
 
    se = 1.96 / np.sqrt(n)
    ax.axhspan(-se, se, color="0.80", alpha=0.6, lw=0,
               label=f"95% band, white-noise null  ±{se:.3f}")
 
    if bartlett:
        bart = 1.96 * np.sqrt((1 + 2 * np.cumsum(np.r_[0, acf[:-1] ** 2])) / n)
        ax.fill_between(lags, -bart, bart, color="blue", alpha=0.45, lw=0,
                        label="95% band, Bartlett")
 
    ax.vlines(lags, 0, acf, color="#2980be", lw=1.3, alpha=0.85)
    ax.plot(lags, acf, "o", ms=3.5, color="#1f77b4")
    ax.axhline(0, color="0.2", lw=0.8)
 
    inside = np.flatnonzero(np.abs(acf) > (bart if bartlett else se))
    if inside.size:
        last = lags[inside[-1]]
        ax.axvline(last, color="#d62728", ls=":", lw=1)
        ax.annotate(f"last lag outside Bartlett band: {last}", xy=(last, 0),
                    xytext=(-8, -34), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8.5, color="#d62728")
 
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Autocorrelation of squared returns")
    ax.set_xlim(lags[0] - 0.5, lags[-1] + 0.5)
    ax.margins(y=0.12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.show()
    return ax.figure, ax

def fit_t(data : pd.Series, floc = None, fscale = None):
    """" Fitting a t-distribution to the data, choose between fixed or variable mu, sigma"""
    x = np.asarray(data)
    kwargs = {}
    
    if floc is not None:
        kwargs["floc"] = floc
    if fscale is not None:
        kwargs["fscale"] = fscale    

    nu, loc, scale = t.fit(x, **kwargs)
    t_fit = {"df": nu, "loc": loc, "scale": scale}
    return t_fit

   


def visualise_returns(data, dists, bins=150, equal_count=False,
                      min_expected=5.0, title=None):
    """
    data   : pd.Series | np.ndarray of returns
    dists  : {label: frozen scipy distribution}, e.g. {"Normal": norm(mu, sd)}
    bins   : int (equal-width) or array of edges
    min_expected : Pearson residuals are masked where expected counts fall below this
    """
    x = np.asarray(data, dtype=float)
    n = len(x)

    edges = np.linspace(x.min(), x.max(), bins + 1)

 
    counts, edges = np.histogram(x, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    density = counts / (n * widths)
 
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07})
 
    ax.bar(centres, density, width=widths, color="0.75", edgecolor="none", label=f"Empirical (n={n:,})")
 
    grid = np.linspace(edges[0], edges[-1], 2000)
    for label, dist in dists.items():
        ax.plot(grid, dist.pdf(grid), lw=1.6, label=label)
 
        # expected counts per bin from the CDF (exact, not a midpoint approximation)
        expected = n * (dist.cdf(edges[1:]) - dist.cdf(edges[:-1]))
        resid = np.where(expected >= min_expected,
                         (counts - expected) / np.sqrt(np.maximum(expected, 1e-12)),
                         np.nan)
        axr.step(centres, resid, where="mid", lw=1.2, label=label)

    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(title or "Portfolio daily returns vs fitted distributions")
 
    for lv, ls in ((0, "-"), (2, ":"), (-2, ":")):
        axr.axhline(lv, color="0.4", ls=ls, lw=0.8)
    axr.set_ylabel(r"Pearson resid. $(O-E)/\sqrt{E}$", fontsize=8)
    axr.set_xlabel("Daily return")
    axr.set_ylim(-6, 6)
 
    return fig, (ax, axr)

def qq_plot(data, dists, clip=None, ax=None):

    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    x_sorted = np.sort(x)
    n = len(x_sorted)
    p = (np.arange(1, n + 1) - 0.5) / n         
 
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))
 
    for label, dist in dists.items():
        ax.scatter(dist.ppf(p), x_sorted, s=4, alpha=0.6, label=label, zorder=2)
 
    lims = clip if clip is not None else (
        min(x_sorted[0], *(d.ppf(p[0]) for d in dists.values())),
        max(x_sorted[-1], *(d.ppf(p[-1]) for d in dists.values())))
    ax.plot(lims, lims, color="0.3", lw=1, zorder=3, label="y = x")
 
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Model quantile")
    ax.set_ylabel("Observed quantile")
    ax.set_title(f"QQ plot, n = {n:,}")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    return ax.figure, ax
 
