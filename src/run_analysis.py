""" Orchestrator for executing the different estimation strategies and consolidating results """
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker
from itertools import cycle
from scipy.stats import  t, norm

from functools import partial

from config import WEIGHTS, CONFIDENCE_LEVELS, NU_FREE, NU_PINNED , MAX_LAG, PORTFOLIO_VALUE, BLOCKS, STRATEGY_ORDER, WINDOWS, ROOT, MC_N_ESTIMATES, MC_N_INTERVALS, BOOTSTRAP_M, SEED 
from diagnostics import  get_acf, plot_acf, fit_t,  visualise_returns, qq_plot, summary_statistics
from portfolio import Portfolio, load_returns
from bootstrap import percentile_ci, bootstrap
from var_historical import historical_from_returns
from var_parametric import parametric_var_es
from var_monte_carlo import t_samples, mc_var_es

def run_diagnostics(portfolio : Portfolio):
    
    diagnostics = summary_statistics(portfolio.portfolio_returns)

    n = len(portfolio.portfolio_returns)
    diagnostics["JB Breakdown"] = {"Skew Component" : diagnostics["Skewness"]**2 * n/6,
                                   "Kurtosis Component" : (diagnostics["Kurtosis"]**2 / 4) * n/6}
    diagnostics["JB Breakdown"]["Share from Kurtosis"] = diagnostics["JB Breakdown"]["Kurtosis Component"]/(
        diagnostics["JB Breakdown"]["Skew Component"] + diagnostics["JB Breakdown"]["Kurtosis Component"])

    diagnostics["Squared ACF"] = {}
    diagnostics["Squared ACF"]["Lag-1"] = portfolio.portfolio_returns.pow(2).autocorr(lag = 1) 
    diagnostics["Squared ACF"]["Lag-126"] = portfolio.portfolio_returns.pow(2).autocorr(lag = MAX_LAG) 

    diagnostics["Nu Free"] = fit_t(portfolio.portfolio_returns, floc = None, fscale = None)["df"]
    diagnostics["Nu Pinned"] = fit_t(portfolio.portfolio_returns, floc = portfolio.mu_p, fscale = portfolio.sigma_p)["df"]

        

    
    return diagnostics

def build_strategies(CL : float, N = MC_N_ESTIMATES):
    strategies = {"Historical" : partial(historical_from_returns, weights = WEIGHTS, confidence = CL) ,
                    "Parametric" : partial(parametric_var_es, weights = WEIGHTS, CL = CL),
                    "Monte Carlo (Normal)" : partial(mc_var_es, weights = WEIGHTS, CL =CL, N = N),
                    "Monte Carlo (t)" : partial(mc_var_es, weights = WEIGHTS, CL =CL, sample_strategy = partial(t_samples, nu = NU_FREE), N = N),
                    "Monte Carlo (t, pinned)" : partial(mc_var_es, weights = WEIGHTS, CL = CL, sample_strategy = partial(t_samples, nu = NU_PINNED), N = N)
                }
    return strategies

def run_estimates(portfolio : Portfolio, confidence_levels : list[float] = CONFIDENCE_LEVELS)-> pd.DataFrame:

    rows = []          
    for CL in confidence_levels:

        strategies = build_strategies(CL)

        for strategy, fn in strategies.items():
            var, es = fn(portfolio.returns)
            rows.append({"Strategy" : strategy, "Confidence" : CL, "VaR": var, "ES" : es, "ES/VaR Ratio": es/var })

    return pd.DataFrame(rows)


def run_intervals(portfolio : Portfolio, confidence_levels : list[float] = CONFIDENCE_LEVELS, N = MC_N_INTERVALS, m = BOOTSTRAP_M) -> pd.DataFrame:

    rows = []

    for CL in confidence_levels:
        strategies = {k: v for k, v in build_strategies(CL, N = N).items() }
        for strategy, fn in strategies.items():
            lo, hi = percentile_ci(bootstrap(portfolio.returns, fn, m = m))
            rows.append({"Strategy": strategy, "Confidence" : CL, "VaR_Lo": lo[0], "VaR_Hi" : hi[0], "ES_Lo": lo[1], "ES_Hi":hi[1]})
        

    return pd.DataFrame(rows)


def run_convergence(portfolio : Portfolio,strategy , confidence_levels : list[float] = CONFIDENCE_LEVELS, N_values : list[int] = [10_000, 100_000, 1_000_000], N_seeds = 30):

    seeds = np.arange(1,N_seeds+1)
    rows = []

    for N_sims in N_values:
        estimates = np.array([strategy(portfolio.returns, seed = seed, N = N_sims, CL = CONFIDENCE_LEVELS[0]) for seed in seeds])
        se = estimates.std(axis = 0, ddof = 1)
        rows.append({"N": N_sims, "VaR_se" : se[0], "ES_se": se[1]})

    return pd.DataFrame(rows)    

def run_window_sensitivity(portfolio : Portfolio, windows : dict, confidence_levels = CONFIDENCE_LEVELS, N = 200_000):
    """ Tracking how VaR and ES estimates change with window selection """
    rows = []
    for name, select in windows.items():
        r_win = select(portfolio.returns)
        p_win = Portfolio.from_returns(r_win)

        for CL in confidence_levels:
            for strategy, fn in build_strategies(CL, N=N).items():
                var, es = fn(p_win.returns)
                rows.append({"Window": name, "n": len(r_win),
                             "sigma_p": p_win.sigma_p, "Confidence": CL,
                             "Strategy": strategy, "VaR": var, "ES": es})
                
    return pd.DataFrame(rows)

def run_block_sensitivity(portfolio, blocks, seed = 42, CL = 0.95):

    strategies = build_strategies(CL = CL)
    strategies.pop("Monte Carlo (Normal)")
    strategies.pop("Monte Carlo (t)")

    rows = []
    
    for strategy, fn in strategies.items():
        for block in blocks:
            lo, hi = percentile_ci(bootstrap(portfolio.returns, fn, block = block, seed = seed))
            rows.append({"Strategy": strategy, "Block": block, "VaR_Lo" : lo[0], "VaR_Hi": hi[0], "ES_Lo" : lo[1], "ES_Hi": hi[1], "VaR_Width": hi[0] - lo[0], "ES_Width" : hi[1] - lo[1]})


    return pd.DataFrame(rows)

def plot_block_sensitivity(ci, strategies=None, metric="VaR",  point_estimates=None):

    lo_col, hi_col = f"{metric}_Lo", f"{metric}_Hi"
    if strategies is None:
        strategies = list(dict.fromkeys(ci["Strategy"]))

    _, ax = plt.subplots(figsize=(7, 4.5))
    colour = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for strategy in strategies:
        data = ci[ci["Strategy"] == strategy].sort_values("Block")
        c = next(colour)
        ax.plot(data["Block"], data[lo_col], marker="o", ms=4, lw=1.2, color=c)
        ax.plot(data["Block"], data[hi_col], marker="o", ms=4, lw=1.2, color=c,
                label=strategy)
        ax.fill_between(data["Block"], data[lo_col], data[hi_col],
                        color=c, alpha=0.25, lw=0)
        if point_estimates and strategy in point_estimates:
            ax.axhline(point_estimates[strategy], color=c, ls="--", lw=0.9)

    ax.set_xscale("log")
    ax.set_xticks(sorted(ci["Block"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax.set_xlabel("Bootstrap block length (trading days)")
    ax.set_ylabel(f"{metric} 95% confidence interval")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    return ax.figure, ax

def plot_window_sensitivity(ws, metric="VaR", strategies=None, sort_by="sigma_p", logy=True, show_sigma=True):

    if strategies is None:
        strategies = [s for s in dict.fromkeys(ws["Strategy"]) if s != "Monte Carlo (Normal)"]     # duplicates Parametric
 
    order = (ws.drop_duplicates("Window").sort_values(sort_by)["Window"].tolist() if sort_by else list(dict.fromkeys(ws["Window"])))
    levels = sorted(ws["Confidence"].unique())
 
    fig, axes = plt.subplots(1, len(levels), figsize=(6.3 * len(levels), 4.8), sharey=True, squeeze=False)
    axes = axes[0]
    xs = np.arange(len(order))
    width = 0.8 / len(strategies)
 
    for ax, CL in zip(axes, levels):
        sub = ws[ws["Confidence"] == CL].set_index(["Window", "Strategy"])
        for j, s in enumerate(strategies):
            vals = [sub.loc[(wn, s), metric] for wn in order]
            ax.bar(xs + (j - (len(strategies) - 1) / 2) * width, vals, width, label=s)
        if show_sigma:
            sig = [ws.loc[ws["Window"] == wn, "sigma_p"].iloc[0] for wn in order]
            ax.plot(xs, sig, "k.--", lw=1, ms=7, label=r"$\sigma_p$ (daily)")
 
        ns = {wn: ws.loc[ws["Window"] == wn, "n"].iloc[0] for wn in order}
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{wn}\nn={ns[wn]:,}" for wn in order], fontsize=8)
        if logy:
            ax.set_yscale("log")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.set_title(f"{metric} at {CL:.0%}")
        ax.grid(axis="y", alpha=0.25, which="both")
        ax.spines[["top", "right"]].set_visible(False)
 
    axes[0].set_ylabel(f"{metric}" + (" (log scale)" if logy else ""))
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Estimation window effect on VaR", y=1.0, fontsize=11)
    fig.tight_layout()
    return fig, axes

def points_at(estimates, CL, metric="VaR"):
    """{strategy: point estimate} at one confidence level."""
    return (estimates.loc[estimates["Confidence"] == CL].set_index("Strategy")[metric].to_dict())

def bootstrap_diagnostic(portfolio, diagnostic, block = 1):
    
    lo, hi = percentile_ci(bootstrap(portfolio.portfolio_returns.pow(2), diagnostic, block = block))
    return lo, hi

def consolidate(estimates : pd.DataFrame, intervals : pd.DataFrame, portfolio_value : float = PORTFOLIO_VALUE):
    results = estimates.merge(intervals, on = ["Strategy", "Confidence"], how = "outer", validate ="one_to_one")

    results["Strategy"] = pd.Categorical(results["Strategy"], categories=STRATEGY_ORDER, ordered=True)
    results = results.sort_values(["Confidence", "Strategy"]).reset_index(drop=True)


    for c in ["VaR", "ES", "VaR_Lo", "VaR_Hi", "ES_Lo", "ES_Hi"]:
        results[f"{c} (EUR)"] = results[c] * PORTFOLIO_VALUE

    return results

def build_figures(portfolio, results, windows_df, blocks_df,
                  max_lag=MAX_LAG + 25, clip=(-0.06, 0.06)):

    r = portfolio.portfolio_returns
    n = len(r)

    nu, loc, scale = fit_t(r).values()
    dists = {"Normal": norm(r.mean(), r.std(ddof=1)),
             f"Student-t ($\\nu$={nu:.2f})": t(nu, loc, scale)}

    figures = {}

    fig, _ = visualise_returns(r, dists)
    figures["Portfolio_returns_fit.png"] = fig

    fig, _ = qq_plot(r, dists, clip=clip)
    figures["qq_plot.png"] = fig

    acf = get_acf(r, min_lag=1, max_lag=max_lag)
    fig, _ = plot_acf(acf, n, min_lag=1, max_lag=max_lag)
    figures["ACF_LagPlot.png"] = fig

    fig, _ = plot_window_sensitivity(windows_df)
    figures["window_sensitivity.png"] = fig

    fig, _ = plot_block_sensitivity(
        blocks_df, strategies=["Historical", "Parametric"],
        point_estimates=points_at(results, 0.99))
    figures["Block_sensitivity_99VaR.png"] = fig

    return figures


def main(portfolio = None):

    if portfolio is None:
        portfolio = Portfolio.from_returns(load_returns())
    
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok = True)

    figdir = ROOT / "figures"
    figdir.mkdir(exist_ok=True)

    diagnostics = run_diagnostics(portfolio)
    estimates   = run_estimates(portfolio)
    intervals   = run_intervals(portfolio)
    results     = consolidate(estimates, intervals, portfolio_value=PORTFOLIO_VALUE)

    for m in ("VaR", "ES"):
        bad = results[(results[m] < results[f"{m}_Lo"]) | (results[m] > results[f"{m}_Hi"])]
        assert bad.empty, f"{m} estimate outside its CI:\n{bad}"

    windows_df  = run_window_sensitivity(portfolio, WINDOWS)
    blocks_df   = run_block_sensitivity(portfolio, BLOCKS, CL=0.99)



    (outdir / "tables.md").write_text(
    "### VaR\n\n" + results_to_table(results, "VaR") +
    "\n\n### ES\n\n" + results_to_table(results, "ES") + "\n"
    )

    for name, df in (("estimates", results), ("window_sensitivity", windows_df),("block_sensitivity", blocks_df)):
        df.to_csv(outdir / f"{name}.csv", index=False, encoding="utf-8")

    (outdir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=float), encoding="utf-8")

    for fname, fig in build_figures(portfolio, results, windows_df, blocks_df).items():
        fig.savefig(figdir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

def results_to_table(results_df, metric = "VaR"):
    
    t = results_df.copy()

    out = pd.DataFrame({"Strategy": t["Strategy"],
                        "Estimate": t[metric].map("{:.3%}".format),
                        "Bootstrap CI (95%)" : t.apply( lambda r: f"[{r[f'{metric}_Lo']:.3%} - {r[f'{metric}_Hi']:.3%}]", axis=1),
                        "Estimate Confidence Level" : t["Confidence"]
                        })

    return out.to_markdown(index = False)

if __name__ == "__main__":
   
    returns_data = load_returns()
    portfolio = Portfolio.from_returns(returns_data, WEIGHTS)

    diagnostics = run_diagnostics(portfolio)

    print(diagnostics)
    #main(portfolio)

    # estimates   = run_estimates(portfolio)
    # intervals   = run_intervals(portfolio)
    # results     = consolidate(estimates, intervals, portfolio_value=PORTFOLIO_VALUE)
    # print(results.head())
