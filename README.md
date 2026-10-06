market-risk is a simple project for modelling VaR using the three different techniques. A sample portfolio was chosen, along with a benchmark index fund. The data is sourced using yfinance, and the metadata can be found within the config.py file.
# Market Risk (VaR)
This repo contains a project in market-risk, specifically assessing Value-at-risk (VaR) and expected shortfall (ES) for a given portfolio of equities.

## Methodologies
There are three main methods for assessing the risk level of a portfolio namely:


1. Historical simulation - this entails utilising the historical daily returns as a sample of possibly outcomes for todays portfolio, and using these to determine the VaR and ES.
2. Parametric - lorem ipsum
3. Monte Carlo - lorem ipsum

Detail sample portfolio here
Data for this portfolio was sources using the yahoo finance api yfinance.
## Results
### Historical VaR
Outputs:
- VaR at 95%, 99%
- Expected Shortfall at same levels
- Both expressed as a% of portfolio and currency amount
- Horizons: 1 ad 10 day. Rport both the sqrt(10) scaled version 
- Stressed VaR: same model calibrated to a stress window (eg. 2008) rather than full sample.

Risk decomposition:
- Component ES: the contribution of each position to total ES
- Marginal VaR: sensitivity to small increase in each position
- Incremental VaR: change in total VaR from removing a position entirely

Backtesting:
- Expception count and rate vs expected over a rolling 250 day window
- Kupiec POF test:
- Christofferson independence test:
- Christofferson conditional coverage:
- Basel traffic light:
- Exception magnitude: How far past VaR breaches went tabulated
- Time series plot: Daily P&L against VaR line with breaches marked (MOST IMPORTANT as its standard)

Estimation diagnostics:
- Window sensitivity
- Rolling VaR time series - VaR through time
- Boostrap confidence intervals on each estimate

## Data Inspection
The first step is to investigate the returns data set. A simple visual inspection of a histogram plot of the data shows it is at the very least approximately normal, however there are excessive tails in both directions. Given that the parametric method for estimating VaR assumes normality in the underlying returns distribution, a more rigourous test must be applied. Typical tests for normality include the Jarque-Bara test, which assumes normality and then rejects this hypothesis if the skewness and kurtosis (equal to 0 and 3 for a normal distribution) are such that the test statistic exceeds a threshold set by the chi-squared critical value for a given degrees of freedom. In the case of large samples the DOF is equal to 2.

The results of these tests are decisive. The data is not normally distributed, and this is almost entirely due to the tails, with the skewness taking a value of 0.23, and the kurtosis being 13.34, yielding a test statistic value of > 36k. This is orders of magnitude above the threshold chi-squared critical value for even a 0.1% CI. Whilst this is not a surprising result, given the power of this test with a sample size of close to 5000, it serves to highlight the limitations of a normality assumption and suggests that much consideration must be given to the distribution tails in marking the risk appropriately.