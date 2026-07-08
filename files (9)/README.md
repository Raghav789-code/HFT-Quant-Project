# Regime-Switching Tactical Asset Allocation

A Hidden Markov Model classifies the market into bull / neutral / bear regimes from daily return, volatility, and VIX features. At each monthly rebalance the current regime is decoded with the Viterbi algorithm, regime-conditional expected returns and covariances are estimated, and a long-only mean-variance portfolio with an explicit turnover (transaction-cost) penalty is solved in CVXPY. The strategy is evaluated with strict walk-forward validation against static benchmarks (SPY buy & hold, 60/40, equal-weight).

```
data (yfinance) → features → HMM regime detection (Viterbi) → CVXPY optimization → walk-forward backtest → results
```

## Repository contents

| File | Description |
|---|---|
| `regime_switching_taa.ipynb` | Full pipeline notebook, runs top to bottom |
| `regime_taa.py` | Same pipeline as a single script (`python regime_taa.py`) |
| `figures/` | Generated exhibits: regimes on price, transition matrix, equity curves, drawdowns, weights |
| `performance_summary.csv` | Auto-generated metrics table (Sharpe, Sortino, MaxDD, Calmar, turnover) |
| `transition_matrix.csv` | Auto-generated regime transition probabilities |
| `PERFORMANCE_SUMMARY.md` | Written summary of strategy and results |

## How to run

```bash
pip install hmmlearn cvxpy yfinance numpy pandas matplotlib scipy
python regime_taa.py          # or open regime_switching_taa.ipynb and Run All
```

Runtime is a few minutes on a laptop (the HMM is refit once per year of the walk-forward, not every month). No API keys are needed — Yahoo Finance and `^VIX` are pulled via `yfinance`. Setting `USE_SYNTHETIC = True` in the config cell runs the identical pipeline on planted-regime synthetic data, useful for offline testing.

## Key design decisions

**Why 3 regimes?** Two states (risk-on / risk-off) merge the "grinding sideways" market into one of the extremes and force the optimizer into binary bets; four or more states fragment the data so that some states get too few observations to estimate a stable covariance matrix (with ~15 years of daily data, a rare 4th state can end up with well under a year of observations). Three states map cleanly onto economically interpretable conditions — low-vol uptrend (bull), elevated-vol chop (neutral), high-vol drawdown (bear) — and BIC on the training window flattens out after 3 components. Crucially, hmmlearn's state labels are arbitrary across fits, so states are relabeled after every fit by realized in-state market Sharpe (best → Bull 0, worst → Bear 2), which makes regimes comparable across walk-forward refits.

**Why these features?** Every feature is computable at time *t* with no look-ahead. Realized 21-day volatility is the single strongest regime separator (vol clusters; bear markets are high-vol almost by definition). The daily market return lets states differ in mean, not just variance. VIX log-changes add a forward-looking fear signal that leads realized vol at turning points. 3-month momentum provides trend context that helps the HMM distinguish "high vol on the way down" from "high vol recovery". Features are standardized using training-window statistics only. Macro series from FRED (CPI surprises, 10y–2y spread) can be appended as extra columns in `build_features` — they were left optional because they are monthly/weekly and require an API key, while the core pipeline stays daily and keyless.

**Why Viterbi on a trailing window?** `GaussianHMM.predict` runs Viterbi, which finds the most likely full state path. Decoding a trailing 1-year window and reading the last day's state gives the smoothed, most-probable current regime while using only past data — this keeps the backtest causal.

**Why mean-variance with an L1 turnover penalty?** The problem `max wᵀμ − (γ/2)wᵀΣw − c‖w − w_prev‖₁` with long-only, fully-invested, 60% single-asset-cap constraints is convex, so CVXPY (Clarabel solver) finds the global optimum in milliseconds. The L1 term prices trading at 10 bps one-way, so the optimizer only trades when the expected regime benefit exceeds the cost — this is what keeps turnover economical instead of bolting costs on after the fact. Risk aversion γ is regime-dependent (bull 5, neutral 15, bear 40): the strategy leans into risk when the decoded regime is benign and de-risks into bonds/gold when it is not. A small ridge term (1e-4·I) is added to Σ for numerical stability.

**Why walk-forward, expanding window, yearly HMM refits?** At every monthly rebalance the model sees only data up to that date; weights are then held for the next 21 trading days strictly out of sample. The window expands (min 3 years) because HMMs benefit from seeing at least one full cycle. The HMM is refit every 12 rebalances (regime *decoding* still happens monthly) — refitting monthly changes results negligibly but multiplies runtime ~12×, and label re-ordering by Sharpe keeps regimes consistent across refits.

**Benchmarks.** SPY buy & hold (raw equity risk), 60/40 SPY/TLT (the standard balanced strawman), and equal-weight across all four assets (the "you didn't need a model" test). All are computed on the identical out-of-sample window.

## Reproducibility

- All random seeds are fixed (`np.random.seed(42)`, `GaussianHMM(random_state=42)`).
- Data is pulled with `auto_adjust=True` from Yahoo Finance for a fixed date range (2007-01-01 to 2025-12-31); note Yahoo occasionally revises adjusted prices, which can shift metrics in the 2nd–3rd decimal.
- `performance_summary.csv` and `transition_matrix.csv` are regenerated on every run — the numbers in `PERFORMANCE_SUMMARY.md` should be refreshed from these after running.
- Exact package versions used: see `requirements.txt` (`pip freeze > requirements.txt` after your run).

## Concepts demonstrated

Hidden Markov Models & the Viterbi algorithm · convex portfolio optimization (CVXPY) · walk-forward validation · Sharpe / Sortino / Calmar ratios · transaction-cost (turnover) modelling.
