# Performance Summary — Regime-Switching Tactical Asset Allocation

## Strategy

A 3-state Gaussian HMM is fit on daily features (SPY return, 21-day realized volatility, VIX log-change, 3-month momentum). States are relabeled by in-state market Sharpe into **Bull / Neutral / Bear**. Every month the current regime is Viterbi-decoded from a trailing 1-year window; regime-conditional annualized mean returns and covariance are estimated from the training window; and portfolio weights over SPY / TLT / GLD / DBC are chosen by a convex program:

max  wᵀμₛ − (γₛ/2)·wᵀΣₛw − c·‖w − w_prev‖₁    s.t.  Σw = 1, 0 ≤ w ≤ 0.6

with regime-dependent risk aversion (Bull γ=5, Neutral γ=15, Bear γ=40) and c = 10 bps one-way transaction costs inside the objective. Evaluation is strictly walk-forward (expanding window, ≥3 years, out-of-sample monthly blocks), with transaction costs deducted from realized P&L.

## Results (out-of-sample, walk-forward)

> Run `python regime_taa.py` (or the notebook top-to-bottom); this table is written automatically to `performance_summary.csv`. Paste the numbers here.

| Strategy | CAGR | Vol | Sharpe | Sortino | Max Drawdown | Calmar | Ann. Turnover |
|---|---|---|---|---|---|---|---|
| **Regime TAA (this work)** | _ | _ | _ | _ | _ | _ | _ |
| SPY buy & hold | _ | _ | _ | _ | _ | _ | — |
| 60/40 (SPY/TLT) | _ | _ | _ | _ | _ | _ | — |
| Equal-weight (4 assets) | _ | _ | _ | _ | _ | _ | — |

## Transition probability matrix (full-sample HMM, Viterbi-labeled)

> Written automatically to `transition_matrix.csv`. Diagonal = persistence; expected duration = 1/(1 − pᵢᵢ) days.

| From \ To | Bull | Neutral | Bear |
|---|---|---|---|
| **Bull** | _ | _ | _ |
| **Neutral** | _ | _ | _ |
| **Bear** | _ | _ | _ |

## Interpretation checklist

- **Risk-adjusted edge:** the strategy targets Sharpe/Sortino/Calmar improvement over 60/40, primarily by cutting exposure in the Bear regime — check the drawdown chart around 2020 (COVID) and 2022 (rates shock).
- **Persistence:** healthy regime models show diagonal transition probabilities well above 0.9 for Bull and Bear (multi-week to multi-month durations). If a state has near-zero persistence it is capturing noise, not a regime.
- **Turnover:** monthly-rebalanced TAA with a turnover penalty should land in the low single digits annualized (≈2–6×). Materially higher means the regime signal is flickering; consider a longer decode window or higher c.
- **Failure mode to note honestly:** regime models lag at sharp V-shaped reversals (e.g., April 2020) — the Bear label persists into the first weeks of recovery, costing some upside. This is the price paid for drawdown protection.

## Exhibits (in `figures/`)

1. `regimes_on_price.png` — Viterbi regime labels shaded over SPY price (log scale)
2. `transition_matrix.png` — transition probability heatmap
3. `equity_curves.png` — walk-forward equity curves vs benchmarks, regime-shaded
4. `drawdowns.png` — drawdown comparison
5. `weights.png` — allocation weights through time
