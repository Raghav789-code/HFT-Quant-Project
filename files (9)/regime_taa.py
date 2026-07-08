"""
Regime-Switching Tactical Asset Allocation
==========================================
Full pipeline: data -> features -> HMM regime detection -> CVXPY optimization
-> walk-forward backtest -> performance report.

Tech stack: Python 3.9+, hmmlearn, CVXPY, NumPy, Pandas, Matplotlib, yFinance, SciPy
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from hmmlearn.hmm import GaussianHMM
from scipy import stats

np.random.seed(42)
plt.rcParams["figure.figsize"] = (12, 6)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TICKERS = ["SPY", "TLT", "GLD", "DBC"]      # equities, long bonds, gold, commodities
BENCH_TICKER = "SPY"
START, END = "2007-01-01", "2025-12-31"
N_REGIMES = 3
TRAIN_MIN_DAYS = 756            # 3 years minimum training window
REBALANCE_EVERY = 21            # trading days (monthly)
TC_BPS = 10                     # one-way transaction cost, basis points
RISK_AVERSION = {0: 15.0, 1: 5.0, 2: 40.0}  # overwritten after regime ordering
ANN = 252

USE_SYNTHETIC = False           # set True only for offline smoke-testing

# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def load_prices():
    if USE_SYNTHETIC:
        # Offline test data: correlated GBM with 3 planted regimes
        n = 4200
        dates = pd.bdate_range("2007-01-02", periods=n)
        regimes = np.zeros(n, dtype=int)
        P = np.array([[0.995, 0.004, 0.001],
                      [0.010, 0.985, 0.005],
                      [0.020, 0.010, 0.970]])
        for t in range(1, n):
            regimes[t] = np.random.choice(3, p=P[regimes[t-1]])
        mu = {0: [0.09, 0.03, 0.05, 0.04], 1: [0.02, 0.05, 0.08, 0.01], 2: [-0.25, 0.10, 0.12, -0.15]}
        vol = {0: [0.12, 0.10, 0.14, 0.16], 1: [0.18, 0.11, 0.16, 0.20], 2: [0.40, 0.16, 0.22, 0.35]}
        rets = np.zeros((n, 4))
        for t in range(n):
            r = regimes[t]
            rets[t] = np.array(mu[r]) / ANN + np.array(vol[r]) / np.sqrt(ANN) * np.random.randn(4)
        prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=TICKERS)
        vix = pd.Series(15 + 20 * (regimes == 2) + 5 * (regimes == 1) + 2 * np.random.randn(n),
                        index=dates, name="VIX").clip(9, 90)
        return prices, vix
    else:
        import yfinance as yf
        px = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"]
        px = px[TICKERS].dropna()
        vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)["Close"]
        vix = vix.squeeze().reindex(px.index).ffill().rename("VIX")
        return px, vix


prices, vix = load_prices()
returns = prices.pct_change().dropna()
vix = vix.reindex(returns.index).ffill()
print(f"Data: {returns.index[0].date()} -> {returns.index[-1].date()}, {len(returns)} days, assets: {list(returns.columns)}")

# ----------------------------------------------------------------------------
# 2. FEATURES for the HMM (built ONLY from information available at time t)
# ----------------------------------------------------------------------------
def build_features(returns, vix):
    mkt = returns[BENCH_TICKER]
    feat = pd.DataFrame(index=returns.index)
    feat["mkt_ret"] = mkt                                     # daily market return
    feat["vol_21"] = mkt.rolling(21).std() * np.sqrt(ANN)     # realized vol
    feat["vix_chg"] = np.log(vix).diff()                      # VIX log-change
    feat["mom_63"] = mkt.rolling(63).sum()                    # 3m momentum
    return feat.dropna()

features = build_features(returns, vix)
returns = returns.loc[features.index]
print(f"Features: {list(features.columns)}, {len(features)} usable days")

# ----------------------------------------------------------------------------
# 3. HMM REGIME DETECTION
# ----------------------------------------------------------------------------
def fit_hmm(feat_train, n_regimes=N_REGIMES, seed=42):
    """Fit a Gaussian HMM on standardized features. Returns model + scaler stats."""
    mu, sd = feat_train.mean(), feat_train.std()
    X = ((feat_train - mu) / sd).values
    model = GaussianHMM(n_components=n_regimes, covariance_type="full",
                        n_iter=500, random_state=seed, tol=1e-4)
    model.fit(X)
    return model, mu, sd

def order_regimes(model, feat_train, mu, sd, ret_train):
    """Relabel states by realized market Sharpe within each state:
       0 = bull (best), 1 = neutral, 2 = bear (worst). Viterbi decoding."""
    X = ((feat_train - mu) / sd).values
    states = model.predict(X)                      # Viterbi path
    sharpe = {}
    for s in np.unique(states):
        r = ret_train[BENCH_TICKER].values[states == s]
        sharpe[s] = r.mean() / (r.std() + 1e-12)
    ranked = sorted(sharpe, key=sharpe.get, reverse=True)      # best -> worst
    mapping = {old: new for new, old in enumerate(ranked)}
    return mapping, states

def decode_current(model, feat_window, mu, sd, mapping):
    """Viterbi-decode the window; return relabeled state of the LAST day."""
    X = ((feat_window - mu) / sd).values
    states = model.predict(X)
    return mapping[states[-1]]

# ----------------------------------------------------------------------------
# 4. CONVEX PORTFOLIO OPTIMIZATION (CVXPY)
# ----------------------------------------------------------------------------
def optimize_weights(mu_vec, cov, w_prev, gamma, tc=TC_BPS / 1e4):
    """Long-only, fully-invested mean-variance with turnover penalty:
         max  w'mu - gamma/2 * w'Σw - tc * ||w - w_prev||_1
    """
    n = len(mu_vec)
    w = cp.Variable(n)
    objective = cp.Maximize(mu_vec @ w
                            - 0.5 * gamma * cp.quad_form(w, cp.psd_wrap(cov))
                            - tc * cp.norm1(w - w_prev))
    cons = [cp.sum(w) == 1, w >= 0, w <= 0.6]     # max 60% in any single asset
    prob = cp.Problem(objective, cons)
    prob.solve(solver=cp.CLARABEL)
    if w.value is None:
        return w_prev
    wv = np.clip(w.value, 0, None)
    return wv / wv.sum()

# ----------------------------------------------------------------------------
# 5. WALK-FORWARD BACKTEST
# ----------------------------------------------------------------------------
def walk_forward(returns, features):
    dates = returns.index
    rebal_points = range(TRAIN_MIN_DAYS, len(dates) - 1, REBALANCE_EVERY)

    w_prev = np.ones(len(TICKERS)) / len(TICKERS)
    weight_log, regime_log, oos_dates = [], [], []
    strat_rets = pd.Series(0.0, index=dates)

    gamma_by_regime = {0: 5.0, 1: 15.0, 2: 40.0}   # bull -> aggressive, bear -> defensive

    for k, t in enumerate(rebal_points):
        train_ret = returns.iloc[:t]
        train_feat = features.iloc[:t]

        # Refit HMM once per year to keep runtime sane; reuse between refits
        if k % 12 == 0:
            model, f_mu, f_sd = fit_hmm(train_feat)
            mapping, _ = order_regimes(model, train_feat, f_mu, f_sd, train_ret)

        regime = decode_current(model, train_feat.iloc[-252:], f_mu, f_sd, mapping)

        # Regime-conditional moments from the training window only
        X = ((train_feat - f_mu) / f_sd).values
        states = pd.Series(model.predict(X), index=train_feat.index).map(mapping)
        in_regime = train_ret.loc[states[states == regime].index]
        if len(in_regime) < 60:                     # too few obs -> fall back to full window
            in_regime = train_ret
        mu_vec = in_regime.mean().values * ANN
        cov = (in_regime.cov().values * ANN
               + 1e-4 * np.eye(len(TICKERS)))       # ridge for numerical stability

        w = optimize_weights(mu_vec, cov, w_prev, gamma_by_regime[regime])

        # Hold w over the next block, out of sample
        block = slice(t + 1, min(t + 1 + REBALANCE_EVERY, len(dates)))
        block_ret = returns.iloc[block] @ w
        # transaction cost charged on the first day of the block
        turnover = np.abs(w - w_prev).sum()
        block_ret.iloc[0] -= turnover * TC_BPS / 1e4
        strat_rets.iloc[block] = block_ret.values

        weight_log.append(w); regime_log.append(regime); oos_dates.append(dates[t])
        w_prev = w

    oos_start = dates[TRAIN_MIN_DAYS + 1]
    strat_rets = strat_rets.loc[oos_start:]
    weights_df = pd.DataFrame(weight_log, index=oos_dates, columns=TICKERS)
    regimes_sr = pd.Series(regime_log, index=oos_dates, name="regime")
    return strat_rets, weights_df, regimes_sr

strat_rets, weights_df, regimes_sr = walk_forward(returns, features)
print(f"Out-of-sample period: {strat_rets.index[0].date()} -> {strat_rets.index[-1].date()}")

# ----------------------------------------------------------------------------
# 6. BENCHMARKS
# ----------------------------------------------------------------------------
oos_ret = returns.loc[strat_rets.index]
bench = pd.DataFrame(index=strat_rets.index)
bench["SPY buy&hold"] = oos_ret[BENCH_TICKER]
bench["60/40"] = 0.6 * oos_ret["SPY"] + 0.4 * oos_ret["TLT"]
bench["Equal-weight"] = oos_ret.mean(axis=1)
bench["Regime TAA"] = strat_rets

# ----------------------------------------------------------------------------
# 7. METRICS
# ----------------------------------------------------------------------------
def perf_metrics(r, weights=None):
    ann_ret = (1 + r).prod() ** (ANN / len(r)) - 1
    ann_vol = r.std() * np.sqrt(ANN)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(ANN)
    sortino = ann_ret / downside if downside > 0 else np.nan
    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    ann_turnover = np.nan
    if weights is not None:
        ann_turnover = weights.diff().abs().sum(axis=1).mean() * (ANN / REBALANCE_EVERY)
    return dict(CAGR=ann_ret, Vol=ann_vol, Sharpe=sharpe, Sortino=sortino,
                MaxDD=max_dd, Calmar=calmar, AnnTurnover=ann_turnover)

summary = pd.DataFrame({name: perf_metrics(bench[name], weights_df if name == "Regime TAA" else None)
                        for name in bench.columns}).T
print("\n===== PERFORMANCE SUMMARY (out-of-sample) =====")
print(summary.round(3).to_string())

# ----------------------------------------------------------------------------
# 8. FINAL FULL-SAMPLE HMM (for the regime-overlay & transition-matrix exhibit)
# ----------------------------------------------------------------------------
model_full, f_mu, f_sd = fit_hmm(features)
mapping_full, raw_states = order_regimes(model_full, features, f_mu, f_sd, returns)
labels_full = pd.Series(raw_states, index=features.index).map(mapping_full)

# transition matrix, reordered to the bull/neutral/bear labeling
P_raw = model_full.transmat_
idx = [k for k, v in sorted(mapping_full.items(), key=lambda kv: kv[1])]
P = P_raw[np.ix_(idx, idx)]
regime_names = ["Bull (0)", "Neutral (1)", "Bear (2)"]
trans_df = pd.DataFrame(P, index=regime_names, columns=regime_names)
print("\n===== TRANSITION PROBABILITY MATRIX =====")
print(trans_df.round(3).to_string())
exp_duration = 1 / (1 - np.diag(P))
print("Expected regime duration (days):", dict(zip(regime_names, exp_duration.round(1))))

# ----------------------------------------------------------------------------
# 9. PLOTS
# ----------------------------------------------------------------------------
import os
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)
colors = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}

# 9a. Regimes on price
fig, ax = plt.subplots(figsize=(14, 6))
px = prices[BENCH_TICKER].reindex(labels_full.index)
ax.plot(px.index, px.values, color="black", lw=0.8)
for s in range(N_REGIMES):
    mask = labels_full == s
    ax.fill_between(px.index, px.min(), px.max(), where=mask, color=colors[s], alpha=0.18)
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s], alpha=0.4) for s in range(N_REGIMES)]
ax.legend(handles, regime_names); ax.set_title(f"HMM Regimes over {BENCH_TICKER} price (Viterbi decoding)")
ax.set_yscale("log"); fig.tight_layout(); fig.savefig(f"{FIGDIR}/regimes_on_price.png", dpi=130); plt.close(fig)

# 9b. Transition matrix heatmap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(3), regime_names); ax.set_yticks(range(3), regime_names)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{P[i, j]:.3f}", ha="center", va="center",
                color="white" if P[i, j] > 0.5 else "black")
ax.set_title("Regime Transition Probability Matrix"); ax.set_xlabel("To"); ax.set_ylabel("From")
fig.colorbar(im); fig.tight_layout(); fig.savefig(f"{FIGDIR}/transition_matrix.png", dpi=130); plt.close(fig)

# 9c. Equity curves with regime shading (out-of-sample)
fig, ax = plt.subplots(figsize=(14, 6))
curves = (1 + bench).cumprod()
for c in curves.columns:
    ax.plot(curves.index, curves[c], lw=1.8 if c == "Regime TAA" else 1.0, label=c)
reg_daily = regimes_sr.reindex(curves.index, method="ffill")
for s in range(N_REGIMES):
    ax.fill_between(curves.index, curves.min().min(), curves.max().max(),
                    where=(reg_daily == s), color=colors[s], alpha=0.10)
ax.set_yscale("log"); ax.legend(); ax.set_title("Walk-Forward Equity Curves (log scale, regime-shaded)")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/equity_curves.png", dpi=130); plt.close(fig)

# 9d. Drawdowns
fig, ax = plt.subplots(figsize=(14, 4))
for c in curves.columns:
    dd = curves[c] / curves[c].cummax() - 1
    ax.plot(dd.index, dd, lw=1.5 if c == "Regime TAA" else 0.9, label=c)
ax.legend(); ax.set_title("Drawdowns"); fig.tight_layout()
fig.savefig(f"{FIGDIR}/drawdowns.png", dpi=130); plt.close(fig)

# 9e. Allocation over time
fig, ax = plt.subplots(figsize=(14, 5))
ax.stackplot(weights_df.index, weights_df.T.values, labels=TICKERS, alpha=0.85)
ax.legend(loc="upper left"); ax.set_ylim(0, 1)
ax.set_title("Portfolio Weights over Time (monthly rebalance)")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/weights.png", dpi=130); plt.close(fig)

summary.round(4).to_csv("performance_summary.csv")
trans_df.round(4).to_csv("transition_matrix.csv")
print(f"\nSaved figures to ./{FIGDIR}/ and CSV outputs. Done.")
