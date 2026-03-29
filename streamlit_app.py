import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.optimize import minimize
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Investment Analytics System",
    page_icon="📈",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Automated Investment Analytics & Portfolio Optimization")
st.markdown("**Bhawini Singh** &nbsp;|&nbsp; Northeastern University")
st.markdown("---")

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Parameters")

st.sidebar.markdown("**Asset Universe**")
st.sidebar.caption("Enter any tickers available on Yahoo Finance, separated by commas. Examples: AAPL, MSFT, TSLA, SPY, GLD, BTC-USD")

raw_input = st.sidebar.text_area(
    "Tickers",
    value="XLK, XLF, XLE, XLV, XLI, XLP, XLU, TLT, GLD, SHY",
    height=100,
    placeholder="AAPL, MSFT, GOOGL, AMZN, TSLA, GLD, TLT"
)

# Parse and validate tickers
def parse_tickers(raw):
    return [t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip()]

raw_tickers = parse_tickers(raw_input)

@st.cache_data(show_spinner=False)
def validate_tickers(tickers, start, end):
    valid, invalid = [], []
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty or len(df) < 60:
                invalid.append(t)
            else:
                valid.append(t)
        except Exception:
            invalid.append(t)
    return valid, invalid

if len(raw_tickers) < 2:
    st.sidebar.error("Enter at least 2 tickers.")
    selected_assets = []
elif len(raw_tickers) > 20:
    st.sidebar.error("Maximum 20 tickers at a time.")
    selected_assets = []
else:
    with st.spinner("Validating tickers..."):
        selected_assets, invalid_tickers = validate_tickers(
            tuple(raw_tickers), str(start_date), str(end_date)
        )
    if invalid_tickers:
        st.sidebar.warning(f"Could not find data for: {', '.join(invalid_tickers)}. They will be skipped.")
    if len(selected_assets) < 2:
        st.sidebar.error("Not enough valid tickers. Please check your input.")
        selected_assets = []
    else:
        st.sidebar.success(f"{len(selected_assets)} valid tickers: {', '.join(selected_assets)}")

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date   = st.sidebar.date_input("End Date",   value=pd.to_datetime("2024-12-31"))

rf_rate = st.sidebar.slider(
    "Risk-Free Rate (%)", min_value=0.0, max_value=6.0, value=2.0, step=0.1
) / 100

min_weight = st.sidebar.slider("Min Weight per Asset (%)", 0, 10, 2) / 100
max_weight = st.sidebar.slider("Max Weight per Asset (%)", 10, 50, 35) / 100

run_btn = st.sidebar.button("🚀 Run Optimization", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("*Data: Yahoo Finance*")

# ── Validation ────────────────────────────────────────────────────────────────
if len(selected_assets) < 2:
    st.stop()

if not run_btn:
    st.info("👈 Enter any tickers in the sidebar (stocks, ETFs, crypto), configure parameters, then click **Run Optimization**.")
    st.markdown("""
    **Example universes to try:**
    - **Mag 7:** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
    - **Sector ETFs:** XLK, XLF, XLE, XLV, XLI, XLP, XLU, TLT, GLD, SHY
    - **Global macro:** SPY, EFA, EEM, TLT, GLD, DJP, UUP
    - **Crypto + equities:** BTC-USD, ETH-USD, AAPL, MSFT, GLD
    - **Your own mix:** any valid Yahoo Finance ticker
    """)
    st.stop()

# ── Data fetch ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_data(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.ffill(limit=5).dropna(how="all")
    return prices

with st.spinner("Fetching market data from Yahoo Finance..."):
    all_tickers = selected_assets + ["SPY"]
    prices_raw  = fetch_data(tuple(all_tickers), str(start_date), str(end_date))

prices    = prices_raw[selected_assets]
spy       = prices_raw[["SPY"]]
returns   = np.log(prices / prices.shift(1)).dropna()
spy_ret   = np.log(spy / spy.shift(1)).dropna()

n  = len(selected_assets)
mu = returns.mean() * 252
sigma = returns.cov() * 252
rf = rf_rate
rf_daily = rf / 252

# ── Helper functions ──────────────────────────────────────────────────────────
def portfolio_stats(w):
    w = np.array(w)
    r = float(w @ mu)
    v = float(np.sqrt(w @ sigma.values @ w))
    s = (r - rf) / v
    return r, v, s

def neg_sharpe(w):
    return -portfolio_stats(w)[2]

bounds      = [(min_weight, max_weight)] * n
constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
w0          = np.ones(n) / n

# ── Optimization ──────────────────────────────────────────────────────────────
with st.spinner("Running portfolio optimization..."):

    # Min Variance
    w_var = cp.Variable(n)
    prob  = cp.Problem(cp.Minimize(cp.quad_form(w_var, sigma.values)),
                       [cp.sum(w_var) == 1, w_var >= min_weight, w_var <= max_weight])
    prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8)
    w_mv = pd.Series(w_var.value, index=selected_assets)

    # Max Sharpe
    res_ms = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})
    w_ms = pd.Series(res_ms.x, index=selected_assets)

    # Risk Parity
    def rp_obj(w):
        w   = np.array(w)
        pv  = np.sqrt(w @ sigma.values @ w)
        mrc = (sigma.values @ w) / pv
        trc = w * mrc
        return np.sum((trc - pv / n) ** 2)

    res_rp = minimize(rp_obj, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1e-14})
    w_rp = pd.Series(res_rp.x, index=selected_assets)

    # Equal Weight
    w_ew = pd.Series(np.ones(n) / n, index=selected_assets)

    strategies = {
        "Min Variance" : w_mv,
        "Max Sharpe"   : w_ms,
        "Risk Parity"  : w_rp,
        "Equal Weight" : w_ew,
    }

    summary = pd.DataFrame({
        name: {
            "Exp. Return %"  : round(portfolio_stats(w)[0] * 100, 3),
            "Volatility %"   : round(portfolio_stats(w)[1] * 100, 3),
            "Sharpe Ratio"   : round(portfolio_stats(w)[2], 3),
        }
        for name, w in strategies.items()
    }).T

# ── Layout ────────────────────────────────────────────────────────────────────
st.success("Optimization complete.")

# ── Section 1: Strategy comparison ───────────────────────────────────────────
st.subheader("📊 Strategy Comparison")
st.dataframe(summary.style.highlight_max(axis=0, color="#c8f7c5")
                          .highlight_min(axis=0, color="#f7c8c8")
                          .format("{:.3f}"),
             use_container_width=True)

# ── Section 2: Cumulative returns ─────────────────────────────────────────────
st.subheader("📈 Cumulative Strategy Returns")

strat_rets = pd.DataFrame({
    name: returns @ w for name, w in strategies.items()
})
cumulative = (1 + strat_rets).cumprod() * 100

fig1, ax1 = plt.subplots(figsize=(12, 4))
colors = {"Min Variance": "#2196F3", "Max Sharpe": "#4CAF50",
          "Risk Parity": "#FF9800", "Equal Weight": "#9E9E9E"}
for col in cumulative.columns:
    ax1.plot(cumulative.index, cumulative[col],
             linewidth=1.6, label=col, color=colors[col])
ax1.set_ylabel("Portfolio Value ($)")
ax1.set_title("Cumulative Returns — Base $100")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig1)

# ── Section 3: Portfolio weights ──────────────────────────────────────────────
st.subheader("⚖️ Portfolio Weights by Strategy")

weights_df = pd.DataFrame(strategies) * 100
fig2, ax2  = plt.subplots(figsize=(12, 4))
x     = np.arange(n)
width = 0.2
bar_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9E9E9E"]

for i, (col, color) in enumerate(zip(weights_df.columns, bar_colors)):
    ax2.bar(x + i * width, weights_df[col], width,
            label=col, color=color, alpha=0.85)

ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(selected_assets, fontsize=9)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_ylabel("Weight (%)")
ax2.set_title("Portfolio Weights by Strategy")
ax2.legend(fontsize=9)
ax2.axhline(min_weight * 100, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
ax2.axhline(max_weight * 100, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)

# ── Section 4: Efficient frontier ─────────────────────────────────────────────
st.subheader("🎯 Efficient Frontier")

np.random.seed(42)
mc_r, mc_v, mc_s = [], [], []
for _ in range(3000):
    w = np.random.dirichlet(np.ones(n))
    w = np.clip(w, min_weight, max_weight)
    w /= w.sum()
    r, v, s = portfolio_stats(w)
    mc_r.append(r * 100); mc_v.append(v * 100); mc_s.append(s)

fig3, ax3 = plt.subplots(figsize=(10, 5))
sc = ax3.scatter(mc_v, mc_r, c=mc_s, cmap="viridis", alpha=0.4, s=6)
plt.colorbar(sc, ax=ax3, label="Sharpe Ratio")

pts = {"Min Var": (w_mv, "blue", "^", 120),
       "Max Sharpe": (w_ms, "green", "*", 200),
       "Risk Parity": (w_rp, "orange", "D", 100),
       "Equal Wt": (w_ew, "gray", "s", 100)}
for label, (w, c, m, s) in pts.items():
    rv, vv, _ = portfolio_stats(w)
    ax3.scatter(vv * 100, rv * 100, color=c, marker=m, s=s, zorder=5, label=label)
    ax3.annotate(label, (vv * 100, rv * 100),
                 textcoords="offset points", xytext=(6, 3), fontsize=8)

ax3.set_xlabel("Volatility (%)"); ax3.set_ylabel("Expected Return (%)")
ax3.set_title("Efficient Frontier (3,000 random portfolios)")
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)

# ── Section 5: Momentum signals ───────────────────────────────────────────────
st.subheader("📡 Momentum Signals (6-Month)")

momentum = returns.rolling(126).sum().iloc[-1] * 100
mom_colors = ["#4CAF50" if v > 0 else "#f44336" for v in momentum.sort_values()]

fig4, ax4 = plt.subplots(figsize=(10, 4))
momentum.sort_values().plot(kind="barh", ax=ax4, color=mom_colors, alpha=0.85)
ax4.axvline(0, color="black", linewidth=0.8)
ax4.set_xlabel("6-Month Cumulative Log Return (%)")
ax4.set_title("Momentum Signal by Asset (most recent 6 months)")
ax4.grid(axis="x", alpha=0.3)
plt.tight_layout()
st.pyplot(fig4)

# ── Section 6: Correlation heatmap ───────────────────────────────────────────
st.subheader("🔗 Correlation Matrix")

corr = returns.corr()
fig5, ax5 = plt.subplots(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax5)
ax5.set_title("Asset Correlation Matrix — Daily Returns")
plt.tight_layout()
st.pyplot(fig5)

# ── Section 7: Risk metrics table ─────────────────────────────────────────────
st.subheader("🛡️ Asset Risk Metrics")

aligned   = returns.join(spy_ret, how="inner")
spy_col   = spy_ret.columns[0]
betas     = {}
for t in selected_assets:
    cov   = np.cov(aligned[t], aligned[spy_col])
    betas[t] = round(cov[0, 1] / cov[1, 1], 3)

def max_dd(s):
    c = (1 + s).cumprod()
    return ((c - c.cummax()) / c.cummax()).min()

risk_metrics = pd.DataFrame({
    "Ann. Vol %"      : (returns.std() * np.sqrt(252) * 100).round(2),
    "Beta vs SPY"     : pd.Series(betas),
    "Max Drawdown %"  : (returns.apply(max_dd) * 100).round(2),
    "6M Momentum %"   : momentum.round(2),
}).sort_values("Ann. Vol %")

st.dataframe(risk_metrics.style.format("{:.2f}"), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built by Bhawini Singh | Northeastern University | For research purposes only. Not investment advice.")
