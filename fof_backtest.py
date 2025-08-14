import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LinearRegression
import warnings

warnings.filterwarnings("ignore")


# Alpha/Beta calculation: fixed window aperture
def compute_alpha_beta(ret_series, benchmark_ret, window=30, end_idx=None):
    if end_idx is not None:
        ret_series = ret_series.iloc[:end_idx]
        benchmark_ret = benchmark_ret.iloc[:end_idx]
    common_index = ret_series.dropna().index.intersection(benchmark_ret.dropna().index)
    if len(common_index) < window:
        return np.nan, np.nan
    idx = common_index[-window:]
    X = benchmark_ret.loc[idx].values.reshape(-1, 1)
    y = ret_series.loc[idx].values
    model = LinearRegression().fit(X, y)
    return model.intercept_, model.coef_[0]


# Rolling backtest and generating weights
def rolling_backtest_weights(ret_df, mu_df, risk_df=None, benchmark=None,
                             lookback=30, holding=3, gap=1, fill_non_rebalance=True,
                             risk_adjust_post=False):
    dates = ret_df.index
    w_hist = []
    t = lookback + gap

    while t < len(dates):
        # Constructing training intervals
        train_start = t - lookback - gap
        train_end = t - gap
        if train_start < 0 or train_end <= train_start:
            t += holding
            continue

        mu_window = mu_df.iloc[train_start:train_end]
        ret_window = ret_df.iloc[train_start + 1:train_end + 1]

        # α/β calculation (fixed window)
        alpha_beta = {}
        if benchmark is not None:
            for fund in ret_df.columns:
                alpha, beta = compute_alpha_beta(
                    ret_df[fund].iloc[:train_end],
                    benchmark.iloc[:train_end]
                )
                alpha_beta[fund] = (alpha, beta)

        # Construct training set (X=μ, α, β, y=real return)
        X_train, y_train = [], []
        for i in range(mu_window.shape[0]):
            mu_row = mu_window.iloc[i].dropna()
            funds = mu_row.index
            if len(funds) == 0:
                continue
            if benchmark is not None:
                alpha = [alpha_beta.get(f, (0, 0))[0] for f in funds]
                beta = [alpha_beta.get(f, (0, 0))[1] for f in funds]
                x = np.stack([mu_row.values, alpha, beta], axis=1)
            else:
                x = mu_row.values.reshape(-1, 1)
            y = ret_window.iloc[i][funds].values
            mask = ~np.isnan(x).any(axis=1) & ~np.isnan(y)
            if mask.sum() > 0:
                X_train.append(x[mask])
                y_train.append(y[mask])

        if len(X_train) == 0:
            t += holding
            continue

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

        # Ridge regression fits the mapping from μ to returns
        model = RidgeCV(alphas=np.logspace(-3, 3, 10))
        model.fit(X_train, y_train)

        # Rebalancing Day Forecast
        mu_today = mu_df.iloc[t].dropna()
        funds = mu_today.index
        if len(funds) == 0:
            t += holding
            continue

        if benchmark is not None:
            alpha = [alpha_beta.get(f, (0, 0))[0] for f in funds]
            beta = [alpha_beta.get(f, (0, 0))[1] for f in funds]
            X_test = np.stack([mu_today.values, alpha, beta], axis=1)
        else:
            X_test = mu_today.values.reshape(-1, 1)

        raw_scores = pd.Series(model.predict(X_test), index=funds)

        # Risk factor adjustment
        if risk_adjust_post and risk_df is not None:
            risk_row_all = risk_df.loc[dates[t]].reindex(funds)

            def zscore(s):
                if s.isna().all() or len(s.dropna()) <= 1:
                    return pd.Series(0, index=s.index)
                z = (s - s.mean()) / (s.std() + 1e-6)
                return z.fillna(0)

            def safe_metric(metric_name):
                try:
                    s = risk_row_all.loc[metric_name].reindex(funds)
                except KeyError:
                    return pd.Series(0, index=funds)

                if s.isna().all():
                    return pd.Series(0, index=funds)

                return s.fillna(0)

            z_vol = zscore(safe_metric("volatility_20"))
            z_cvar = zscore(safe_metric("cvar_20"))
            z_mdd = zscore(safe_metric("mdd_20"))

            # Composite Risk Score (Volatility 40% + CVaR 30% + MDD 20%)
            risk_score = 0.4 * z_vol + 0.3 * z_cvar + 0.2 * z_mdd
            if (risk_score.abs() < 1e-8).all():
                risk_score[:] = 0

            # Quantile Clipping & Min-Max Normalization
            risk_score = risk_score.clip(
                lower=risk_score.quantile(0.05),
                upper=risk_score.quantile(0.95)
            )
            risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min() + 1e-6)

            # The inverse of risk as an adjustment factor
            risk_adj = 1 / (risk_score + 1e-6)
            risk_adj = risk_adj.clip(lower=0.5, upper=2.0)

            raw_scores *= risk_adj

        # Weight normalization
        weights = raw_scores.clip(lower=0)
        weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(weights), index=weights.index)

        w_all = pd.Series(0.0, index=ret_df.columns, name=dates[t])
        w_all[weights.index] = weights
        w_hist.append(w_all)

        t += holding

    W = pd.DataFrame(w_hist)
    if fill_non_rebalance:
        W = W.reindex(dates).ffill().fillna(0)
    return W


# Backtesting Logic (T+1 Execution)
def run_backtest_for_weights(weight_df: pd.DataFrame,
                             ret_df: pd.DataFrame,
                             cost_rate: float = 0.001,
                             t_plus_delay: int = 1) -> pd.Series:
    weight_df = weight_df.copy().sort_index()
    ret_df = ret_df.copy().sort_index()

    common_dates = weight_df.index.intersection(ret_df.index)
    weight_df = weight_df.loc[common_dates]
    ret_df = ret_df.loc[common_dates]

    nav = [1.0]
    prev_w = weight_df.iloc[0]

    # Initial position cost
    turnover_init = np.abs(prev_w).sum()
    init_cost = turnover_init * cost_rate
    nav[0] = nav[0] * (1 - init_cost)

    for i in range(len(weight_df) - t_plus_delay):
        w = weight_df.iloc[i]  # 调仓日权重
        r_next = ret_df.iloc[i + t_plus_delay]  # 下一日收益

        # Turnover cost
        turnover = np.abs(w - prev_w).sum()
        cost = turnover * cost_rate

        port_ret = np.dot(w.values, r_next.values)
        nav.append(nav[-1] * (1 + port_ret - cost))
        prev_w = w

    strategy_name = getattr(weight_df, "name", None) or "strategy"
    return pd.Series(nav, index=weight_df.index[:len(nav)], name=strategy_name)


# Indicator calculation
def calc_metrics(nav_df):
    nav_ret_df = nav_df.pct_change().dropna()
    perf = []
    for col in nav_ret_df.columns:
        ret = nav_ret_df[col]
        ann_ret = (1 + ret).prod() ** (252 / len(ret)) - 1
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum = (1 + ret).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
        perf.append({
            "Strategy": col,
            "Annual Return": ann_ret,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
            "Calmar": calmar
        })
    return pd.DataFrame(perf).set_index("Strategy")

# Draw the equity curve
def plot_nav_series(*nav_series_list, title="Net Asset Value Curve"):
    plt.figure(figsize=(12, 5))
    for nav in nav_series_list:
        plt.plot(nav, label=nav.name)
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../out/Strategy Performance.png", dpi=300)
    plt.show()


def main():
    ret_df = pd.read_csv("../returns_log.csv", index_col=0, parse_dates=True)
    df = pd.read_csv("../xgb_preds_daily.csv", parse_dates=["date"])
    mu_df = df.pivot_table(index="date", columns="fund", values="pred")

    df_risk = pd.read_csv("../fund_risk_daily.csv", parse_dates=["date"])
    risk_df = df_risk.pivot_table(
        index="date",
        columns="fund",
        values=["volatility_20", "cvar_20", "mdd_20"]
    ).sort_index()

    # Align trading days
    common_dates = ret_df.index.intersection(mu_df.index)

    ret_df = ret_df.loc[common_dates].sort_index()
    mu_df = mu_df.loc[common_dates].sort_index()

    if risk_df is not None:
        risk_df = risk_df.reindex(common_dates).sort_index()

    # Benchmark Return
    try:
        benchmark = pd.read_csv("../benchmark_ret.csv", index_col=0, parse_dates=True).iloc[:, 0]
        benchmark = benchmark.loc[common_dates]
    except Exception:
        benchmark = None

    print(f"Data alignment successful, {mu_df.shape[0]} trading days")

    # Ridge regression strategy (risk adjustment after prediction)
    weights_ridge = rolling_backtest_weights(ret_df, mu_df, risk_df, benchmark,
                                             lookback=30, holding=3, gap=1,
                                             fill_non_rebalance=True,
                                             risk_adjust_post=True)
    nav_ridge = run_backtest_for_weights(weights_ridge, ret_df, t_plus_delay=1)
    nav_ridge.name = "ridge_regression"
    nav_series_list = [nav_ridge]

    # Equal Weight Strategy
    equal_w = pd.DataFrame(1.0 / weights_ridge.shape[1],
                           index=weights_ridge.index,
                           columns=weights_ridge.columns)
    equal_nav = run_backtest_for_weights(equal_w, ret_df, t_plus_delay=1)
    equal_nav.name = "Equal_Weighted"
    nav_series_list.append(equal_nav)

    strategy_names = []
    # Read other strategies in the portfolio_weights directory
    portfolio_dir = "../portfolio_weights"
    if os.path.exists(portfolio_dir):
        for file in os.listdir(portfolio_dir):
            if file.endswith(".csv"):
                path = os.path.join(portfolio_dir, file)
                weight_df = pd.read_csv(path, index_col=0, parse_dates=True)
                strategy_name = os.path.splitext(file)[0]
                weight_df.name = strategy_name
                nav = run_backtest_for_weights(weight_df, ret_df, t_plus_delay=1)
                nav.name = strategy_name
                nav_series_list.append(nav)
                strategy_names.append(strategy_name)

    # Merge all equity curves and calculate indicators
    nav_df = pd.concat(nav_series_list, axis=1)
    metrics_df = calc_metrics(nav_df)
    print(metrics_df)

    os.makedirs("../out", exist_ok=True)
    plot_nav_series(*nav_series_list, title="All Strategies NAV Comparison")
    nav_df.to_csv("../out/nav_all_strategies.csv")
    metrics_df.to_csv("../out/performance_metrics_all.csv")


if __name__ == "__main__":
    main()
