import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

def compute_alpha_beta(ret_series, benchmark_ret):
    common_index = ret_series.dropna().index.intersection(benchmark_ret.dropna().index)
    X = benchmark_ret.loc[common_index].values.reshape(-1, 1)
    y = ret_series.loc[common_index].values
    if len(X) < 20:
        return np.nan, np.nan
    model = LinearRegression().fit(X, y)
    return model.intercept_, model.coef_[0]


def rolling_backtest(ret_df, mu_df, risk_df=None, benchmark=None,
                     lookback=30, holding=3, cost_rate=0.001, gap=1):

    dates = ret_df.index
    port_nav = [0.0]
    w_hist = []
    w_dates = []
    t = lookback

    while t + holding < len(dates):
        train_start = t - lookback - gap
        train_end = t - gap
        if train_start < 0 or train_end <= train_start:
            t += holding
            continue

        mu_window = mu_df.iloc[train_start:train_end]
        ret_window = ret_df.iloc[train_start + 1:train_end + 1]

        alpha_beta = {}
        if benchmark is not None:
            for fund in ret_df.columns:
                alpha, beta = compute_alpha_beta(
                    ret_df[fund].iloc[train_start:train_end],
                    benchmark.iloc[train_start:train_end]
                )
                alpha_beta[fund] = (alpha, beta)

        X_train, y_train = [], []
        for i in range(mu_window.shape[0]):
            mu_row = mu_window.iloc[i].dropna()
            funds = mu_row.index
            if len(funds) == 0:
                continue

            if risk_df is not None:
                risk_row_all = risk_df.loc[dates[train_start + 1 + i]].reindex(funds)
                risk_vol = risk_row_all.get("volatility_20", pd.Series(0, index=funds)).fillna(0)
                risk_cvar = risk_row_all.get("cvar_20", pd.Series(0, index=funds)).fillna(0)
                risk_mdd = risk_row_all.get("mdd_20", pd.Series(0, index=funds)).fillna(0)
                risk_corr = risk_row_all.get("corr_20", pd.Series(0, index=funds)).fillna(0)

                z_vol = (risk_vol - risk_vol.mean()) / (risk_vol.std() + 1e-6)
                z_cvar = (risk_cvar - risk_cvar.mean()) / (risk_cvar.std() + 1e-6)
                z_mdd = (risk_mdd - risk_mdd.mean()) / (risk_mdd.std() + 1e-6)
                z_corr = (risk_corr - risk_corr.mean()) / (risk_corr.std() + 1e-6)

                risk_score = (
                    0.4 * z_vol + 0.3 * z_cvar + 0.2 * z_mdd + 0.1 * z_corr
                ).fillna(0)
                mu_risk = mu_row / (risk_score + 1e-6)
            else:
                mu_risk = pd.Series(0.0, index=funds)

            if benchmark is not None:
                alpha = [alpha_beta.get(f, (0, 0))[0] for f in funds]
                beta = [alpha_beta.get(f, (0, 0))[1] for f in funds]
                x = np.stack([mu_row.values, mu_risk.values, alpha, beta], axis=1)
            else:
                x = np.stack([mu_row.values, mu_risk.values], axis=1)

            y = ret_window.iloc[i][funds].values
            if np.isnan(y).all():
                continue

            mask = ~np.isnan(x).any(axis=1) & ~np.isnan(y)
            if mask.sum() == 0:
                continue

            X_train.append(x[mask])
            y_train.append(y[mask])

        if len(X_train) == 0:
            print(f"t={t}, no valid training data in window {train_start}:{train_end}")
            t += holding
            continue

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

        if len(X_train) == 0:
            print(f"t={t}, all samples have NaNs after filtering")
            t += holding
            continue

        model = RidgeCV(alphas=np.logspace(-3, 3, 10))
        model.fit(X_train, y_train)

        mu_today = mu_df.iloc[t].dropna()
        funds = mu_today.index
        if len(funds) == 0:
            print(f"t={t}, no valid mu for prediction")
            t += holding
            continue

        if risk_df is not None:
            risk_row_all = risk_df.loc[dates[t]].reindex(funds)
            risk_vol = risk_row_all.get("volatility_20", pd.Series(0, index=funds)).fillna(0)
            risk_cvar = risk_row_all.get("cvar_20", pd.Series(0, index=funds)).fillna(0)
            risk_mdd = risk_row_all.get("mdd_20", pd.Series(0, index=funds)).fillna(0)
            risk_corr = risk_row_all.get("corr_20", pd.Series(0, index=funds)).fillna(0)

            z_vol = (risk_vol - risk_vol.mean()) / (risk_vol.std() + 1e-6)
            z_cvar = (risk_cvar - risk_cvar.mean()) / (risk_cvar.std() + 1e-6)
            z_mdd = (risk_mdd - risk_mdd.mean()) / (risk_mdd.std() + 1e-6)
            z_corr = (risk_corr - risk_corr.mean()) / (risk_corr.std() + 1e-6)

            risk_score = (
                0.4 * z_vol + 0.3 * z_cvar + 0.2 * z_mdd + 0.1 * z_corr
            ).fillna(0)
            mu_risk = mu_today / (risk_score + 1e-6)
        else:
            mu_risk = pd.Series(0.0, index=funds)

        if benchmark is not None:
            alpha = [alpha_beta.get(f, (0, 0))[0] for f in funds]
            beta = [alpha_beta.get(f, (0, 0))[1] for f in funds]
            X_test = np.stack([mu_today.values, mu_risk.values, alpha, beta], axis=1)
        else:
            X_test = np.stack([mu_today.values, mu_risk.values], axis=1)

        raw_scores = pd.Series(model.predict(X_test), index=funds)
        adjusted_scores = raw_scores / (risk_score + 1e-6)
        weights = adjusted_scores.clip(lower=0)
        weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(weights), index=weights.index)

        w_all = pd.Series(0.0, index=ret_df.columns, name=dates[t])
        w_all[weights.index] = weights
        w_hist.append(w_all)
        w_dates.append(dates[t])

        future_ret = ret_df.iloc[t + 1][weights.index].fillna(0.0)
        port_ret = np.dot(weights.values, future_ret.values)

        if len(w_hist) >= 2:
            prev_w = w_hist[-2][weights.index].fillna(0)
            turnover = np.sum(np.abs(weights.values - prev_w.values))
        else:
            turnover = 0.0
        cost = turnover * cost_rate

        port_nav.append(port_nav[-1] + port_ret - cost)
        t += holding

    if len(w_dates) == 0:
        raise ValueError("There are no valid training rounds.")

    nav_series = pd.Series(np.exp(port_nav[1:]), index=w_dates, name="Ridge_Weighted")
    W = pd.DataFrame(w_hist)
    return nav_series, W


def calc_metrics(nav_df):
    nav_ret_df = nav_df.pct_change().dropna()
    perf = []
    for col in nav_ret_df:
        ret = nav_ret_df[col]
        ann_ret = (1 + ret.mean()) ** 252 - 1
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


def run_backtest_for_weights(weight_df: pd.DataFrame,
                             ret_df: pd.DataFrame,
                             cost_rate: float = 0.001) -> pd.Series:

    weight_df = weight_df.copy().sort_index()
    ret_df = ret_df.copy().sort_index()

    common_dates = weight_df.index.intersection(ret_df.index)
    weight_df = weight_df.loc[common_dates]
    ret_df = ret_df.loc[common_dates]

    nav = [0.0]
    prev_w = None

    for i, date in enumerate(weight_df.index):
        w = weight_df.loc[date].fillna(0)
        r = ret_df.loc[date].fillna(0)

        if prev_w is None:
            turnover = 0.0
        else:
            aligned_prev = prev_w.reindex(w.index).fillna(0)
            turnover = np.abs(w - aligned_prev).sum()

        cost = turnover * cost_rate
        port_ret = np.dot(w.values, r.values)
        nav.append(nav[-1] + port_ret - cost)
        prev_w = w

    nav_series = pd.Series(np.exp(nav[1:]), index=weight_df.index, name=weight_df.name if hasattr(weight_df, 'name') else "strategy")
    return nav_series


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
    risk_df = df_risk.pivot(index="date", columns="fund")

    common_dates = ret_df.index.intersection(mu_df.index).intersection(risk_df.index)
    ret_df = ret_df.loc[common_dates].sort_index()
    mu_df = mu_df.loc[common_dates].sort_index()
    risk_df = risk_df.loc[common_dates].sort_index()

    try:
        benchmark = pd.read_csv("../benchmark_ret.csv", index_col=0, parse_dates=True).iloc[:, 0]
        benchmark = benchmark.loc[common_dates]
    except Exception:
        benchmark = None

    assert mu_df.shape[0] == ret_df.shape[0], "The number of rows in mu_df and ret_df is inconsistent. "
    print(f"Data alignment successful, totaling {mu_df.shape[0]} trading days")

    nav_ridge, weights = rolling_backtest(
        ret_df, mu_df, risk_df=risk_df, benchmark=benchmark,
        lookback=30, holding=3, gap=1
    )

    strategy_dir = "../portfolio_weights"
    nav_series_list = []
    strategy_names = []

    for file in os.listdir(strategy_dir):
        if file.endswith(".csv"):
            path = os.path.join(strategy_dir, file)
            weight_df = pd.read_csv(path, index_col=0, parse_dates=True)
            strategy_name = os.path.splitext(file)[0]
            weight_df.name = strategy_name

            nav = run_backtest_for_weights(weight_df, ret_df)
            nav.name = strategy_name

            nav_series_list.append(nav)
            strategy_names.append(strategy_name)

    nav_ridge.name = "ridge_regression"
    nav_series_list.append(nav_ridge)
    strategy_names.append("ridge_regression")

    equal_w = pd.DataFrame(1.0 / weights.shape[1], index=weights.index, columns=weights.columns)
    equal_ret = (ret_df * equal_w).sum(axis=1)
    equal_nav = (1 + equal_ret).cumprod()
    equal_nav.name = "Equal_Weighted"
    nav_series_list.append(equal_nav)
    strategy_names.append("Equal_Weighted")

    plot_nav_series(*nav_series_list, title="All Strategies NAV Comparison")
    nav_df = pd.concat(nav_series_list, axis=1)

    metrics_list = []
    for col in nav_df.columns:
        m = calc_metrics(nav_df[[col]])
        m.index = [col]
        metrics_list.append(m)
    metrics_df = pd.concat(metrics_list)

    print("All Strategy Performance Metrics:\n")
    print(metrics_df)

    os.makedirs("../out", exist_ok=True)
    nav_df.to_csv("../out/nav_all_strategies.csv")
    metrics_df.to_csv("../out/performance_metrics_all.csv")


if __name__ == "__main__":
    main()