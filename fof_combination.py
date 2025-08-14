import pandas as pd
import numpy as np
import os

# Loading data
df_preds = pd.read_csv("../xgb_preds_daily.csv", parse_dates=["date"])
df_mu = df_preds.pivot_table(index="date", columns="fund", values="pred").sort_index()
df_mu = df_mu.ffill().dropna(how="all")

df_risk_raw = pd.read_csv("../fund_risk_daily.csv", parse_dates=["date"])
risk_types = ["volatility_20", "cvar_20", "mdd_20"]

risk_dict = {}
for r in risk_types:
    risk_dict[r] = df_risk_raw.pivot(index="date", columns="fund", values=r).sort_index()  # 滞后 1 日

df_benchmark = pd.read_csv("../benchmark_ret.csv", index_col=0, parse_dates=True)["benchmark"]

ret_df = pd.read_csv("../returns_log.csv", index_col=0, parse_dates=True)

# Align the dates of all data
common_dates = df_mu.index
common_dates = common_dates.intersection(df_benchmark.index).intersection(ret_df.index)
for r in risk_dict:
    common_dates = common_dates.intersection(risk_dict[r].index)

df_mu = df_mu.loc[common_dates]
df_benchmark = df_benchmark.loc[common_dates]
ret_df = ret_df.loc[common_dates]
for r in risk_dict:
    risk_dict[r] = risk_dict[r].loc[common_dates]

# Alpha/beta rolling calculation function (based on historical real returns and benchmark returns)
def compute_alpha_beta_series(ret_df, benchmark, window=60, min_periods=20):
    alphas = pd.DataFrame(index=ret_df.index, columns=ret_df.columns, dtype=float)
    betas = pd.DataFrame(index=ret_df.index, columns=ret_df.columns, dtype=float)
    for i in range(window, len(ret_df)):
        idx = ret_df.index[i - window:i]
        for fund in ret_df.columns:
            y = ret_df.loc[idx, fund].dropna()
            x = benchmark.loc[y.index].dropna()
            common_idx = y.index.intersection(x.index)
            if len(common_idx) >= min_periods:
                model = np.polyfit(x.loc[common_idx], y.loc[common_idx], 1)
                betas.iloc[i, ret_df.columns.get_loc(fund)] = model[0]
                alphas.iloc[i, ret_df.columns.get_loc(fund)] = model[1]
    return alphas, betas

# Compute α and β, lagged forward by 1 day (to avoid future information leakage)
alphas_df, betas_df = compute_alpha_beta_series(ret_df, df_benchmark, window=60)
alphas_df = alphas_df.shift(1)
betas_df = betas_df.shift(1)

# Core: Rolling generation weights (combining μ, α, β, risk)
def rolling_alpha_beta_mu_weights(
    mu_df, alphas_df, betas_df, risk_df,
    lookback=30, holding=3,
    risk_clip=(0.3, 3.0),
    alpha_weight=0.4
):
    dates = mu_df.index
    weights_list = []
    last_w = pd.Series(0.0, index=mu_df.columns)

    for i, date in enumerate(dates):
        if i < lookback:
            weights_list.append(last_w)
            continue

        if i % holding != 0:
            weights_list.append(last_w)
            continue

        valid_funds = mu_df.columns[mu_df.loc[date].notna()]
        if len(valid_funds) == 0:
            weights_list.append(last_w)
            continue

        # α score (intercept, non-negative)
        alpha_score = alphas_df.loc[date, valid_funds].fillna(0).clip(lower=0)
        # β score (take the inverse, the smaller the β, the better, non-negative)
        beta_score = (1 / (betas_df.loc[date, valid_funds].abs() + 1e-6)).clip(lower=0)

        # μ score (predicted return, non-negative and normalized)
        pred_score = mu_df.loc[date, valid_funds].clip(lower=0)
        pred_score /= pred_score.sum() if pred_score.sum() > 0 else 1

        # α, β score normalization
        alpha_score /= alpha_score.sum() if alpha_score.sum() > 0 else 1
        beta_score /= beta_score.sum() if beta_score.sum() > 0 else 1
        pred_score /= pred_score.sum() if pred_score.sum() > 0 else 1

        # Composite α/β score (equally weighted)
        alpha_beta_score = 0.5 * alpha_score + 0.5 * beta_score

        # Weighted mixture of αβ score and μ score
        combined = alpha_weight * alpha_beta_score + (1 - alpha_weight) * pred_score

        # Risk adjustment (inverse of risk factor)
        risk_today = risk_df.loc[date, valid_funds]
        risk_adj = (1 / (risk_today + 1e-6)).clip(lower=risk_clip[0], upper=risk_clip[1])
        combined *= risk_adj

        # Convert to weight (normalized)
        w = combined / combined.sum() if combined.sum() > 0 else pd.Series(1 / len(valid_funds), index=valid_funds)

        # Extreme loss of smooth protection
        # if len(valid_funds) < max(3, len(mu_df.columns) // 5):
        #     w = 0.7 * last_w + 0.3 * w

        w = w.reindex(mu_df.columns).fillna(0)
        last_w = w
        weights_list.append(w)

    return pd.DataFrame(weights_list, index=dates)

def risk_parity_weights(mu_df, risk_df):
    eps = 1e-6

    risk_df = risk_df.shift(1)

    risk_df = risk_df.apply(lambda row: row.fillna(row.mean()), axis=1)
    risk_df = risk_df.fillna(risk_df.mean().mean())

    # Risk Countdown
    inv_risk = 1 / (risk_df + 1e-6)

    # Combined with the predicted returns (allowing negative fine-tuning, but adding eps to avoid all zeros)
    score = (mu_df + eps) * inv_risk

    # clip prevents extreme negative weights
    score = score.clip(lower=0)

    # Normalized to weights
    weights = score.div(score.sum(axis=1), axis=0).fillna(0)

    return weights

os.makedirs("../portfolio_weights", exist_ok=True)

w_alpha_beta_vol = rolling_alpha_beta_mu_weights(df_mu, alphas_df, betas_df, risk_dict["volatility_20"])
w_alpha_beta_vol.to_csv("../portfolio_weights/alpha_beta_vol_weights_daily.csv")

w_alpha_beta_cvar = rolling_alpha_beta_mu_weights(df_mu, alphas_df, betas_df, risk_dict["cvar_20"])
w_alpha_beta_cvar.to_csv("../portfolio_weights/alpha_beta_cvar_weights_daily.csv")

w_risk_parity = risk_parity_weights(df_mu, risk_dict["volatility_20"])
w_risk_parity.to_csv("../portfolio_weights/risk_parity_weights_daily.csv")
