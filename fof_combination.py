import pandas as pd
import numpy as np
import os

# Loading forecast returns
df = pd.read_csv("../xgb_preds_daily.csv", parse_dates=["date"])
df_mu = df.pivot_table(index="date", columns="fund", values="pred").sort_index()
df_mu = df_mu.ffill().dropna(how="all")

# Loading risk factor data
df_risk = pd.read_csv("../fund_risk_daily.csv", parse_dates=["date"])
df_vol = df_risk.pivot(index="date", columns="fund", values="volatility_20").ffill()
df_cvar = df_risk.pivot(index="date", columns="fund", values="cvar_20").ffill()

# Loading Benchmark Returns
df_benchmark = pd.read_csv("../benchmark_ret.csv", index_col=0, parse_dates=True)["benchmark"]

# Alignment Date
common_dates = df_mu.index.intersection(df_vol.index).intersection(df_cvar.index).intersection(df_benchmark.index)
df_mu = df_mu.loc[common_dates]
df_vol = df_vol.loc[common_dates]
df_cvar = df_cvar.loc[common_dates]
df_benchmark = df_benchmark.loc[common_dates]

# Weight calculation function
def calc_weights(preds: pd.DataFrame, method: str = None, risk: pd.DataFrame = None,
                 benchmark: pd.Series = None, alpha_weight: float = 0.7) -> pd.DataFrame:
    weights_list = []

    for date, row in preds.iterrows():
        valid = row.dropna()
        if len(valid) == 0:
            continue

        elif method == "risk_parity":
            hist = preds.loc[:date].tail(20)
            stds = hist.std()
            stds = stds[valid.index]
            inv_risk = 1 / (stds + 1e-6)
            w = inv_risk / inv_risk.sum()

        elif method == "alpha_beta":
            if risk is None or benchmark is None:
                raise ValueError("Alpha_beta strategy requires risk + benchmark")

            alphas = []
            betas = []

            for fund in valid.index:
                y = preds[fund].loc[:date].dropna()
                x = benchmark.loc[y.index]

                if len(x) < 20:
                    alpha, beta = np.nan, np.nan
                else:
                    model = np.polyfit(x, y, 1)
                    beta = model[0]
                    alpha = model[1]

                alphas.append(alpha)
                betas.append(beta)

            alpha_score = pd.Series(alphas, index=valid.index).fillna(0)
            beta_score = pd.Series(betas, index=valid.index).fillna(0)

            alpha_score = alpha_score.clip(lower=0)
            beta_score = (1 / (beta_score.abs() + 1e-6)).clip(lower=0)

            alpha_score = alpha_score / alpha_score.sum() if alpha_score.sum() > 0 else alpha_score
            beta_score = beta_score / beta_score.sum() if beta_score.sum() > 0 else beta_score

            combined = alpha_weight * alpha_score + (1 - alpha_weight) * beta_score
            w = combined / combined.sum() if combined.sum() > 0 else pd.Series(1 / len(valid), index=valid.index)

        else:
            raise ValueError("Unknown method")

        w.name = date
        weights_list.append(w)

    return pd.DataFrame(weights_list)

# Generate weights for the two strategies
w_risk = calc_weights(df_mu, method="risk_parity")
w_alpha_beta_cvar = calc_weights(df_mu, method="alpha_beta", risk=df_cvar, benchmark=df_benchmark, alpha_weight=0.7)

os.makedirs("../portfolio_weights", exist_ok=True)
w_risk.to_csv("../portfolio_weights/risk_parity_weights_daily.csv")
w_alpha_beta_cvar.to_csv("../portfolio_weights/alpha_beta_cvar_weights_daily.csv")
