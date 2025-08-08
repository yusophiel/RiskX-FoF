import pandas as pd
import numpy as np

# Predicting earnings for the next few days
horizon = 3
# The length of the rolling window when calculating risk indicators (volatility, CVaR, maximum drawdown, correlation)
roll_feat_win = 15

# Loading raw data
ret_df = pd.read_csv("../returns_log.csv", index_col=0, parse_dates=True)
benchmark = ret_df.mean(axis=1)
benchmark.name = "benchmark"
benchmark.to_csv("../benchmark_ret.csv")

df_nav = pd.read_csv("../clean_unit_nav.csv", index_col=0, parse_dates=True)
market_ret = ret_df.mean(axis=1)

# Cyclic construction of single fund characteristics
features = []
for col in ret_df.columns:
    df = pd.DataFrame(index=ret_df.index)
    # Historical return factors
    df["ret"] = ret_df[col]
    df["mom_5"] = df["ret"].rolling(5).mean()
    df["mom_10"] = df["ret"].rolling(10).mean()
    df["std_10"] = df["ret"].rolling(10).std()
    df["mean_20"] = df["ret"].rolling(20).mean()

    # Risk factors
    df["volatility_20"] = df["ret"].rolling(roll_feat_win).std() * np.sqrt(252)

    nav = df_nav[col]
    cum = (1 + df["ret"]).cumprod()
    roll_max = cum.rolling(roll_feat_win).max()
    drawdown = (cum - roll_max) / roll_max
    df["mdd_20"] = drawdown.rolling(roll_feat_win).min()

    def cvar_95(x):
        var = np.percentile(x, 5)
        return x[x <= var].mean()

    df["cvar_20"] = df["ret"].rolling(roll_feat_win).apply(cvar_95, raw=True)
    df["corr_20"] = df["ret"].rolling(roll_feat_win).corr(market_ret)
    df["target"] = np.log(df_nav[col].shift(-horizon) / df_nav[col])
    df["fund"] = col
    features.append(df.dropna())

# Combine all fund characteristics
all_data = pd.concat(features)
all_data.reset_index(inplace=True)
all_data.rename(columns={"index": "date"}, inplace=True)
all_data.to_csv("../xgb_features_with_risk.csv", index=False)

# Extract daily fund risk profile
df_risk = all_data[[
    "date", "fund",
    "volatility_20",
    "cvar_20",
    "mdd_20",
    "corr_20"
]].dropna()

df_risk_daily = df_risk.sort_values("date").drop_duplicates(subset=["date", "fund"], keep="last")
df_risk_daily.to_csv("../fund_risk_daily.csv", index=False)


