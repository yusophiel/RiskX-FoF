import pandas as pd
import numpy as np

# Forecast days of future earnings
horizon = 1
# Rolling window length for calculating risk indicators
roll_feat_win = 15

# Loading raw data
ret_df = pd.read_csv("../returns_log.csv", index_col=0, parse_dates=True)
benchmark = ret_df.mean(axis=1)
benchmark.name = "benchmark"
benchmark.to_csv("../benchmark_ret.csv")

df_nav = pd.read_csv("../clean_unit_nav.csv", index_col=0, parse_dates=True)
market_ret = ret_df.mean(axis=1)

all_dates = ret_df.index

# Constructing the characteristics of a single fund
features = []
for col in ret_df.columns:
    df = pd.DataFrame(index=all_dates)
    df["mom_5"] = ret_df[col].shift(1).rolling(5).mean() # Average return over the past 5 days
    df["mom_10"] = ret_df[col].shift(1).rolling(10).mean() # Average return over the past 10 days
    df["std_10"] = ret_df[col].shift(1).rolling(10).std() # Standard deviation of returns over the past 10 days
    df["mean_20"] = ret_df[col].shift(1).rolling(20).mean() # Average return over the past 20 days

    # Annualized volatility (20 days)
    df["volatility_20"] = ret_df[col].shift(1).rolling(roll_feat_win).std() * np.sqrt(252)

    # Maximum Drawdown (MDD)
    cum = (1 + ret_df[col].shift(1)).cumprod()
    roll_max = cum.shift(1).rolling(roll_feat_win).max()
    drawdown = (cum.shift(1) - roll_max) / roll_max
    df["mdd_20"] = drawdown.rolling(roll_feat_win).min()

    # CVaR
    def cvar_95(x):
        var = np.percentile(x, 5)
        return x[x <= var].mean()
    df["cvar_20"] = ret_df[col].shift(1).rolling(roll_feat_win).apply(cvar_95, raw=True)

    # Prediction target (target)
    df["target"] = np.log(df_nav[col].shift(-horizon) / df_nav[col])

    df["fund"] = col
    features.append(df)

# Combined features of all funds
all_data = pd.concat(features)
all_data.reset_index(inplace=True)
all_data.rename(columns={"index": "date"}, inplace=True)
all_data.to_csv("../xgb_features_with_risk.csv", index=False)

# Save daily risk factor data
df_risk = all_data[[
    "date", "fund",
    "volatility_20",
    "cvar_20",
    "mdd_20"
]].dropna()

df_risk_daily = df_risk.sort_values("date").drop_duplicates(subset=["date", "fund"], keep="last")
df_risk_daily.to_csv("../fund_risk_daily.csv", index=False)

print("Signature file saved：../xgb_features_with_risk.csv")
print("Risk factor file saved：../fund_risk_daily.csv")
