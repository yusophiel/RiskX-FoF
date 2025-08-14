import pandas as pd
import numpy as np
import os
import re

# The path of the fund net value file to be read
file_paths = [
    "../data/fund1.xlsx",
    "../data/fund2.xlsx",
    "../data/fund3.xlsx",
    "..."
]

# Read and normalize a single fund file
def load_and_standardize(filepath):
    engine = "openpyxl" if filepath.endswith("xlsx") else None
    df = pd.read_excel(filepath, engine=engine)
    df.columns = df.columns.str.strip()

    def pick(col_regex):
        cols = [c for c in df.columns if re.search(col_regex, c)]
        return cols[0] if cols else None

    date_col = pick(r"date")
    unit_col = pick(r"Unit\s*value")
    acc_col = pick(r"Cumulative\s*value")

    if date_col is None or unit_col is None:
        raise ValueError(f"❌ {filepath} The Date or Unit Net Value column is missing, please check")

    name = os.path.basename(filepath).split('.')[0]
    use_cols = [date_col, unit_col]
    cols_new = ["date", f"{name}_Unit_value"]
    if acc_col is not None:
        use_cols.append(acc_col)
        cols_new.append(f"{name}_Cumulative_value")

    df = df[use_cols].copy()
    df.columns = cols_new
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df

# Batch read all fund files and merge
merged = None
for p in file_paths:
    try:
        df_i = load_and_standardize(p)
        merged = df_i if merged is None else merged.join(df_i, how="outer")
    except Exception as e:
        print(f"[SKIP] Read failed: {p} - {e}")

merged = merged.sort_index()

unit_nav_df = merged.filter(regex="_Unit_value$")

# Data Cleaning Steps
# 1. Delete all empty dates
unit_nav_df = unit_nav_df.dropna(how="all")
# 2. Delete funds with historical effective values less than 50 days
unit_nav_df = unit_nav_df.loc[:, unit_nav_df.notna().sum() >= 50]
# 3. Delete days when the number of valid funds per day is less than 3
unit_nav_df = unit_nav_df[unit_nav_df.notna().sum(axis=1) >= 3]
# 4. Forward filling of missing values + removal of outliers (≤0 or =0)
unit_nav_df = unit_nav_df.ffill()
unit_nav_df = unit_nav_df.replace(0, np.nan)
unit_nav_df = unit_nav_df.where(unit_nav_df > 0).dropna(how="all")
# 5. Calculate logarithmic returns
ret_df = np.log(unit_nav_df).diff().dropna()

# Market Portfolio Returns (Simple Equal Weighting)
market_ret = ret_df.mean(axis=1)

# Calculating Alpha/Beta Factors
alphas = {}
betas = {}

for fund in ret_df.columns:
    aligned = pd.concat([ret_df[fund], market_ret], axis=1).dropna()
    if len(aligned) < 30:
        continue
    X = aligned.iloc[:, 1].values.reshape(-1, 1)
    y = aligned.iloc[:, 0].values
    X = np.concatenate([np.ones_like(X), X], axis=1)

    # Least Squares Regression [Intercept=Alpha, Slope=Beta]
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    alphas[fund] = coef[0]
    betas[fund] = coef[1]

alpha_beta_df = pd.DataFrame({"Alpha": alphas, "Beta": betas})
alpha_beta_df.index.name = "Fund"
alpha_beta_df.to_csv("../fund_alpha_beta.csv")

unit_nav_df.to_csv("../clean_unit_nav.csv")
ret_df.to_csv("../returns_log.csv")

print("Cleaning is completed and Alpha/Beta factors are output")
print(f"Amount of Reserved Funds: {unit_nav_df.shape[1]}")
print(f"Transaction date range: {unit_nav_df.index.min().date()} → {unit_nav_df.index.max().date()}")
