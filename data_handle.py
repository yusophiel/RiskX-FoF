import pandas as pd
import numpy as np
import os
import re

file_paths = [
    "../data/fund_1.xlsx",
    "../data/fund_2.xlsx",
    "../data/fund_3.xlsx",
    # ...
]


def load_and_standardize(filepath):
    engine = "openpyxl" if filepath.endswith("xlsx") else None
    df = pd.read_excel(filepath, engine=engine)
    df.columns = df.columns.str.strip()

    def pick(col_regex):
        cols = [c for c in df.columns if re.search(col_regex, c)]
        return cols[0] if cols else None

    date_col = pick(r"date")
    unit_col = pick(r"Unit\s*Value")
    acc_col = pick(r"Accumulated\s*Value")

    if date_col is None or unit_col is None:
        raise ValueError(f"❌ {filepath} lack data")

    name = os.path.basename(filepath).split('.')[0]
    use_cols = [date_col, unit_col]
    cols_new = ["date", f"{name}_UnitValue"]
    if acc_col is not None:
        use_cols.append(acc_col)
        cols_new.append(f"{name}_AccumulatedValue")

    df = df[use_cols].copy()
    df.columns = cols_new
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df


merged = None
for p in file_paths:
    try:
        df_i = load_and_standardize(p)
        merged = df_i if merged is None else merged.join(df_i, how="outer")
    except Exception as e:
        print(f"Pass：{p} - {e}")

merged = merged.sort_index()

unit_nav_df = merged.filter(regex="_UnitValue$")

unit_nav_df = unit_nav_df.dropna(how="all")

unit_nav_df = unit_nav_df.loc[:, unit_nav_df.notna().sum() >= 50]

unit_nav_df = unit_nav_df[unit_nav_df.notna().sum(axis=1) >= 3]

unit_nav_df = unit_nav_df.ffill()
unit_nav_df = unit_nav_df.replace(0, np.nan)
unit_nav_df = unit_nav_df.where(unit_nav_df > 0).dropna(how="all")

ret_df = np.log(unit_nav_df).diff().dropna()

market_ret = ret_df.mean(axis=1)

alphas = {}
betas = {}

for fund in ret_df.columns:
    aligned = pd.concat([ret_df[fund], market_ret], axis=1).dropna()
    if len(aligned) < 30:
        continue
    X = aligned.iloc[:, 1].values.reshape(-1, 1)
    y = aligned.iloc[:, 0].values
    X = np.concatenate([np.ones_like(X), X], axis=1)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    alphas[fund] = coef[0]
    betas[fund] = coef[1]

alpha_beta_df = pd.DataFrame({"Alpha": alphas, "Beta": betas})
alpha_beta_df.index.name = "Fund"

alpha_beta_df.to_csv("../fund_alpha_beta.csv")
unit_nav_df.to_csv("../clean_unit_nav.csv")
ret_df.to_csv("../returns_log.csv")
