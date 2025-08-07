import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error

train_window = 40
holding_window = 3
step_size = 3
min_train_size = 20

df_all = pd.read_csv("../xgb_features_with_risk.csv", parse_dates=["date"])
features = [col for col in df_all.columns if col not in ["target", "date", "fund"]]
df_all = df_all.sort_values("date")

xgb_params = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
    "nthread": -1,
    "verbosity": 0
}

all_preds = []
booster = None

for start in range(0, len(df_all) - train_window - holding_window, step_size):
    train_slice = df_all.iloc[start: start + train_window]
    test_slice = df_all.iloc[start + train_window: start + train_window + holding_window]

    if len(train_slice) < min_train_size or len(test_slice) == 0:
        continue

    X_train = train_slice[features]
    y_train = train_slice["target"]
    X_test = test_slice[features]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=20,
    )

    y_pred = booster.predict(dtest)
    test_info = test_slice[["date", "fund"]].copy()
    test_info["pred"] = y_pred
    all_preds.append(test_info)

df_mu = pd.concat(all_preds).sort_values("date")
df_mu.to_csv("../xgb_preds_daily.csv", index=False)

