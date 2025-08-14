import pandas as pd
import xgboost as xgb

# Training window length (days)
train_window = 30
# Position/forecast window length (days)
holding_window = 1
# Sliding window step size
step_size = 1
# Minimum number of valid training samples
min_train_size = 20
# Optional: "sliding" (sliding window), "expanding" (expanding window), "single" (single training)
mode = "single"

# Loading feature data
df_all = pd.read_csv("../xgb_features_with_risk.csv", parse_dates=["date"])
features = [col for col in df_all.columns if col not in ["target", "date", "fund"]]
df_all = df_all.sort_values("date")

# XGBoost parameter configuration
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

# Sliding window training mode
if mode == "sliding":
    for start in range(0, len(df_all) - train_window - holding_window, step_size):
        train_slice = df_all.iloc[start: start + train_window]
        test_slice = df_all.iloc[start + train_window : start + train_window + holding_window]
        if len(train_slice) < min_train_size or len(test_slice) == 0:
            continue

        train_non_na = train_slice.dropna(subset=features + ["target"])
        if len(train_non_na) < min_train_size:
            continue

        X_train = train_non_na[features]
        y_train = train_non_na["target"]

        X_test = test_slice[features]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        booster = xgb.train(xgb_params, dtrain, num_boost_round=40)
        y_pred = booster.predict(dtest)

        test_info = test_slice[["date", "fund"]].copy()
        test_info["pred"] = y_pred
        all_preds.append(test_info)

# Extended Window Training Mode
elif mode == "expanding":
    train_end = train_window
    while train_end < len(df_all) - holding_window:
        train_slice = df_all.iloc[:train_end]
        test_slice = df_all.iloc[train_end : train_end + holding_window]

        train_non_na = train_slice.dropna(subset=features + ["target"])
        if len(train_non_na) < min_train_size:
            train_end += holding_window
            continue

        X_train = train_non_na[features]
        y_train = train_non_na["target"]

        X_test = test_slice[features]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        booster = xgb.train(xgb_params, dtrain, num_boost_round=40)
        y_pred = booster.predict(dtest)

        test_info = test_slice[["date", "fund"]].copy()
        test_info["pred"] = y_pred
        all_preds.append(test_info)

        train_end += holding_window

# Single training mode
elif mode == "single":
    train_slice = df_all.iloc[:train_window] # First train_window days of training
    test_slice = df_all.iloc[train_window:] # All the rest are tests

    train_non_na = train_slice.dropna(subset=features + ["target"])
    X_train = train_non_na[features]
    y_train = train_non_na["target"]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    booster = xgb.train(xgb_params, dtrain, num_boost_round=40)

    X_test = test_slice[features]
    dtest = xgb.DMatrix(X_test)
    y_pred = booster.predict(dtest)

    test_info = test_slice[["date", "fund"]].copy()
    test_info["pred"] = y_pred
    all_preds.append(test_info)

df_mu = pd.concat(all_preds).sort_values("date")
df_mu.to_csv("../xgb_preds_daily.csv", index=False)
print(f"Prediction completed, mode: {mode}, results saved to ../xgb_preds_daily.csv")
