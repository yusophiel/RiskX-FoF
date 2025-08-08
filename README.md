## Project Overview
This model focuses on the design and implementation of a `multi-strategy fund-of-funds (FOF) portfolio optimization framework` that combines machine learning-based return forecasting with various weighting strategies.

Specifically, I developed a rolling forecasting and backtesting system that uses `XGBoost` regression for return forecasting and combines static and dynamic portfolio allocation strategies. Static strategies include `risk parity`, `alpha-beta + CVaR`, and an `equal-weighted baseline`, while dynamic strategies `Ridge` optimize fund weights using predicted returns (μ) and multiple risk metrics, such as annualized volatility, conditional value-at-risk (CVaR), maximum drawdown, and market correlation.

I also refined the entire `model training loop`, `portfolio weight generation`, `backtesting engine`, and `performance visualization`. This includes rolling window training, daily weight calculation, Sharpe ratio estimation, turnover calculation, and simulation validation based on unseen market data.

---

## Features

- XGBoost-based rolling return forecasting
- `Multi-factor risk profiling` using volatility, CVaR, drawdown, and market correlation
- Multiple portfolio strategies
    - Risk Parity
    - Alpha-Beta + CVaR
    - Ridge multi-factor regression (dynamic allocation)
    - Equal-weight baseline 
- Simulate a real transaction environment, taking into account transaction cost and turnover
- Fully parameterized pipeline for flexible holding period, and lookback window
- Fast computation with pandas & numpy

---

## Module Justification

### `data_handle.py`
- Loads and cleans raw fund NAV data
- Computes daily log returns for all funds
- Generates equal-weight benchmark returns for Alpha/Beta regression

### `build_xgb_dataset.py`
- Constructs feature matrix from historical fund returns
- Includes momentum (mom_5, mom_10), volatility (volatility_20), drawdown (mdd_20), CVaR (cvar_20), and correlation (corr_20)
- Generates multi-factor risk profiles for each fund on each date

### `train_xgb_models.py`
- Implements **rolling-window training** of XGBoost regression models to predict fund returns over the next `holding_window` days
- Saves predicted return matrix for portfolio allocation

### `fof_combination.py`
- Generates portfolio weights for multiple strategies:
    - `Risk Parity`: Inverse volatility weighting
    - `Alpha-Beta + CVaR`: Combines intercept and inverse beta from return-vs-benchmark regression, filtered by CVaR
    - `Ridge Multi-factor Regression`: Learns optimal weights from predicted returns and risk factors using Ridge regression
- Saves portfolio weights for backtesting

### `fof_backtest.py`
- Runs backtests for all strategies using historical returns and generated weights
- Computes performance metrics:
    - Annualized return
    - Sharpe ratio
    - Max drawdown
    - Calmar ratio
- Plots NAV curves for strategy comparison
- Calculates turnover and transaction costs

---

## Example Output

- All Strategies NAV Comparison
    ![Strategy_Performance](images/Strategy_Performance.png)
- Performance Metrics Example
    ![Performance_Metrics_Example](images/performance_metrics_all.png)

---

## Requirements

- Python >= 3.8
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
