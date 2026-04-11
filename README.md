# Zindi Liquidity Stress Solution V3

This repo contains my solution to "AI4EAC Liquidity Stress Early Warning Challenge ". I won the first place (3k + ticket to the award ceremony). It predicts the probability that a mobile money customer will experience liquidity stress in the next 30 days. It uses the customer’s last 6 months of behavior, creates features that describe changes in income, spending, balance, and activity, then trains several models and chooses the best final prediction strategy.

## Overview

This package contains the code and documentation for the `submission_v3_auto.csv` solution. The goal of the model is to predict the probability that a mobile money customer will experience liquidity stress within the next 30 days. The solution is designed as a probability estimation pipeline, with strong attention to calibrated predictions because the competition score gives more weight to log loss than to ranking alone.

## Methodology

The solution starts from the monthly customer activity variables provided in the competition data. Each record contains six months of behavior covering inflows, deposits, withdrawals, transfers, balances, merchant and bill payments, and activity intensity. The central idea behind the solution is that liquidity stress is better detected through changes in behavior over time than through raw transaction totals alone.

The first stage of the pipeline is feature engineering. The code builds a large set of derived variables from the six-month histories. These include summary statistics such as mean, standard deviation, coefficient of variation, range, slope, and recency ratios. The pipeline also creates cashflow and pressure variables such as inflow totals, outflow totals, net cashflow, withdrawal-to-balance ratios, deposit-to-withdraw ratios, received-to-send ratios, and activity intensity.

Additional features were engineered to capture behavioral deterioration. These include recent-versus-historical comparisons, recent volatility against older volatility, drawdown from prior peak values, changes in transaction diversity, repeated negative cashflow, zero-balance behavior, recent pressure flags, and recent balance buffer. A final `v3` feature block adds stress-oriented features such as balance drawdown percentage, pressure acceleration, activity drop, inflow drop, outflow rise, recent payment stop flags, and paycheck-to-pressure style combinations. The final engineered dataset contains 926 features in total: 921 numeric features and 5 categorical features.

The modeling stage uses three base models:

- CatBoost
- Histogram Gradient Boosting
- Logistic Regression anchor model

All models are trained using stratified 5-fold cross-validation with a fixed random seed of `42`. Out-of-fold predictions are collected for model evaluation and for ensemble construction. The final `v3_auto` solution uses the top two performing models, CatBoost and Histogram Gradient Boosting, in a weighted blend. The original selected blend weights from the saved `v3_auto` run are:

- CatBoost: `0.9093763221312825`
- HistGradientBoosting: `0.09062367786871744`

The blended probabilities are then calibrated using isotonic calibration. This calibrated weighted top-2 blend is the final strategy used for the packaged `v3_auto` solution.

The recorded local validation result for this run was approximately:

- Log Loss: `0.2335`
- ROC-AUC: `0.9180`
- Competition Score: `0.8271`

The final submission file writes the same predicted probability into both required competition target columns.

## Reproducibility

This package is structured so that a reviewer can reproduce the training pipeline from start to finish.

- Global random seed: `42`
- Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Environment used for packaging: local macOS, CPU execution, Python `3.9.6`, architecture `x86_64`
- Required package versions are pinned in `requirements.txt`

The package also includes the exact saved `v3_auto` model artifacts, the packaged submission file, and the saved run summary from the original run.

## Data Used

The solution uses only the competition files included in the `data/` folder:

- `Train.csv`
- `Test.csv`
- `data_dictionary.csv`
- `SampleSubmission.csv`

No external data is used.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From inside the `my_solution_v3/` folder, run:

```bash
python3 solution.py
```

If you prefer a notebook, you can open and run:

```bash
solution.ipynb
```

When the script or notebook starts, it should report:

```bash
Engineered train shape: (40000, 928)
Engineered test shape: (30000, 928)
Numeric features: 921 | Categorical features: 5
```

## Output

The main submission file is:

```bash
submissions/submission.csv
```

The package also contains:

```bash
run_summary_v3_auto.json
```

This file records the final configuration and validation metrics for the packaged `v3_auto` solution.
