from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import Bounds, minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

SEED = 42
N_SPLITS = 5
FEATURE_SET = "v3"
TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
EPS = 1e-6

FINAL_BLEND_WEIGHTS = {
    "catboost": 0.9093763221312825,
    "hist_gb": 0.09062367786871744,
}

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submissions"
SUMMARY_PATH = ROOT / "run_summary_v3_auto.json"


def competition_score(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float, float]:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    ll = log_loss(y_true, prob)
    auc = roc_auc_score(y_true, prob)
    score = 0.60 * (1.0 - ll) + 0.40 * auc
    return ll, auc, score


def month_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    pattern = re.compile(r"^m([1-6])_(.+)$")
    groups: Dict[str, List[str]] = {}
    for col in df.columns:
        match = pattern.match(col)
        if not match:
            continue
        suffix = match.group(2)
        groups.setdefault(suffix, []).append(col)
    for suffix, cols in groups.items():
        groups[suffix] = sorted(cols, key=lambda c: int(pattern.match(c).group(1)), reverse=True)
    return groups


def add_monthly_summary_features(df: pd.DataFrame, monthly_groups: Dict[str, List[str]]) -> pd.DataFrame:
    feature_data: Dict[str, np.ndarray] = {}
    for suffix, cols in monthly_groups.items():
        values = df[cols].to_numpy(dtype=float)
        oldest = values[:, 0]
        newest = values[:, -1]
        mean = values.mean(axis=1)
        std = values.std(axis=1)
        min_ = values.min(axis=1)
        max_ = values.max(axis=1)
        median = np.median(values, axis=1)
        slope = np.polyfit(np.arange(values.shape[1]), values.T, deg=1)[0]
        recent_3 = values[:, -3:].mean(axis=1)
        old_3 = values[:, :3].mean(axis=1)
        non_zero = (values > 0).sum(axis=1)
        last_positive_month = np.where(
            (values > 0).any(axis=1),
            values.shape[1] - np.argmax(values[:, ::-1] > 0, axis=1),
            0,
        )
        zero_share = (values == 0).mean(axis=1)

        prefix = f"agg_{suffix}"
        feature_data[f"{prefix}_mean"] = mean
        feature_data[f"{prefix}_std"] = std
        feature_data[f"{prefix}_cv"] = std / (np.abs(mean) + EPS)
        feature_data[f"{prefix}_min"] = min_
        feature_data[f"{prefix}_max"] = max_
        feature_data[f"{prefix}_median"] = median
        feature_data[f"{prefix}_range"] = max_ - min_
        feature_data[f"{prefix}_newest"] = newest
        feature_data[f"{prefix}_oldest"] = oldest
        feature_data[f"{prefix}_newest_to_oldest_ratio"] = newest / (oldest + EPS)
        feature_data[f"{prefix}_recent3_to_old3_ratio"] = recent_3 / (old_3 + EPS)
        feature_data[f"{prefix}_recent3_minus_old3"] = recent_3 - old_3
        feature_data[f"{prefix}_slope"] = slope
        feature_data[f"{prefix}_non_zero_months"] = non_zero
        feature_data[f"{prefix}_zero_share"] = zero_share
        feature_data[f"{prefix}_last_positive_month_index"] = last_positive_month
    return pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)


def add_cross_feature_ratios(df: pd.DataFrame) -> pd.DataFrame:
    feature_data: Dict[str, np.ndarray] = {}
    month_idx = range(1, 7)
    for i in month_idx:
        inflow = (
            df[f"m{i}_received_total_value"]
            + df[f"m{i}_deposit_total_value"]
            + df[f"m{i}_transfer_from_bank_total_value"]
        )
        outflow = (
            df[f"m{i}_withdraw_total_value"]
            + df[f"m{i}_mm_send_total_value"]
            + df[f"m{i}_paybill_total_value"]
            + df[f"m{i}_merchantpay_total_value"]
        )
        feature_data[f"m{i}_inflow_total"] = inflow
        feature_data[f"m{i}_outflow_total"] = outflow
        feature_data[f"m{i}_net_cashflow"] = inflow - outflow
        feature_data[f"m{i}_inflow_to_outflow_ratio"] = inflow / (outflow + EPS)
        feature_data[f"m{i}_withdraw_to_balance_ratio"] = df[f"m{i}_withdraw_total_value"] / (
            df[f"m{i}_daily_avg_bal"] + EPS
        )
        feature_data[f"m{i}_deposit_to_withdraw_ratio"] = df[f"m{i}_deposit_total_value"] / (
            df[f"m{i}_withdraw_total_value"] + EPS
        )
        feature_data[f"m{i}_received_to_send_ratio"] = df[f"m{i}_received_total_value"] / (
            df[f"m{i}_mm_send_total_value"] + EPS
        )
        feature_data[f"m{i}_merchantpay_to_paybill_ratio"] = df[f"m{i}_merchantpay_total_value"] / (
            df[f"m{i}_paybill_total_value"] + EPS
        )
        feature_data[f"m{i}_avg_withdraw_amt"] = df[f"m{i}_withdraw_total_value"] / (df[f"m{i}_withdraw_volume"] + EPS)
        feature_data[f"m{i}_avg_deposit_amt"] = df[f"m{i}_deposit_total_value"] / (df[f"m{i}_deposit_volume"] + EPS)
        feature_data[f"m{i}_avg_received_amt"] = df[f"m{i}_received_total_value"] / (df[f"m{i}_received_volume"] + EPS)
        feature_data[f"m{i}_avg_send_amt"] = df[f"m{i}_mm_send_total_value"] / (df[f"m{i}_mm_send_volume"] + EPS)
        feature_data[f"m{i}_activity_intensity"] = (
            df[f"m{i}_paybill_volume"]
            + df[f"m{i}_merchantpay_volume"]
            + df[f"m{i}_transfer_from_bank_volume"]
            + df[f"m{i}_mm_send_volume"]
            + df[f"m{i}_received_volume"]
            + df[f"m{i}_deposit_volume"]
            + df[f"m{i}_withdraw_volume"]
        )

    temp = pd.DataFrame(feature_data, index=df.index)
    feature_data["agg_total_inflow_6m"] = temp[[f"m{i}_inflow_total" for i in month_idx]].sum(axis=1)
    feature_data["agg_total_outflow_6m"] = temp[[f"m{i}_outflow_total" for i in month_idx]].sum(axis=1)
    feature_data["agg_total_net_cashflow_6m"] = temp[[f"m{i}_net_cashflow" for i in month_idx]].sum(axis=1)
    feature_data["agg_inflow_to_outflow_6m"] = feature_data["agg_total_inflow_6m"] / (
        feature_data["agg_total_outflow_6m"] + EPS
    )
    feature_data["agg_activity_6m"] = temp[[f"m{i}_activity_intensity" for i in month_idx]].sum(axis=1)
    feature_data["agg_activity_recent3_to_old3"] = temp[[f"m{i}_activity_intensity" for i in [1, 2, 3]]].sum(axis=1) / (
        temp[[f"m{i}_activity_intensity" for i in [4, 5, 6]]].sum(axis=1) + EPS
    )
    feature_data["agg_balance_recent3_to_old3"] = df[[f"m{i}_daily_avg_bal" for i in [1, 2, 3]]].mean(axis=1) / (
        df[[f"m{i}_daily_avg_bal" for i in [4, 5, 6]]].mean(axis=1) + EPS
    )
    feature_data["agg_withdraw_recent3_to_old3"] = df[[f"m{i}_withdraw_total_value" for i in [1, 2, 3]]].sum(axis=1) / (
        df[[f"m{i}_withdraw_total_value" for i in [4, 5, 6]]].sum(axis=1) + EPS
    )
    feature_data["agg_deposit_recent3_to_old3"] = df[[f"m{i}_deposit_total_value" for i in [1, 2, 3]]].sum(axis=1) / (
        df[[f"m{i}_deposit_total_value" for i in [4, 5, 6]]].sum(axis=1) + EPS
    )
    feature_data["agg_received_recent3_to_old3"] = df[[f"m{i}_received_total_value" for i in [1, 2, 3]]].sum(axis=1) / (
        df[[f"m{i}_received_total_value" for i in [4, 5, 6]]].sum(axis=1) + EPS
    )
    feature_data["agg_send_recent3_to_old3"] = df[[f"m{i}_mm_send_total_value" for i in [1, 2, 3]]].sum(axis=1) / (
        df[[f"m{i}_mm_send_total_value" for i in [4, 5, 6]]].sum(axis=1) + EPS
    )
    feature_data["agg_paycheck_to_pressure_ratio"] = (
        feature_data["agg_total_inflow_6m"] + df["arpu"] * 6
    ) / (feature_data["agg_total_outflow_6m"] + EPS)
    return pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)


def add_entropy_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_data: Dict[str, np.ndarray] = {}
    tx_types = ["paybill", "merchantpay", "transfer_from_bank", "mm_send", "received", "deposit", "withdraw"]
    for i in range(1, 7):
        values = df[[f"m{i}_{tx}_volume" for tx in tx_types]].to_numpy(dtype=float)
        probs = values / (values.sum(axis=1, keepdims=True) + EPS)
        feature_data[f"m{i}_tx_entropy"] = -(probs * np.log(probs + EPS)).sum(axis=1)
        feature_data[f"m{i}_tx_diversity"] = (values > 0).sum(axis=1)

    temp = pd.DataFrame(feature_data, index=df.index)
    entropy_cols = [f"m{i}_tx_entropy" for i in range(1, 7)]
    diversity_cols = [f"m{i}_tx_diversity" for i in range(1, 7)]
    feature_data["agg_tx_entropy_mean"] = temp[entropy_cols].mean(axis=1)
    feature_data["agg_tx_entropy_slope"] = np.polyfit(np.arange(6), temp[entropy_cols].to_numpy(dtype=float).T, deg=1)[0]
    feature_data["agg_tx_diversity_mean"] = temp[diversity_cols].mean(axis=1)
    feature_data["agg_tx_diversity_recent3_to_old3"] = temp[[f"m{i}_tx_diversity" for i in [1, 2, 3]]].mean(axis=1) / (
        temp[[f"m{i}_tx_diversity" for i in [4, 5, 6]]].mean(axis=1) + EPS
    )
    return pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)


def add_behavioral_shift_features(df: pd.DataFrame, monthly_groups: Dict[str, List[str]]) -> pd.DataFrame:
    feature_data: Dict[str, np.ndarray] = {}
    focus_suffixes = [
        "daily_avg_bal",
        "deposit_total_value",
        "withdraw_total_value",
        "received_total_value",
        "mm_send_total_value",
        "paybill_total_value",
        "merchantpay_total_value",
        "activity_intensity",
        "net_cashflow",
        "inflow_total",
        "outflow_total",
        "tx_entropy",
        "tx_diversity",
    ]
    for suffix in focus_suffixes:
        if suffix not in monthly_groups:
            continue
        values = df[monthly_groups[suffix]].to_numpy(dtype=float)
        deltas = np.diff(values, axis=1)
        prefix = f"shift_{suffix}"
        feature_data[f"{prefix}_max_drop"] = deltas.min(axis=1)
        feature_data[f"{prefix}_max_jump"] = deltas.max(axis=1)
        feature_data[f"{prefix}_drop_count"] = (deltas < 0).sum(axis=1)
        feature_data[f"{prefix}_jump_count"] = (deltas > 0).sum(axis=1)
        feature_data[f"{prefix}_sign_changes"] = np.sum(np.sign(deltas[:, 1:]) != np.sign(deltas[:, :-1]), axis=1)
        feature_data[f"{prefix}_recent_volatility"] = values[:, -3:].std(axis=1)
        feature_data[f"{prefix}_old_volatility"] = values[:, :3].std(axis=1)
        feature_data[f"{prefix}_recent_to_old_volatility"] = feature_data[f"{prefix}_recent_volatility"] / (
            feature_data[f"{prefix}_old_volatility"] + EPS
        )
        feature_data[f"{prefix}_latest_gap_from_mean"] = values[:, -1] - values.mean(axis=1)
        feature_data[f"{prefix}_drawdown_from_peak"] = values[:, -1] - values.max(axis=1)
        feature_data[f"{prefix}_distance_from_floor"] = values[:, -1] - values.min(axis=1)

    recent_balance = df["m1_daily_avg_bal"].to_numpy(dtype=float)
    recent_outflow = df["m1_outflow_total"].to_numpy(dtype=float)
    recent_inflow = df["m1_inflow_total"].to_numpy(dtype=float)
    recent_net = df[[f"m{i}_net_cashflow" for i in range(1, 7)]].to_numpy(dtype=float)
    recent_balances = df[[f"m{i}_daily_avg_bal" for i in range(1, 7)]].to_numpy(dtype=float)
    zero_balance = (recent_balances <= 0).astype(int)
    pressure_flag = (recent_outflow > recent_inflow).astype(int)
    negative_cashflow = (recent_net < 0).astype(int)

    feature_data["agg_negative_cashflow_months"] = negative_cashflow.sum(axis=1)
    feature_data["agg_recent_negative_cashflow_months"] = negative_cashflow[:, :3].sum(axis=1)
    feature_data["agg_balance_pressure_months"] = (
        df[[f"m{i}_withdraw_to_balance_ratio" for i in range(1, 7)]].to_numpy(dtype=float) > 1
    ).sum(axis=1)
    feature_data["agg_zero_balance_months"] = zero_balance.sum(axis=1)
    feature_data["agg_recent_pressure_flag"] = pressure_flag
    feature_data["agg_recent_balance_buffer"] = recent_balance / (recent_outflow + EPS)
    feature_data["agg_recent_net_to_balance"] = df["m1_net_cashflow"].to_numpy(dtype=float) / (recent_balance + EPS)
    feature_data["agg_recent_inflow_minus_outflow"] = recent_inflow - recent_outflow
    return pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)


def add_v3_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_data: Dict[str, np.ndarray] = {}
    balance = df[[f"m{i}_daily_avg_bal" for i in range(1, 7)]].to_numpy(dtype=float)
    inflow = df[[f"m{i}_inflow_total" for i in range(1, 7)]].to_numpy(dtype=float)
    outflow = df[[f"m{i}_outflow_total" for i in range(1, 7)]].to_numpy(dtype=float)
    net = df[[f"m{i}_net_cashflow" for i in range(1, 7)]].to_numpy(dtype=float)
    withdraw_ratio = df[[f"m{i}_withdraw_to_balance_ratio" for i in range(1, 7)]].to_numpy(dtype=float)
    activity = df[[f"m{i}_activity_intensity" for i in range(1, 7)]].to_numpy(dtype=float)

    feature_data["v3_balance_drawdown_pct"] = (balance[:, -1] - balance.max(axis=1)) / (np.abs(balance.max(axis=1)) + EPS)
    feature_data["v3_balance_floor_to_peak_ratio"] = balance.min(axis=1) / (balance.max(axis=1) + EPS)
    feature_data["v3_recent3_balance_min"] = balance[:, -3:].min(axis=1)
    feature_data["v3_recent3_balance_mean"] = balance[:, -3:].mean(axis=1)
    feature_data["v3_recent3_balance_std"] = balance[:, -3:].std(axis=1)
    feature_data["v3_recent3_cashflow_sum"] = net[:, -3:].sum(axis=1)
    feature_data["v3_recent3_cashflow_mean"] = net[:, -3:].mean(axis=1)
    feature_data["v3_recent3_negative_cashflow_streak"] = (net[:, -3:] < 0).sum(axis=1)
    feature_data["v3_recent3_inflow_to_outflow"] = inflow[:, -3:].sum(axis=1) / (outflow[:, -3:].sum(axis=1) + EPS)
    feature_data["v3_recent3_balance_to_outflow"] = balance[:, -3:].mean(axis=1) / (outflow[:, -3:].mean(axis=1) + EPS)
    feature_data["v3_recent3_pressure_share"] = (withdraw_ratio[:, -3:] > 1).mean(axis=1)
    feature_data["v3_pressure_acceleration"] = withdraw_ratio[:, -3:].mean(axis=1) - withdraw_ratio[:, :3].mean(axis=1)
    feature_data["v3_activity_drop_pct"] = (
        activity[:, -3:].mean(axis=1) - activity[:, :3].mean(axis=1)
    ) / (activity[:, :3].mean(axis=1) + EPS)
    feature_data["v3_inflow_drop_pct"] = (
        inflow[:, -3:].mean(axis=1) - inflow[:, :3].mean(axis=1)
    ) / (inflow[:, :3].mean(axis=1) + EPS)
    feature_data["v3_outflow_rise_pct"] = (
        outflow[:, -3:].mean(axis=1) - outflow[:, :3].mean(axis=1)
    ) / (outflow[:, :3].mean(axis=1) + EPS)
    feature_data["v3_latest_balance_vs_arpu"] = df["m1_daily_avg_bal"].to_numpy(dtype=float) / (
        df["arpu"].to_numpy(dtype=float) + EPS
    )
    feature_data["v3_latest_outflow_vs_arpu"] = df["m1_outflow_total"].to_numpy(dtype=float) / (
        df["arpu"].to_numpy(dtype=float) + EPS
    )
    feature_data["v3_latest_inflow_vs_arpu"] = df["m1_inflow_total"].to_numpy(dtype=float) / (
        df["arpu"].to_numpy(dtype=float) + EPS
    )
    feature_data["v3_recent_bill_stop_flag"] = (
        (df["m1_paybill_volume"].to_numpy(dtype=float) == 0)
        & (df[[f"m{i}_paybill_volume" for i in [4, 5, 6]]].sum(axis=1).to_numpy(dtype=float) > 0)
    ).astype(int)
    feature_data["v3_recent_merchant_stop_flag"] = (
        (df["m1_merchantpay_volume"].to_numpy(dtype=float) == 0)
        & (df[[f"m{i}_merchantpay_volume" for i in [4, 5, 6]]].sum(axis=1).to_numpy(dtype=float) > 0)
    ).astype(int)
    feature_data["v3_recent_deposit_stop_flag"] = (
        (df["m1_deposit_volume"].to_numpy(dtype=float) == 0)
        & (df[[f"m{i}_deposit_volume" for i in [4, 5, 6]]].sum(axis=1).to_numpy(dtype=float) > 0)
    ).astype(int)
    feature_data["v3_recent_received_stop_flag"] = (
        (df["m1_received_volume"].to_numpy(dtype=float) == 0)
        & (df[[f"m{i}_received_volume" for i in [4, 5, 6]]].sum(axis=1).to_numpy(dtype=float) > 0)
    ).astype(int)
    feature_data["v3_paycheck_pressure_combo"] = df["agg_paycheck_to_pressure_ratio"].to_numpy(dtype=float) * (
        1.0 / (df["agg_recent_balance_buffer"].to_numpy(dtype=float) + EPS)
    )
    return pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    combined = pd.concat(
        [train_df.assign(_dataset="train"), test_df.assign(_dataset="test")],
        axis=0,
        ignore_index=True,
    )
    monthly_groups = month_columns(combined)
    combined = add_monthly_summary_features(combined, monthly_groups)
    combined = add_cross_feature_ratios(combined)
    monthly_groups = month_columns(combined)
    combined = add_entropy_features(combined)
    monthly_groups = month_columns(combined)
    combined = add_behavioral_shift_features(combined, monthly_groups)
    combined = add_v3_features(combined)

    categorical_cols = ["gender", "region", "smartphone", "segment", "earning_pattern"]
    for col in categorical_cols:
        combined[col] = combined[col].astype("category")

    train_fe = combined.loc[combined["_dataset"] == "train"].drop(columns=["_dataset"]).reset_index(drop=True)
    test_fe = combined.loc[combined["_dataset"] == "test"].drop(columns=["_dataset"]).reset_index(drop=True)

    feature_cols = [c for c in train_fe.columns if c not in {TARGET, ID_COL}]
    numeric_cols = train_fe[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return train_fe, test_fe, numeric_cols, categorical_cols


def build_logistic_anchor(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")),
        ]
    )


def build_hist_model(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_depth=6,
                    max_iter=300,
                    min_samples_leaf=40,
                    l2_regularization=0.2,
                    max_leaf_nodes=31,
                    random_state=SEED,
                ),
            ),
        ]
    )


def build_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=400,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        bagging_temperature=0.0,
        task_type="CPU",
        devices="0",
        random_seed=SEED,
        verbose=False,
    )


def fit_fold_model(
    model_name: str,
    model: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_cols: List[str],
) -> object:
    if model_name == "catboost":
        cat_idx = [x_train.columns.get_loc(col) for col in categorical_cols]
        model.fit(
            x_train,
            y_train,
            eval_set=(x_valid, y_valid),
            cat_features=cat_idx,
            use_best_model=True,
            early_stopping_rounds=50,
        )
        return model

    model.fit(x_train, y_train)
    return model


def cross_validate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, List[object]]]:
    feature_cols = [c for c in train_df.columns if c not in {TARGET, ID_COL}]
    X = train_df[feature_cols]
    y = train_df[TARGET]
    X_test = test_df[feature_cols]

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    model_builders = {
        "catboost": lambda: build_catboost(),
        "hist_gb": lambda: build_hist_model(numeric_cols, categorical_cols),
        "logistic_anchor": lambda: build_logistic_anchor(numeric_cols, categorical_cols),
    }

    oof = pd.DataFrame(index=train_df.index)
    test_preds = pd.DataFrame(index=test_df.index)
    score_map: Dict[str, Dict[str, float]] = {}
    saved_models: Dict[str, List[object]] = {}

    for model_name, builder in model_builders.items():
        print(f"\nTraining model: {model_name}")
        model_oof = np.zeros(len(train_df))
        fold_test_preds = []
        saved_models[model_name] = []
        fold_scores = []

        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
            x_train, x_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = builder()
            model = fit_fold_model(model_name, model, x_train, y_train, x_valid, y_valid, categorical_cols)

            val_pred = np.clip(model.predict_proba(x_valid)[:, 1], 1e-5, 1 - 1e-5)
            test_pred = np.clip(model.predict_proba(X_test)[:, 1], 1e-5, 1 - 1e-5)

            model_oof[valid_idx] = val_pred
            fold_test_preds.append(test_pred)
            ll, auc, comp = competition_score(y_valid.to_numpy(), val_pred)
            print(f"Fold {fold} | LogLoss: {ll:.4f} | AUC: {auc:.4f} | Comp: {comp:.4f}")
            fold_scores.append(comp)
            saved_models[model_name].append(model)

        ll, auc, comp = competition_score(y.to_numpy(), model_oof)
        print(
            f"{model_name} OOF | LogLoss: {ll:.4f} | AUC: {auc:.4f} | Comp: {comp:.4f} | "
            f"CV mean±std: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}"
        )
        oof[model_name] = model_oof
        test_preds[model_name] = np.mean(fold_test_preds, axis=0)
        score_map[model_name] = {
            "logloss": ll,
            "auc": auc,
            "competition_score": comp,
        }

    return oof, test_preds, score_map, saved_models


def optimize_blend_weights(oof_preds: pd.DataFrame, y_true: pd.Series) -> np.ndarray:
    initial = np.full(oof_preds.shape[1], 1.0 / oof_preds.shape[1])

    def objective(weights: np.ndarray) -> float:
        weights = np.clip(weights, 0, 1)
        weights = weights / weights.sum()
        blended = np.dot(oof_preds.to_numpy(), weights)
        _, _, comp = competition_score(y_true.to_numpy(), blended)
        return -comp

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = Bounds(0.0, 1.0)
    result = minimize(objective, x0=initial, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x if result.success else initial
    weights = np.clip(weights, 0, 1)
    weights = weights / weights.sum()
    return weights


def isotonic_calibrate(
    oof_blend: np.ndarray,
    y_true: pd.Series,
    test_blend: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], object]:
    sigmoid_calibrator = CalibratedClassifierCV(LogisticRegression(max_iter=2000), method="sigmoid", cv=5)
    isotonic_calibrator = CalibratedClassifierCV(LogisticRegression(max_iter=2000), method="isotonic", cv=5)

    raw_feature = oof_blend.reshape(-1, 1)
    test_feature = test_blend.reshape(-1, 1)

    sigmoid_calibrator.fit(raw_feature, y_true)
    sigmoid_oof = sigmoid_calibrator.predict_proba(raw_feature)[:, 1]
    _, _, sigmoid_score = competition_score(y_true.to_numpy(), sigmoid_oof)

    isotonic_calibrator.fit(raw_feature, y_true)
    isotonic_oof = isotonic_calibrator.predict_proba(raw_feature)[:, 1]
    isotonic_test = isotonic_calibrator.predict_proba(test_feature)[:, 1]
    _, _, isotonic_score = competition_score(y_true.to_numpy(), isotonic_oof)

    metrics = {
        "sigmoid": sigmoid_score,
        "isotonic": isotonic_score,
    }
    return isotonic_oof, isotonic_test, metrics, isotonic_calibrator


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "Train.csv")
    test_df = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")

    train_fe, test_fe, numeric_cols, categorical_cols = engineer_features(train_df, test_df)
    print(f"Engineered train shape: {train_fe.shape}")
    print(f"Engineered test shape: {test_fe.shape}")
    print(f"Numeric features: {len(numeric_cols)} | Categorical features: {len(categorical_cols)}")

    oof_preds, test_preds, model_scores, cv_models = cross_validate_models(
        train_fe,
        test_fe,
        numeric_cols,
        categorical_cols,
    )

    optimized_weights = optimize_blend_weights(oof_preds[["catboost", "hist_gb"]], train_fe[TARGET])
    blend_oof = np.dot(oof_preds[["catboost", "hist_gb"]].to_numpy(), np.array([
        FINAL_BLEND_WEIGHTS["catboost"],
        FINAL_BLEND_WEIGHTS["hist_gb"],
    ]))
    blend_test = np.dot(test_preds[["catboost", "hist_gb"]].to_numpy(), np.array([
        FINAL_BLEND_WEIGHTS["catboost"],
        FINAL_BLEND_WEIGHTS["hist_gb"],
    ]))

    calibrated_oof, calibrated_test, calibration_metrics, calibrator = isotonic_calibrate(
        blend_oof,
        train_fe[TARGET],
        blend_test,
    )
    ll, auc, comp = competition_score(train_fe[TARGET].to_numpy(), calibrated_oof)

    submission = sample.copy()
    submission["TargetLogLoss"] = np.clip(calibrated_test, 0.001, 0.999)
    if "TargetRAUC" in submission.columns:
        submission["TargetRAUC"] = np.clip(calibrated_test, 0.001, 0.999)
    else:
        submission["Target RAUC"] = np.clip(calibrated_test, 0.001, 0.999)

    submission_path = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)

    metadata = {
        "feature_count": len([c for c in train_fe.columns if c not in {TARGET, ID_COL}]),
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        "feature_set": FEATURE_SET,
        "n_splits": N_SPLITS,
        "model_scores": model_scores,
        "selected_strategy": {
            "name": "weighted_top2_calibrated",
            "base_name": "weighted_top2",
            "mode": "calibrated",
            "members": ["catboost", "hist_gb"],
            "weights": FINAL_BLEND_WEIGHTS,
            "optimized_weights_from_rerun": {
                "catboost": float(optimized_weights[0]),
                "hist_gb": float(optimized_weights[1]),
            },
        },
        "catboost_params": {"task_type": "CPU", "devices": "0"},
        "hist_params": {},
        "lightgbm_params": {},
        "xgboost_params": {},
        "selected_oof": {"logloss": ll, "auc": auc, "competition_score": comp},
        "calibration_candidates": calibration_metrics,
        "mean_abs_gap_full_vs_cv_test": 0.0,
        "submission_path": str(submission_path),
    }
    SUMMARY_PATH.write_text(json.dumps(metadata, indent=2))

    joblib.dump(
        {"selected_strategy": "weighted_top2_calibrated", "metadata": metadata, "calibrator": calibrator},
        MODEL_DIR / "ensemble_v3_auto.joblib",
    )
    for model_name, models in cv_models.items():
        joblib.dump(models, MODEL_DIR / f"{model_name}_cv_v3_auto.joblib")

    print(f"\nSaved submission to: {submission_path}")
    print(f"Saved metadata to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
