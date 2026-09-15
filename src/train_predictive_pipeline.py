"""Train and evaluate hostel meal turnout forecasting models.

The script uses a chronological holdout to reduce temporal leakage, compares a
linear baseline against Random Forest and XGBoost ensemble regressors, and
converts predicted turnout into an operational food-preparation recommendation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


TARGET = "student_turnout"
CATEGORICAL_FEATURES = ["meal_category", "menu_name", "day_name"]
NUMERIC_FEATURES = [
    "year",
    "month",
    "day_of_week",
    "day_of_year",
    "residents_present",
    "advance_bookings",
    "weekend_flag",
    "exam_flag",
    "vacation_flag",
    "festival_flag",
    "special_event_flag",
    "temperature_c",
    "rain_index",
    "menu_popularity_score",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load input data, add safe calendar features, and enforce core invariants."""

    df = pd.read_csv(path, parse_dates=["date"])
    required = set(FEATURES).difference({"day_of_year"}) | {
        "date",
        TARGET,
        "food_consumed_kg",
        "food_prepared_kg",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input dataset is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Input dataset is empty.")

    df = df.sort_values(["date", "meal_category"]).reset_index(drop=True)
    df["day_of_year"] = df["date"].dt.dayofyear.astype(int)

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Modeling columns contain missing values.")
    if (df[TARGET] < 0).any():
        raise ValueError("Student turnout cannot be negative.")
    if (df[TARGET] > df["residents_present"]).any():
        raise ValueError("Student turnout cannot exceed residents present.")
    return df


def chronological_split(df: pd.DataFrame, train_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by whole dates so the holdout strictly occurs after the training set."""

    if not 0.60 <= train_fraction <= 0.90:
        raise ValueError("train_fraction must be between 0.60 and 0.90.")

    unique_dates = np.array(sorted(df["date"].dt.normalize().unique()))
    split_idx = int(len(unique_dates) * train_fraction)
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    cutoff = pd.Timestamp(unique_dates[split_idx - 1])

    train_df = df[df["date"] <= cutoff].copy()
    test_df = df[df["date"] > cutoff].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty partition.")
    return train_df, test_df


def build_preprocessor() -> ColumnTransformer:
    """Create a preprocessing graph compatible with all candidate regressors."""

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models(seed: int) -> Dict[str, Pipeline]:
    """Return baseline and ensemble model pipelines with reproducible settings."""

    estimators = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=450,
            max_depth=18,
            min_samples_leaf=2,
            max_features=0.80,
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.035,
            max_depth=6,
            min_child_weight=2.0,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.05,
            reg_lambda=1.25,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        ),
    }

    return {
        name: Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics requested for the capstone."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def consumption_lookup(train_df: pd.DataFrame) -> Dict[str, float]:
    """Estimate robust meal-specific kilograms consumed per attending student."""

    safe = train_df[train_df[TARGET] > 0].copy()
    safe["kg_per_student"] = safe["food_consumed_kg"] / safe[TARGET]
    lookup = safe.groupby("meal_category")["kg_per_student"].median().to_dict()
    if not lookup:
        raise ValueError("Could not estimate per-student food consumption.")
    return {str(k): float(v) for k, v in lookup.items()}


def create_operational_predictions(
    test_df: pd.DataFrame,
    predicted_turnout: np.ndarray,
    kg_lookup: Dict[str, float],
    safety_buffer: float,
) -> pd.DataFrame:
    """Create actionable preparation recommendations and retrospective impact estimates."""

    if not 0.0 <= safety_buffer <= 0.20:
        raise ValueError("safety_buffer must be between 0 and 0.20.")

    out = test_df[
        [
            "date",
            "meal_category",
            "menu_name",
            "residents_present",
            "advance_bookings",
            TARGET,
            "food_consumed_kg",
            "food_prepared_kg",
            "food_wasted_kg",
        ]
    ].copy()

    # Turnout predictions are rounded and bounded by feasible hostel population.
    bounded = np.minimum(
        np.maximum(np.rint(predicted_turnout), 0), out["residents_present"].to_numpy()
    ).astype(int)
    out["predicted_turnout"] = bounded
    out["turnout_abs_error"] = (out[TARGET] - out["predicted_turnout"]).abs()

    kg_per_student = out["meal_category"].map(kg_lookup).astype(float)
    out["recommended_food_prep_kg"] = (
        out["predicted_turnout"] * kg_per_student * (1.0 + safety_buffer)
    ).round(2)

    # Retrospective operational simulation. Shortage is tracked separately from waste.
    out["simulated_leftover_kg"] = np.maximum(
        out["recommended_food_prep_kg"] - out["food_consumed_kg"], 0.0
    ).round(2)
    out["simulated_shortage_kg"] = np.maximum(
        out["food_consumed_kg"] - out["recommended_food_prep_kg"], 0.0
    ).round(2)
    out["prep_reduction_vs_actual_kg"] = (
        out["food_prepared_kg"] - out["recommended_food_prep_kg"]
    ).round(2)
    return out


def summarize_operations(predictions: pd.DataFrame) -> Dict[str, float]:
    """Aggregate operational simulation metrics over the holdout period."""

    actual_prepared = float(predictions["food_prepared_kg"].sum())
    recommended = float(predictions["recommended_food_prep_kg"].sum())
    actual_leftover = float(predictions["food_wasted_kg"].sum())
    simulated_leftover = float(predictions["simulated_leftover_kg"].sum())
    simulated_shortage = float(predictions["simulated_shortage_kg"].sum())

    return {
        "holdout_actual_prepared_kg": round(actual_prepared, 2),
        "holdout_recommended_prepared_kg": round(recommended, 2),
        "prepared_food_change_pct": round(
            100.0 * (recommended - actual_prepared) / max(actual_prepared, 1e-9), 3
        ),
        "holdout_actual_recorded_waste_kg": round(actual_leftover, 2),
        "holdout_simulated_leftover_kg": round(simulated_leftover, 2),
        "simulated_leftover_reduction_pct": round(
            100.0 * (actual_leftover - simulated_leftover) / max(actual_leftover, 1e-9),
            3,
        ),
        "holdout_simulated_shortage_kg": round(simulated_shortage, 2),
        "shortage_rate_vs_consumption_pct": round(
            100.0
            * simulated_shortage
            / max(float(predictions["food_consumed_kg"].sum()), 1e-9),
            3,
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/hostel_food_waste.csv"),
        help="Input CSV generated by generate_synthetic_data.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument(
        "--safety-buffer",
        type=float,
        default=0.03,
        help="Extra food above estimated consumption, expressed as a fraction.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Train candidates, choose the lowest-RMSE model, and save all artifacts."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_validate(args.data)
    train_df, test_df = chronological_split(df, args.train_fraction)
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    metrics_rows = []
    fitted_models: Dict[str, Pipeline] = {}
    test_predictions: Dict[str, np.ndarray] = {}

    for name, pipeline in build_models(args.seed).items():
        pipeline.fit(X_train, y_train)
        raw_pred = pipeline.predict(X_test)
        pred = np.clip(raw_pred, 0, test_df["residents_present"].to_numpy())
        metrics = regression_metrics(y_test, pred)
        metrics_rows.append({"model": name, **metrics})
        fitted_models[name] = pipeline
        test_predictions[name] = pred
        print(
            f"{name:16s} | MAE={metrics['MAE']:.3f} | "
            f"RMSE={metrics['RMSE']:.3f} | R2={metrics['R2']:.4f}"
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE").reset_index(drop=True)
    best_name = str(metrics_df.iloc[0]["model"])
    best_model = fitted_models[best_name]
    best_pred = test_predictions[best_name]

    kg_lookup = consumption_lookup(train_df)
    operational = create_operational_predictions(
        test_df, best_pred, kg_lookup, args.safety_buffer
    )
    operational_summary = summarize_operations(operational)

    metrics_df.to_csv(args.output_dir / "model_metrics.csv", index=False)
    operational.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    joblib.dump(best_model, args.model_dir / "best_turnout_model.joblib")

    metadata = {
        "synthetic_data_notice": (
            "Evaluation uses generated synthetic data; results are demonstration results, "
            "not empirical evidence from an operating hostel."
        ),
        "best_model": best_name,
        "train_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "train_start": str(train_df["date"].min().date()),
        "train_end": str(train_df["date"].max().date()),
        "test_start": str(test_df["date"].min().date()),
        "test_end": str(test_df["date"].max().date()),
        "safety_buffer": args.safety_buffer,
        "kg_per_student_by_meal": kg_lookup,
        "operational_simulation": operational_summary,
    }
    (args.model_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\nBest model: {best_name}")
    print(f"Saved metrics -> {args.output_dir / 'model_metrics.csv'}")
    print(f"Saved predictions -> {args.output_dir / 'holdout_predictions.csv'}")
    print(f"Saved model -> {args.model_dir / 'best_turnout_model.joblib'}")
    print(json.dumps(operational_summary, indent=2))


if __name__ == "__main__":
    main()
