"""Run single-meal turnout inference and calculate a food-preparation recommendation.

Example:
    python src/predict.py --input examples/next_meal.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_predictive_pipeline import FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON file with pre-meal features.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/best_turnout_model.joblib"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("models/model_metadata.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    model = joblib.load(args.model)

    date = pd.Timestamp(payload["date"])
    row = {
        **payload,
        "year": int(date.year),
        "month": int(date.month),
        "day_of_week": int(date.dayofweek),
        "day_of_year": int(date.dayofyear),
        "day_name": date.day_name(),
    }
    missing = [feature for feature in FEATURES if feature not in row]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")

    frame = pd.DataFrame([{feature: row[feature] for feature in FEATURES}])
    raw = float(model.predict(frame)[0])
    predicted_turnout = int(
        np.clip(round(raw), 0, int(row["residents_present"]))
    )

    kg_lookup = metadata["kg_per_student_by_meal"]
    meal = str(row["meal_category"])
    if meal not in kg_lookup:
        raise ValueError(f"No consumption profile stored for meal category: {meal}")

    safety_buffer = float(metadata["safety_buffer"])
    recommended_kg = predicted_turnout * float(kg_lookup[meal]) * (1.0 + safety_buffer)

    result = {
        "date": payload["date"],
        "meal_category": meal,
        "predicted_turnout": predicted_turnout,
        "recommended_food_prep_kg": round(recommended_kg, 2),
        "safety_buffer_pct": round(safety_buffer * 100.0, 2),
        "model": metadata["best_model"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
