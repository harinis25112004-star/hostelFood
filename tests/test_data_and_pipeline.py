"""Fast unit tests for the synthetic data and operational recommendation logic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_synthetic_data import GeneratorConfig, generate_dataset, validate_dataset
from train_predictive_pipeline import create_operational_predictions


def test_generator_creates_more_than_1500_valid_records() -> None:
    config = GeneratorConfig(start_date="2024-01-01", end_date="2025-12-31", seed=7)
    df = generate_dataset(config)
    validate_dataset(df, config)
    assert len(df) > 1500
    assert set(df["meal_category"].unique()) == {"Breakfast", "Lunch", "Dinner"}
    assert (df["food_wasted_kg"] >= 0).all()


def test_recommendation_is_nonnegative_and_bounded() -> None:
    test_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "meal_category": ["Lunch"],
            "menu_name": ["Test Menu"],
            "residents_present": [500],
            "advance_bookings": [420],
            "student_turnout": [410],
            "food_consumed_kg": [220.0],
            "food_prepared_kg": [245.0],
            "food_wasted_kg": [25.0],
        }
    )
    result = create_operational_predictions(
        test_df=test_df,
        predicted_turnout=np.array([430.2]),
        kg_lookup={"Lunch": 0.54},
        safety_buffer=0.03,
    )
    assert int(result.loc[0, "predicted_turnout"]) <= 500
    assert float(result.loc[0, "recommended_food_prep_kg"]) >= 0.0
