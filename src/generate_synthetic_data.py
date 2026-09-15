"""Generate a realistic synthetic hostel meal-demand and food-waste dataset.

The generated data are intended for engineering development, demonstrations,
algorithm validation, and academic capstone experimentation. They are synthetic
and must not be presented as measurements collected from a real hostel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


MEALS = ("Breakfast", "Lunch", "Dinner")


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for reproducible synthetic dataset generation."""

    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"
    hostel_capacity: int = 520
    seed: int = 42


MEAL_PROFILE: Dict[str, Dict[str, float]] = {
    "Breakfast": {
        "base_turnout_rate": 0.74,
        "portion_kg": 0.36,
        "prep_buffer": 0.105,
        "cost_per_kg": 82.0,
    },
    "Lunch": {
        "base_turnout_rate": 0.88,
        "portion_kg": 0.54,
        "prep_buffer": 0.085,
        "cost_per_kg": 105.0,
    },
    "Dinner": {
        "base_turnout_rate": 0.82,
        "portion_kg": 0.49,
        "prep_buffer": 0.095,
        "cost_per_kg": 112.0,
    },
}


MENU_LIBRARY = {
    "Breakfast": [
        ("Idli-Sambar", 0.92),
        ("Dosa-Chutney", 0.98),
        ("Pongal-Vada", 0.89),
        ("Poori-Masala", 0.95),
        ("Upma-Chutney", 0.77),
        ("Ragi-Dosa", 0.82),
    ],
    "Lunch": [
        ("Sambar-Rice-Veg", 0.86),
        ("Veg-Biryani-Raita", 0.98),
        ("Tomato-Rice-Curd", 0.84),
        ("Lemon-Rice-Poriyal", 0.80),
        ("Meals-Kootu-Rasam", 0.90),
        ("Chapati-Kurma-Rice", 0.94),
    ],
    "Dinner": [
        ("Chapati-Kurma", 0.91),
        ("Dosa-Sambar", 0.96),
        ("Rice-Rasam-Curry", 0.82),
        ("Idiyappam-Kurma", 0.93),
        ("Variety-Rice-Curd", 0.85),
        ("Parotta-Kurma", 0.99),
    ],
}


def _date_in_ranges(date: pd.Timestamp, ranges: Iterable[tuple[str, str]]) -> bool:
    """Return True when *date* is inside any inclusive YYYY-MM-DD date range."""

    return any(pd.Timestamp(start) <= date <= pd.Timestamp(end) for start, end in ranges)


def _exam_flag(date: pd.Timestamp) -> int:
    """Approximate two examination windows per academic year."""

    year = date.year
    ranges = [
        (f"{year}-04-15", f"{year}-05-15"),
        (f"{year}-11-10", f"{year}-12-10"),
    ]
    return int(_date_in_ranges(date, ranges))


def _vacation_flag(date: pd.Timestamp) -> int:
    """Approximate semester/summer vacation periods for synthetic behavior."""

    year = date.year
    ranges = [
        (f"{year}-05-20", f"{year}-06-20"),
        (f"{year}-12-21", f"{year}-12-31"),
        (f"{year}-01-01", f"{year}-01-05"),
    ]
    return int(_date_in_ranges(date, ranges))


def _festival_flag(date: pd.Timestamp) -> int:
    """Mark a few fixed-date synthetic low-attendance festival periods."""

    md = (date.month, date.day)
    return int(
        md in {(1, 14), (1, 15), (1, 16), (8, 15), (10, 2), (12, 25)}
    )


def _seasonal_temperature(date: pd.Timestamp, rng: np.random.Generator) -> float:
    """Generate a plausible warm-climate daily temperature in degrees Celsius."""

    day = date.dayofyear
    seasonal = 29.0 + 3.8 * np.sin(2 * np.pi * (day - 45) / 365.25)
    return float(np.clip(seasonal + rng.normal(0, 1.5), 21.0, 39.0))


def _rain_index(date: pd.Timestamp, rng: np.random.Generator) -> float:
    """Generate a 0-1 rain-intensity index with monsoon seasonality."""

    month_factor = {
        1: 0.15,
        2: 0.10,
        3: 0.12,
        4: 0.20,
        5: 0.30,
        6: 0.40,
        7: 0.45,
        8: 0.48,
        9: 0.58,
        10: 0.78,
        11: 0.72,
        12: 0.40,
    }[date.month]
    wet_day = rng.random() < month_factor
    if not wet_day:
        return float(rng.uniform(0.0, 0.08))
    return float(np.clip(rng.beta(2.2, 3.0), 0.08, 1.0))


def _resident_presence(
    capacity: int,
    weekend: int,
    exam: int,
    vacation: int,
    festival: int,
    rng: np.random.Generator,
) -> int:
    """Estimate students physically present in the hostel on a given day."""

    factor = 0.965
    factor *= 0.94 if weekend else 1.0
    factor *= 0.985 if exam else 1.0
    factor *= 0.34 if vacation else 1.0
    factor *= 0.74 if festival else 1.0
    noisy = capacity * factor + rng.normal(0, 9)
    return int(np.clip(round(noisy), 60, capacity))


def generate_dataset(config: GeneratorConfig) -> pd.DataFrame:
    """Create the complete meal-level synthetic dataset."""

    rng = np.random.default_rng(config.seed)
    dates = pd.date_range(config.start_date, config.end_date, freq="D")
    rows = []
    record_id = 1

    for date in dates:
        weekend = int(date.dayofweek >= 5)
        exam = _exam_flag(date)
        vacation = _vacation_flag(date)
        festival = _festival_flag(date)
        temperature = _seasonal_temperature(date, rng)
        rain_index = _rain_index(date, rng)
        residents_present = _resident_presence(
            config.hostel_capacity,
            weekend,
            exam,
            vacation,
            festival,
            rng,
        )

        # Special events occasionally increase/decrease meal participation.
        special_event = int((not vacation) and rng.random() < 0.055)
        special_event_effect = rng.choice([0.94, 1.06, 1.10]) if special_event else 1.0

        for meal in MEALS:
            profile = MEAL_PROFILE[meal]
            menu_name, base_popularity = MENU_LIBRARY[meal][
                int(rng.integers(0, len(MENU_LIBRARY[meal])))
            ]
            menu_popularity = float(
                np.clip(base_popularity + rng.normal(0, 0.035), 0.65, 1.08)
            )

            # Attendance drivers. These values are synthetic engineering assumptions.
            turnout_rate = profile["base_turnout_rate"]
            turnout_rate *= 0.93 if weekend and meal == "Breakfast" else 1.0
            turnout_rate *= 0.97 if weekend and meal == "Lunch" else 1.0
            turnout_rate *= 1.015 if exam and meal in {"Lunch", "Dinner"} else 1.0
            turnout_rate *= 0.96 if exam and meal == "Breakfast" else 1.0
            turnout_rate *= 0.985 if rain_index > 0.60 else 1.0
            turnout_rate *= 0.88 if festival else 1.0
            turnout_rate *= special_event_effect
            turnout_rate *= menu_popularity
            turnout_rate += rng.normal(0, 0.025)
            turnout_rate = float(np.clip(turnout_rate, 0.28, 0.99))

            attendance = int(
                np.clip(round(residents_present * turnout_rate), 25, residents_present)
            )

            # Portion consumption slightly varies with meal, temperature, and exams.
            portion_multiplier = 1.0
            portion_multiplier *= 0.975 if temperature > 35 else 1.0
            portion_multiplier *= 1.018 if exam and meal == "Dinner" else 1.0
            portion_multiplier *= float(np.clip(rng.normal(1.0, 0.035), 0.90, 1.10))
            avg_consumed_per_student_kg = profile["portion_kg"] * portion_multiplier
            food_consumed_kg = attendance * avg_consumed_per_student_kg

            # Kitchen preparation is based on expected turnout with a conservative buffer.
            planning_noise = rng.normal(1.0, 0.035)
            prep_buffer = max(0.035, profile["prep_buffer"] + rng.normal(0, 0.018))
            food_prepared_kg = food_consumed_kg * planning_noise * (1.0 + prep_buffer)
            food_prepared_kg = max(food_prepared_kg, food_consumed_kg * 1.01)

            # Waste is the prepared-but-unconsumed amount plus a small plate-waste component.
            kitchen_leftover_kg = max(food_prepared_kg - food_consumed_kg, 0.0)
            plate_waste_rate = float(np.clip(rng.normal(0.025, 0.008), 0.006, 0.055))
            plate_waste_kg = food_consumed_kg * plate_waste_rate
            food_wasted_kg = kitchen_leftover_kg + plate_waste_kg
            waste_percentage = 100.0 * food_wasted_kg / max(food_prepared_kg, 1e-9)

            # Simulated pre-meal booking/reservation signal available to prediction models.
            booking_rate = float(np.clip(turnout_rate + rng.normal(0.01, 0.045), 0.20, 1.0))
            advance_bookings = int(
                np.clip(round(residents_present * booking_rate), 0, residents_present)
            )

            rows.append(
                {
                    "record_id": record_id,
                    "date": date.date().isoformat(),
                    "year": int(date.year),
                    "month": int(date.month),
                    "day_of_week": int(date.dayofweek),
                    "day_name": date.day_name(),
                    "meal_category": meal,
                    "menu_name": menu_name,
                    "hostel_capacity": config.hostel_capacity,
                    "residents_present": residents_present,
                    "advance_bookings": advance_bookings,
                    "weekend_flag": weekend,
                    "exam_flag": exam,
                    "vacation_flag": vacation,
                    "festival_flag": festival,
                    "special_event_flag": special_event,
                    "temperature_c": round(temperature, 2),
                    "rain_index": round(rain_index, 3),
                    "menu_popularity_score": round(menu_popularity, 3),
                    "student_turnout": attendance,
                    "avg_consumed_per_student_kg": round(avg_consumed_per_student_kg, 4),
                    "food_consumed_kg": round(food_consumed_kg, 2),
                    "food_prepared_kg": round(food_prepared_kg, 2),
                    "food_wasted_kg": round(food_wasted_kg, 2),
                    "waste_percentage": round(waste_percentage, 2),
                    "estimated_waste_cost_inr": round(
                        food_wasted_kg * profile["cost_per_kg"], 2
                    ),
                }
            )
            record_id += 1

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "meal_category"]).reset_index(drop=True)


def validate_dataset(df: pd.DataFrame, config: GeneratorConfig) -> None:
    """Run hard validation checks before the dataset is written to disk."""

    required = {
        "date",
        "meal_category",
        "student_turnout",
        "food_prepared_kg",
        "food_wasted_kg",
        "exam_flag",
        "weekend_flag",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(df) <= 1500:
        raise ValueError(f"Expected more than 1,500 records; generated {len(df)}")
    if df.isna().any().any():
        raise ValueError("Synthetic dataset unexpectedly contains missing values.")
    if (df["student_turnout"] > df["residents_present"]).any():
        raise ValueError("Turnout cannot exceed residents present.")
    if (df["food_wasted_kg"] < 0).any():
        raise ValueError("Food waste cannot be negative.")
    if (df["food_prepared_kg"] <= 0).any():
        raise ValueError("Prepared food must be strictly positive.")
    if df["hostel_capacity"].nunique() != 1 or int(df["hostel_capacity"].iloc[0]) != config.hostel_capacity:
        raise ValueError("Hostel capacity is inconsistent with generator configuration.")


def save_outputs(df: pd.DataFrame, output_path: Path, config: GeneratorConfig) -> None:
    """Persist CSV data and a compact metadata/quality summary."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df.copy()
    df_to_save["date"] = df_to_save["date"].dt.date.astype(str)
    df_to_save.to_csv(output_path, index=False)

    summary = {
        "synthetic_data_notice": (
            "This dataset is simulated and must not be represented as empirical hostel measurements."
        ),
        "generator_config": asdict(config),
        "record_count": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "meal_counts": {k: int(v) for k, v in df["meal_category"].value_counts().items()},
        "mean_student_turnout": round(float(df["student_turnout"].mean()), 3),
        "mean_waste_percentage": round(float(df["waste_percentage"].mean()), 3),
        "total_food_prepared_kg": round(float(df["food_prepared_kg"].sum()), 2),
        "total_food_wasted_kg": round(float(df["food_wasted_kg"].sum()), 2),
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hostel_food_waste.csv"),
        help="CSV output path.",
    )
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--hostel-capacity", type=int, default=520)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = GeneratorConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        hostel_capacity=args.hostel_capacity,
        seed=args.seed,
    )
    df = generate_dataset(config)
    validate_dataset(df, config)
    save_outputs(df, args.output, config)
    print(f"Generated {len(df):,} synthetic meal records -> {args.output}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Mean food-waste rate: {df['waste_percentage'].mean():.2f}%")


if __name__ == "__main__":
    main()
