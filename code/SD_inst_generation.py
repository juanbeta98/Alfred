#!/usr/bin/env python3
"""
simulate_artificial_day.py

Create simulated single-day instances sampling real services from a source month,
preserving per-city proportions and static/dynamic mix.

Outputs (per seed, per scenario):
  instances/simu_inst/N{n_services}/{scenario}/seed_{seed}/
    - labors_simulated.csv
    - labors_static_simulated.csv
    - labors_dynamic_simulated.csv
    - metadata/summary.json
    - metadata/sampling_plan.csv

Author: (yours)
"""

import os
import json
import argparse
from typing import Dict, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# PROJECT imports (adjust paths as needed)
from src.data.data_load import load_tables
from src.utils.inst_generation_utils import filter_invalid_services, create_hist_directory
from src.config.SD_experimentation_config import (
    n_services,
    scenarios,
    seeds
)

# ---------------------------------------------------------------------
# Time-shift helper you provided (keeps the local hour, handles tz)
# ---------------------------------------------------------------------
def _shift_to_new_day(orig_ts, new_day, tz: str = "America/Bogota"):
    """
    Move a timestamp to a new calendar day preserving the local clock time.
    Handles numpy.datetime64 by coercing to pandas.Timestamp.
    """
    if pd.isna(orig_ts) or pd.isna(new_day):
        return pd.NaT

    orig_ts = pd.Timestamp(orig_ts)
    new_day = pd.Timestamp(new_day)

    # Normalize original to tz
    if orig_ts.tzinfo is None:
        orig_local = orig_ts.tz_localize(tz)
    else:
        orig_local = orig_ts.tz_convert(tz)

    # Base of the new day in tz
    base = pd.Timestamp(new_day)
    if base.tzinfo is None:
        base = base.tz_localize(tz)
    else:
        base = base.tz_convert(tz)
    base = base.normalize()

    shifted = base + pd.Timedelta(
        hours=orig_local.hour,
        minutes=orig_local.minute,
        seconds=orig_local.second,
        microseconds=orig_local.microsecond
    )
    return shifted

# ---------------------------------------------------------------------
# Utilities & modular helpers
# ---------------------------------------------------------------------
def classify_static_dynamic_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean columns 'is_static' and 'is_dynamic' based on DATE only.
    static: created_at.date < schedule_date.date
    dynamic: created_at.date == schedule_date.date
    """
    df = df.copy()
    df["created_date_only"] = pd.to_datetime(df["created_at"]).dt.date
    # choose the schedule date column name used in your data:
    # some of your code used 'labor_start_date' for schedule_date
    # make both available if necessary
    if "schedule_date" in df.columns:
        sched_col = "schedule_date"
    else:
        raise KeyError("No schedule date column found ('schedule_date')")

    df["schedule_date_only"] = pd.to_datetime(df[sched_col]).dt.date
    df["is_static"] = df["created_date_only"] < df["schedule_date_only"]
    df["is_dynamic"] = df["created_date_only"] == df["schedule_date_only"]
    return df


def compute_city_plan(
    classified_df: pd.DataFrame,
    n_services: int,
    scenario: str,
    rng: np.random.Generator,
    scenario_multiplier_map: Dict[str, Tuple[float,float]] = None
) -> Dict[str, dict]:
    """
    Compute sampling plan per city:
      - number of samples per city (proportional to city share),
      - historic static/dynamic means,
      - simulated static proportion = historic_static_mean * sampled_multiplier (bounded 0..1),
      - derived numbers n_static, n_dynamic (integers, sum to n_city).
    """
    if scenario_multiplier_map is None:
        scenario_multiplier_map = {
            "easy": (0.6, 0.8),
            "normal": (0.9, 1.1),
            "hard": (1.2, 1.4),
        }

    if scenario not in scenario_multiplier_map:
        raise ValueError(f"Unknown scenario '{scenario}' (valid: {list(scenario_multiplier_map.keys())})")

    low, high = scenario_multiplier_map[scenario]

    city_counts = classified_df["city"].value_counts().sort_index()
    city_share = city_counts / city_counts.sum()
    n_city_samples = (city_share * n_services).round().astype(int)

    # adjust rounding diff to keep total exactly n_services
    diff = n_services - n_city_samples.sum()
    if diff != 0:
        # add diff to largest city share
        top_city = city_share.idxmax()
        n_city_samples[top_city] += diff

    plan = {}
    for city, n_city in n_city_samples.items():
        city_df = classified_df[classified_df["city"] == city]
        historic_static_mean = float(city_df["is_static"].mean()) if len(city_df) > 0 else 0.0
        historic_dynamic_mean = float(city_df["is_dynamic"].mean()) if len(city_df) > 0 else 0.0

        # sample multiplier in [low, high]
        mult = float(rng.uniform(low, high))
        static_prop_sim = historic_static_mean * mult
        static_prop_sim = min(max(static_prop_sim, 0.0), 1.0)  # clamp
        dynamic_prop_sim = 1.0 - static_prop_sim

        n_static = int(round(static_prop_sim * n_city))
        n_dynamic = int(n_city - n_static)

        # fix edge rounding differences (if sample counts exceed availability we'll correct later)
        plan[city] = {
            "n_total": int(n_city),
            "historic_static_mean": historic_static_mean,
            "historic_dynamic_mean": historic_dynamic_mean,
            "multiplier": mult,
            "static_prop_sim": static_prop_sim,
            "dynamic_prop_sim": dynamic_prop_sim,
            "n_static": n_static,
            "n_dynamic": n_dynamic
        }

    return plan

def sample_city_services(city_df: pd.DataFrame, plan_entry: dict, seed: int) -> pd.DataFrame:
    """
    Sample the requested number of services for a single city, respecting static/dynamic groups.
    Returns concatenated sampled dataframe (static + dynamic).
    If not enough records in either group, sample as many as available and adjust.
    """
    rng = np.random.default_rng(seed)
    static_df = city_df[city_df["is_static"]].copy()
    dynamic_df = city_df[city_df["is_dynamic"]].copy()

    n_static = min(plan_entry["n_static"], len(static_df))
    n_dynamic = min(plan_entry["n_dynamic"], len(dynamic_df))

    # if not enough overall, try to top-up from the other group (but keep the split as much as possible)
    total_needed = plan_entry["n_total"]
    total_available = len(static_df) + len(dynamic_df)
    n_static = min(n_static, total_available)
    n_dynamic = min(n_dynamic, max(0, total_needed - n_static))

    # final check/adjust
    if n_static + n_dynamic < total_needed:
        # try to fill using available from either group
        remaining = total_needed - (n_static + n_dynamic)
        available_else = (len(static_df) - n_static) + (len(dynamic_df) - n_dynamic)
        take_more = min(remaining, available_else)
        # greedily take from the larger group
        if len(dynamic_df) - n_dynamic >= take_more:
            n_dynamic += take_more
        else:
            add_from_dyn = max(0, len(dynamic_df) - n_dynamic)
            n_dynamic += add_from_dyn
            n_static += min(take_more - add_from_dyn, len(static_df) - n_static)

    samples = []
    if n_static > 0:
        samples.append(static_df.sample(n=n_static, replace=False, random_state=seed))
    if n_dynamic > 0:
        samples.append(dynamic_df.sample(n=n_dynamic, replace=False, random_state=seed + 1))

    if samples:
        return pd.concat(samples, ignore_index=True)
    else:
        return pd.DataFrame(columns=city_df.columns)


def preserve_deltas_and_shift(
    df: pd.DataFrame,
    sim_day: str,
    tz: str = "America/Bogota",
    schedule_col: str = "labor_start_date",
    created_col: str = "created_at",
    extra_datetime_cols: Tuple[str] = ("labor_start_date", "labor_end_date", "schedule_date", "actual_start", "actual_end")
) -> pd.DataFrame:
    """
    Shift schedule_date to sim_day and move created_at preserving delta days:
      new_schedule_date = sim_day at same local time as original schedule_date
      created_delta_days = (original_schedule_date.date - created_at.date)
      new_created_at = new_schedule_date - created_delta_days (preserving time of day)
    Also shift other datetime columns that are offsets relative to schedule_date preserving their day offsets.
    """
    df = df.copy()

    # ensure schedule col exists
    if schedule_col not in df.columns:
        raise KeyError(f"schedule_col '{schedule_col}' not found in dataframe")

    # Normalize input datetimes
    df[schedule_col] = pd.to_datetime(df[schedule_col], errors="coerce")
    df[created_col] = pd.to_datetime(df[created_col], errors="coerce")

    target_day = pd.Timestamp(sim_day)

    # compute delta in days between schedule_date and created_at (as integer days)
    # delta_days = (schedule_date.date - created_at.date)
    delta_days = (df[schedule_col].dt.floor("D") - df[created_col].dt.floor("D")).dt.days

    # Shift schedule_date to sim_day preserving the original time-of-day using _shift_to_new_day
    df[schedule_col] = df[schedule_col].apply(lambda ts: _shift_to_new_day(ts, target_day, tz))

    # New created_at = new schedule_date - delta_days (preserve time-of-day)
    def compute_new_created(row):
        sched_new = row[schedule_col]
        d = int(row["_delta_days"]) if not pd.isna(row["_delta_days"]) else 0
        # shift schedule_new back by d days (keeping time) by making new_day = sched_new - d days (midnight base)
        new_day_for_created = (pd.Timestamp(sched_new) - pd.Timedelta(days=d)).normalize()
        return _shift_to_new_day(row["_orig_created_at"], new_day_for_created, tz)

    df["_delta_days"] = delta_days
    df["_orig_created_at"] = df[created_col]

    df[created_col] = df.apply(compute_new_created, axis=1)

    # Shift other datetime columns that are offsets relative to original schedule_date.
    # Rule: if a column exists and is datetime, compute days_offset = (col_date.floor('D') - schedule_floor_old)
    # then new_col = new_schedule_date + days_offset preserving time-of-day.
    # We'll attempt for common columns; skip missing or non-datetime columns.
    # Build mapping of original schedule base for offset calculation: use original schedule date floor from original data:
    # For this we saved the original schedule in '_orig_schedule' temporarily.
    df["_orig_schedule"] = pd.to_datetime(df[schedule_col], errors="coerce")  # currently already shifted; we need the pre-shift original
    # To get original pre-shift schedule times we must reconstitute from _delta_days and created_orig? Simpler approach:
    # We stored delta days and original created_at; but original schedule original = created_orig + delta_days
    df["_orig_schedule"] = df["_orig_created_at"] + df["_delta_days"].apply(lambda d: pd.Timedelta(days=int(d)))

    # For each extra column: compute day offset between original column and original schedule, then shift
    for col in extra_datetime_cols:
        if col in df.columns:
            # coerce to datetime
            df[col] = pd.to_datetime(df[col], errors="coerce")
            # compute day offset
            orig_offset = (df[col].dt.floor("D") - df["_orig_schedule"]).dt.floor("D").dt.days
            # create new column value: new_schedule_date + offset days preserving time-of-day
            def compute_shifted_col(r, orig_col=col):
                if pd.isna(r[orig_col]):
                    return pd.NaT
                try:
                    new_day_base = pd.Timestamp(r[schedule_col]).normalize() + pd.Timedelta(days=int(r["_offset_tmp"]))
                    return _shift_to_new_day(r[orig_col], new_day_base, tz)
                except Exception:
                    return pd.NaT
            # attach temporary offsets to df for compute
            df["_offset_tmp"] = orig_offset.fillna(0).astype(int)
            df[col] = df.apply(compute_shifted_col, axis=1)
            df.drop(columns=["_offset_tmp"], inplace=True, errors=True)

    # cleanup temporary cols
    df.drop(columns=["_delta_days", "_orig_created_at", "_orig_schedule"], inplace=True, errors=True)
    return df

def ensure_directory_layout(base_path: str, n_services: int, scenario: str, seed: int):
    base_dir = os.path.join(base_path, "instances", "simu_inst", f"N{n_services}", scenario, f"seed_{seed}")
    meta_dir = os.path.join(base_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    return base_dir, meta_dir

def save_outputs(
    simulated_df: pd.DataFrame, 
    hist_directory: pd.DataFrame,
    sampling_plan: dict, 
    data_path: str,
    n_services: int, 
    scenario: str, 
    seed: int, 
    sim_day: str
):
    base_dir, meta_dir = ensure_directory_layout(data_path, n_services, scenario, seed)
    # main files
    labors_path = os.path.join(base_dir, "labors_sim_df.csv")
    labors_static_path = os.path.join(base_dir, "labors_sim_static_df.csv")
    labors_dynamic_path = os.path.join(base_dir, "labors_sim_dynamic_df.csv")
    hist_directory_path = os.path.join(base_dir, "directorio_hist_df.csv")

    simulated_df.to_csv(labors_path, index=False)
    simulated_df[simulated_df["is_static"]].to_csv(labors_static_path, index=False)
    simulated_df[simulated_df["is_dynamic"]].to_csv(labors_dynamic_path, index=False)
    hist_directory.to_csv(hist_directory_path, index=False)

    # sampling plan CSV
    sampling_rows = []
    for city, values in sampling_plan.items():
        row = values.copy()
        row["city"] = city
        sampling_rows.append(row)
    sampling_df = pd.DataFrame(sampling_rows)
    sampling_csv_path = os.path.join(meta_dir, "sampling_plan.csv")
    sampling_df.to_csv(sampling_csv_path, index=False)

    # summary JSON
    summary = {
        "sim_day": sim_day,
        "n_services": n_services,
        "scenario": scenario,
        "seed": seed,
        # "timestamp": datetime.utcnow().isoformat() + "Z",
        "per_city": sampling_plan,
    }
    summary_path = os.path.join(meta_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    return {
        "labors_path": labors_path,
        "labors_static_path": labors_static_path,
        "labors_dynamic_path": labors_dynamic_path,
        "sampling_csv_path": sampling_csv_path,
        "summary_path": summary_path
    }

# ---------------------------------------------------------------------
# Main orchestration function
# ---------------------------------------------------------------------
def simulate_artificial_day(
    data_path: str,
    month: str,
    sim_day: str,
    n_services: int,
    seed: int = 42,
    scenario: str = "normal",
    tz: str = "America/Bogota"
):
    """
    Full pipeline to create a simulated single day instance.
    """
    rng = np.random.default_rng(seed)

    # Load tables (use your project's load_tables)
    directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path, generate_labors=False)

    # Filter invalid services using your util
    labors_filtered_df = filter_invalid_services(labors_raw_df, min_delay_minutes=120, only_unilabor_services=False)
    # Make sure relevant cols are datetime
    for col in ['schedule_date', "created_at", "labor_start_date", "labor_end_date"]:
        if col in labors_filtered_df.columns:
            labors_filtered_df[col] = pd.to_datetime(labors_filtered_df[col], errors="coerce")

    # Select month subset
    if "schedule_date" not in labors_filtered_df.columns:
        raise KeyError("Expected 'schedule_date' in loaded labors data")

    month_df = labors_filtered_df[
        labors_filtered_df["schedule_date"].dt.strftime("%Y-%m") == month
    ].copy()

    if month_df.empty:
        raise ValueError(f"No services found for month {month}")

    # classify static/dynamic by date-only
    classified = classify_static_dynamic_by_date(month_df)

    # compute per-city sampling plan
    sampling_plan = compute_city_plan(classified, n_services, scenario, rng)

    # sample per city
    sampled_frames = []
    for city, plan in sampling_plan.items():
        city_df = classified[classified["city"] == city]
        if city_df.empty:
            continue
        sampled_city = sample_city_services(city_df, plan, seed + hash(city) % 99999)
        sampled_frames.append(sampled_city)

    if not sampled_frames:
        raise RuntimeError("No samples were drawn for any city")

    simulated_df = pd.concat(sampled_frames, ignore_index=True)

    # After sampling, ensure we have exact intended total: if not, trim or augment minimally
    if len(simulated_df) > n_services:
        simulated_df = simulated_df.sample(n=n_services, random_state=seed).reset_index(drop=True)
    elif len(simulated_df) < n_services:
        # try to top-up from month_df (within city proportions)
        deficit = n_services - len(simulated_df)
        fallback_pool = month_df.drop(index=simulated_df.index, errors="ignore")
        extra = fallback_pool.sample(n=min(deficit, len(fallback_pool)), random_state=seed + 999)
        simulated_df = pd.concat([simulated_df, extra], ignore_index=True)

    # record is_static / is_dynamic for the sampled set (in case sampling adjusted)
    simulated_df = classify_static_dynamic_by_date(simulated_df)

    # SHIFT DATES: preserve deltas and shift schedule_date -> sim_day
    # we assume schedule col is 'labor_start_date' and created col is 'created_at'
    simulated_shifted = preserve_deltas_and_shift(
        simulated_df,
        sim_day,
        tz=tz,
        schedule_col="schedule_date",
        created_col="created_at",
        extra_datetime_cols=("labor_start_date", "labor_end_date")
    )

    # rename schedule column if you want a common name 'schedule_date'
    if "schedule_date" not in simulated_shifted.columns and "labor_start_date" in simulated_shifted.columns:
        simulated_shifted["schedule_date"] = simulated_shifted["labor_start_date"]

    # Recompute any columns you want (e.g., simulated_day tag)
    simulated_shifted["simulated_day"] = pd.to_datetime(sim_day)

    # ------ Create historic directory of drivers ------
    hist_directory = create_hist_directory(simulated_shifted)

    # ------ Saving instance ------
    out_paths = save_outputs(
        simulated_shifted, 
        hist_directory,
        sampling_plan, 
        data_path,
        n_services, 
        scenario, 
        seed, 
        sim_day
    )

    # Return dataframe and paths for further use
    return simulated_shifted, sampling_plan, out_paths

# ---------------------------------------------------------------------
# CLI / Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate an artificial single-day instance (v2).")
    data_path = '../data'
    parser.add_argument("--month", default="2025-06", help="Source month (YYYY-MM)")
    parser.add_argument("--sim_day", default="2026-11-11", help="Simulated day (YYYY-MM-DD)")
    args = parser.parse_args()

    for n_serv in n_services:
        for scenario in scenarios:
            for seed in seeds:
                df_sim, plan, paths = simulate_artificial_day(
                    data_path=data_path,
                    month=args.month,
                    sim_day=args.sim_day,
                    n_services=n_serv,
                    seed=seed,
                    scenario=scenario
                )
                
                print(f' ✅ {n_serv} -\t{scenario} -\t{seed}')
