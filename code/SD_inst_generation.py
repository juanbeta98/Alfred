"""
Simulate an artificial single-day instance by sampling real services proportionally per city.

Example usage:
    python simulate_artificial_day.py \
        --month 2025-05 \
        --sim_day 2026-11-11 \
        --n_services 2000 \
        --seed 42
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# ---- Project imports ----
from src.data.data_load import load_tables
from src.config.config import *
from src.utils.inst_generation_utils import filter_invalid_services


# ==========================================================
# ========== Helper: Shift timestamps to sim_day ===========
# ==========================================================
def shift_timestamp_to_day(ts: pd.Timestamp, target_day: str) -> pd.Timestamp:
    """Shift timestamp to the same time of day but on a new target date."""
    if pd.isna(ts):
        return pd.NaT
    # Ensure ts is datetime
    ts = pd.to_datetime(ts, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    time_part = ts.time()
    return pd.to_datetime(f"{target_day} {time_part}")


# ==========================================================
# ========== Core simulation function ======================
# ==========================================================
def simulate_artificial_day(
    data_path: str,
    month: str,
    sim_day: str,
    n_services: int,
    seed: int = 42,
):
    """
    Simulate an artificial single-day instance by sampling services from a given month.

    Each city contributes proportionally to its share of total services that month.
    All timestamps are shifted to the same simulated date, preserving time-of-day patterns.

    Parameters
    ----------
    data_path : str
        Base Alfred data path.
    month : str
        Month to sample from, e.g. "2025-05".
    sim_day : str
        Target simulated date, e.g. "2026-11-11".
    n_services : int
        Total number of labors to include in the simulated day.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Simulated labors dataframe.
    """
    print(f"\n===================== Simulating Artificial Day =====================")
    print(f"📆 Month source: {month}")
    print(f"🧪 Simulated day: {sim_day}")
    print(f"🔢 Target total services: {n_services}")
    print(f"🎲 Seed: {seed}")
    print(f"====================================================================\n")

    # --------------------------------------------------------------
    # 1️⃣ Load and filter
    # --------------------------------------------------------------
    directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(
        data_path, generate_labors=False
    )

    labors_filtered_df = filter_invalid_services(
        labors_raw_df,
        min_delay_minutes=0,
        only_unilabor_services=True,
    )

    # Ensure datetime
    labors_filtered_df["labor_start_date"] = pd.to_datetime(labors_filtered_df["labor_start_date"], errors="coerce")

    # --------------------------------------------------------------
    # 2️⃣ Filter for the target month
    # --------------------------------------------------------------
    month_period = pd.Period(month)
    month_df = labors_filtered_df[
        labors_filtered_df["labor_start_date"].dt.to_period("M") == month_period
    ].copy()

    if month_df.empty:
        raise ValueError(f"No services found for month {month} — check data and date format.")

    # --------------------------------------------------------------
    # 3️⃣ Compute per-city proportions
    # --------------------------------------------------------------
    city_counts = month_df["city"].value_counts()
    city_share = city_counts / city_counts.sum()
    n_city_samples = (city_share * n_services).round().astype(int)

    # Adjust rounding to match total exactly
    diff = n_services - n_city_samples.sum()
    if diff != 0:
        top_city = city_share.idxmax()
        n_city_samples[top_city] += diff

    print("📊 Sampling plan per city:")
    print(n_city_samples)

    # --------------------------------------------------------------
    # 4️⃣ Sample per city
    # --------------------------------------------------------------
    rng = np.random.default_rng(seed)
    sampled_dfs = []
    for i, (city, n) in enumerate(n_city_samples.items()):
        city_df = month_df[month_df["city"] == city]
        n = min(n, len(city_df))  # prevent oversampling

        sampled_city_df = city_df.sample(
            n=n, replace=False, random_state=seed + i
        ).copy()
        sampled_dfs.append(sampled_city_df)

    simulated_df = pd.concat(sampled_dfs, ignore_index=True)

    # --------------------------------------------------------------
    # 5️⃣ Shift timestamps to sim_day
    # --------------------------------------------------------------
    time_cols = [col for col in simulated_df.columns if "time" in col or "date" in col or "start" in col or "end" in col]
    time_cols = [c for c in time_cols if simulated_df[c].dtype == "datetime64[ns]"]

    for col in time_cols:
        simulated_df[col] = simulated_df[col].apply(lambda ts: shift_timestamp_to_day(ts, sim_day))

    simulated_df["simulated_day"] = pd.to_datetime(sim_day)

    # --------------------------------------------------------------
    # 6️⃣ Save
    # --------------------------------------------------------------
    output_dir = os.path.join(data_path, "instances", "simu_inst", f'N{n_services}')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"labors_simulated_df_seed{seed}.csv")
    simulated_df.to_csv(output_path, index=False)

    print(f"\n💾 Simulated day saved to: {output_path}")
    print(f"✅ Total simulated labors: {len(simulated_df)}")
    print(f"✅ Cities represented: {simulated_df['city'].nunique()}")
    print(f"====================================================================\n")

    return simulated_df


# ==========================================================
# ========== CLI ===========================================
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate an artificial single-day instance.")

    args = parser.parse_args()

    data_path = f'{REPO_PATH}/data'
    month = '2025-06'
    sim_day = '2026-11-11'
    n_services = 150

    for seed in range(50):
        simulate_artificial_day(
            data_path=data_path,
            month=month,
            sim_day=sim_day,
            n_services=n_services,
            seed=seed,
        )
