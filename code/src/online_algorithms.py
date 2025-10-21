import pandas as pd
import numpy as np


from datetime import datetime, timedelta

import random

from typing import Tuple, Optional, Dict, Any, List, Union, Callable

from src.filtering import flexible_filter
from src.distance_utils import distance


''' Evaluating driver feasibility functions '''
def evaluate_driver_feasibility(
    new_labor: pd.Series,
    driver:float,
    moves_driver_df: pd.DataFrame,
    directory_df: pd.DataFrame,
    ALFRED_SPEED: float,
    VEHICLE_TRANSPORT_SPEED: float,
    TIEMPO_ALISTAR: float,
    TIEMPO_FINALIZACION: float,
    TIEMPO_GRACIA: float,
    EARLY_BUFFER: int = 30
) -> Tuple[bool, str, Optional[dict]]:
    """
    Determine whether a driver can insert a new labor into their route *and* build a full insertion plan.

    Returns:
    --------
    feasible : bool
        Whether the insertion is possible.
    reason : str
        If infeasible, description of the reason.
    insertion_plan : dict or None
        A structured plan describing how the driver’s schedule would look after insertion.
    """

    feasible = False
    infeasible_log = ''
    insertion_plan = None

    # ---------------------------------------------------------------------------------------
    # Case 1: Driver has no other labors assigned
    # ---------------------------------------------------------------------------------------
    if moves_driver_df.empty:
        feasible, infeasible_log, insertion_plan = _direct_insertion_empty_driver(
            new_labor=new_labor,
            driver=driver,
            directory_df=directory_df,
            alfred_speed=ALFRED_SPEED,
            VEHICLE_TRANSPORT_SPEED=VEHICLE_TRANSPORT_SPEED,
            TIEMPO_ALISTAR=TIEMPO_ALISTAR,
            TIEMPO_FINALIZACION=TIEMPO_FINALIZACION,
            EARLY_BUFFER=EARLY_BUFFER
        )
        return feasible, infeasible_log, insertion_plan


    # ---------------------------------------------------------------------------------------
    # Case 2: insertion before first labor
    # ---------------------------------------------------------------------------------------
    is_before_first_labor = _is_creation_before_first_labor(
        moves_driver_df,
        directory_df,
        driver
    )
    if is_before_first_labor:
        # --- Skip if next labor starts before new labor’s starts moving
        if new_labor['schedule_date'] <= moves_driver_df.loc[2,'schedule_date']:
            feasible, infeasible_log, insertion_plan = _evaluate_and_execute_insertion_before_first_labor(
                new_labor=new_labor,
                moves_driver_df=moves_driver_df,
                driver=driver,
                directory_df=directory_df,
                vehicle_transport_speed=VEHICLE_TRANSPORT_SPEED,
                alfred_speed=ALFRED_SPEED,
                tiempo_alistar=TIEMPO_ALISTAR,
                tiempo_finalizacion=TIEMPO_FINALIZACION,
                tiempo_gracia=TIEMPO_GRACIA,
                early_buffer=EARLY_BUFFER
            )

            return feasible, infeasible_log, insertion_plan

    labor_iter = 2

              
    # ---------------------------------------------------------------------------------------
    # Case 3: Insertion between existing labors
    # ---------------------------------------------------------------------------------------
    n_rows = len(moves_driver_df)

    while labor_iter < len(moves_driver_df):
        # --- Break if the current labor is the next labor
        if labor_iter + 3 > len(moves_driver_df):
                break
        
        # --- Context: previous labor & next labor
        curr_end_time, curr_end_pos, next_start_time, next_start_pos = \
            _get_driver_context(moves_driver_df, labor_iter)

        curr_labor_id = moves_driver_df.loc[labor_iter, "labor_id"]
        next_labor_id_candidate = moves_driver_df.loc[labor_iter + 3, "labor_id"]

        # --- Skip if next labor starts before new labor’s schedule
        if next_start_time < new_labor['schedule_date']:
            labor_iter += 3
            continue

        # --- Compute arrival to new service
        would_arrive_at, dist_to_new_service, travel_time_to_new = _compute_arrival_to_next_labor(
            current_end_time=curr_end_time,
            current_end_pos=curr_end_pos, 
            target_pos=new_labor['start_address_point'],
            speed=ALFRED_SPEED
        )

        # --- Break if driver wouldn't arrive on time to new labor
        if would_arrive_at > new_labor['latest_arrival_time']:
            infeasible_log = "Driver would not arrive on time to the new labor."
            break

        # --- Adjust if arriving too early (driver waits)
        real_arrival_time, start_move_time = _adjust_for_early_arrival(
            would_arrive_at=would_arrive_at,
            travel_time=travel_time_to_new,
            schedule_date=new_labor['schedule_date'], 
            early_buffer=EARLY_BUFFER
        )

        # --- Compute new labor end time
        finish_new_labor_time, finish_new_labor_pos, new_labor_duration, new_labor_distance = \
            _compute_service_end_time(
                arrival_time=real_arrival_time,
                start_pos=new_labor['start_address_point'],
                end_pos=new_labor['end_address_point'],
                vehicle_speed=VEHICLE_TRANSPORT_SPEED,
                prep_time=TIEMPO_ALISTAR,
                finish_time=TIEMPO_FINALIZACION
            )

        # --- Check feasibility with next scheduled labor
        feasible_next, would_arrive_next, dist_to_next_labor, travel_time_to_next = \
            _can_reach_next_labor(
                new_finish_time=finish_new_labor_time, 
                new_finish_pos=finish_new_labor_pos,
                next_start_time=next_start_time, 
                next_start_pos=next_start_pos,
                driver_speed=ALFRED_SPEED, 
                grace_time=TIEMPO_GRACIA
            )

        if not feasible_next:
            infeasible_log = (
                "Driver would not make it to the next scheduled labor in time "
                "if new labor is inserted."
            )
            break
        
        # --- DOWNSTREAM FEASIBILITY CHECK
        downstream_ok, downstream_shifts = _simulate_downstream_shift(
            moves_driver_df=moves_driver_df,
            start_idx=labor_iter+3,
            next_labor_id=next_labor_id_candidate,
            next_labor_arrival=would_arrive_next,
            next_labor_start_pos=finish_new_labor_pos,
            ALFRED_SPEED=ALFRED_SPEED,
            VEHICLE_TRANSPORT_SPEED=VEHICLE_TRANSPORT_SPEED,
            TIEMPO_ALISTAR=TIEMPO_ALISTAR,
            TIEMPO_FINALIZACION=TIEMPO_FINALIZACION,
            TIEMPO_GRACIA=TIEMPO_GRACIA,
            EARLY_BUFFER=EARLY_BUFFER
        )

        if not downstream_ok:
            infeasible_log = "Downstream labors become infeasible after insertion."
            break

        # --- FEASIBLE INSERTION FOUND
        feasible = True

        # Precompute updated moves for the new labor and next labor
        new_moves = _build_new_moves_block(
            new_labor=new_labor,
            driver=driver,
            start_free_time=curr_end_time,
            start_move_time=start_move_time,
            move_duration_min=travel_time_to_new,
            real_arrival_time=real_arrival_time,
            new_labor_distance=new_labor_distance,
            new_labor_duration=new_labor_duration,
            finish_new_time=finish_new_labor_time,
            start_pos=curr_end_pos,
            start_address=new_labor["start_address_point"],
            end_address=new_labor["end_address_point"],
            dist_to_new=dist_to_new_service,
        )

        # Build insertion plan dict (the full blueprint)
        insertion_plan = {
            "driver_id": driver,
            "new_labor_id": new_labor["labor_id"],
            "prev_labor_id": curr_labor_id,
            "next_labor_id": next_labor_id_candidate,
            "dist_to_new_service": dist_to_new_service,
            "dist_to_next_labor": dist_to_next_labor,
            "new_moves": new_moves,
            "new_labor_timing": {
                "arrival_time": real_arrival_time,
                "finish_time": finish_new_labor_time,
                'total_duration': new_labor_duration,
                "travel_distance": new_labor_distance,
            },
            "updated_labors": [{
                "labor_id": next_labor_id_candidate,
                "new_start_time": would_arrive_next,
                "new_free_time": finish_new_labor_time,
            }],
            "affected_labors": [next_labor_id_candidate] + [s.keys() for s in downstream_shifts],
            "downstream_shifts": downstream_shifts,
            "first_assignment": False
        }

        break  # stop after finding first feasible insertion
    

    # ---------------------------------------------------------------------------------------
    # Case 4: Insert labor at the end of the shift
    # ---------------------------------------------------------------------------------------
    if not feasible and infeasible_log == '' and labor_iter >= n_rows - 3:
        # feasible = True
        prev_labor_id = moves_driver_df.loc[labor_iter, "labor_id"]

        # Compute basic timing info
        curr_end_time = moves_driver_df.loc[labor_iter, "actual_end"]
        curr_end_pos = moves_driver_df.loc[labor_iter, "end_point"]
        would_arrive_at, dist_to_new_service, travel_time_to_new = _compute_arrival_to_next_labor(
            curr_end_time,
            curr_end_pos,
            new_labor["start_address_point"],
            ALFRED_SPEED,
        )

        # Check if driver would arrive on time to the new labor
        if would_arrive_at > new_labor["latest_arrival_time"]:
            infeasible_log = "Driver would arrive too late to the new labor."
            return feasible, infeasible_log, insertion_plan


        real_arrival_time, start_move_time = _adjust_for_early_arrival(
            would_arrive_at=would_arrive_at,
            travel_time = travel_time_to_new,
            schedule_date=new_labor["schedule_date"], 
            early_buffer=EARLY_BUFFER
        )

        finish_new_labor_time, finish_new_labor_pos, new_labor_duration, new_labor_distance = \
            _compute_service_end_time(
                arrival_time=real_arrival_time,
                start_pos=new_labor["start_address_point"],
                end_pos=new_labor["end_address_point"],
                vehicle_speed=VEHICLE_TRANSPORT_SPEED,
                prep_time=TIEMPO_ALISTAR,
                finish_time=TIEMPO_FINALIZACION,
            )

        new_moves = _build_new_moves_block(
            new_labor=new_labor,
            driver=driver,
            start_free_time=curr_end_time,
            start_move_time=start_move_time,
            move_duration_min=travel_time_to_new,
            real_arrival_time=real_arrival_time,
            new_labor_distance=new_labor_distance,
            new_labor_duration=new_labor_duration,
            finish_new_time=finish_new_labor_time,
            start_pos=curr_end_pos,
            start_address=new_labor["start_address_point"],
            end_address=new_labor["end_address_point"],
            dist_to_new=dist_to_new_service,
        )

        insertion_plan = {
            "driver_id": driver,
            "new_labor_id": new_labor["labor_id"],
            "prev_labor_id": prev_labor_id,
            "next_labor_id": None,
            "dist_to_new_service": dist_to_new_service,
            "dist_to_next_labor": None,
            "new_moves":  new_moves,
            "new_labor_timing": {
                "arrival_time": real_arrival_time,
                "finish_time": finish_new_labor_time,
                "total_duration": new_labor_duration,
                "travel_distance": new_labor_distance,
            },
            "updated_labors": [],
            "affected_labors": [],
            "downstream_shifts": [],
            "first_assignment": False
        }


    # ============================================================
    # FINAL RETURN
    # ============================================================

    return feasible, infeasible_log, insertion_plan


def _direct_insertion_empty_driver(
    new_labor: pd.Series,
    driver:float,
    directory_df:pd.DataFrame,
    alfred_speed: float,
    VEHICLE_TRANSPORT_SPEED: float,
    TIEMPO_ALISTAR: float,
    TIEMPO_FINALIZACION: float,
    EARLY_BUFFER: float,

) -> Tuple:
    home_pos = directory_df.loc[directory_df['alfred'] == driver, 'address_point'].iloc[0]

    # 1. Compute start and end times for the new labor (driver starts “from base” or assumed start)
    start_time = new_labor["schedule_date"]
    arrival_time = start_time - timedelta(minutes=EARLY_BUFFER)  # no travel, assume immediate availability

    _, dist_to_new_service, travel_time_to_new = _compute_arrival_to_next_labor(
        current_end_time=arrival_time, 
        current_end_pos=home_pos,
        target_pos=new_labor['start_address_point'],
        speed=alfred_speed
    )
    
    start_move_time = arrival_time - timedelta(minutes=travel_time_to_new)
    
    finish_new_labor_time, finish_new_labor_pos, new_labor_duration, new_labor_distance = \
        _compute_service_end_time(
        arrival_time,
        new_labor["start_address_point"],
        new_labor["end_address_point"],
        VEHICLE_TRANSPORT_SPEED,
        TIEMPO_ALISTAR,
        TIEMPO_FINALIZACION,
    )

    # Precompute updated moves for the new labor and next labor
    new_moves = _build_new_moves_block(
        new_labor=new_labor,
        driver=driver,
        start_free_time=None,
        start_move_time=start_move_time,
        move_duration_min=travel_time_to_new,
        real_arrival_time=arrival_time,
        new_labor_distance=new_labor_distance,
        new_labor_duration=new_labor_duration,
        finish_new_time=finish_new_labor_time,
        start_pos=home_pos,
        start_address=new_labor["start_address_point"],
        end_address=new_labor["end_address_point"],
        dist_to_new=dist_to_new_service,
    )

    # 3. Prepare minimal insertion plan
    insertion_plan = {
        "driver_id": driver,
        "new_labor_id": new_labor["labor_id"],
        "prev_labor_id": None,
        "next_labor_id": None,
        "dist_to_new_service": 0,
        "dist_to_next_labor": None,
        "new_moves": new_moves,
        "new_labor_timing": {
            "arrival_time": arrival_time,
            "finish_time": finish_new_labor_time,
            "total_duration": new_labor_duration,
            "travel_distance": new_labor_distance
        },
        "updated_labors": [],
        "affected_labors": [],
        "downstream_shifts": [],
        "first_assignment": False
    }

    return True, "", insertion_plan


def _is_creation_before_first_labor(
    moves_driver_df: pd.DataFrame,
    directory_df: pd.DataFrame,
    driver: str,
) -> bool:
    """
    Determine if a new labor can be inserted *before* the driver's first scheduled labor.

    Logic:
    -------
    If the first movement (the '_free' segment) starts from the same location
    as the driver's registered starting address in the directory, we can consider
    the driver as 'at base' before any assigned labor → insertion before first labor is valid.

    Returns
    -------
    bool : True if insertion before first labor is possible.
    """
    # --- Extract driver's base location from directory ---
    try:
        driver_base = (
            directory_df.loc[directory_df["alfred"] == driver, "address_point"]
            .dropna()
            .iloc[0]
        )
    except IndexError:
        # Driver not found in directory → cannot determine base position
        return False

    first_start = str(moves_driver_df.iloc[0]["start_point"]).strip()
    driver_base = str(driver_base).strip()

    # --- Compare normalized positions ---
    return first_start == driver_base


def _evaluate_and_execute_insertion_before_first_labor(
    new_labor, 
    moves_driver_df,
    driver: str,
    directory_df: pd.DataFrame,
    vehicle_transport_speed: float,
    alfred_speed: float,
    tiempo_alistar: float,
    tiempo_finalizacion: float,
    tiempo_gracia: float,
    early_buffer: float,
) -> Tuple:
    
    home_pos = directory_df.loc[directory_df['alfred'] == driver, 'address_point'].iloc[0]

    arrival_time = new_labor["schedule_date"] - timedelta(minutes=early_buffer)
    
    _, dist_to_new_service, travel_time_to_new = _compute_arrival_to_next_labor(
        current_end_time=arrival_time, 
        current_end_pos=home_pos,
        target_pos=new_labor['start_address_point'],
        speed=alfred_speed)
    
    start_move_time = arrival_time - timedelta(minutes=travel_time_to_new)

    next_labor_id_candidate = moves_driver_df.loc[0, 'labor_id']

    # Compute new labor end time
    finish_new_labor_time, finish_new_labor_pos, new_labor_duration, new_labor_distance = \
        _compute_service_end_time(
            arrival_time=arrival_time,
            start_pos=new_labor['start_address_point'],
            end_pos=new_labor['end_address_point'],
            vehicle_speed=vehicle_transport_speed,
            prep_time=tiempo_alistar,
            finish_time=tiempo_finalizacion
        )
    
    # --- Check feasibility with next scheduled labor
    feasible_next, would_arrive_next, dist_to_next_labor, travel_time_to_next = \
        _can_reach_next_labor(
            new_finish_time=finish_new_labor_time, 
            new_finish_pos=finish_new_labor_pos,
            next_start_time=moves_driver_df.loc[0, 'schedule_date'],
            next_start_pos=moves_driver_df.loc[0, 'start_point'],
            driver_speed=alfred_speed, 
            grace_time=tiempo_gracia,
        )

    if not feasible_next:
        infeasible_log = (
            "Driver would not make it to the next scheduled labor in time "
            "if new labor is inserted."
        )
        return False, infeasible_log, None

    new_moves = _build_new_moves_block(
        new_labor=new_labor,
        driver=driver,
        start_free_time=None,
        start_move_time=start_move_time,
        move_duration_min=travel_time_to_new,
        real_arrival_time=arrival_time,
        new_labor_distance=new_labor_distance,
        new_labor_duration=new_labor_duration,
        finish_new_time=finish_new_labor_time,
        start_pos=home_pos,
        start_address=new_labor["start_address_point"],
        end_address=new_labor["end_address_point"],
        dist_to_new=dist_to_new_service,
        )
    
    # Now simulate downstream shift from the FIRST labor
    downstream_ok, downstream_shifts = _simulate_downstream_shift(
        moves_driver_df=moves_driver_df,
        start_idx=0,
        next_labor_id=moves_driver_df.iloc[0]["labor_id"],
        next_labor_arrival=finish_new_labor_time,
        next_labor_start_pos=finish_new_labor_pos,
        ALFRED_SPEED=alfred_speed,
        VEHICLE_TRANSPORT_SPEED=vehicle_transport_speed,
        TIEMPO_ALISTAR=tiempo_alistar,
        TIEMPO_FINALIZACION=tiempo_finalizacion,
        TIEMPO_GRACIA=tiempo_gracia,
        EARLY_BUFFER=early_buffer
    )

    if not downstream_ok:
        return False, "Downstream labors become infeasible after early insertion.", None
    
    # Build insertion plan same as normal
    insertion_plan = {
        "driver_id": driver,
        "new_labor_id": new_labor["labor_id"],
        "prev_labor_id": None,
        "next_labor_id": next_labor_id_candidate,
        "dist_to_new_service": None,
        "dist_to_next_labor": dist_to_next_labor,
        "new_moves": new_moves,
        "new_labor_timing": {
            "arrival_time": arrival_time,
            "finish_time": finish_new_labor_time,
            "travel_distance": dist_to_next_labor,
        },
        "updated_labors": [{
            "labor_id": next_labor_id_candidate,
            "new_start_time": would_arrive_next,
            "new_free_time": finish_new_labor_time,
        }],
        "affected_labors": [next_labor_id_candidate] + [s.keys() for s in downstream_shifts],
        "downstream_shifts": downstream_shifts,
        "first_assignment": True
    }

    return True, "", insertion_plan


def _build_new_moves_block(
    new_labor: pd.Series,
    driver: str,
    start_free_time: datetime,
    start_move_time: datetime,
    move_duration_min: float,
    real_arrival_time: datetime,
    new_labor_distance: float,
    new_labor_duration: float,
    finish_new_time: datetime,
    start_pos: Optional[str],
    start_address: str,
    end_address: str,
    dist_to_new: float,
) -> pd.DataFrame:
    """
    Construct the three standardized rows (FREE_TIME, DRIVER_MOVE, LABOR)
    associated with a new labor insertion.

    All labors are represented by exactly three rows:
        - {labor_id}_free   → FREE_TIME
        - {labor_id}_move   → DRIVER_MOVE
        - {labor_id}_labor  → Actual LABOR

    This ensures consistent triplet structure across all labors in moves_df.

    Parameters
    ----------
    new_labor : pd.Series
        Row from labors_dynamic_df containing new labor details.
    driver : str
        Driver identifier.
    start_free_time, start_move_time, real_arrival_time, finish_new_time : datetime
        Precomputed timestamps marking free, move, and labor segments.
    move_duration_min, new_labor_distance, new_labor_duration : float
        Precomputed durations (min) and distances (km) for move and labor.
    start_pos, start_address, end_address : str
        WKT/point strings for spatial start and end of each segment.
    dist_to_new : float
        Distance from previous endpoint to new labor start (km).


    Returns
    -------
    pd.DataFrame
        3-row standardized DataFrame for the new labor segment,
        ready for concatenation into moves_df.
    """
    new_labor_service_id = new_labor.get("service_id", None)
    new_labor_labor_id = new_labor["labor_id"]
    new_labor_schedule_date = new_labor["schedule_date"]


    # --- Compute move timing ---
    move_end_time = real_arrival_time

    # --- Compute free timing ---
    if start_free_time == None or start_free_time > start_move_time:
        # No waiting time — zero-duration free
        start_free_time = end_free_time = start_move_time
    else:
        end_free_time = start_move_time

    rows = []

    # ============================================================
    # 1️⃣ FREE TIME ROW
    # ============================================================
    rows.append({
        "service_id": new_labor_service_id,
        "labor_id": new_labor_labor_id,
        "labor_context_id": f"{new_labor['labor_id']}_free",
        "labor_name": "FREE_TIME",
        "labor_category": "FREE_TIME",
        "assigned_driver": driver,
        "schedule_date": new_labor_schedule_date,
        "actual_start": start_free_time,
        "actual_end": end_free_time,
        "start_point": start_pos if start_pos is not None else start_address,
        "end_point": start_pos if start_pos is not None else start_address,
        "distance_km": 0.0,
        "duration_min": (end_free_time - start_free_time).total_seconds() / 60.0,
        "city": new_labor["city"],
        "date": new_labor.get("date", new_labor["schedule_date"].date()),
    })

    # ============================================================
    # 2️⃣ DRIVER MOVE ROW
    # ============================================================
    rows.append({
        "service_id": new_labor_service_id,
        "labor_id": new_labor_labor_id,
        "labor_context_id": f"{new_labor['labor_id']}_move",
        "labor_name": "DRIVER_MOVE",
        "labor_category": "DRIVER_MOVE",
        "assigned_driver": driver,
        "schedule_date": new_labor_schedule_date,
        "actual_start": start_move_time,
        "actual_end": move_end_time,
        "start_point": start_pos if start_pos is not None else start_address,
        "end_point": start_address,
        "distance_km": dist_to_new,
        "duration_min": move_duration_min,
        "city": new_labor["city"],
        "date": new_labor.get("date", new_labor["schedule_date"].date()),
    })

    # ============================================================
    # 3️⃣ LABOR ROW
    # ============================================================
    rows.append({
        "service_id": new_labor_service_id,
        "labor_id": new_labor_labor_id,
        "labor_context_id": f"{new_labor['labor_id']}_labor",
        "labor_name": new_labor["labor_name"],
        "labor_category": new_labor["labor_category"],
        "assigned_driver": driver,
        "schedule_date": new_labor_schedule_date,
        "actual_start": real_arrival_time,
        "actual_end": finish_new_time,
        "start_point": start_address,
        "end_point": end_address,
        "distance_km": new_labor_distance,
        "duration_min": new_labor_duration,
        "city": new_labor["city"],
        "date": new_labor.get("date", new_labor["schedule_date"].date()),
    })

    # ============================================================
    # ✅ Return clean and aligned DataFrame
    # ============================================================
    cols = [
        "service_id", "labor_id", "labor_context_id", "labor_name",
        "labor_category", "assigned_driver", "schedule_date", "actual_start",
        "actual_end", "start_point", "end_point", "distance_km",
        "duration_min", "city", "date"
    ]
    return pd.DataFrame(rows)[cols]


def _simulate_downstream_shift(
    moves_driver_df: pd.DataFrame,
    start_idx: int,
    next_labor_id,
    next_labor_arrival,
    next_labor_start_pos: str,
    ALFRED_SPEED: float,
    VEHICLE_TRANSPORT_SPEED: float,
    TIEMPO_ALISTAR: float,
    TIEMPO_FINALIZACION: float,
    TIEMPO_GRACIA: float,
    EARLY_BUFFER: int
    ):
    """
    Simulates how downstream labors are affected after inserting a new labor
    (either before the first labor or between two existing labors).
    """
    
    curr_labor_id = next_labor_id
    
    # --- Initialization ---
    downstream_shifts = []     # Store info of shifts per labor
    feasible = True            # Default to feasible until proven otherwise

    curr_labor_actual_start = moves_driver_df.loc[start_idx,'actual_start']

    # --- Early exit condition ---
    shift = (next_labor_arrival - curr_labor_actual_start).total_seconds() / 60
    if abs(shift) < 1:
        # No temporal shift → no downstream propagation
        return True, []

    # --- Prepare iteration variables ---
    curr_end_time, curr_end_pos, total_duration, dist = _compute_service_end_time(
        arrival_time=next_labor_arrival,
        start_pos=next_labor_start_pos,
        end_pos=moves_driver_df.loc[start_idx, 'end_point'],
        vehicle_speed=VEHICLE_TRANSPORT_SPEED,
        prep_time=TIEMPO_ALISTAR,
        finish_time=TIEMPO_FINALIZACION
    )

    labor_rows = moves_driver_df[
    moves_driver_df["labor_context_id"].astype(str).str.endswith("_labor")
    ].index.tolist()

    # Find current labor position among them
    try:
        start_pos = labor_rows.index(start_idx)
    except ValueError:
        # Fallback in case start_idx was a move/free index
        start_pos = 0

    # --- Iterate through all subsequent labors ---
    for i in labor_rows[start_pos + 1:]:
    # for i in range(start_idx + 3, len(moves_driver_df), 3):  # Move by triplets
        # iterate naturally over downstream labor indices
        next_start_time = moves_driver_df.loc[i, 'schedule_date']
        next_start_pos = moves_driver_df.loc[i, 'start_point']
        next_end_pos = moves_driver_df.loc[i, 'end_point']

        # --- Compute arrival to new service
        would_arrive_at, dist_to_next_service, travel_time_to_next = _compute_arrival_to_next_labor(
            current_end_time=curr_end_time,
            current_end_pos=curr_end_pos, 
            target_pos=next_start_pos,
            speed=ALFRED_SPEED
        )

        # --- Break if driver wouldn't arrive on time to new labor
        next_labor_max_arrival_time = next_start_time + timedelta(minutes=TIEMPO_GRACIA)
        if would_arrive_at > next_labor_max_arrival_time:
            return False, []

        # --- Adjust if arriving too early (driver waits)
        real_arrival_time, start_move_time = _adjust_for_early_arrival(
            would_arrive_at=would_arrive_at, 
            travel_time= travel_time_to_next,
            schedule_date=next_start_time,
            early_buffer=EARLY_BUFFER
        )
        
        # --- Compute new labor end time
        finish_next_labor_time, finish_next_labor_pos, _, dist_service = \
            _compute_service_end_time(
                arrival_time=real_arrival_time,
                start_pos=next_start_pos,
                end_pos=next_end_pos,
                vehicle_speed=VEHICLE_TRANSPORT_SPEED,
                prep_time=TIEMPO_ALISTAR,
                finish_time=TIEMPO_FINALIZACION
            )
        
        # --- Save all the changes ---
        free_time = {'start': curr_end_time, 'end': start_move_time}
        move_time = {'start': start_move_time, 'end': real_arrival_time}
        labor_time = {'start': real_arrival_time, 'end': finish_next_labor_time}
        
        # --- Update tracking variables ---
        current_labor_id = moves_driver_df.loc[i, 'labor_id']
        curr_end_time = finish_next_labor_time
        curr_end_pos = finish_next_labor_pos

        # --- Log downstream changes ---
        downstream_shifts.append({current_labor_id:[free_time, move_time, labor_time]})

        shift = (labor_time['end'] - moves_driver_df.loc[i, 'actual_end']).total_seconds() / 60
        if abs(shift) < 1:
            break

    # --- Return final result ---
    return feasible, downstream_shifts


def _get_driver_context(
    moves_driver_df: pd.DataFrame,
    idx: int,
) -> Tuple:
    """Return the current and next labor context for driver."""
    # The most recent labor is still in place
    curr_end_time = moves_driver_df.loc[idx, 'actual_end']
    curr_end_pos = moves_driver_df.loc[idx, 'end_point']
    next_start_time = moves_driver_df.loc[idx + 3, 'schedule_date']
    next_start_pos = moves_driver_df.loc[idx + 3, 'start_point']
    
    return curr_end_time, curr_end_pos, next_start_time, next_start_pos


def _compute_arrival_to_next_labor(
    current_end_time, 
    current_end_pos, 
    target_pos: str, 
    speed: float, 
) -> Tuple:
    """Compute when driver would arrive at the target position."""
    dist, _ = distance(current_end_pos, target_pos, method='haversine')
    travel_time = dist / speed * 60
    return current_end_time + timedelta(minutes=travel_time), dist, travel_time


def _adjust_for_early_arrival(
    would_arrive_at,
    travel_time,
    schedule_date, 
    early_buffer: float = 30):
    """
    Adjusts arrival time if driver would arrive too early.
    Ensures driver waits to arrive no earlier than (schedule_date - early_buffer).
    """
    earliest_allowed = schedule_date - timedelta(minutes=early_buffer)

    real_arrival_time = max(would_arrive_at, earliest_allowed)
    move_start_time =  real_arrival_time - timedelta(minutes=travel_time)

    return real_arrival_time, move_start_time


def _compute_service_end_time(
    arrival_time, 
    start_pos: str, 
    end_pos: str, 
    vehicle_speed: float, 
    prep_time: float, 
    finish_time,
) -> Tuple:
    """Compute finish time and position of performing the new service."""
    labor_distance, _ = distance(start_pos, end_pos, method='haversine')
    travel_time = labor_distance / vehicle_speed * 60
    labor_total_duration = prep_time + travel_time + finish_time
    finish_time = arrival_time + timedelta(minutes=labor_total_duration)
    return finish_time, end_pos, labor_total_duration, labor_distance


def _can_reach_next_labor(
    new_finish_time, 
    new_finish_pos: str, 
    next_start_time, 
    next_start_pos: str, 
    driver_speed: float, 
    grace_time: float,
) -> Tuple:
    """Check if driver can arrive to next labor in time after finishing new service."""
    dist, _ = distance(new_finish_pos, next_start_pos, method='haversine')
    travel_time = dist / driver_speed * 60
    would_arrive_next = new_finish_time + timedelta(minutes=travel_time)
    feasible = would_arrive_next <= next_start_time + timedelta(minutes=grace_time)
    return feasible, would_arrive_next, dist, travel_time


''' Filtering functions '''
def filter_dynamic_df(labors_dynamic_df, city, fecha):
    labors_dynamic_filtered_df = flexible_filter(
        labors_dynamic_df,
        city=city,
        schedule_date=fecha
        ).sort_values(['created_at', 'schedule_date', 'labor_start_date']).reset_index(drop=True)

    return labors_dynamic_filtered_df


def get_drivers(labors_algo_df, city, fecha):
    labors_algo_filtered_df = flexible_filter(
        labors_algo_df,
        city=city,
        schedule_date=fecha
    )

    drivers = (
        labors_algo_filtered_df['assigned_driver']
        .dropna()                                 # Remove NaN values
        .astype(str)                              # Ensure all are strings
    )

    # Remove empty strings and pure whitespace
    drivers = [d for d in drivers.unique() if d.strip() != '']

    return drivers


def filter_dfs_for_insertion(
    labors_algo_df: pd.DataFrame,
    moves_algo_df: pd.DataFrame,
    city: str,
    fecha,
    driver,
    created_at
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare filtered labors and moves for evaluating insertions for a given driver / day.

    Behaviour summary
    -----------------
    1. Filter labors and moves by city, date and driver.
    2. Keep only labors whose actual_end > created_at (future or ongoing labors).
    3. For moves, keep rows whose labor_id starts with any of the active labor ids
       (so free/move/labor triplets are preserved).
    4. If the first remaining moves row corresponds to a triplet whose previous labor
       was filtered out (i.e. the driver is in the _free/_move/_labor of an already-started
       block), and the created_at occurs *before* the end of that first row, then
       prepend the whole previous triplet (previous labor + its moves) so we retain
       the context where the driver currently is.

    Returns
    -------
    (labors_algo_filtered_df, moves_algo_filtered_df)
    """
    # --- Step 1: Prefilter by city, date and driver ---
    labors_pref = flexible_filter(
        labors_algo_df,
        city=city,
        schedule_date=fecha,
        assigned_driver=driver
    ).sort_values(['created_at', 'schedule_date', 'labor_start_date']).reset_index(drop=True)

    moves_pref = flexible_filter(
        moves_algo_df,
        city=city,
        schedule_date=fecha,
        assigned_driver=driver
    ).sort_values(['schedule_date', 'actual_start', 'actual_end'])

    # --- Step 2: Keep labors that end after created_at (ongoing + future) ---
    labors_filtered = (
        labors_pref[labors_pref["actual_end"] > created_at]
        .sort_values(["schedule_date", "actual_start", "actual_end"])
        .reset_index(drop=True)
    )

    # If there are no future/ongoing labors, just return empty labors and empty moves
    if labors_filtered.empty:
        return labors_filtered, pd.DataFrame(columns=moves_pref.columns)

    # --- Step 3: Build set of active (base) labor_ids to keep triplets in moves ---
    # Ensure labour ids are strings
    active_base_ids = {str(lid) for lid in labors_filtered["labor_id"].astype(str).unique()}

    # Keep moves whose labor_id starts with any of the base ids
    # Use vectorized string operations for performance
    import re 
    
    pattern = "|".join([f"^{re.escape(bid)}" for bid in active_base_ids])  # anchors to start
    # NOTE: pandas' `.str` methods expect strings and can handle NaN safely
    moves_filtered = moves_pref[moves_pref["labor_id"].astype(str).str.match(pattern, na=False)]

    # Keep original ordering but don't reset index yet — we need original positions
    moves_filtered = moves_filtered.sort_values(["schedule_date", "actual_start", "actual_end"])

    if moves_filtered.empty:
        # No moves matching active labors (unlikely) — return labors and empty moves
        return labors_filtered, pd.DataFrame(columns=moves_pref.columns)

    # --- Step 4: If the first remaining move row is not the very first in the prefiltered moves,
    # and created_at occurs before the 'end' of that first remaining move, then the previous triplet
    # is relevant and must be prepended.
    first_index_label = moves_filtered.index[0]

    # find positional location of that label within the original prefiltered moves (moves_pref)
    try:
        pos_in_pref = moves_pref.index.get_loc(first_index_label)
    except KeyError:
        # If for some reason the index label is not found in moves_pref (shouldn't happen),
        # fallback to simple reset + check:
        pos_in_pref = 0

    # Only if there *is* a previous row in the prefiltered moves we can consider adding it
    if pos_in_pref > 0:
        # We compare created_at with the end of the first kept move row
        first_move_end = moves_filtered.iloc[0]["actual_end"]
        # If created_at is before that end => the new order was created while the driver was still
        # in that triplet (free / move / labor). We should include the previous triplet to recover context.
        if pd.notna(first_move_end) and created_at < first_move_end:
            # previous row label and its labor_id
            prev_index_label = moves_pref.index[pos_in_pref - 1]
            prev_labor_id_raw = str(moves_pref.loc[prev_index_label, "labor_id"])

            # Extract base id (strip suffixes like _free/_move). Keep everything up to first underscore
            base_prev_id = prev_labor_id_raw.split("_")[0]

            # Attempt to find the previous labor row in labors_pref using the base id
            prev_labors = labors_pref[labors_pref["labor_id"].astype(str) == base_prev_id]
            if not prev_labors.empty:
                # Prepend the previous labor to labors_filtered if not already present
                if base_prev_id not in set(labors_filtered["labor_id"].astype(str)):
                    labors_filtered = pd.concat([prev_labors, labors_filtered], ignore_index=True)
                    labors_filtered = labors_filtered.sort_values(
                        ["schedule_date", "actual_start", "actual_end"]
                    ).reset_index(drop=True)

                # Grab all moves belonging to that previous labor triplet (prefix match)
                prev_triplet_mask = moves_pref["labor_id"].astype(str).str.startswith(base_prev_id)
                prev_triplet = moves_pref[prev_triplet_mask].sort_values(
                    ["schedule_date", "actual_start", "actual_end"]
                )

                # Prepend them to the filtered moves (avoid duplicate rows)
                # ensure we don't duplicate if prev_triplet rows are already present
                combined = pd.concat([prev_triplet, moves_filtered], ignore_index=True)
                combined = combined.drop_duplicates(subset=[*moves_pref.columns], keep="first")
                moves_filtered = combined.sort_values(
                    ["schedule_date", "actual_start", "actual_end"]
                ).reset_index(drop=True)
            else:
                # previous labor not found in labors_pref — skip adding
                moves_filtered = moves_filtered.reset_index(drop=True)
        else:
            moves_filtered = moves_filtered.reset_index(drop=True)
    else:
        # first kept row is the first in the prefitered set — nothing to prepend
        moves_filtered = moves_filtered.reset_index(drop=True)

    return labors_filtered, moves_filtered


# def get_best_insertion(candidate_insertions, selection_mode="min_total_distance", random_state=None):
#     """
#     Selects the best driver among feasible insertion options.

#     Parameters
#     ----------
#     candidate_insertions : list of tuples or dicts
#         Expected tuple structure:
#         (driver, prev_labor_id, next_labor_id,
#          dist_to_new_labor, dist_to_next_labor,
#          feasible, downstream_shifts, would_arrive_next)
#     selection_mode : str, optional
#         Criterion for selecting among feasible insertions:
#         - "random": choose a random driver
#         - "min_total_distance": minimize (dist_to_new_labor + dist_to_next_labor)
#         - "min_dist_to_new_labor": minimize distance to the new labor
#     random_state : int, optional
#         For reproducibility in random selection.

#     Returns
#     -------
#     selected_driver : str or None
#         ID of the chosen driver.
#     insertion_point : tuple or None
#         (prev_labor_id, next_labor_id) pair indicating insertion location.
#     selection_df : pd.DataFrame
#         DataFrame summarizing all candidates and metrics.
#     """

#     if not candidate_insertions:
#         return None, None, pd.DataFrame()

#     # Normalize input to DataFrame
#     columns = [
#         "driver",
#         "prev_labor_id",
#         "next_labor_id",
#         "dist_to_new_labor",
#         "dist_to_next_labor",
#         "feasible",
#         "downstream_shifts",
#         "would_arrive_next"
#     ]

#     selection_df = pd.DataFrame(candidate_insertions, columns=columns[:len(candidate_insertions[0])])

#     # Keep only feasible candidates (if 'feasible' column exists)
#     if "feasible" in selection_df.columns:
#         selection_df = selection_df[selection_df["feasible"] == True]

#     if selection_df.empty:
#         return None, None, pd.DataFrame(columns=columns)

#     # Handle NaNs in distance columns safely
#     for col in ["dist_to_new_labor", "dist_to_next_labor"]:
#         if col in selection_df.columns:
#             selection_df[col] = selection_df[col].fillna(0).astype(float)
#         else:
#             selection_df[col] = 0.0

#     # Compute total distance
#     selection_df["total_distance"] = (
#         selection_df["dist_to_new_labor"] + selection_df["dist_to_next_labor"]
#     )

#     # --- Selection logic ---
#     if selection_mode == "random":
#         if random_state is not None:
#             random.seed(random_state)
#         chosen_row = selection_df.sample(1, random_state=random_state).iloc[0]

#     elif selection_mode == "min_dist_to_new_labor":
#         chosen_row = selection_df.loc[selection_df["dist_to_new_labor"].idxmin()]

#     elif selection_mode == "min_total_distance":
#         chosen_row = selection_df.loc[selection_df["total_distance"].idxmin()]

#     else:
#         raise ValueError(f"Unknown selection_mode '{selection_mode}'")

#     selected_driver = chosen_row["driver"]
#     insertion_point = (
#         chosen_row.get("prev_labor_id"),
#         chosen_row.get("next_labor_id")
#     )

#     return selected_driver, insertion_point, selection_df


def get_best_insertion(
    candidate_insertions: List[Dict[str, Any]],
    selection_mode: str = "min_total_distance",
    random_state: Optional[int] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Selects the best insertion plan among feasible options.

    Parameters
    ----------
    candidate_insertions : list of dict
        Each element is an insertion_plan produced by evaluate_driver_feasibility().
        Must include at least:
          - 'driver_id'
          - 'dist_to_new_service'
          - 'dist_to_next_labor'
          - 'feasible' (optional but recommended)
          - (others like downstream_shifts, new_moves, etc.)
    selection_mode : str, optional
        Criterion for selection:
          - "min_total_distance" (default): minimize (dist_to_new_service + dist_to_next_labor)
          - "min_dist_to_new_labor": minimize dist_to_new_service
          - "random": choose randomly among feasible insertions
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    best_plan : dict or None
        The chosen insertion_plan ready for commit_new_labor_insertion.
    selection_df : pd.DataFrame or None
        DataFrame summary of all feasible candidates (for diagnostics or logging).
    """

    if not candidate_insertions:
        return None, pd.DataFrame()

    # --- Convert to DataFrame for easy filtering / comparison ---
    selection_records = []
    for driver, plan in candidate_insertions:
        selection_records.append({
            "driver_id": plan.get("alfred"),
            "prev_labor_id": plan.get("prev_labor_id"),
            "next_labor_id": plan.get("next_labor_id"),
            "dist_to_new_service": plan.get("dist_to_new_service", np.nan),
            "dist_to_next_labor": plan.get("dist_to_next_labor", np.nan),
            "feasible": plan.get("feasible", True)
        })

    selection_df = pd.DataFrame(selection_records)

    # --- Keep only feasible ones ---
    if "feasible" in selection_df.columns:
        selection_df = selection_df[selection_df["feasible"] == True]
    if selection_df.empty:
        return None, selection_df

    # # --- Compute total distance ---
    # selection_df["total_distance"] = (
    #     selection_df["dist_to_new_service"].fillna(0).astype(float)
    #     + selection_df["dist_to_next_labor"].fillna(0).astype(float)
    # )
    selection_df["dist_to_new_service"] = pd.to_numeric(selection_df["dist_to_new_service"], errors="coerce").fillna(0)
    selection_df["dist_to_next_labor"] = pd.to_numeric(selection_df["dist_to_next_labor"], errors="coerce").fillna(0)
    selection_df["total_distance"] = selection_df["dist_to_new_service"] + selection_df["dist_to_next_labor"]

    # --- Select best plan according to selection_mode ---
    if selection_mode == "random":
        chosen_idx = (
            selection_df.sample(1, random_state=random_state).index[0]
            if not selection_df.empty else None
        )

    elif selection_mode == "min_dist_to_new_labor":
        chosen_idx = selection_df["dist_to_new_service"].idxmin()

    elif selection_mode == "min_total_distance":
        chosen_idx = selection_df["total_distance"].idxmin()

    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    if chosen_idx is None:
        return None, selection_df

    # --- Retrieve the corresponding plan directly ---
    driver, best_plan = candidate_insertions[chosen_idx]

    return driver, best_plan, selection_df


''' Updating dataframe functions'''
def commit_new_labor_insertion():
    pass




from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def _ensure_datetime_col(df: pd.DataFrame, col: str) -> None:
    """Ensure column is datetime dtype (inplace)."""
    if col not in df.columns:
        return
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")



def commit_new_labor_insertion(
    labors_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    driver: str,
    insertion_plan: Dict[str, Any],
    new_labor: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Commit a new labor insertion into the global labors and moves DataFrames.

    This function integrates the new labor and its corresponding triplet of
    move records (FREE_TIME, DRIVER_MOVE, LABOR) into the driver’s schedule,
    along with any downstream updates from the insertion plan.

    All timing and distance values are assumed to be precomputed and
    validated by `evaluate_driver_feasibility`.

    Parameters
    ----------
    labors_df : pd.DataFrame
        Global labors dataframe containing all scheduled labors.
    moves_df : pd.DataFrame
        Global moves dataframe containing all driver movement rows.
    insertion_plan : dict
        Output from `evaluate_driver_feasibility`, containing:
          - 'driver_id'
          - 'new_labor_id'
          - 'new_labor_timing' (dict with arrival/finish)
          - 'new_moves' (DataFrame from `_build_new_moves_block`)
          - 'updated_labors' (list of dicts with timing adjustments)
          - 'downstream_shifts' (list of dicts for triplet re-timing)
          - 'first_assignment' (bool)
    new_labor : pd.Series
        Row from `labors_df` corresponding to the labor being inserted.

    Returns
    -------
    (labors_df_new, moves_df_new)
        Updated and re-sorted copies of the input DataFrames.
    """

    # ============================================================
    # 1️⃣ BASIC EXTRACTION
    # ============================================================
    # driver = insertion_plan["driver_id"]
    new_labor_id = new_labor['labor_id']
    new_moves_df = insertion_plan.get("new_moves", pd.DataFrame())
    updated_labors = insertion_plan.get("updated_labors", [])
    downstream_shifts = insertion_plan.get("downstream_shifts", [])
    timing = insertion_plan.get("new_labor_timing", {})

    arrival_time = timing.get("arrival_time")
    finish_time = timing.get("finish_time")

    # ============================================================
    # 2️⃣ ADD NEW LABOR TO labors_df
    # ============================================================
    labors_new = labors_df.copy()

    # clone and fill the new labor row
    new_lab_row = new_labor.copy()
    new_lab_row["assigned_driver"] = driver
    new_lab_row["actual_start"] = arrival_time
    new_lab_row["actual_end"] = finish_time

    # align columns and append
    for col in labors_new.columns:
        if col not in new_lab_row.index:
            new_lab_row[col] = np.nan
    new_lab_row = new_lab_row[labors_new.columns]
    labors_new = pd.concat([labors_new, pd.DataFrame([new_lab_row])], ignore_index=True)

    # ============================================================
    # 3️⃣ APPEND NEW MOVES BLOCK
    # ============================================================
    moves_new = moves_df.copy()

    if not new_moves_df.empty:
        # Ensure schema alignment
        for c in moves_new.columns:
            if c not in new_moves_df.columns:
                new_moves_df[c] = np.nan

        new_moves_df = new_moves_df[moves_new.columns]
        moves_new = pd.concat([moves_new, new_moves_df], ignore_index=True)

    # ============================================================
    # 4️⃣ APPLY UPDATED LABOR TIMINGS (Immediate neighbor)
    # ============================================================
    for upd in updated_labors:
        lid = str(upd["labor_id"])
        new_start = upd.get("new_start_time")
        new_free_end = upd.get("new_free_time")

        mask_prefix = moves_new["labor_context_id"].astype(str).str.startswith(lid)
        if not mask_prefix.any():
            continue

        free_mask = mask_prefix & moves_new["labor_context_id"].str.endswith("_free")
        labor_mask = mask_prefix & moves_new["labor_context_id"].str.endswith("_labor")

        if free_mask.any():
            moves_new.loc[free_mask, "actual_end"] = new_free_end
            moves_new.loc[free_mask, "duration_min"] = (
                (moves_new.loc[free_mask, "actual_end"] - moves_new.loc[free_mask, "actual_start"])
                .dt.total_seconds() / 60
            )

        if labor_mask.any():
            moves_new.loc[labor_mask, "actual_start"] = new_start
            moves_new.loc[labor_mask, "duration_min"] = (
                (moves_new.loc[labor_mask, "actual_end"] - moves_new.loc[labor_mask, "actual_start"])
                .dt.total_seconds() / 60
            )

    # ============================================================
    # 5️⃣ APPLY DOWNSTREAM SHIFTS (Full triplet timing rewrites)
    # ============================================================
    for shift_entry in downstream_shifts:
        for labor_key, triplet in shift_entry.items():
            if not isinstance(triplet, list) or len(triplet) != 3:
                continue

            free_times, move_times, labor_times = triplet
            mask = moves_new["labor_context_id"].astype(str).str.startswith(str(labor_key))

            if not mask.any():
                continue

            # Apply each timing
            free_mask = mask & moves_new["labor_context_id"].str.endswith("_free")
            move_mask = mask & moves_new["labor_context_id"].str.endswith("_move")
            labor_mask = mask & moves_new["labor_context_id"].str.endswith("_labor")

            # FREE
            if free_mask.any():
                moves_new.loc[free_mask, ["actual_start", "actual_end"]] = [
                    free_times["start"], free_times["end"]
                ]
                moves_new.loc[free_mask, "duration_min"] = (
                    (moves_new.loc[free_mask, "actual_end"] - moves_new.loc[free_mask, "actual_start"])
                    .dt.total_seconds() / 60
                )

            # MOVE
            if move_mask.any():
                moves_new.loc[move_mask, ["actual_start", "actual_end"]] = [
                    move_times["start"], move_times["end"]
                ]
                moves_new.loc[move_mask, "duration_min"] = (
                    (moves_new.loc[move_mask, "actual_end"] - moves_new.loc[move_mask, "actual_start"])
                    .dt.total_seconds() / 60
                )

            # LABOR
            if labor_mask.any():
                moves_new.loc[labor_mask, ["actual_start", "actual_end"]] = [
                    labor_times["start"], labor_times["end"]
                ]
                moves_new.loc[labor_mask, "duration_min"] = (
                    (moves_new.loc[labor_mask, "actual_end"] - moves_new.loc[labor_mask, "actual_start"])
                    .dt.total_seconds() / 60
                )

    # ============================================================
    # 6️⃣ FINAL SORTING & CLEANUP
    # ============================================================
    # Ensure datetime dtype consistency
    for col in ["actual_start", "actual_end"]:
        if col in moves_new.columns:
            moves_new[col] = pd.to_datetime(moves_new[col], errors="coerce")
    for col in ["actual_start", "actual_end"]:
        if col in labors_new.columns:
            labors_new[col] = pd.to_datetime(labors_new[col], errors="coerce")

    # Sort moves by driver + time
    moves_new = (
        moves_new.sort_values(["assigned_driver", "schedule_date", "actual_start"], na_position="last")
        .reset_index(drop=True)
    )

    # Sort labors by time
    labors_new = (
        labors_new.sort_values(["assigned_driver", "schedule_date", "actual_start"], na_position="last")
        .reset_index(drop=True)
    )

    return labors_new, moves_new


