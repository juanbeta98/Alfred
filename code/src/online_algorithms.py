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
            ALFRED_SPEED=ALFRED_SPEED,
            VEHICLE_TRANSPORT_SPEED=VEHICLE_TRANSPORT_SPEED,
            TIEMPO_ALISTAR=TIEMPO_ALISTAR,
            TIEMPO_FINALIZACION=TIEMPO_FINALIZACION,
            EARLY_BUFFER=EARLY_BUFFER
        )
        return feasible, infeasible_log, insertion_plan


    # ---------------------------------------------------------------------------------------
    # Case 2: insertion before first labor
    # ---------------------------------------------------------------------------------------
    is_before_first_labor = _is_creation_before_first_labor(moves_driver_df)
    if is_before_first_labor:
        # --- Skip if next labor starts before new labor’s schedule
        if new_labor['schedule_date'] < moves_driver_df.loc[0,'actual_start']:
            feasible, infeasible_log, insertion_plan = _evaluate_and_execute_insertion_before_first_labor(
                new_labor,
                moves_driver_df,
                driver,
                VEHICLE_TRANSPORT_SPEED,
                ALFRED_SPEED,
                TIEMPO_ALISTAR,
                TIEMPO_FINALIZACION,
                TIEMPO_GRACIA,
                EARLY_BUFFER
            )

            return feasible, infeasible_log, insertion_plan
        else:
            labor_iter = 0
    else:
        # Find the first 'labor' type row in the dataframe dynamically
        labor_iter = moves_driver_df.index[
            moves_driver_df["labor_id"].astype(str).str.match(r"^\d+$")
        ][0]

              
    # ---------------------------------------------------------------------------------------
    # Case 3: Insertion between existing labors
    # ---------------------------------------------------------------------------------------
    n_rows = len(moves_driver_df)

    while labor_iter + 3 <= n_rows:
        # --- Context: previous labor & next labor
        curr_end_time, curr_end_pos, next_start_time, next_start_pos = \
            _get_driver_context(moves_driver_df, labor_iter)

        curr_labor_id = moves_driver_df.loc[labor_iter, "labor_id"]
        next_labor_id_candidate = moves_driver_df.loc[labor_iter + 3, "labor_id"]

        # --- Skip if next labor starts before new labor’s schedule
        if next_start_time <= new_labor['schedule_date']:
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
        real_arrival_time = _adjust_for_early_arrival(
            would_arrive_at, 
            new_labor['schedule_date'], 
            EARLY_BUFFER
        )

        # --- Compute new labor end time
        finish_new_labor_time, finish_new_labor_pos, _, dist_service = \
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
            new_labor,
            driver=driver,
            start_free_time=curr_end_time,
            real_arrival_time=real_arrival_time,
            finish_new_time=finish_new_labor_time,
            start_pos=curr_end_pos,
            start_address=new_labor["start_address_point"],
            end_address=new_labor["end_address_point"],
            dist_to_new=dist_to_new_service,
            dist_service=dist_service,
            alfred_speed=ALFRED_SPEED
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
                "travel_distance": dist_service,
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
        prev_labor_id = moves_driver_df.loc[n_rows - 1, "labor_id"]

        # Compute basic timing info
        curr_end_time = moves_driver_df.loc[n_rows - 1, "actual_end"]
        curr_end_pos = moves_driver_df.loc[n_rows - 1, "end_point"]
        would_arrive_at, dist_to_new_service, _ = _compute_arrival_to_next_labor(
            curr_end_time,
            curr_end_pos,
            new_labor["start_address_point"],
            ALFRED_SPEED,
        )

        # Check if driver would arrive on time to the new labor
        if would_arrive_at > new_labor["latest_arrival_time"]:
            infeasible_log = "Driver would arrive too late to the new labor."
            return feasible, infeasible_log, insertion_plan


        real_arrival_time = _adjust_for_early_arrival(
            would_arrive_at, new_labor["schedule_date"], EARLY_BUFFER
        )

        finish_new_labor_time, finish_new_labor_pos, total_duration, dist_service = \
            _compute_service_end_time(
                real_arrival_time,
                new_labor["start_address_point"],
                new_labor["end_address_point"],
                VEHICLE_TRANSPORT_SPEED,
                TIEMPO_ALISTAR,
                TIEMPO_FINALIZACION,
            )

        new_moves = _build_new_moves_block(
            new_labor,
            start_free_time=curr_end_time,
            real_arrival_time=real_arrival_time,
            finish_new_time=finish_new_labor_time,
            start_pos=curr_end_pos,
            start_address=new_labor["start_address_point"],
            end_address=new_labor["end_address_point"],
            dist_to_new=dist_to_new_service,
            dist_service=dist_service
        )

        insertion_plan = {
            "driver_id": new_labor["driver_id"],
            "new_labor_id": new_labor["labor_id"],
            "prev_labor_id": prev_labor_id,
            "next_labor_id": None,
            "dist_to_new_service": dist_to_new_service,
            "dist_to_next_labor": None,
            "new_moves":  new_moves,
            "new_labor_timing": {
                "arrival_time": real_arrival_time,
                "finish_time": finish_new_labor_time,
                "total_duration": total_duration,
                "travel_distance": dist_service,
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
    ALFRED_SPEED:float,
    VEHICLE_TRANSPORT_SPEED: float,
    TIEMPO_ALISTAR: float,
    TIEMPO_FINALIZACION: float,
    EARLY_BUFFER: float,

) -> Tuple:    
    # 1. Compute start and end times for the new labor (driver starts “from base” or assumed start)
    start_time = new_labor["schedule_date"]
    arrival_time =start_time - timedelta(minutes=EARLY_BUFFER)  # no travel, assume immediate availability
    
    finish_new_time, finish_new_pos, total_duration, dist_service = _compute_service_end_time(
        arrival_time,
        new_labor["start_address_point"],
        new_labor["end_address_point"],
        VEHICLE_TRANSPORT_SPEED,
        TIEMPO_ALISTAR,
        TIEMPO_FINALIZACION,
    )

    # 2. Build the 3-block structure: free (optional), move (optional), labor
    new_moves = _build_new_moves_block(
        new_labor,
        driver,
        start_free_time=None,  # no previous labor
        real_arrival_time=arrival_time,
        finish_new_time=finish_new_time,
        start_pos=None,
        start_address=new_labor["start_address_point"],
        end_address=new_labor["end_address_point"],
        dist_to_new=0,
        dist_service=dist_service,
        alfred_speed=ALFRED_SPEED
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
            "finish_time": finish_new_time,
            "total_duration": total_duration,
            "travel_distance": dist_service
        },
        "updated_labors": [],
        "affected_labors": [],
        "downstream_shifts": [],
        "first_assignment": False
    }

    return True, "", insertion_plan


def _is_creation_before_first_labor(
    moves_driver_df: pd.DataFrame
) -> bool:
    return not str(moves_driver_df.iloc[0]["labor_id"]).endswith("_free")


def _evaluate_and_execute_insertion_before_first_labor(
    new_labor, 
    moves_driver_df,
    driver,
    VEHICLE_TRANSPORT_SPEED: float,
    ALFRED_SPEED: float,
    TIEMPO_ALISTAR: float,
    TIEMPO_FINALIZACION: float,
    TIEMPO_GRACIA: float,
    EARLY_BUFFER: float,
) -> Tuple:
    
    real_arrival_time = new_labor["schedule_date"] - timedelta(minutes=EARLY_BUFFER)

    next_labor_id_candidate = moves_driver_df.loc[0, 'labor_id']

    # Compute new labor end time
    finish_new_labor_time, finish_new_labor_pos, _, dist_service = \
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
            next_start_time=moves_driver_df.loc[0, 'schedule_date'],
            next_start_pos=moves_driver_df.loc[0, 'start_point'],
            driver_speed=ALFRED_SPEED, 
            grace_time=TIEMPO_GRACIA
        )

    if not feasible_next:
        infeasible_log = (
            "Driver would not make it to the next scheduled labor in time "
            "if new labor is inserted."
        )
        return False, infeasible_log, None

    # Now simulate downstream shift from the FIRST labor
    downstream_ok, downstream_shifts = _simulate_downstream_shift(
        driver_df=moves_driver_df,
        from_labor_id=moves_driver_df.iloc[0]["labor_id"],
        start_shift_time=finish_new_labor_time,
        start_shift_pos=finish_new_labor_pos,
        ALFRED_SPEED=ALFRED_SPEED,
        TIEMPO_GRACIA=TIEMPO_GRACIA,
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
        "new_moves": None,
        "new_labor_timing": {
            "arrival_time": real_arrival_time,
            "finish_time": finish_new_labor_time,
            "travel_distance": dist_service,
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
    driver: float,
    start_free_time: Optional[datetime],
    real_arrival_time: datetime,
    finish_new_time: datetime,
    start_pos: Optional[Tuple[float, float]],
    start_address: Tuple[float, float],
    end_address: Tuple[float, float],
    dist_to_new: float,
    dist_service: float,
    alfred_speed: float,
) -> pd.DataFrame:
    """
    Construct the three move rows (free, move, labor) associated with a new labor.

    Parameters
    ----------
    new_labor : pd.Series
        Row from labors_dynamic_df with info about the new labor.
    start_free_time : datetime or None
        When the previous labor actually ended. None if this is the first labor.
    real_arrival_time : datetime
        When the driver actually arrives at the new labor start.
    finish_new_time : datetime
        When the driver finishes performing the new labor.
    start_pos : (float, float) or None
        Coordinates of previous labor's end. None if first assignment.
    start_address : (float, float)
        Start point of the new service.
    end_address : (float, float)
        End point of the new service.
    dist_to_new : float
        Distance from previous labor’s end to new labor’s start (km).
    dist_service : float
        Distance from new labor’s start to its end (km).

    Returns
    -------
    pd.DataFrame
        With three rows: `_free`, `_move`, and actual labor.
    """

    rows = []

    if start_free_time is not None:
        # Driver is coming from a previous labor
        # ============================================================
        # 1️⃣ FREE ROW
        # ============================================================
        free_row = {
            "labor_id": f"{new_labor['labor_id']}_free",
            "driver_id": driver,
            "type": "free",
            "schedule_date": new_labor['schedule_date'],
            "actual_start": start_free_time,
            "actual_end": real_arrival_time - timedelta(minutes=(dist_to_new / alfred_speed) * 60),
            "start_point": start_pos,
            "end_point": start_pos,
            "distance_km": 0,
            "duration_min": (real_arrival_time - start_free_time).total_seconds() / 60.0,
            "city": new_labor["city"],
        }
        rows.append(free_row)

        # ============================================================
        # 2️⃣ MOVE ROW
        # ============================================================
        move_start_time = real_arrival_time - timedelta(minutes=(dist_to_new / alfred_speed) * 60)
        move_end_time = real_arrival_time
        move_duration = (move_end_time - move_start_time).total_seconds() / 60.0

        move_row = {
            "labor_id": f"{new_labor['labor_id']}_move",
            "driver_id": driver,
            "type": "move",
            "schedule_date": new_labor['schedule_date'],
            "actual_start": move_start_time,
            "actual_end": move_end_time,
            "start_point": start_pos if start_pos is not None else start_address,
            "end_point": start_address,
            "distance_km": dist_to_new,
            "duration_min": move_duration,
            "city": new_labor["city"],
        }
        rows.append(move_row)

    # ============================================================
    # 3️⃣ LABOR ROW
    # ============================================================
    labor_row = {
        "labor_id": new_labor["labor_id"],
        "driver_id": driver,
        "type": "labor",
        "schedule_date": new_labor["schedule_date"],
        "actual_start": real_arrival_time,
        "actual_end": finish_new_time,
        "start_point": start_address,
        "end_point": end_address,
        "distance_km": dist_service,
        "duration_min": (finish_new_time - real_arrival_time).total_seconds() / 60.0,
        "city": new_labor["city"],
    }
    rows.append(labor_row)

    # ============================================================
    # 4️⃣ RETURN AS DATAFRAME
    # ============================================================
    new_moves_df = pd.DataFrame(rows)

    # ensure correct column order for merging later
    cols = [
        "labor_id", "driver_id", "type", "schedule_date",
        "actual_start", "actual_end", "start_point", "end_point",
        "distance_km", "duration_min", "city",
    ]
    return new_moves_df[cols]


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
    
    current_labor_id = next_labor_id
    
    # --- Initialization ---
    downstream_shifts = []     # Store info of shifts per labor
    feasible = True            # Default to feasible until proven otherwise

    next_labor_actual_start = moves_driver_df.loc[start_idx,'actual_start']

    # --- Early exit condition ---
    shift = (next_labor_arrival - next_labor_actual_start).total_seconds() / 60
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
    ~moves_driver_df["labor_id"].astype(str).str.contains("_")
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
        real_arrival_time = _adjust_for_early_arrival(
            would_arrive_at, 
            next_start_time,
            EARLY_BUFFER
        )

        start_moving_time = real_arrival_time - timedelta(minutes=travel_time_to_next)
        
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
        free_time = {'start': curr_end_time, 'end': start_moving_time}
        move_time = {'start': start_moving_time, 'end': real_arrival_time}
        labor_time = {'start': real_arrival_time, 'end': finish_next_labor_time}

        curr_labor_id = moves_driver_df.loc[i+3, 'labor_id']
        downstream_shifts.append({curr_labor_id:[free_time, move_time, labor_time]})

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
    scheduled_date, 
    early_buffer: float = 30):
    """
    Adjusts arrival time if driver would arrive too early.
    Ensures driver waits to arrive no earlier than (schedule_date - early_buffer).
    """
    earliest_allowed = scheduled_date - timedelta(minutes=early_buffer)
    return max(would_arrive_at, earliest_allowed)


def _compute_service_end_time(
    arrival_time, 
    start_pos: str, 
    end_pos: str, 
    vehicle_speed: float, 
    prep_time: float, 
    finish_time,
) -> Tuple:
    """Compute finish time and position of performing the new service."""
    dist, _ = distance(start_pos, end_pos, method='haversine')
    travel_time = dist / vehicle_speed * 60
    total_duration = prep_time + travel_time + finish_time
    finish_time = arrival_time + timedelta(minutes=total_duration)
    return finish_time, end_pos, total_duration, dist


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


# ''' Insertion functions'''
# def get_best_insertion(candidate_insertions, selection_mode="min_total_distance", random_state=None):
#     """
#     Selects the best driver among feasible insertions based on a chosen criterion.

#     Parameters
#     ----------
#     candidate_insertions : list of tuples
#         Each tuple should be:
#         (driver, insertion_point, dist_to_new_labor, dist_to_next_labor)
#     selection_mode : str, optional
#         Selection criterion:
#         - "random": choose a random driver
#         - "min_total_distance": minimize (dist_to_new_labor + dist_to_next_labor)
#         - "min_dist_to_new_labor": minimize distance to the new labor
#     random_state : int, optional
#         For reproducibility when using random selection.

#     Returns
#     -------
#     selected_driver : str
#         The chosen driver ID.
#     insertion_point : int
#         Where to insert the new labor in the driver's schedule.
#     selection_df : pd.DataFrame
#         Table summarizing all candidate metrics (for analysis/debugging).
#     """

#     if len(candidate_insertions) == 0:
#         return None, None, pd.DataFrame()

#     # Convert to DataFrame for easier computation
#     selection_df = pd.DataFrame(candidate_insertions, columns=[
#         "driver", "prev_labor_id", 'next_labor_id', "dist_to_new_labor", "dist_to_next_labor"
#     ])

#     # Replace None/NaN manually using np.where (bypasses .fillna())
#     selection_df["dist_to_new_labor"] = np.where(
#         selection_df["dist_to_new_labor"].isna(),
#         0,
#         selection_df["dist_to_new_labor"]
#     ).astype(float)

#     selection_df["dist_to_next_labor"] = np.where(
#         selection_df["dist_to_next_labor"].isna(),
#         0,
#         selection_df["dist_to_next_labor"]
#     ).astype(float)

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
#     insertion_point = (chosen_row["prev_labor_id"], chosen_row['next_labor_id'])

#     return selected_driver, insertion_point, selection_df

def get_best_insertion(candidate_insertions, selection_mode="min_total_distance", random_state=None):
    """
    Selects the best driver among feasible insertion options.

    Parameters
    ----------
    candidate_insertions : list of tuples or dicts
        Expected tuple structure:
        (driver, prev_labor_id, next_labor_id,
         dist_to_new_labor, dist_to_next_labor,
         feasible, downstream_shifts, would_arrive_next)
    selection_mode : str, optional
        Criterion for selecting among feasible insertions:
        - "random": choose a random driver
        - "min_total_distance": minimize (dist_to_new_labor + dist_to_next_labor)
        - "min_dist_to_new_labor": minimize distance to the new labor
    random_state : int, optional
        For reproducibility in random selection.

    Returns
    -------
    selected_driver : str or None
        ID of the chosen driver.
    insertion_point : tuple or None
        (prev_labor_id, next_labor_id) pair indicating insertion location.
    selection_df : pd.DataFrame
        DataFrame summarizing all candidates and metrics.
    """

    if not candidate_insertions:
        return None, None, pd.DataFrame()

    # Normalize input to DataFrame
    columns = [
        "driver",
        "prev_labor_id",
        "next_labor_id",
        "dist_to_new_labor",
        "dist_to_next_labor",
        "feasible",
        "downstream_shifts",
        "would_arrive_next"
    ]

    selection_df = pd.DataFrame(candidate_insertions, columns=columns[:len(candidate_insertions[0])])

    # Keep only feasible candidates (if 'feasible' column exists)
    if "feasible" in selection_df.columns:
        selection_df = selection_df[selection_df["feasible"] == True]

    if selection_df.empty:
        return None, None, pd.DataFrame(columns=columns)

    # Handle NaNs in distance columns safely
    for col in ["dist_to_new_labor", "dist_to_next_labor"]:
        if col in selection_df.columns:
            selection_df[col] = selection_df[col].fillna(0).astype(float)
        else:
            selection_df[col] = 0.0

    # Compute total distance
    selection_df["total_distance"] = (
        selection_df["dist_to_new_labor"] + selection_df["dist_to_next_labor"]
    )

    # --- Selection logic ---
    if selection_mode == "random":
        if random_state is not None:
            random.seed(random_state)
        chosen_row = selection_df.sample(1, random_state=random_state).iloc[0]

    elif selection_mode == "min_dist_to_new_labor":
        chosen_row = selection_df.loc[selection_df["dist_to_new_labor"].idxmin()]

    elif selection_mode == "min_total_distance":
        chosen_row = selection_df.loc[selection_df["total_distance"].idxmin()]

    else:
        raise ValueError(f"Unknown selection_mode '{selection_mode}'")

    selected_driver = chosen_row["driver"]
    insertion_point = (
        chosen_row.get("prev_labor_id"),
        chosen_row.get("next_labor_id")
    )

    return selected_driver, insertion_point, selection_df
