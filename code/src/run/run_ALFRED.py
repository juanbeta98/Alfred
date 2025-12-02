import os
import pickle
import pandas as pd
from datetime import timedelta

import json

from time import perf_counter
from tqdm import tqdm
from itertools import product

from src.algorithms.ALFRED_algorithm import (
    generate_alfred_parameters, 
    load_alfred_parameters,
    generate_time_intervals,
    alfred_algorithm_assignment
)

from src.utils.utils import prep_online_algorithm_inputs, compute_workday_end
from src.config.experimentation_config import *
from src.config.config import *

from src.utils.filtering import flexible_filter



def run_ALFRED(
    instance: str,
    optimization_obj: str,
    distance_method: str,
    time_previous_freeze: int,
    batch_interval_minutes: int = 30,
    start_of_day_str: str = "07:00",
    save_results: bool = True,
    multiprocessing: bool = True,
    n_processes: int = None,
    experiment_type: str = 'online_operation'
):

    (
        data_path,
        assignment_type,
        driver_init_mode,
        duraciones_df,
        valid_cities, 
        labors_real_df,
        directorio_hist_df,
        global_dist_dict,
        fechas,
        alpha,
        labors_dynamic_df,
        labors_algo_dynamic_df,
        moves_algo_dynamic_df,
        postponed_labors
    ) = prep_online_algorithm_inputs(
        instance, 
        distance_method, 
        optimization_obj)
    
    decision_times = generate_time_intervals(
        strat_time = '07:00',
        end_time = '20:00',
        increment = 30)

    for fecha in fechas:
        print(f"{'-'*120}\n▶ Processing date: {fecha} / {fechas[-1]}")
        
        for city in valid_cities:

            dist_dict_city = global_dist_dict.get(city, {})

            for decision_point in decision_times:
                
                generate_alfred_parameters(
                    decision_point,
                    experiment_type=experiment_type,
                    dist_method = distance_method,
                    instance=instance,
                    return_params = False
                    )
                
                alfred_parameters = load_alfred_parameters(
                    experiment_type=experiment_type,
                    dist_method = distance_method,
                    instance=instance,
                )

                labors_algo_dynamic_df, moves_algo_dynamic_df, _ =  alfred_algorithm_assignment(
                    alfred_parameters
                )

if __name__ == '__main__':
    run_ALFRED()