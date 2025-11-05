import pandas as pd
import pickle
from datetime import timedelta
import os

from time import perf_counter
from tqdm import tqdm

from src.data_load import load_inputs, load_distances
from src.filtering import filter_labors_by_date, flexible_filter
from src.utils import get_max_drivers, print_iteration_header

from src.config import * # type: ignore
from src.experimentation_config import *
from src.pipeline import prepare_algo_baseline_pipeline, run_city_pipeline
from src.offline_algorithms import run_assignment_algorithm
from src.metrics import collect_results_from_dicts, compute_metrics_with_moves, compute_iteration_metrics, concat_run_results
from src.alpha_tuning_utils import should_update_incumbent, update_incumbent_state, \
    metrics, dist_norm_factor, extra_time_norm_factor
from src.utils import get_city_name_from_code

import multiprocessing as mp

# ——————————————————————————
# Configuración previa
# ——————————————————————————
def run_online_hist_baseline(
    instance: str,
    distance_method: str,
    save_results: bool
):
    data_path = f'{REPO_PATH}/data'
    
    distance_type = 'osrm'              # Options: ['osrm', 'manhattan']

    driver_init_mode = 'historic_directory' # Options: ['historic_directory', 'driver_directory']

    save_path = f'{data_path}/resultados'

    # Load shared inputs only once
    (directorio_df, labor_raw_df, cities_df, duraciones_df,
        valid_cities, labors_real_df, directorio_hist_df) = load_inputs(data_path, instance)

    # Parámetros de fecha
    fechas = fechas_map(instance)
    initial_day = int(fechas[0].rsplit('-')[2])

    run_results = []
    for start_date in fechas:
        dist_dict = load_distances(data_path, distance_type, instance, distance_method)
        df_day = filter_labors_by_date(
            labors_real_df, start_date=start_date, end_date='one day lag'
        )

        for city in valid_cities:

            directorio_hist_filtered_df = flexible_filter(
                        directorio_hist_df, city=city, date=start_date
                    )
            
            df_cleaned_template = prepare_algo_baseline_pipeline(
                city_code=city, 
                start_date=start_date, 
                df_dist=df_day, 
                dist_method=distance_method,
                DIST_DICT=dist_dict.get(city, {})
            )

            # 7. Ejecutar el algorithmo de assignación
            results_df, moves_df = run_assignment_algorithm(  
                df_cleaned_template=df_cleaned_template,
                directorio_df=directorio_hist_filtered_df,
                duraciones_df=duraciones_df,
                day_str=start_date, 
                ciudad=get_city_name_from_code(city),
                assignment_type='historic',
                dist_method=distance_method,
                dist_dict=dist_dict.get(city, {}),
                alpha=0,
                instance=instance,
                driver_init_mode=driver_init_mode
            )
            
            results_df['date'] = start_date
            moves_df['city'] = city
            
            moves_df['date'] = start_date
            run_results.append((results_df, moves_df))

    # Build consolidated dfs for this run
    results_df, moves_df = concat_run_results(run_results)

    # Guardar en pickle
    # ------ Ensure output directory exists ------
    if save_results:
        output_dir = os.path.join(
            save_path, 
            "online_operation", 
            instance,
            distance_method
        )
        os.makedirs(output_dir, exist_ok=True)  # Creates folder if missing

        with open(os.path.join(output_dir, f'res_hist.pkl'), 'wb') as f:
            pickle.dump([results_df, moves_df], f)

    print(f' ✅ Successfully ran online historic baseline scheduling \n')
    return True


def run_online_algo_baseline(
    instance: str,
    optimization_obj: str,
    distance_method: str,
    save_results: bool
):
    data_path = f'{REPO_PATH}/data'
    
    ''' START RUN PARAMETERS'''
    distance_type = 'osrm'              # Options: ['osrm', 'manhattan']

    assignment_type = 'algorithm'

    driver_init_mode = 'historic_directory' # Options: ['historic_directory', 'driver_directory']
    ''' END RUN PARAMETERS'''

    fechas = fechas_map(instance)
    initial_day = int(fechas[0].rsplit('-')[2])

    # Load shared inputs only once
    (directorio_df, labor_raw_df, cities_df, duraciones_df,
        valid_cities, labors_real_df, directorio_hist_df) = load_inputs(data_path, instance)

    ### Optimization config
    print_iteration_header(
        details='algorithm baseline scheduling')

    alpha = hyperparameter_selection[optimization_obj]

    start = perf_counter()

    
    run_results = []
    for start_date in fechas:
        print('----------------------------------------------------------------------------')

        dist_dict = load_distances(data_path, distance_type, instance, distance_method)
        df_day = filter_labors_by_date(
            labors_real_df, start_date=start_date, end_date='one day lag'
        )
        
        for city in valid_cities:
            directorio_hist_filtered_df = flexible_filter(
                directorio_hist_df, city=city, date=start_date
            )

            if df_day[df_day['city']==city].empty:
                continue

            inc_vt_labors = 0
            inc_extra_time = 1e9
            inc_dist = 1e9
            inc_state = None

            # Iterative search 
            for iter in tqdm(range(1, max_iterations[city] + 1), desc=f"{city}/{start_date}", unit="iter", leave=False):
                ''' START RUN ITERATION FOR SET ALPHA '''
                df_cleaned_template = prepare_algo_baseline_pipeline(
                    city_code=city, 
                    start_date=start_date, 
                    df_dist=df_day, 
                    dist_method=distance_method,
                    DIST_DICT=dist_dict.get(city, {})
                )

                # 7. Ejecutar el algorithmo de assignación
                results_df, moves_df = run_assignment_algorithm(  
                    df_cleaned_template=df_cleaned_template,
                    directorio_df=directorio_hist_filtered_df,
                    duraciones_df=duraciones_df,
                    day_str=start_date, 
                    ciudad=get_city_name_from_code(city),
                    assignment_type=assignment_type,
                    dist_method=distance_method,
                    dist_dict=dist_dict,
                    alpha=alpha,
                    instance=instance,
                    driver_init_mode=driver_init_mode
                )

                
                metrics = compute_metrics_with_moves(
                    results_df, 
                    moves_df,
                    fechas=[start_date],
                    dist_dict=dist_dict,
                    workday_hours=8,
                    city=city,
                    skip_weekends=False,
                    assignment_type=assignment_type,
                    dist_method=distance_method
                ) 

                iter_vt_labors, iter_extra_time, iter_dist = compute_iteration_metrics(metrics)

                ''' UPDATE INCUMBENT '''
                if iter_vt_labors >= inc_vt_labors:
                    update = False
                    
                    update, new_score, inc_score = should_update_incumbent(
                    optimization_obj,
                    iter_dist=iter_dist,
                    iter_extra_time=iter_extra_time,
                    inc_dist=inc_dist,
                    inc_extra_time=inc_extra_time
                    )

                    results_df['date'] = start_date
                    moves_df['city'] = city
                    moves_df['date'] = start_date

                    if update:
                        inc_state = update_incumbent_state(
                            iter_idx=iter,
                            iter_vt_labors=iter_vt_labors,
                            iter_extra_time=iter_extra_time,
                            iter_dist=iter_dist,
                            results_df=results_df,
                            moves_df=moves_df,
                            metrics=metrics,
                            start_time=start
                        )

                    # unpack into your variables (if you still want separate vars)
                    inc_vt_labors = inc_state["vt_labors"]
                    inc_extra_time = inc_state["extra_time"]
                    inc_dist = inc_state["dist"]
                    inc_values = inc_state["values"]

                    ''' CHECKPOINT SAVE '''
                    if iter in iterations_nums[city]:
                        duration = round(perf_counter() - start, 1)
                        tqdm.write(  # ✅ ensures checkpoint is printed on its own line
                            f"{start_date} \t{city} \t{iter} \t{inc_values[0]} \t"
                            f"{round(duration)}s \t{inc_vt_labors} \t"
                            f"{round(inc_extra_time/60,1)}h \t{inc_dist} km"
                        )

            if inc_state:
                run_results.append([inc_state['results'], inc_state['moves']])
            ''' END RUN ITERATION '''

    # Build consolidated dfs for this run
    results_df, moves_df = concat_run_results(run_results)

    if save_results:
        # ------ Ensure output directory exists ------
        output_dir = os.path.join(
            data_path, 
            "resultados", 
            "online_operation", 
            instance,
            distance_method
        )
        os.makedirs(output_dir, exist_ok=True)  # Creates folder if missing

        with open(os.path.join(output_dir, f'res_algo_OFFLINE.pkl'), 'wb') as f:
            duration = round(perf_counter() - start, 1)
            pickle.dump([results_df, moves_df], f)

    print(f' ✅ Successfully ran online static scheduling \n')
    return True



from datetime import datetime
from typing import Dict, Any

def run_single_iteration(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single iteration of the assignment algorithm.
    Args is a dict (since Pool.map only passes one argument).
    Returns all iteration results & metrics.
    """
    city = args["city"]
    start_date = args["start_date"]
    iter_idx = args["iter_idx"]
    df_day = args["df_day"]
    dist_dict = args["dist_dict"]
    directorio_hist_filtered_df = args["directorio_hist_filtered_df"]
    duraciones_df = args["duraciones_df"]
    distance_method = args["distance_method"]
    assignment_type = args["assignment_type"]
    alpha = args["alpha"]
    instance = args["instance"]
    driver_init_mode = args["driver_init_mode"]

    # --- Algorithm execution ---
    df_cleaned_template = prepare_algo_baseline_pipeline(
        city_code=city,
        start_date=start_date,
        df_dist=df_day,
        dist_method=distance_method,
        DIST_DICT=dist_dict.get(city, {})
    )

    results_df, moves_df = run_assignment_algorithm(
        df_cleaned_template=df_cleaned_template,
        directorio_df=directorio_hist_filtered_df,
        duraciones_df=duraciones_df,
        day_str=start_date,
        ciudad=get_city_name_from_code(city),
        assignment_type=assignment_type,
        dist_method=distance_method,
        dist_dict=dist_dict,
        alpha=alpha,
        instance=instance,
        driver_init_mode=driver_init_mode
    )

    metrics = compute_metrics_with_moves(
        results_df,
        moves_df,
        fechas=[start_date],
        dist_dict=dist_dict,
        workday_hours=8,
        city=city,
        skip_weekends=False,
        assignment_type=assignment_type,
        dist_method=distance_method
    )

    vt_labors, extra_time, dist = compute_iteration_metrics(metrics)

    return {
        "city": city,
        "date": start_date,
        "iter": iter_idx,
        "vt_labors": vt_labors,
        "extra_time": extra_time,
        "dist": dist,
        "results": results_df,
        "moves": moves_df,
        "metrics": metrics,
    }

def select_best_iteration(df_results: pd.DataFrame, optimization_obj: str) -> int:
    """
    Return index of best (incumbent) iteration.
    """
    if optimization_obj == "driver_distance":
        return df_results["dist"].idxmin()
    elif optimization_obj == "driver_extra_time":
        return df_results["extra_time"].idxmin()
    # elif optimization_obj == "vt_labors":
    #     return df_results["vt_labors"].idxmax()
    else:
        raise ValueError(f"Unknown optimization objective: {optimization_obj}")


def run_online_algo_baseline_parallel(
    instance: str,
    optimization_obj: str,
    distance_method: str,
    save_results: bool,
    n_processes: int = None  # defaults to all available CPUs
):
    data_path = f'{REPO_PATH}/data'
    distance_type = 'osrm'
    assignment_type = 'algorithm'
    driver_init_mode = 'historic_directory'

    fechas = fechas_map(instance)
    (
        directorio_df, labor_raw_df, cities_df, duraciones_df,
        valid_cities, labors_real_df, directorio_hist_df
    ) = load_inputs(data_path, instance)

    alpha = hyperparameter_selection[optimization_obj]
    start = perf_counter()

    run_results = []

    for start_date in fechas:
        print(f"\n{'-'*70}\n▶️ Processing date: {start_date}")

        dist_dict = load_distances(data_path, distance_type, instance, distance_method)
        df_day = filter_labors_by_date(labors_real_df, start_date=start_date, end_date='one day lag')

        for city in valid_cities:
            if df_day[df_day['city'] == city].empty:
                continue

            directorio_hist_filtered_df = flexible_filter(
                directorio_hist_df, city=city, date=start_date
            )

            max_iter = max_iterations[city]
            iter_args = [
                {
                    "city": city,
                    "start_date": start_date,
                    "iter_idx": i,
                    "df_day": df_day,
                    "dist_dict": dist_dict,
                    "directorio_hist_filtered_df": directorio_hist_filtered_df,
                    "duraciones_df": duraciones_df,
                    "distance_method": distance_method,
                    "assignment_type": assignment_type,
                    "alpha": alpha,
                    "instance": instance,
                    "driver_init_mode": driver_init_mode,
                }
                for i in range(1, max_iter + 1)
            ]

            print(f"  ⏩ Running {max_iter} iterations in parallel for city {city}...")

            # --- Run in parallel ---
            from tqdm import tqdm
            with mp.Pool(processes=n_processes) as pool:
                results = list(tqdm(pool.imap(run_single_iteration, iter_args), total=len(iter_args)))


            # --- Select incumbent ---
            df_results = pd.DataFrame(results)
            best_idx = select_best_iteration(df_results, optimization_obj)
            inc_state = df_results.iloc[best_idx]

            print(
                f"  ✅ {city}: Best iter={inc_state['iter']} | "
                f"vt={inc_state['vt_labors']} | "
                f"extra={round(inc_state['extra_time']/60,1)}h | "
                f"dist={inc_state['dist']:.1f} km"
            )

            run_results.append([inc_state["results"], inc_state["moves"]])

    # --- Combine all results ---
    results_df, moves_df = concat_run_results(run_results)

    if save_results:
        output_dir = os.path.join(data_path, "resultados", "online_operation", instance, distance_method)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'res_algo_OFFLINE.pkl'), "wb") as f:
            pickle.dump([results_df, moves_df], f)

    print(f"\n✅ Completed in {round(perf_counter() - start, 1)}s total.")
    return True





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    args = parser.parse_args()

    run_online_hist_baseline(args.instance)
    run_online_algo_baseline(args.instance)    