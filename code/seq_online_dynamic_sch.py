import pandas as pd
import numpy as np

import ipywidgets as widgets
from ipywidgets import interact

from datetime import timedelta
import pickle
import os


from src.data_load import load_tables, load_online_instance, load_directorio_hist_df

from src.online_algorithms import evaluate_driver_feasibility
from src.online_algorithms import filter_dynamic_df, filter_dfs_for_insertion
from src.online_algorithms import get_drivers, get_best_insertion, commit_new_labor_insertion

from src.metrics import collect_hist_baseline_dfs
from src.experimentation_config import *
from src.config import *

data_path = f'{REPO_PATH}/data'

instance = 'instAD1'
distance_method = 'haversine'

inst_path = f'{data_path}/resultados/online_operation/{instance}'

directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path, generate_labors=False)
labors_real_df, labors_static_df, labors_dynamic_df = load_online_instance(data_path, instance, labors_raw_df)
directorio_hist_df = load_directorio_hist_df(data_path, instance)
labors_dynamic_df['latest_arrival_time'] = labors_dynamic_df['schedule_date'] + timedelta(minutes=TIEMPO_GRACIA)

fechas = fechas_dict[instance]

hist_inst = f'{instance[:5]}S{instance[6:]}'
labors_hist_df, moves_hist_df = collect_hist_baseline_dfs(data_path, hist_inst, fechas, distance_method)

metrics = ['hybrid']
alphas = [0]

def collect_alpha_results_to_df(data_path: str, instance: str, dist_method: str, metrics: list, alphas: list):
    labors_algo_df = pd.DataFrame()
    moves_algo_df = pd.DataFrame()

    for metric in metrics: 
        for alpha in alphas:
            upload_path = f'{inst_path}/res_{metric}_{alpha:.1f}_static.pkl'

            if not os.path.exists(upload_path):
                continue
            with open(upload_path, "rb") as f:
                res = pickle.load(f)
                inc_values, duration, results_df, moves_df, metrics_df = res

            if not results_df.empty:
                results_df = results_df.sort_values(["city", "date", "service_id", "labor_id"])
            if not moves_df.empty:
                moves_df = moves_df.sort_values(["city", "date", "service_id", "labor_id"])

            # Normalize datetime columns to Bogotá tz
            datetime_cols = [
                "labor_created_at",
                "labor_start_date",
                "labor_end_date",
                "created_at",
                "schedule_date",
                "actual_start", 
                "actual_end"
                ]


            for df in (results_df, moves_df):
                for col in datetime_cols:
                    if col in df.columns:
                        df[col] = (
                            pd.to_datetime(df[col], errors="coerce", utc=True)
                            .dt.tz_convert("America/Bogota")
                        )
                
                for col in ['city', 'alfred', 'service_id', 'assigned_driver']:
                    if col in df.columns:
                        df[col] = (
                            df[col]
                            .apply(lambda x: '' if (pd.isna(x) or x == '') else str(int(float(x))))
                        )

            results_df['labor_id'] = (
                results_df['labor_id']
                .apply(lambda x: '' if (pd.isna(x) or x == '') else str(int(float(x))))
            )

            labors_algo_df = pd.concat([labors_algo_df,results_df])
            moves_algo_df = pd.concat([moves_algo_df,moves_df])
    
    return labors_algo_df, moves_algo_df

labors_algo_df, moves_algo_df = collect_alpha_results_to_df(data_path, instance, 'haversine', metrics, alphas)


labors_algo_dynamic_df = labors_algo_df.copy()
moves_algo_dynamic_df = moves_algo_df.copy()

unassigned_labors = {}

for city in valid_cities:
    print(city)
    for fecha in fechas:
        print(' ' + fecha)
        # if fecha != '2026-01-08':
        #     continue
        unassigned_labors[(city,fecha)] = []

        labors_dynamic_filtered_df = filter_dynamic_df(
            labors_dynamic_df=labors_dynamic_df,
            city=city,
            fecha=fecha
            )
        
        drivers = get_drivers(
            labors_algo_df=labors_algo_df,
            city=city,
            fecha=fecha)
        
        for i, new_labor in labors_dynamic_filtered_df.iterrows():
            # print(f'\t {i}')
            candidate_insertions = []

            for driver in drivers:  
                # print(f'\t \t {driver}')        
                labors_driver_df, moves_driver_df = filter_dfs_for_insertion(
                    labors_algo_df=labors_algo_df,
                    moves_algo_df=moves_algo_df,
                    city=city,
                    fecha=fecha,
                    driver=driver,
                    created_at=new_labor['created_at']
                    )

                feasible, infeasible_log, insertion_plan = evaluate_driver_feasibility(
                    new_labor=new_labor,
                    driver=driver,
                    moves_driver_df=moves_driver_df,
                    directory_df=directorio_hist_df,
                    ALFRED_SPEED=ALFRED_SPEED,
                    VEHICLE_TRANSPORT_SPEED=VEHICLE_TRANSPORT_SPEED,
                    TIEMPO_ALISTAR=TIEMPO_ALISTAR,
                    TIEMPO_FINALIZACION=TIEMPO_FINALIZACION, 
                    TIEMPO_GRACIA=TIEMPO_GRACIA,
                    )

                if feasible: 
                    candidate_insertions.append((driver, insertion_plan))
            
            if len(candidate_insertions)==0:
                unassigned_labors[(city,fecha)].append(new_labor)
                continue
            
            selected_driver, insertion_point, selection_df = get_best_insertion(
                candidate_insertions, 
                selection_mode="min_total_distance", 
                random_state=None)

            labors_algo_dynamic_df, moves_algo_dynamic_df = commit_new_labor_insertion(    
                labors_df=labors_algo_df,
                moves_df=moves_algo_dynamic_df,
                driver=selected_driver,
                insertion_plan=insertion_point,
                new_labor=new_labor
            )

print(f'Number of unassigned_labors: {len(unassigned_labors)}')

with open(f'{inst_path}/res_dynamic.pkl', 'wb') as f:
    pickle.dump([labors_algo_dynamic_df, moves_algo_dynamic_df], f)