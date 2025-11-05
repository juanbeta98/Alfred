import os
import pickle
import pandas as pd
from datetime import timedelta

from tqdm import tqdm
from itertools import product

from src.data_load import load_tables, load_online_instance, load_directorio_hist_df, load_duraciones_df, upload_ONLINE_static_solution
from src.online_algorithms import (
    commit_labor_insertion,
    commit_nontransport_labor_insertion,
    filter_dynamic_df,
    get_drivers,
    get_nontransport_labor_duration
)
from src.experimentation_config import *
from src.config import *


def run_INSERT(
    instance: str,
    optimization_obj: str,
    distance_method: str):
    """
    Ejecuta la simulación online dinámica para una instancia dada.
    Puede correrse como módulo independiente o ser llamado desde otro script.
    """
    data_path = f"{REPO_PATH}/data"

    # --- Parámetros de ejecución ---
    distance_type = "osrm"
    driver_init_mode = "historic_directory"

    # --- Cargar datos base ---
    directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path, generate_labors=False)
    labors_real_df, labors_static_df, labors_dynamic_df = load_online_instance(data_path, instance, labors_raw_df)
    directorio_hist_df = load_directorio_hist_df(data_path, instance)
    duraciones_df = load_duraciones_df(data_path)
    labors_dynamic_df["latest_arrival_time"] = labors_dynamic_df["schedule_date"] + timedelta(minutes=TIEMPO_GRACIA)

    fechas = fechas_map(instance)

    # --- Consolidar resultados previos (estáticos) ---
    labors_algo_static_df, moves_algo_static_df = upload_ONLINE_static_solution(
        data_path, 
        instance, 
        distance_method, 
        optimization_obj
    )

    labors_algo_dynamic_df = labors_algo_static_df.copy()
    moves_algo_dynamic_df = moves_algo_static_df.copy()
    unassigned_services = []

    # --- Ejecución por ciudad y fecha ---
    for city, fecha in tqdm(product(valid_cities, fechas),
                        total=len(valid_cities) * len(fechas),
                        desc="Processing city/date pairs",
                        unit="iteration"):

        labors_dynamic_filtered_df = filter_dynamic_df(
            labors_dynamic_df=labors_dynamic_df, city=city, fecha=fecha
        )

        drivers = get_drivers(
            labors_algo_df=labors_algo_dynamic_df,
            directorio_hist_df=directorio_hist_df,
            city=city,
            fecha=fecha,
            get_all=True,
        )

        # for _, new_labor in labors_dynamic_filtered_df.iterrows():
        for service_id, service_df in labors_dynamic_filtered_df.groupby('service_id'):

            # Unilabor service
            if len(service_df) == 1:
                
                new_labor = service_df.iloc[0,:]
                success_flag, labors_algo_dynamic_df, moves_algo_dynamic_df, curr_end_time_iter, \
                    curr_end_pos_iter, unassigned_services = commit_labor_insertion(
                            labors_algo_df=labors_algo_dynamic_df,
                            moves_algo_df=moves_algo_dynamic_df,
                            new_labor=new_labor,
                            directorio_hist_df=directorio_hist_df,
                            unassigned_services=unassigned_services,
                            drivers=drivers,
                            city=city,
                            fecha=fecha,
                            vehicle_transport_speed=VEHICLE_TRANSPORT_SPEED,
                            alfred_speed=ALFRED_SPEED,
                            tiempo_alistar=TIEMPO_ALISTAR,
                            tiempo_finalizacion=TIEMPO_FINALIZACION,
                            tiempo_gracia=TIEMPO_GRACIA,
                            early_buffer=TIEMPO_GRACIA,
                            selection_mode = 'min_total_distance',
                        )
                
                if curr_end_time_iter != None:
                    curr_end_time = curr_end_time_iter
                    curr_end_pos = curr_end_pos_iter
            
            # Multilabor service
            else:                
                # Assign first labor, retrieve end_time
                first_labor = service_df.iloc[0]

                temp_labors_algo_df = labors_algo_dynamic_df.copy()
                temp_moves_algo_df = moves_algo_dynamic_df.copy()

                success_flag, temp_labors_algo_df, temp_moves_algo_df, curr_end_time_iter, curr_end_pos_iter, _ = \
                    commit_labor_insertion(
                        labors_algo_df=temp_labors_algo_df,
                        moves_algo_df=temp_moves_algo_df,
                        new_labor=first_labor,
                        directorio_hist_df=directorio_hist_df,
                        unassigned_services=unassigned_services,
                        drivers=drivers,
                        city=city,
                        fecha=fecha,
                        vehicle_transport_speed=VEHICLE_TRANSPORT_SPEED,
                        alfred_speed=ALFRED_SPEED,
                        tiempo_alistar=TIEMPO_ALISTAR,
                        tiempo_finalizacion=TIEMPO_FINALIZACION,
                        tiempo_gracia=TIEMPO_GRACIA,
                        early_buffer=TIEMPO_GRACIA,
                        selection_mode = 'min_total_distance'
                    )
                
                if not success_flag:
                    unassigned_services.append(service_df)
                    continue
                
                curr_end_time = curr_end_time_iter
                curr_end_pos = curr_end_pos_iter
            
                def _get_first_labor_end_status(labors_df, first_labor_id):
                    row = labors_df.loc[labors_df['labor_id'] == first_labor_id].iloc[0]
                    return row['actual_end'], row['end_address_point']

                curr_end_time, curr_end_pos = _get_first_labor_end_status(
                    labors_df=temp_labors_algo_df, 
                    first_labor_id=first_labor['labor_id'])
                
                if curr_end_time_iter != curr_end_time: 
                    print('yes indeed')

                # For all other labors in sequential order:
                for _, subsequent_labor in service_df.iloc[1:].iterrows():
                    labor_category = subsequent_labor['labor_category']
                    if labor_category == 'VEHICLE_TRANSPORTATION':
                        
                        success_flag, temp_labors_algo_df, temp_moves_algo_df, curr_end_time_iter, curr_end_pos_iter, _ = commit_labor_insertion(
                            labors_algo_df=temp_labors_algo_df,
                            moves_algo_df=temp_moves_algo_df,
                            new_labor=first_labor,
                            directorio_hist_df=directorio_hist_df,
                            unassigned_services=unassigned_services,
                            drivers=drivers,
                            city=city,
                            fecha=fecha,
                            vehicle_transport_speed=VEHICLE_TRANSPORT_SPEED,
                            alfred_speed=ALFRED_SPEED,
                            tiempo_alistar=TIEMPO_ALISTAR,
                            tiempo_finalizacion=TIEMPO_FINALIZACION,
                            tiempo_gracia=TIEMPO_GRACIA,
                            early_buffer=TIEMPO_GRACIA,
                            selection_mode = 'min_total_distance',
                            forced_start_time=curr_end_time
                        )

                        if curr_end_time_iter != None:
                            curr_end_time = curr_end_time_iter
                            curr_end_pos = curr_end_pos_iter
                    else:

                        labor_duration = get_nontransport_labor_duration(duraciones_df, city, subsequent_labor['labor_type'])

                        labor_pos = subsequent_labor['end_address_point']

                        temp_labors_algo_df, temp_moves_algo_df, end_time = commit_nontransport_labor_insertion(
                            labors_df=temp_labors_algo_df,
                            moves_df=temp_moves_algo_df,
                            new_labor=subsequent_labor,
                            start_pos=labor_pos,
                            start_time=curr_end_time,
                            duration=labor_duration,
                            fecha=fecha
                        )

                        curr_end_time = end_time
                        curr_end_pos = labor_pos

                    if not success_flag:
                        unassigned_services.append(service_df)
                        break
                        
                if success_flag:
                    labors_algo_dynamic_df = temp_labors_algo_df.copy()
                    moves_algo_dynamic_df = temp_moves_algo_df.copy()


    print(f"Number of unassigned_labors: {len(unassigned_services)}")

    # ------ Ensure output directory exists ------
    output_dir = os.path.join(data_path, 
                                "resultados", 
                                "online_operation", 
                                instance,
                                distance_method
                                )
    os.makedirs(output_dir, exist_ok=True)  # Creates folder if missing
    
    with open(os.path.join(output_dir, f'res_algo_INSERT.pkl'), 'wb') as f:
        pickle.dump([labors_algo_dynamic_df, moves_algo_dynamic_df], f)

    with open(os.path.join(output_dir, f'unassigned_INSERT.pkl'), 'wb') as f:
        pickle.dump(unassigned_services, f)

    print(f' ✅ Successfully ran INSERT algorithm \n')
    return {"unassigned_labors": len(unassigned_services)}


# --- Allow running standalone ---
if __name__ == "__main__":
    instance='instAD2b'
    optimization_obj='hybrid'
    distance_method='haversine'

    import argparse

    parser = argparse.ArgumentParser(description="Run online dynamic insertion simulation")
    parser.add_argument("--instance", required=True, help="Instance name (e.g. instAD2b)")
    args = parser.parse_args()

    run_INSERT(
        # args.instance
        instance=instance,
        optimization_obj=optimization_obj,
        distance_method=distance_method)
