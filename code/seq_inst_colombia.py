import pandas as pd
import pickle
from datetime import timedelta

# sys.path.append(os.path.abspath(os.path.join('../src')))  # Adjust as needed

from src.data_load import load_tables, load_instance, load_distances # type: ignore
from src.filtering import filter_labors_by_date, filter_labors_by_city, filter_labores # type: ignore
from src.utils import codificacion_ciudades
from src.config import * # type: ignore
from src.experimentation_config import *
from src.pipeline import run_city_pipeline


# ——————————————————————————
# Configuración previa
# ——————————————————————————
if __name__ == "__main__":
    data_path = f'{REPO_PATH}/data'
    instance_input = input('Input the instance -inst{}- to run?: ')
    instance = f'inst{instance_input}'
    assert instance in instance_map.keys(), f'Non-existing instance "{instance}"!'
    instance_type = instance_map[instance]
    
    distance_type = 'osrm'              # Options: ['osrm', 'manhattan']
    distance_method = 'precalced'      # Options: ['precalced', 'haversine']
    
    assignment_types = ['algorithm', 'historic']  # 👈 choose one or both
    save_path = f'{data_path}/resultados/{instance_type}_inst/{instance}'

    print(f'------------------ Running -{instance}- ------------------')
    print(f'------- Run mode: sequential')
    print(f'------- Distances: {distance_method}\n')
    # print(f'------- Assignment types: {assignment_types}\n')

    # Load shared inputs only once
    directorio_df, labor_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path)
    labors_real_df = load_instance(data_path, instance, labor_raw_df)

    # Parámetros de fecha
    fechas = fechas_dict[instance]
    initial_day = int(fechas[0].rsplit('-')[2])

    for assignment_type in assignment_types:
        print(f"\n=== Running assignment_type = {assignment_type} ===")

        for start_date in fechas:
            dist_dict = load_distances(data_path, distance_type, instance)

            df_day = filter_labors_by_date(labors_real_df, start_date=start_date, end_date='one day lag')

            # Corrida secuencial
            results = []
            for city in valid_cities:

                base_day = pd.to_datetime(start_date).date()
                max_drivers_dict = max_drivers.get(instance, {}).get(city, None)
                if max_drivers_dict:
                    max_drivers_num = max_drivers_dict[base_day.day - initial_day]
                
                res = run_city_pipeline(
                    city, start_date, df_day, directorio_df, duraciones_df,
                    assignment_type, 
                    alpha=0,
                    DIST_DICT=dist_dict.get(city, {}),
                    dist_method=distance_method,
                    max_drivers_num=max_drivers_num,
                    instance=instance,
                    )
                results.append(res)
                print('       ✅ ' + city)

            # Resultados como diccionario
            results_by_city = {}
            for i in range(len(results)):
                if len(results[i]) > 0:
                    city, df_cleaned, df_moves, n_drivers = results[i]
                    results_by_city[city] = (df_cleaned, df_moves, n_drivers) 

            # Guardar en pickle
            if assignment_type == 'algorithm':
                save_full_path = f'{save_path}/res_{start_date}.pkl'
            elif assignment_type == 'historic':
                save_full_path = f'{save_path}/res_hist_{start_date}.pkl'

            with open(save_full_path, 'wb') as f:
                pickle.dump(results_by_city, f)

            print('  ✅ ' + start_date)

    print("\n✅ Finished all runs")


