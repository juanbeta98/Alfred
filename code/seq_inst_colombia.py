import pandas as pd
import pickle
from datetime import timedelta
from multiprocessing import Pool

import sys
import os

# sys.path.append(os.path.abspath(os.path.join('../src')))  # Adjust as needed

from src.data_load import load_tables, load_artificial_instance, load_distances # type: ignore
from src.filtering import filter_labors_by_date, filter_labors_by_city, filter_labores # type: ignore
from src.utils import codificacion_ciudades
from src.config import * # type: ignore
from src.pipeline import run_city_pipeline


# ——————————————————————————
# Configuración previa
# ——————————————————————————
if __name__ == "__main__":
    data_path = '/Users/juanbeta/Library/CloudStorage/GoogleDrive-juan.beta98@gmail.com/My Drive/Work/Alfred/Alfred/data'
    instance='inst3'
    distance_type = 'real'
    # assignment_type = 'algorithm'
    assignment_type = 'historic'
    save_path = f'{data_path}/resultados/artif_col_inst/{instance}'

    print(f'------------------ Running -{instance}- ------------------')
    print(f'------- Run mode: sequential')
    print(f'------- Distances: {distance_type}')
    print(f'------- Assignment: {assignment_type}\n')

    # data_path = '../data'
    directorio_df, labor_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path)
    labors_real_df = load_artificial_instance(data_path, instance, labor_raw_df)
    dist_dict = load_distances(data_path, 'real', instance)

    # Parámetros de fecha    
    fechas = pd.date_range("2025-09-08", "2025-09-13").strftime("%Y-%m-%d").tolist()

    for start_date in fechas:
        print(start_date)
        df_day = filter_labors_by_date(labors_real_df, start_date=start_date, end_date='one day lag')

        # Corrida secuencial
        results = []
        for city in valid_cities:
            # print(f'\t - {city}')
            res = run_city_pipeline(city, start_date, df_day, directorio_df, 
                                    duraciones_df, assignment_type, 1, dist_dict)
            results.append(res)

        # Resultados como diccionario
        results_by_city = {city: (df_cleaned, df_moves, n_drivers) for city, df_cleaned, df_moves, n_drivers in results}

        # Guardar en pickle
        if assignment_type == 'algorithm':
            save_full_path = f'{save_path}/res_{start_date}.pkl'
        elif assignment_type == 'historic':
            save_full_path = f'{save_path}/res_hist_{start_date}.pkl'
        with open(save_full_path, 'wb') as f:
            pickle.dump(results_by_city, f)
    
    print(f'-------- Successfully ran -{instance}- --------')

