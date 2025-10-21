# distance_utils.py
import pandas as pd

from datetime import datetime

import math
import requests

import pickle
import os
import csv

from .config import OSRM_URL
from .experimentation_config import instance_map

def parse_point(s):
    """
    Extrae latitud y longitud de un texto con formato 'POINT(lon lat)'.

    Parámetros:
        s (str): Texto de coordenadas.

    Retorna:
        tuple: (latitud, longitud) o (None, None) si no es válido.
    """
    if pd.isna(s) or not s.strip().startswith("POINT"):
        return None, None
    lon, lat = map(float, s.lstrip('POINT').strip(' ()').split())
    return lat, lon


def distance(p1, p2, method, dist_dict=None, timeout=5, **kwargs):
    """
    Calcula la distancia entre dos puntos geográficos según el método indicado.

    Parámetros:
        p1, p2 (str): Puntos en formato 'POINT(lon lat)'.
        method (str): 'precalced', 'haversine', 'osrm', 'manhattan'.
        dist_dict (dict): Diccionario precalculado, requerido si method='precalced'.
        timeout (int): Tiempo máximo en segundos para consultas OSRM.

    Retorna:
        float: Distancia en kilómetros, NaN si no es calculable.
    """
    if method == 'precalced':
        if (p1, p2) in dist_dict:
            return dist_dict.get((p1, p2), float('nan')), dist_dict
        else:
            new_distance, _ = distance(p1, p2, 'osrm')
            updated_dist_dict = _update_distance_dictionary(kwargs['instance'], kwargs['city_code'], 
                                                    p1, p2, new_distance, log=False)
            return new_distance, updated_dist_dict

    lat1, lon1 = parse_point(p1)
    lat2, lon2 = parse_point(p2)
    if None in (lat1, lon1, lat2, lon2):
        return float('nan')

    if method == 'haversine':
        φ1, φ2 = map(math.radians, (lat1, lat2))
        dφ = math.radians(lat2 - lat1)
        dλ = math.radians(lon2 - lon1)
        a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
        return 2 * 6371 * math.atan2(math.sqrt(a), math.sqrt(1-a)), dist_dict

    if method == 'osrm':
        coords = f"{lon1},{lat1};{lon2},{lat2}"
        try:
            r = requests.get(OSRM_URL + coords + "?overview=false", timeout=timeout)
            r.raise_for_status()
            return r.json()['routes'][0]['distance'] / 1000, {}
        except:
            print(f'A DISTANCE WAS NOT APPROPIATELY GENERATED')
            return distance(p1, p2, method='haversine')

    # Manhattan
    KM_PER_DEG_LAT = 111.32
    mean_lat = math.radians((lat1 + lat2) / 2)
    dlat = abs(lat1 - lat2) * KM_PER_DEG_LAT
    dlon = abs(lon1 - lon2) * KM_PER_DEG_LAT * math.cos(mean_lat)
    
    return dlat + dlon


def _update_distance_dictionary(
    instance: str, 
    city_code: str, 
    p1: tuple, 
    p2: tuple, 
    new_distance: float, 
    log: bool = True
) -> dict:
    """
    Updates the distance dictionary pickle with a new computed distance.
    Distances are stored per city_code inside the pickle.
    Optionally logs the update with newest entries at the top of the log file.
    """
    import os, pickle, csv
    from datetime import datetime

    data_path = '/Users/juanbeta/Library/CloudStorage/GoogleDrive-juan.beta98@gmail.com/My Drive/Work/Alfred/Alfred/data'
    inst_path = f'{data_path}/instances/{instance_map[instance]}_inst/dist'
    pickle_file = f'{inst_path}/osrm_dist_dict.pkl'
    log_file = f'{inst_path}/distance_log.csv'

    # --- Load existing dictionary ---
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            dist_dict = pickle.load(f)
    else:
        dist_dict = {}

    # Ensure this city has its own dict
    if city_code not in dist_dict:
        dist_dict[city_code] = {}

    # --- Update city-specific dictionary ---
    dist_dict[city_code][(p1, p2)] = new_distance
    dist_dict[city_code][(p2, p1)] = new_distance

    # --- Save back pickle ---
    with open(pickle_file, "wb") as f:
        pickle.dump(dist_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # --- Optional logging (prepend new row) ---
    if log:
        new_row = [
            datetime.now().isoformat(timespec="seconds"),
            city_code,
            str(p1),
            str(p2),
            round(new_distance, 6)
        ]

        if os.path.exists(log_file):
            with open(log_file, "r", newline="") as f:
                rows = list(csv.reader(f))

            header = rows[0] if rows else ["timestamp", "city", "p1", "p2", "distance_km"]
            existing_rows = rows[1:] if len(rows) > 1 else []

            rows = [header, new_row] + existing_rows
        else:
            rows = [["timestamp", "city", "p1", "p2", "distance_km"], new_row]

        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    return dist_dict


