# distance_utils.py
import math
import requests
import pandas as pd
from .config import OSRM_URL


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


def distance(p1, p2, method, dist_dict=None, timeout=5):
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
        return dist_dict.get((p1, p2), float('nan'))

    lat1, lon1 = parse_point(p1)
    lat2, lon2 = parse_point(p2)
    if None in (lat1, lon1, lat2, lon2):
        return float('nan')

    if method == 'haversine':
        φ1, φ2 = map(math.radians, (lat1, lat2))
        dφ = math.radians(lat2 - lat1)
        dλ = math.radians(lon2 - lon1)
        a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
        return 2 * 6371 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    if method == 'osrm':
        coords = f"{lon1},{lat1};{lon2},{lat2}"
        try:
            r = requests.get(OSRM_URL + coords + "?overview=false", timeout=timeout)
            r.raise_for_status()
            return r.json()['routes'][0]['distance'] / 1000
        except:
            return distance(p1, p2, method='haversine')

    # Manhattan
    KM_PER_DEG_LAT = 111.32
    mean_lat = math.radians((lat1 + lat2) / 2)
    dlat = abs(lat1 - lat2) * KM_PER_DEG_LAT
    dlon = abs(lon1 - lon2) * KM_PER_DEG_LAT * math.cos(mean_lat)
    
    return dlat + dlon
