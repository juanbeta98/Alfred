import pandas as pd

from datetime import datetime
# sys.path.append(os.path.abspath(os.path.join('../src')))  # Adjust as needed

from src.distance_utils import distance
from src.data_load import load_tables, load_artificial_instance
from src.filtering import filter_labors_by_date, filter_labors_by_city, filter_labores
from src.metrics import collect_vt_metrics_range, show_day_report_dayonly, compute_indicators
from src.preprocessing import remap_to_base_date, build_services_map_df, process_group
from src.plotting import plot_weekly_services, plot_daily_services_week
from src.config import *
from src.inst_generation_utils import top_service_days, create_artificial_week

import pandas as pd
import numpy as np
import requests
import math

data_path = '/Users/juanbeta/Library/CloudStorage/GoogleDrive-juan.beta98@gmail.com/My Drive/Work/Alfred/Alfred/data'
instance = 'inst3'
directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path,generate_labors=False)
labors_real_df = load_artificial_instance(data_path, instance, labors_raw_df)


OSRM_URL = "http://localhost:5000/table/v1/driving/"

def parse_point(p: str):
    """
    Parse 'POINT(lon lat)' into (lat, lon).
    """
    try:
        coords = p.strip().replace("POINT (", "").replace(")", "").split()
        lon, lat = float(coords[0]), float(coords[1])
        return lat, lon
    except Exception:
        return None, None


def haversine(lat1, lon1, lat2, lon2):
    """
    Simple haversine distance (km).
    """
    φ1, φ2 = map(math.radians, (lat1, lat2))
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * 6371 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_transport_distances(labors_real_df, method="osrm", timeout=5, chunk_size=50):
    """
    Compute all unique transport distances using OSRM table API (fast) with fallback.
    
    Returns
    -------
    dict: {(start_id, end_id): distance_km}
    """
    dist_dict = {}

    # Extract unique pairs
    pairs = labors_real_df[
        ['start_address_id', 'end_address_id',
         'start_address_point', 'end_address_point']
    ].dropna().drop_duplicates()

    # Build unique node list
    unique_points = {}
    bad_ids = []
    for _, row in pairs.iterrows():
        for aid, point in [
            (row['start_address_id'], row['start_address_point']),
            (row['end_address_id'], row['end_address_point'])
        ]:
            if aid not in unique_points:
                lat, lon = parse_point(point)
                if lat is not None and lon is not None:
                    unique_points[aid] = (lon, lat)  # OSRM expects lon,lat
                else:
                    bad_ids.append(aid)

    if bad_ids:
        print(f"⚠️ Skipped {len(bad_ids)} ids with invalid coordinates: {bad_ids[:5]}...")

    ids = list(unique_points.keys())
    coords = [unique_points[i] for i in ids]
    id_to_idx = {i: idx for idx, i in enumerate(ids)}

    # Precompute distance matrix via OSRM (chunked)
    n = len(coords)
    distances = np.full((n, n), np.nan)

    if method == "osrm" and n > 0:
        for i in range(0, n, chunk_size):
            chunk_ids = ids[i:i + chunk_size]
            chunk_coords = coords[i:i + chunk_size]
            coord_str = ";".join([f"{lon},{lat}" for lon, lat in chunk_coords])

            url = OSRM_URL + coord_str + "?annotations=distance"
            try:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                dist_matrix = r.json()["distances"]

                # Fill global matrix
                for local_i, global_i in enumerate(range(i, i + len(chunk_coords))):
                    for local_j, global_j in enumerate(range(i, i + len(chunk_coords))):
                        d = dist_matrix[local_i][local_j]
                        if d is not None:
                            distances[global_i, global_j] = d / 1000.0  # to km
            except Exception as e:
                print(f"⚠️ OSRM table failed for chunk {i}:{i+chunk_size}, reason={e}")

    # Build final dictionary with fallback
    for _, row in pairs.iterrows():
        sid, did = row["start_address_id"], row["end_address_id"]

        if sid not in id_to_idx or did not in id_to_idx:
            print(f"⚠️ Missing mapping for pair ({sid}, {did}) – skipping.")
            continue

        si, di = id_to_idx[sid], id_to_idx[did]
        d = distances[si, di]

        if np.isnan(d):  # fallback to haversine
            lat1, lon1 = parse_point(row["start_address_point"])
            lat2, lon2 = parse_point(row["end_address_point"])
            if None not in (lat1, lon1, lat2, lon2):
                d = haversine(lat1, lon1, lat2, lon2)
            else:
                d = float("nan")

        dist_dict[(sid, did)] = d

    return dist_dict

dist_dict = compute_transport_distances(labors_real_df, method = 'osrm', timeout=5, chunk_size=100)
