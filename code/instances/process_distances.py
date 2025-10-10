import pandas as pd
import numpy as np

import pickle

from datetime import datetime

from time import perf_counter
# sys.path.append(os.path.abspath(os.path.join('../src')))  # Adjust as needed

from src.distance_utils import distance
from src.data_load import load_tables, load_instance, load_real_instance
from src.config import *
from src.experimentation_config import instance_map

data_path = '/Users/juanbeta/Library/CloudStorage/GoogleDrive-juan.beta98@gmail.com/My Drive/Work/Alfred/Alfred/data'
# instance = 'inst4b'
instance = 'inst_r1'
instance_type = instance_map[instance]

directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path, generate_labors=False)
labors_real_df = load_instance(data_path, instance, labors_raw_df)


# OSRM_URL = "http://localhost:5000/table/v1/driving/"

def compute_transport_distances(labors_real_df: pd.DataFrame, method="osrm", timeout=5):
    """
    Compute distances only between:
      - Nodes actually appearing in transport labors
      - All pairwise distances *only between shops* (non-null shop ids)
    
    Distances are grouped by city, not by date.
    Returns {city: dict}.

    Parameters
    ----------
    labors_real_df : pd.DataFrame
        DataFrame with transport labors (must include city, start/end address info).
    method : str, default "osrm"
        Distance computation method ("osrm" or "haversine").
    timeout : int, default 5
        Timeout for OSRM calls.

    Returns
    -------
    dict
        Dictionary of dicts: {city: {(id1, id2): distance_km, (p1, p2): distance_km, ...}}
    """
    dist_dict_by_city = {}

    # --- Ensure schedule_date is datetime
    if "schedule_date" in labors_real_df:
        if not pd.api.types.is_datetime64_any_dtype(labors_real_df["schedule_date"]):
            labors_real_df["schedule_date"] = pd.to_datetime(labors_real_df["schedule_date"])

    # --- Group by city ---
    grouped = labors_real_df.groupby("city")

    for city, df_sub in grouped:
        start = perf_counter()
        dist_dict = {}

        # Collect start nodes
        start_nodes = df_sub[["start_address_id", "start_address_point"]].rename(
            columns={"start_address_id": "address_id", "start_address_point": "address_point"}
        )

        # Collect end nodes
        end_nodes = df_sub[["end_address_id", "end_address_point"]].rename(
            columns={"end_address_id": "address_id", "end_address_point": "address_point"}
        )

        # Merge them
        nodes = pd.concat([start_nodes, end_nodes], ignore_index=True).dropna().drop_duplicates()

        # --- 1. Compute distances for actual log pairs ---
        for _, row in df_sub.iterrows():
            sid, sp = row["start_address_id"], row["start_address_point"]
            did, dp = row["end_address_id"], row["end_address_point"]

            if pd.isna(sp) or pd.isna(dp):
                continue

            if (sp, dp) not in dist_dict:
                d = distance(sp, dp, method=method, timeout=timeout)
                dist_dict[(sp, dp)] = d
                dist_dict[(dp, sp)] = d
            if sid and did and (sid, did) not in dist_dict:
                dist_dict[(sid, did)] = dist_dict[(sp, dp)]
                dist_dict[(did, sid)] = dist_dict[(dp, sp)]

        dist_dict_by_city[city] = dist_dict
        if city == '1':
            print(f"✅ Finished {city}  \t\t{round(perf_counter()-start,2)} \t{len(nodes)} nodes \t{len(dist_dict)} distances")
        else:
            print(f"✅ Finished {city}  \t{round(perf_counter()-start,2)} \t{len(nodes)} nodes \t{len(dist_dict)} distances")

    return dist_dict_by_city

print(f'------------------ Generating distances for for -{instance}- ------------------')
print(f'------- Instance: {instance} \n')
manhattan_distances = compute_transport_distances(labors_real_df, method='Manhattan')

with open(f'{data_path}/instances/{instance_type}_inst/{instance}/dist/manhattan_{instance}_dist_dict.pkl', "wb") as f:
    pickle.dump(manhattan_distances, f)

print('\n--------- REAL DISTANCES')
real_distances = compute_transport_distances(labors_real_df, method='osrm')

with open(f'{data_path}/instances/{instance_type}_inst/{instance}/dist/osrm_{instance}_dist_dict.pkl', "wb") as f:
    pickle.dump(real_distances, f)

print(f'\n------------------ Distances generated succesfully ------------------')