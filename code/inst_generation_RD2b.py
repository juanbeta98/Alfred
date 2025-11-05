import pandas as pd
from datetime import datetime, timedelta

import os

from src.data_load import load_tables
from src.config import *
from src.inst_generation_utils import split_static_dynamic, create_hist_directory, filter_invalid_services

data_path = '/Users/juanbeta/Library/CloudStorage/GoogleDrive-juan.beta98@gmail.com/My Drive/Work/Alfred/Alfred/data'
instance = 'instRD2b'
directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(
    data_path, generate_labors=False
)

labors_filtered_df = filter_invalid_services(
    labors_raw_df, 
    min_delay_minutes=0,
    only_unilabor_services=True
)

# ---  Identify the last full week available ---
labors_filtered_df['labor_start_date'] = pd.to_datetime(labors_filtered_df['labor_start_date'])
max_date = labors_filtered_df['labor_start_date'].max().normalize()

# Align to the last Sunday
last_sunday = max_date - pd.Timedelta(days=max_date.weekday() + 1) if max_date.weekday() != 6 else max_date
last_monday = last_sunday - pd.Timedelta(days=6)

print(f"Using week: {last_monday.date()} → {last_sunday.date()}")

# --- Slice data for that week ---
mask = (labors_filtered_df['labor_start_date'] >= last_monday) & \
       (labors_filtered_df['labor_start_date'] <= last_sunday)
labors_inst_df = labors_filtered_df.loc[mask].copy()

# ------ Create historic directory of drivers ------
hist_directory = create_hist_directory(labors_inst_df)

labors_inst_static_df, labors_inst_dynamic_df = split_static_dynamic(labors_inst_df)

# ------ Ensure output directory exists ------
output_dir = os.path.join(data_path, "instances", "real_inst", instance)
os.makedirs(output_dir, exist_ok=True)

labors_inst_static_df.to_csv(os.path.join(output_dir, f"labors_{instance}_static_df.csv"), index=False)
labors_inst_dynamic_df.to_csv(os.path.join(output_dir, f"labors_{instance}_dynamic_df.csv"), index=False)
labors_inst_df.to_csv(os.path.join(output_dir, f"labors_{instance}_df.csv"), index=False)
hist_directory.to_csv(os.path.join(output_dir, "directorio_hist_df.csv"), index=False)

print(f'-------- Instance -{instance}- generated successfully --------\n')
