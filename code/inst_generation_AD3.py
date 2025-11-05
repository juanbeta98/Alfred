import pandas as pd

from datetime import datetime, timedelta

import os

from src.data_load import load_tables
from src.config import *
from src.inst_generation_utils import (
    top_service_days, 
    create_artificial_dynamic_week, 
    split_static_dynamic,
    create_hist_directory, 
    filter_invalid_services
)


# ------ Load the data ------
data_path = '/Users/juanbeta/Library/CloudStorage/GoogleDrive-juan.beta98@gmail.com/My Drive/Work/Alfred/Alfred/data'
instance = 'instAD3'
directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path, generate_labors=False)

# ------ Filter invalid services ------
labors_filtered_df = filter_invalid_services(labors_raw_df, min_delay_minutes=30)

top7_df = top_service_days(labors_filtered_df, city_col="city", date_col="labor_start_date", 
                           top_n=7, starting_year=2025)

labors_inst_df, mapping_df = create_artificial_dynamic_week(labors_filtered_df, top7_df, 
                                                    seed=52, starting_date="2026-01-05")

# ------ Create historic directory of drivers ------
hist_directory = create_hist_directory(labors_inst_df)

labors_inst_static_df, labors_inst_dynamic_df = split_static_dynamic(labors_inst_df)

# ------ Ensure output directory exists ------
output_dir = os.path.join(data_path, "instances", "artif_inst", instance)
os.makedirs(output_dir, exist_ok=True)  # Creates folder if missing

# ------ Saving instance ------
labors_inst_static_df.to_csv(os.path.join(output_dir, f"labors_{instance}_static_df.csv"), index=False)
labors_inst_dynamic_df.to_csv(os.path.join(output_dir, f"labors_{instance}_dynamic_df.csv"), index=False)
labors_inst_df.to_csv(os.path.join(output_dir, f"labors_{instance}_df.csv"), index=False)
hist_directory.to_csv(os.path.join(output_dir, "directorio_hist_df.csv"), index=False)
mapping_df.to_csv(os.path.join(output_dir, f"mapping_{instance}.csv"), index=False)

print(f'-------- Instance -{instance}- generated successfully --------\n')
