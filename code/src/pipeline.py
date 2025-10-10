import pandas as pd

from .distance_utils import distance # type: ignore
from .data_load import load_tables # type: ignore
from .filtering import filter_labors_by_date, filter_labors_by_city, filter_labores # type: ignore
from .metrics import collect_vt_metrics_range, show_day_report_dayonly, compute_indicators # type: ignore
from .preprocessing import remap_to_base_date, build_services_map_df, process_group # type: ignore
from .plotting import plot_results # type: ignore
from .config import *
from .experimentation_config import max_drivers
from .algorithms import remove_drivers, compute_avg_times, run_assignment_algorithm, init_drivers # type: ignore
from .utils import get_city_name_from_code

# ——————————————————————————
# Función maestra por ciudad
# ——————————————————————————
def run_city_pipeline(  city_code, 
                        start_date, 
                        df_dist, 
                        directorio_df, 
                        duraciones_df,
                        assignment_type,
                        alpha=1,
                        dist_method='haversine',
                        DIST_DICT=None,
                        max_drivers_num=None,
                        **kwargs
                      ) :
    """
    Ejecuta TODO el flujo para una ciudad y fecha dada.
    start_date: string o Timestamp, mismo para todas las ciudades.
    Devuelve: (city_code, df_cleaned, df_moves)
    """
    # 1. Filtrar por ciudad
    df_city = filter_labors_by_city(df_dist, str(city_code))
    
    # 3. Quitar cancelados y ordenar
    df_city_filtered = (
        df_city.query("state_service != 'CANCELED'")
        .sort_values(['service_id', 'labor_start_date'])
        .reset_index(drop=True)
    )

    # 4. Remapear fechas al día base
    base_day = pd.to_datetime(start_date).date()
    df_city_remaped = remap_to_base_date(
        df_city_filtered, 
        ['schedule_date', 'labor_start_date', 'labor_end_date'], 
        base_day
    )

    # 5. Construir mapa de servicios
    services_map_df = build_services_map_df(df_city_remaped)

    # 6. Procesar grupos
    cleaned = []
    for _, grp in df_city_remaped.groupby('service_id', sort=False):
        kwargs['city_code'] = city_code
        cleaned.append(process_group(grp, dist_method=dist_method, 
                                     dist_dict=DIST_DICT, **kwargs))

    if len(cleaned)==0:
        return ()
    
    df_cleaned_template = pd.concat([c for c in cleaned if not c.empty], ignore_index=True)
    df_cleaned_template = df_cleaned_template.merge(
        services_map_df, 
        on=['service_id', 'labor_id'], 
        how='left'
    )

    df_cleaned_template = filter_labores(df_cleaned_template, hour_threshold=0)
    # avg_times_map = compute_avg_times(df_dist)

    #TODO: DEPENDING ON WHICH 
    # max_drivers_num = max_drivers.get(instance, None)
    # if max_drivers_num:
    #     max_drivers_num = max_drivers_num[base_day.day - initial_day]

    # 7. Ejecutar el algorithmo de assignación
    df_result, df_moves, n_drivers = run_assignment_algorithm(  df_cleaned_template=df_cleaned_template,
                                                                directorio_df=directorio_df,
                                                                duraciones_df=duraciones_df,
                                                                day_str=start_date, 
                                                                ciudad=get_city_name_from_code(city_code),
                                                                dist_dict=DIST_DICT,
                                                                dist_method=dist_method,
                                                                assignment_type=assignment_type,
                                                                alpha=alpha,
                                                                max_drivers=max_drivers_num,
                                                                **kwargs
                                                      )

    # 8. Devolver resultados
    return city_code, df_result, df_moves, n_drivers