import pandas as pd

from src.experimentation_config import instance_map

codificacion_ciudades = {
                            '149':'BOGOTA', 
                            '1':'MEDELLIN', 
                            '126':'BARRANQUILLA',
                            '150':'CARTAGENA',
                            '844':'BUCARAMANGA',
                            '830':'PEREIRA',
                            '1004':'CALI'
                        }

def get_city_name(city_code: str, cities_df: pd.DataFrame) -> str:
    """
    Retorna el nombre de la ciudad dado su código.

    Parámetros
    ----------
    city_code : str
        Código de la ciudad (columna 'cod_ciudad').
    cities_df : pd.DataFrame
        DataFrame con las columnas 'cod_ciudad' y 'ciudad'.

    Retorna
    -------
    str
        Nombre de la ciudad correspondiente al código. 
        Si el código no se encuentra, devuelve 'DESCONOCIDO'.
    """
    match = cities_df.loc[cities_df['cod_ciudad'].astype(str) == str(city_code), 'ciudad']
    if not match.empty:
        return match.iloc[0]
    return "DESCONOCIDO"


def get_city_name_from_code(city_code):
    if city_code == 'ALL':
        return 'Global'
    return codificacion_ciudades.get(city_code, 'DESCONOCIDO')


def process_instance(instance_input): 
    instance = f'inst{instance_input}'
    assert instance in instance_map.keys(), f'Non-existing instance "{instance}"!'
    instance_type = instance_map[instance]

    return instance, instance_type


def process_dynamic_instance(instance_input):
    if instance_input=='r':
        instance = f'instRD1'
    elif instance_input=='a':
        instance = f'instAD1'
    else:
        raise ValueError('Select a valid instance')
    instance_type = instance_map[instance]

    return instance, instance_type


def get_max_drivers(instance, city, max_drivers, start_date, initial_day):
    base_day = pd.to_datetime(start_date).date()
    max_drivers_dict = max_drivers.get(instance, {}).get(city, None)
    if max_drivers_dict:
        max_drivers_num = max_drivers_dict[base_day.day - initial_day]
    else:
        max_drivers_num = None
    
    return max_drivers_num


def consolidate_run_results(results):
    results_by_city = {}
    for i in range(len(results)):
        if len(results[i]) > 0:
            city, df_cleaned, df_moves, n_drivers = results[i]
            results_by_city[city] = (df_cleaned, df_moves, n_drivers)
    
    return results_by_city