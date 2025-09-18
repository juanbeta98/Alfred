import pandas as pd
import random

import math
from datetime import datetime, timedelta, time
from typing import Tuple, Dict, Any, List

from .distance_utils import parse_point, distance
from .config import *



'''
MAIN EXECUTION ALGORITHM
'''
def run_assignment_algorithm(   df_cleaned_template: pd.DataFrame,
                                directorio_df: pd.DataFrame, 
                                duraciones_df: pd.DataFrame,
                                day_str: str, 
                                ciudad: str,
                                assignment_type: str = 'algorithm',
                                dist_dict = None,
                                alpha = 1, 
                                max_drivers = None
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Ejecuta la simulación de asignación de labores de manera cronológica para una ciudad en un día específico.

    Usa la duración estimada (percentil 75) de cada tipo de labor por ciudad. Si no existe,
    usa el promedio del percentil 75 en otras ciudades. Como último recurso, usa `TIMEPO_OTHER`.
    """

    if df_cleaned_template.empty:
        return pd.DataFrame(), pd.DataFrame(), 0

    df_sorted = prepare_iteration_data(df_cleaned_template)
    assigned = [pd.NA] * len(df_cleaned_template)
    starts = [pd.NaT] * len(df_cleaned_template)
    ends = [pd.NaT] * len(df_cleaned_template)
    service_end_times = {}

    drivers = init_drivers_wrapper(df_sorted, directorio_df, ciudad, assignment_type=assignment_type, 
                                   max_drivers=max_drivers)

    # --- Lógica de asignación principal ---
    for _, row in df_sorted.iterrows():
        original_idx = row['index']
        service_id = row['service_id']

        if service_end_times.get(service_id) is pd.NaT:
            continue
        prev_end = service_end_times.get(service_id)

        if row['labor_category'] == 'VEHICLE_TRANSPORTATION':
            pick = get_driver_wrapper(drivers, row, prev_end, TIEMPO_PREVIO, TIEMPO_GRACIA,
                                      ALFRED_SPEED, 'haversine', dist_dict, 
                                      alpha, assignment_type)

            if not pick:
                service_end_times[service_id] = pd.NaT
                continue

            drv = pick['drv']
            is_last = not df_cleaned_template[
                (df_cleaned_template['service_id'] == service_id) &
                (df_cleaned_template['schedule_date'] > row['schedule_date'])
            ].empty

            # TODO -> Backlog 1 
            astart, aend = assign_task_to_driver(
                drivers[drv], pick['arrival'],
                prev_end or (row['schedule_date'] - timedelta(minutes=TIEMPO_PREVIO)),
                row['map_start_point'], row['map_end_point'], 
                is_last, TIEMPO_ALISTAR, TIEMPO_FINALIZACION,
                VEHICLE_TRANSPORT_SPEED, 'haversine', 
                dist_dict
            )
            assigned[original_idx], starts[original_idx], ends[original_idx] = drv, astart, aend
            service_end_times[service_id] = aend

        else:
            # --- Labores no transporte: usar p75 de duraciones_df ---
            astart = prev_end or row['schedule_date']

            # Buscar primero por ciudad + labor_type
            dur_row = duraciones_df[
                (duraciones_df["city"] == ciudad) &
                (duraciones_df["labor_type"] == row["labor_type"])
            ]

            if not dur_row.empty:
                duration_min = float(dur_row["p75_min"].iloc[0])
            else:
                # Si no existe en esa ciudad, sacar promedio global para ese labor_type
                labor_rows = duraciones_df[duraciones_df["labor_type"] == row["labor_type"]]
                if not labor_rows.empty:
                    duration_min = float(labor_rows["p75_min"].mean())
                else:
                    # Fallback final
                    duration_min = TIEMPO_OTHER

            aend = astart + timedelta(minutes=duration_min)
            starts[original_idx], ends[original_idx] = astart, aend
            service_end_times[service_id] = aend

    # --- Construcción DataFrame final ---
    df_result = df_cleaned_template.copy()
    if assignment_type == 'algorithm':
        df_result['assigned_driver'] = assigned
        df_result['actual_start'] = pd.to_datetime(starts)
        df_result['actual_end'] = pd.to_datetime(ends)
    elif assignment_type == 'historic':
        df_result['historic_driver'] = assigned
        df_result['historic_start'] = pd.to_datetime(starts)
        df_result['historic_end'] = pd.to_datetime(ends)

    # — Reconstrucción de movimientos y tiempos libres —
    df_moves = build_driver_movements(df_result, directorio_df, day_str, 'haversine', 
                                      ALFRED_SPEED, ciudad, dist_dict=dist_dict, 
                                      assignment_type=assignment_type)

    return df_result, df_moves, len(drivers)


'''
OPERATORS
'''
def get_candidate_drivers(  drivers: Dict[str, Dict], 
                            row: pd.Series,
                            prev_end: pd.Timestamp, 
                            TIEMPO_PREVIO: int,
                            TIEMPO_GRACIA: int, 
                            ALFRED_SPEED: float,
                            method: str,
                            dist_dict: dict
                          ) -> List[Dict[str, Any]]:
    """
    Genera lista de conductores candidatos para una labor VT.

    Parámetros
    ----------
    drivers : dict
        Diccionario con información de cada conductor (posición, disponibilidad).
    row : pd.Series
        Fila del DataFrame con información de la labor.
    prev_end : Timestamp
        Fin del servicio anterior para este `service_id`.
    TIEMPO_PREVIO : int
        Minutos antes de la hora programada para permitir llegada anticipada.
    TIEMPO_GRACIA : int
        Minutos de tolerancia para llegada tardía.
    ALFRED_SPEED : float
        Velocidad media del vehículo en km/h.
    method : str
        Método de cálculo de distancia.

    Retorna
    -------
    list of dict
        Lista de conductores con hora estimada de llegada.
    """
    sched = row['schedule_date']
    early = prev_end or (sched - timedelta(minutes=TIEMPO_PREVIO))
    late  = (prev_end + timedelta(minutes=TIEMPO_GRACIA)) if prev_end else (sched + timedelta(minutes=TIEMPO_GRACIA))
    
    cands = []
    for name, drv in drivers.items():
        av = drv['available']
        if av.time() < drv['work_start']:
            av = pd.Timestamp(datetime.combine(av.date(), drv['work_start']), tz=av.tz)

        dkm = distance(drv['position'], row['map_start_point'], method=method, dist_dict=dist_dict)
        arr = av + timedelta(minutes=(0 if math.isnan(dkm) else dkm/ALFRED_SPEED*60))

        if arr <= late:
            cands.append({'drv': name, 'arrival': arr})
    return cands


def select_from_candidates(cands, ALPHA):
    """
    Selecciona un candidato de una lista usando el criterio GRASP (Greedy Randomized Adaptive Search Procedure)
    con una Lista Restringida de Candidatos (RCL, Restricted Candidate List).

    El algoritmo:
      1. Calcula el "costo" de cada candidato (aquí basado en la marca de tiempo de llegada).
      2. Determina el umbral de inclusión en la RCL usando el parámetro `ALPHA`.
         - ALPHA = 0 → comportamiento completamente codicioso (elige el mínimo costo)
         - ALPHA = 1 → comportamiento completamente aleatorio
      3. Selecciona aleatoriamente un candidato de la RCL.

    Parámetros
    ----------
    cands : list of dict
        Lista de candidatos. Cada candidato debe tener la clave 'arrival' con un objeto datetime.
    ALPHA : float
        Parámetro de control de aleatoriedad en [0,1]. 0 = codicioso, 1 = aleatorio.

    Retorna
    -------
    dict o None
        Candidato seleccionado de la RCL o None si la lista de candidatos está vacía.

    Notas
    -----
    - Si todos los candidatos tienen el mismo costo, se selecciona uno al azar.
    - Esta función es útil en heurísticas GRASP para problemas de asignación o ruteo.
    """
    if not cands:
        return None
    costs = [c['arrival'].timestamp() for c in cands]
    if len(costs) == 1:
        return cands[0]
    min_cost, max_cost = min(costs), max(costs)
    if min_cost == max_cost:
        return random.choice(cands)
    thr = max_cost - ALPHA * (max_cost - min_cost)
    RCL = [c for c, cost in zip(cands, costs) if cost <= thr]
    return random.choice(RCL)


'''
AUXILIARY METHODS FOR MAIN ALGORITHMS
'''
def init_drivers_wrapper(df_labors, directorio_df, ciudad, assignment_type="algorithm", max_drivers=None):
    if assignment_type == "algorithm":
        return init_drivers(df_labors, directorio_df, ciudad, ignore_schedule=True, max_drivers=max_drivers)
    elif assignment_type == "historic":
        return get_historic_drivers(df_labors)
    else:
        raise ValueError(f"Unknown assignment mode: {assignment_type}")


def _init_drivers(df_labores, directorio_df, ciudad='BOGOTA', ignore_schedule=False):
    """
    Inicializa la posición y disponibilidad inicial de los conductores para la simulación.

    Parámetros
    ----------
    df_labores : pd.DataFrame
        DataFrame con las labores programadas, debe incluir la columna 'schedule_date'.
    df_directorio : pd.DataFrame
        DataFrame con la información de los conductores, incluyendo:
        - 'city' (ciudad de operación)
        - 'ALFRED'S' (nombre del conductor)
        - 'latitud', 'longitud' (coordenadas iniciales)
        - 'start_time' (hora de inicio de jornada, formato 'HH:MM:SS')
    ciudad : str, opcional
        Ciudad para filtrar conductores (por defecto 'BOGOTA').
    ignore_schedule : bool, opcional
        Si True, los conductores estarán disponibles desde medianoche (00:00:00).

    Retorna
    -------
    dict
        Diccionario de conductores, cada clave es el nombre del conductor y el valor es un dict con:
        - 'position' : str → coordenadas iniciales en formato WKT 'POINT (lon lat)'
        - 'available' : pd.Timestamp → hora inicial de disponibilidad con zona horaria
        - 'work_start' : datetime.time → hora de inicio de jornada
    """
    
    if df_labores.empty:
        return {}
    
    # Zona horaria a partir de las labores
    tz = df_labores['schedule_date'].dt.tz
    if tz is None:
        raise ValueError("La columna 'schedule_date' no tiene zona horaria asignada.")
    
    # Fecha mínima de las labores
    first_date = df_labores['schedule_date'].dt.date.min()
    if pd.isna(first_date):
        raise ValueError("No se pudo determinar la fecha mínima en 'schedule_date'.")
    
    # Filtrar conductores por ciudad
    df_ciudad = directorio_df[directorio_df['city'] == ciudad]
    
    conductores = {}
    for _, conductor in df_ciudad.iterrows():
        
        if pd.isna(conductor['latitud']) or pd.isna(conductor['longitud']):
            continue
        
        if ignore_schedule:
            hora_inicio = time(0, 0, 0)  # medianoche
        else:
            try:
                hora_inicio = datetime.strptime(conductor['start_time'], '%H:%M:%S').time()
            except ValueError:
                raise ValueError(f'Formato inválido de hora para el conductor {conductor["ALFRED'S"]}')
        
        disponibilidad = datetime.combine(first_date, hora_inicio)
        
        conductores[conductor["ALFRED'S"]] = {
            'position': f"POINT ({conductor['longitud']} {conductor['latitud']})",
            'available': pd.Timestamp(disponibilidad).tz_localize(tz),
            'work_start': hora_inicio
        }
    
    return conductores


def init_drivers(df_labores: pd.DataFrame,
                 directorio_df: pd.DataFrame,
                 ciudad: str = 'BOGOTA',
                 ignore_schedule: bool = False,
                 max_drivers = None) -> dict:
    """
    Inicializa la posición y disponibilidad inicial de los conductores para la simulación.

    Parámetros
    ----------
    df_labores : pd.DataFrame
        DataFrame con las labores programadas, debe incluir la columna 'schedule_date'.
    directorio_df : pd.DataFrame
        DataFrame con la información de los conductores, incluyendo:
        - 'city' (ciudad de operación)
        - 'ALFRED'S' (nombre del conductor)
        - 'latitud', 'longitud' (coordenadas iniciales)
        - 'start_time' (hora de inicio de jornada, formato 'HH:MM:SS')
    ciudad : str, opcional
        Ciudad para filtrar conductores (por defecto 'BOGOTA').
    ignore_schedule : bool, opcional
        Si True, los conductores estarán disponibles desde medianoche (00:00:00).
    max_drivers : int o None, opcional
        Número máximo de conductores a incluir. 
        - Si None, se incluyen todos los disponibles en la ciudad.
        - Si menor que el total, se seleccionan los primeros `n`.
        - Si mayor o igual al total, se incluyen todos.

    Retorna
    -------
    dict
        Diccionario de conductores, cada clave es el nombre del conductor y el valor es un dict con:
        - 'position' : str → coordenadas iniciales en formato WKT 'POINT (lon lat)'
        - 'available' : pd.Timestamp → hora inicial de disponibilidad con zona horaria
        - 'work_start' : datetime.time → hora de inicio de jornada
    """
    
    if df_labores.empty:
        return {}
    
    # Zona horaria a partir de las labores
    tz = df_labores['schedule_date'].dt.tz
    if tz is None:
        raise ValueError("La columna 'schedule_date' no tiene zona horaria asignada.")
    
    # Fecha mínima de las labores
    first_date = df_labores['schedule_date'].dt.date.min()
    if pd.isna(first_date):
        raise ValueError("No se pudo determinar la fecha mínima en 'schedule_date'.")
    
    # Filtrar conductores por ciudad
    df_ciudad = directorio_df[directorio_df['city'] == ciudad].copy()

    # Limitar número de conductores si corresponde
    if max_drivers is not None and max_drivers < len(df_ciudad):
        df_ciudad = df_ciudad.iloc[:max_drivers]
    
    conductores = {}
    for _, conductor in df_ciudad.iterrows():
        if pd.isna(conductor['latitud']) or pd.isna(conductor['longitud']):
            continue
        
        if ignore_schedule:
            hora_inicio = time(0, 0, 0)  # medianoche
        else:
            try:
                hora_inicio = datetime.strptime(conductor['start_time'], '%H:%M:%S').time()
            except ValueError:
                raise ValueError(f'Formato inválido de hora para el conductor {conductor["ALFRED'S"]}')
        
        disponibilidad = datetime.combine(first_date, hora_inicio)
        
        conductores[conductor["ALFRED'S"]] = {
            'position': f"POINT ({conductor['longitud']} {conductor['latitud']})",
            'available': pd.Timestamp(disponibilidad).tz_localize(tz),
            'work_start': hora_inicio
        }
    
    
    return conductores


def prepare_iteration_data(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    return (
        df_cleaned.copy()
        .reset_index()
        .sort_values("schedule_date", kind="stable")  # stable keeps tie order
        .reset_index(drop=True)
    )



def assign_task_to_driver(  driver_data: Dict, 
                            arrival: pd.Timestamp, 
                            early: pd.Timestamp,
                            start_point: str, 
                            end_point: str,
                            is_last_in_service: bool, 
                            TIEMPO_ALISTAR: int,
                            tiempo_finalizacion: int, 
                            vehicle_speed: float,
                            method: str, 
                            dist_dict = None
                          ) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Asigna una tarea a un conductor y actualiza su disponibilidad y posición.

    Retorna
    -------
    (inicio, fin) : tuple of pd.Timestamp
    """
    astart = max(arrival, early)
    d2 = distance(start_point, end_point, method=method, dist_dict=dist_dict)
    dur = TIEMPO_ALISTAR + (0 if math.isnan(d2) else d2/vehicle_speed*60) \
          + (tiempo_finalizacion if not is_last_in_service else 0)
    aend = astart + timedelta(minutes=dur)

    driver_data['available'] = aend
    driver_data['position'] = end_point
    return astart, aend


'''
SOLUTION RECONSTRUCTION METHODS
'''
def get_historic_drivers(df_cleaned: pd.DataFrame) -> dict:
    """
    Inicializa los conductores a partir de la asignación histórica (df_cleaned).
    
    Cada conductor inicia disponible a medianoche (00:00:00) de la primera fecha
    presente en df_cleaned. La posición inicial se toma como el primer punto
    registrado (columna 'addres_point') para cada conductor.
    
    Parámetros
    ----------
    df_cleaned : pd.DataFrame
        DataFrame con las labores reales, debe contener:
        - 'schedule_date' (datetime con tz)
        - 'alfred' (código único del conductor)
        - 'addres_point' (posición inicial en WKT, ej. 'POINT (lon lat)')

    Retorna
    -------
    dict
        Diccionario de conductores, cada clave es el código 'alfred' y el valor un dict con:
        - 'position' : str → coordenadas iniciales en formato WKT
        - 'available' : pd.Timestamp → hora inicial de disponibilidad (tz-aware)
        - 'work_start' : datetime.time → hora de inicio de jornada (00:00:00 fijo)
    """
    
    # --- Validaciones básicas ---
    required_cols = {"schedule_date", "alfred", "address_point"}
    missing_cols = required_cols - set(df_cleaned.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas en df_cleaned: {missing_cols}")

    if df_cleaned.empty:
        return {}

    # Verificar que schedule_date tenga zona horaria
    tz = df_cleaned["schedule_date"].dt.tz
    if tz is None:
        raise ValueError("La columna 'schedule_date' no tiene zona horaria asignada.")

    # Obtener fecha mínima
    first_date = df_cleaned["schedule_date"].dt.date.min()
    if pd.isna(first_date):
        raise ValueError("No se pudo determinar la fecha mínima en 'schedule_date'.")

    # --- Construcción de conductores ---
    conductores = {}
    for driver, df_driver in df_cleaned.groupby("alfred"):
        # Posición inicial: primer punto válido
        pos_list = df_driver["address_point"].dropna().unique().tolist()
        if not pos_list:
            continue  # ignorar conductor sin posición
        position = pos_list[0]

        # Disponibilidad: medianoche de la fecha mínima
        hora_inicio = time(0, 0, 0)
        disponibilidad = datetime.combine(first_date, hora_inicio)

        conductores[driver] = {
            "position": position,
            "available": pd.Timestamp(disponibilidad).tz_localize(tz),
            "work_start": hora_inicio,
        }

    return conductores


def get_driver_wrapper(drivers: Dict[str, Dict],
                row: pd.Series,
                prev_end: pd.Timestamp,
                TIEMPO_PREVIO: int,
                TIEMPO_GRACIA: int,
                ALFRED_SPEED: float,
                method: str,
                dist_dict: dict,
                ALPHA: float,
                assignment_type: str = 'algorithm') -> Dict[str, Any]:
    """
    Wrapper para seleccionar un conductor, usando ya sea:
    - El algoritmo (RCL con get_candidate_drivers + select_from_candidates), o
    - El conductor real asignado (get_real_driver).

    Parámetros
    ----------
    drivers : dict
        Diccionario de conductores inicializados.
    row : pd.Series
        Fila de la labor, debe incluir 'alfred' si use_real=True.
    prev_end : Timestamp
        Hora de finalización del servicio previo.
    TIEMPO_PREVIO : int
        Minutos antes de la hora programada para permitir llegada anticipada.
    TIEMPO_GRACIA : int
        Minutos de tolerancia para llegada tardía.
    ALFRED_SPEED : float
        Velocidad media de movimiento en km/h.
    method : str
        Método de cálculo de distancias.
    dist_dict : dict
        Diccionario de distancias precalculadas.
    ALPHA : float
        Parámetro GRASP para seleccionar en modo algoritmo.
    use_real : bool, default False
        Si True → usar el conductor real (columna 'alfred').
        Si False → usar la lógica de candidatos del algoritmo.

    Retorna
    -------
    dict o None
        - Modo real: {'drv': alfred_code, 'arrival': timestamp} o None
        - Modo algoritmo: candidato elegido de la RCL o None
    """

    if assignment_type == 'historic':
        return get_historic_driver(
            drivers, row, prev_end,
            TIEMPO_PREVIO, TIEMPO_GRACIA,
            ALFRED_SPEED, method, dist_dict
        )
    elif assignment_type == 'algorithm':
        cands = get_candidate_drivers(
            drivers, row, prev_end,
            TIEMPO_PREVIO, TIEMPO_GRACIA,
            ALFRED_SPEED, method, dist_dict
        )
        return select_from_candidates(cands, ALPHA)
    else:
        raise ValueError(f"Unknown assignment mode: {assignment_type}")


def get_historic_driver(drivers: Dict[str, Dict], 
                    row: pd.Series, 
                    prev_end: pd.Timestamp, 
                    TIEMPO_PREVIO: int,
                    TIEMPO_GRACIA: int,
                    ALFRED_SPEED: float, 
                    method: str, 
                    dist_dict: dict) -> Dict[str, Any]:
    """
    Obtiene el conductor REAL asignado a un servicio y calcula su hora estimada de llegada,
    respetando la misma lógica de ventanas de tiempo usada en get_candidate_drivers.

    Parámetros
    ----------
    drivers : dict
        Diccionario de conductores (como en init_drivers o get_historic_drivers).
    row : pd.Series
        Fila con información de la labor, debe incluir 'alfred', 'map_start_point' y 'schedule_date'.
    prev_end : pd.Timestamp
        Fin del servicio anterior del mismo service_id (o None si es el primero).
    TIEMPO_PREVIO : int
        Minutos antes de la hora programada para permitir llegada anticipada.
    TIEMPO_GRACIA : int
        Minutos de tolerancia para llegada tardía.
    ALFRED_SPEED : float
        Velocidad promedio del conductor en km/h.
    method : str
        Método de cálculo de distancias ('osrm', 'haversine', etc).
    dist_dict : dict
        Diccionario de distancias precalculadas (cuando aplique).

    Retorna
    -------
    dict o None
        Diccionario con {'drv': alfred_code, 'arrival': timestamp} 
        o None si no se encuentra al conductor o si llega demasiado tarde.
    """

    # Conductor real asignado
    drv_code = row.get("alfred")
    if drv_code not in drivers:
        return None
    drv = drivers[drv_code]

    sched = row['schedule_date']

    # Ventana de tiempo (igual que en get_candidate_drivers)
    early = prev_end or (sched - timedelta(minutes=TIEMPO_PREVIO))
    late  = (prev_end + timedelta(minutes=TIEMPO_GRACIA)) if prev_end else (sched + timedelta(minutes=TIEMPO_GRACIA))

    # Disponibilidad del conductor
    av = drv['available']
    if av.time() < drv['work_start']:
        av = pd.Timestamp(datetime.combine(av.date(), drv['work_start']), tz=av.tz)

    # Distancia hasta el punto inicial de la labor
    dkm = distance(drv['position'], row['map_start_point'], method=method, dist_dict=dist_dict)

    # Tiempo de viaje en minutos
    travel_min = 0 if math.isnan(dkm) else dkm / ALFRED_SPEED * 60

    # Hora de llegada
    arrival = av + timedelta(minutes=travel_min)

    # Validar contra ventana de tolerancia
    if arrival <= late:
        return {'drv': drv_code, 'arrival': arrival}
    else:
        return None



'''
AUXILIARY FUNCTIONS
'''
def build_driver_movements(df_result: pd.DataFrame,
                           directorio_df: pd.DataFrame,
                           day_str: str,
                           DISTANCE_METHOD: str,
                           ALFRED_SPEED: float,
                           ciudad: str,
                           assignment_type: str = 'algorithm',
                           dist_dict: dict = None) -> pd.DataFrame:
    """
    Reconstruye actividades de tipo DRIVER_MOVE y FREE_TIME a partir de las labores,
    incluyendo la distancia recorrida (km) en los movimientos.

    Parámetros
    ----------
    df_result : pd.DataFrame
        DataFrame de labores.
        - Simulación: ['assigned_driver', 'actual_start', 'actual_end', ...]
        - Real: ['alfred', 'labor_start_date', 'labor_end_date', ...]
    directorio_df : pd.DataFrame o None
        Solo necesario en modo simulado. Directorio con columnas
        ['ALFRED'S', 'latitud', 'longitud', 'city', 'start_time'].
    day_str : str
        Día de la simulación en formato 'YYYY-MM-DD'.
    DISTANCE_METHOD : str
        Método de cálculo de distancias.
    ALFRED_SPEED : float
        Velocidad media de los conductores (km/h).
    ciudad : str
        Ciudad a procesar (solo en modo simulado).
    assignment_type : str, default 'algorithm'
        'algorithm' → usa columnas simuladas, 'historic' → usa asignación real.
    dist_dict : dict, optional
        Diccionario de distancias precalculadas.

    Retorna
    -------
    pd.DataFrame
        DataFrame con labores originales + movimientos + tiempos libres,
        con columnas adicionales 'duration_min' y 'distance_km'.
    """

    # --- 1. Mapear nombres de columnas según origen ---
    if assignment_type == 'algorithm':
        driver_col, start_col, end_col = "assigned_driver", "actual_start", "actual_end"
    elif assignment_type == 'historic':
        driver_col, start_col, end_col = "historic_driver", "historic_start", "historic_end"
    else:
        raise ValueError("assignment_type debe ser 'historic' o 'algorithm'")

    records = []

    # --- 2. Zona horaria ---
    tz = df_result['schedule_date'].dt.tz
    if tz is None and not df_result[start_col].dropna().empty:
        tz = df_result[start_col].dropna().iloc[0].tz

    # --- 3. Fecha base ---
    valid_dates = df_result['schedule_date'].dropna().dt.date
    first_day = pd.to_datetime(day_str).date() if valid_dates.empty else valid_dates.min()

    # --- 4. Inicializar posición/hora de inicio ---
    driver_pos, driver_end = {}, {}

    if assignment_type == 'algorithm':
        df_dir_city = directorio_df[directorio_df['city'] == ciudad]
        for _, d in df_dir_city.iterrows():
            if pd.isna(d['latitud']):
                continue
            name = d["ALFRED'S"]
            driver_pos[name] = f"POINT ({d['longitud']} {d['latitud']})"
            st = datetime.combine(first_day, datetime.strptime(d['start_time'], '%H:%M:%S').time())
            driver_end[name] = pd.Timestamp(st, tz=tz)
    else:
        for drv in df_result[driver_col].dropna().unique():
            first_row = (
                df_result[df_result[driver_col] == drv]
                .dropna(subset=[start_col])
                .sort_values(start_col)
                .iloc[0]
            )
            driver_pos[drv] = first_row['map_start_point']
            driver_end[drv] = first_row[start_col]

    # --- 5. Recorrer labores ordenadas ---
    for _, row in df_result.dropna(subset=[start_col]).sort_values(start_col).iterrows():
        drv = row[driver_col]

        if pd.notna(drv) and row['labor_category'] == 'VEHICLE_TRANSPORTATION':
            prev_e, prev_p = driver_end[drv], driver_pos[drv]

            if row[start_col] > prev_e:
                # Distancia entre ubicaciones
                dkm = distance(prev_p, row['map_start_point'], method=DISTANCE_METHOD, dist_dict=dist_dict)
                t_move = timedelta(minutes=(0 if math.isnan(dkm) else dkm / ALFRED_SPEED * 60))
                depart = max(prev_e, row[start_col] - t_move)

                # FREE_TIME
                if depart > prev_e:
                    records.append({
                        'service_id': row['service_id'],
                        'labor_id': f"{int(row['labor_id'])}_free",
                        'labor_name': 'FREE_TIME',
                        'labor_category': 'FREE_TIME',
                        driver_col: drv,
                        'schedule_date': row['schedule_date'],
                        start_col: prev_e,
                        end_col: depart,
                        'start_point': prev_p,
                        'end_point': prev_p,
                        'distance_km': 0.0
                    })

                # DRIVER_MOVE
                records.append({
                    'service_id': row['service_id'],
                    'labor_id': f"{int(row['labor_id'])}_move",
                    'labor_name': 'DRIVER_MOVE',
                    'labor_category': 'DRIVER_MOVE',
                    driver_col: drv,
                    'schedule_date': row['schedule_date'],
                    start_col: depart,
                    end_col: row[start_col],
                    'start_point': prev_p,
                    'end_point': row['map_start_point'],
                    'distance_km': dkm
                })

        # Labor original (no distancia → NaN)
        records.append({
            'service_id': row['service_id'],
            'labor_id': int(row['labor_id']),
            'labor_name': row['labor_name'],
            'labor_category': row['labor_category'],
            driver_col: drv,
            'schedule_date': row['schedule_date'],
            start_col: row[start_col],
            end_col: row[end_col],
            'start_point': row['map_start_point'],
            'end_point': row['map_end_point'],
            'distance_km': float("nan")
        })

        # Actualizar estado del conductor
        if pd.notna(drv) and pd.notna(row[end_col]):
            driver_end[drv] = row[end_col]
            driver_pos[drv] = row['map_end_point']

    # --- 6. Convertir a DataFrame y calcular duración ---
    df_moves = pd.DataFrame(records)
    if not df_moves.empty:
        df_moves[start_col] = pd.to_datetime(df_moves[start_col])
        df_moves[end_col] = pd.to_datetime(df_moves[end_col])
        df_moves['duration_min'] = (
            (df_moves[end_col] - df_moves[start_col]).dt.total_seconds() / 60
        ).round(1)

    return df_moves





def remove_drivers(df, driver_names):
    """
    Devuelve una copia del DataFrame excluyendo a los conductores indicados.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con información de labores.
    driver_names : str o list
        Nombre(s) de los conductores a eliminar.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame sin los conductores especificados.
    """
    if isinstance(driver_names, str):
        driver_names = [driver_names]
    return df[~df["ALFRED'S"].isin(driver_names)].copy()


def compute_avg_times(df: pd.DataFrame) -> dict:
    """
    Recalcula los tiempos promedio por tipo de labor a partir de las fechas de inicio y fin.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'labor_name', 'labor_start_date', 'labor_end_date'.

    Retorna
    -------
    dict
        Diccionario {labor_name: promedio_duracion_en_minutos}.
    """
    df_temp = (
        df.dropna(subset=['labor_start_date','labor_end_date'])
          .assign(duration_td=lambda d: d['labor_end_date'] - d['labor_start_date'])
    )
    df_temp = df_temp[df_temp['duration_td'] <= pd.Timedelta(days=1)]
    df_temp['duration_min'] = df_temp['duration_td'].dt.total_seconds() / 60
    avg_times_map = df_temp.groupby('labor_name')['duration_min'].mean().to_dict()
    return avg_times_map



