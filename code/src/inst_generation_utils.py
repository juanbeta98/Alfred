import pandas as pd
import numpy as np


def top_service_days(df, 
                     city_col="city", 
                     date_col="labor_start_date", 
                     top_n=7,
                     starting_year: int = 2025):
    """
    Encuentra los días con mayor número de servicios para cada ciudad.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas `service_id`, ciudad (`city_col`) y fecha (`date_col`).
    city_col : str, default="city"
        Nombre de la columna con el código de ciudad.
    date_col : str, default="labor_start_date"
        Nombre de la columna con la fecha de inicio del labor.
    top_n : int, default=7
        Número de días con más servicios que se desean extraer por ciudad.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas ['city', 'rank', 'date', '# services'].
    """

    # Asegurar datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Filtrar por año
    df = df[df['schedule_date'].dt.year >= starting_year]

    # Evitar contar el mismo servicio varias veces si tiene múltiples labores
    df_unique = df.drop_duplicates(subset=["service_id", city_col])

    # Contar servicios por ciudad y fecha
    daily_counts = (
        df_unique.groupby([city_col, df_unique[date_col].dt.date])["service_id"]
        .nunique()
        .reset_index(name="# services")
        .rename(columns={date_col: "date"})
    )

    # Ranking top_n por ciudad
    daily_counts["rank"] = (
        daily_counts.groupby(city_col)["# services"]
        .rank(method="first", ascending=False)
    )

    # Filtrar solo los top_n
    top_days = daily_counts[daily_counts["rank"] <= top_n].copy()
    top_days["rank"] = top_days["rank"].astype(int)

    # Ordenar para mejor lectura
    top_days = top_days.sort_values([city_col, "rank"]).reset_index(drop=True)

    return top_days


def _enforce_tz(s: pd.Series, tz: str) -> pd.Series:
    """Asegura que una serie datetime sea tz-aware en la zona horaria indicada."""
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize(tz)
    return s.dt.tz_convert(tz)


def _shift_to_new_day(orig_ts, new_day, tz: str):
    """
    Mueve un timestamp a un nuevo día calendario, preservando la hora local.
    Maneja correctamente objetos numpy.datetime64 convirtiéndolos a Timestamp.

    Parámetros
    ----------
    orig_ts : pd.Timestamp | datetime-like
        Timestamp original.
    new_day : datetime-like
        Nuevo día base (sin importar si viene como numpy.datetime64 o Timestamp).
    tz : str
        Zona horaria destino.

    Retorna
    -------
    pd.Timestamp | NaT
        Timestamp ajustado al nuevo día en la zona horaria indicada.
    """
    if pd.isna(orig_ts) or pd.isna(new_day):
        return pd.NaT

    orig_ts = pd.Timestamp(orig_ts)
    new_day = pd.Timestamp(new_day)  # evita error con numpy.datetime64

    # Ajustar original al tz
    if orig_ts.tzinfo is None:
        orig_local = orig_ts.tz_localize(tz)
    else:
        orig_local = orig_ts.tz_convert(tz)

    # Base del nuevo día a medianoche
    base = new_day
    if base.tzinfo is None:
        base = base.tz_localize(tz)
    else:
        base = base.tz_convert(tz)
    base = base.normalize()

    shifted = base + pd.Timedelta(
        hours=orig_local.hour,
        minutes=orig_local.minute,
        seconds=orig_local.second,
        microseconds=orig_local.microsecond
    )
    return shifted


def create_artificial_week(
    df, top_days_df, city_col="city",
    start_col="labor_start_date", end_col="labor_end_date",
    schedule_col="schedule_date", seed=42,
    starting_date="2026-01-05"
):
    """
    Crea una semana artificial re-asignando los 7 días más cargados por ciudad
    a la semana que inicia el lunes 2025-09-08. Todas las fechas se fuerzan
    a la zona horaria America/Bogota.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original con schedule_date, labor_start_date y labor_end_date.
    top_days_df : pd.DataFrame
        DataFrame con los 7 días top por ciudad. Columnas mínimas: 
        [city, rank, date, '# services'].
    city_col : str, default="city"
        Columna de ciudad.
    start_col : str, default="labor_start_date"
        Columna de inicio de labor.
    end_col : str, default="labor_end_date"
        Columna de fin de labor.
    schedule_col : str, default="schedule_date"
        Columna de fecha de programación.
    seed : int, default=42
        Semilla para reproducibilidad del shuffle.

    Retorna
    -------
    df_artificial : pd.DataFrame
        DataFrame con fechas reubicadas en la semana artificial.
    mapping_df : pd.DataFrame
        Mapeo de fechas originales a fechas artificiales por ciudad.
    """
    tz = "America/Bogota"

    # Asegurar datetimes con TZ Bogotá
    df[start_col] = _enforce_tz(df[start_col], tz)
    if end_col in df.columns:
        df[end_col] = _enforce_tz(df[end_col], tz)
    if schedule_col in df.columns:
        df[schedule_col] = _enforce_tz(df[schedule_col], tz)

    # Semana artificial (Lun–Dom desde 2025-09-08) → con TZ Bogotá
    artificial_week = pd.date_range(starting_date, periods=7, freq="D", tz=tz)

    rng = np.random.default_rng(seed)
    mappings = []
    df_list = []

    # Procesar por ciudad
    for city in top_days_df[city_col].unique():
        # Forzar fechas originales a tz-aware Bogotá
        city_days = (
            pd.to_datetime(top_days_df[top_days_df[city_col] == city]["date"])
            .dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
            .dt.normalize()
            .tolist()
        )

        # Asignación aleatoria de los 7 días a la semana artificial
        shuffled_targets = rng.choice(artificial_week.to_pydatetime(), size=len(city_days), replace=False)

        city_mapping = pd.DataFrame({
            city_col: city,
            "original_date": city_days,
            "artificial_date": shuffled_targets
        })
        mappings.append(city_mapping)

        # Aplicar el mapeo
        for orig, new in zip(city_days, shuffled_targets):
            mask = (df[city_col] == city) & (df[start_col].dt.normalize() == orig)
            df_temp = df.loc[mask].copy()

            df_temp[start_col] = df_temp[start_col].apply(lambda x: _shift_to_new_day(x, new, tz))
            if end_col in df.columns:
                df_temp[end_col] = df_temp[end_col].apply(lambda x: _shift_to_new_day(x, new, tz))
            if schedule_col in df.columns:
                df_temp[schedule_col] = df_temp[schedule_col].apply(lambda x: _shift_to_new_day(x, new, tz))

            df_list.append(df_temp)

    df_artificial = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    mapping_df = pd.concat(mappings, ignore_index=True)

    return df_artificial, mapping_df


def create_artificial_dynamic_week(
    df, top_days_df, city_col="city",
    start_col="labor_start_date", end_col="labor_end_date",
    schedule_col="schedule_date", created_col="created_at",
    seed=42, starting_date="2026-01-05"
):
    """
    Create an artificial week by reassigning the 7 busiest days (per city)
    to a new week starting at `starting_date`. All dates are localized to
    America/Bogota timezone. 

    Additionally, the 'created_at' column is adjusted so that the relative 
    difference between schedule_date and created_at is preserved. This ensures 
    that if a service was created X days before the schedule date, it will also 
    appear X days before the new artificial schedule date.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe containing schedule_date, labor_start_date, 
        labor_end_date, and optionally created_at.
    top_days_df : pd.DataFrame
        DataFrame with the top 7 busiest days per city.
        Must contain columns: [city, rank, date, '# services'].
    city_col : str, default="city"
        Column that identifies the city.
    start_col : str, default="labor_start_date"
        Column with labor start times.
    end_col : str, default="labor_end_date"
        Column with labor end times.
    schedule_col : str, default="schedule_date"
        Column with scheduled dates.
    created_col : str, default="created_at"
        Column with creation timestamps.
    seed : int, default=42
        Random seed for reproducibility of the mapping.
    starting_date : str, default="2026-01-05"
        Start date for the artificial week (Monday).

    Returns
    -------
    df_artificial : pd.DataFrame
        DataFrame with rescheduled artificial week dates.
    mapping_df : pd.DataFrame
        Mapping of original dates to artificial dates per city.

    Notes
    -----
    - The relative delta between created_at and schedule_date is preserved.
    - Example: if a service was created 2 days before its schedule_date, 
      it will remain 2 days before the new artificial schedule_date.
    """
    tz = "America/Bogota"

    # Ensure datetime with Bogotá TZ
    df[start_col] = _enforce_tz(df[start_col], tz)
    if end_col in df.columns:
        df[end_col] = _enforce_tz(df[end_col], tz)
    if schedule_col in df.columns:
        df[schedule_col] = _enforce_tz(df[schedule_col], tz)
    if created_col in df.columns:
        df[created_col] = _enforce_tz(df[created_col], tz)

    # Build artificial week (Mon–Sun) starting from given date
    artificial_week = pd.date_range(starting_date, periods=7, freq="D", tz=tz)

    rng = np.random.default_rng(seed)
    mappings = []
    df_list = []

    # Process city by city
    for city in top_days_df[city_col].unique():
        # Extract original busy days for this city
        city_days = (
            pd.to_datetime(top_days_df[top_days_df[city_col] == city]["date"])
            .dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
            .dt.normalize()
            .tolist()
        )

        # Randomly assign them to new artificial week
        shuffled_targets = rng.choice(
            artificial_week.to_pydatetime(), 
            size=len(city_days), 
            replace=False
        )

        # Build mapping record
        city_mapping = pd.DataFrame({
            city_col: city,
            "original_date": city_days,
            "artificial_date": shuffled_targets
        })
        mappings.append(city_mapping)

        # Apply the mapping
        for orig, new in zip(city_days, shuffled_targets):
            mask = (df[city_col] == city) & (df[start_col].dt.normalize() == orig)
            df_temp = df.loc[mask].copy()

            # --- preserve deltas BEFORE shifting schedule ---
            if created_col in df.columns:
                original_sched = df.loc[mask, schedule_col]
                original_created = df.loc[mask, created_col]
                deltas = original_sched - original_created
            else:
                deltas = None

            # --- shift schedule/start/end to new artificial date ---
            df_temp[start_col] = df_temp[start_col].apply(lambda x: _shift_to_new_day(x, new, tz))
            if end_col in df.columns:
                df_temp[end_col] = df_temp[end_col].apply(lambda x: _shift_to_new_day(x, new, tz))
            if schedule_col in df.columns:
                df_temp[schedule_col] = df_temp[schedule_col].apply(lambda x: _shift_to_new_day(x, new, tz))

            # --- shift created_at keeping relative offset ---
            if created_col in df.columns and deltas is not None:
                df_temp[created_col] = df_temp[schedule_col] - deltas.values

            df_list.append(df_temp)

    # Final outputs
    df_artificial = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    mapping_df = pd.concat(mappings, ignore_index=True)

    return df_artificial, mapping_df



def split_static_dynamic(df_artificial: pd.DataFrame,
                         city_col: str = "city",
                         schedule_col: str = "schedule_date",
                         created_col: str = "created_at"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split artificial labors into static and dynamic subsets:
    - Static: created before the schedule_date (any number of days earlier).
    - Dynamic: created on the same day as the schedule_date.
    Comparison is done at calendar day level (ignores hours/minutes).

    Also prints a summary per city and schedule_date with counts.
    
    Returns
    -------
    labors_inst_static_df, labors_inst_dynamic_df : pd.DataFrame
    """
    df = df_artificial.copy()

    # Normalize to dates (strip time)
    sched_dates = df[schedule_col].dt.normalize()
    created_dates = df[created_col].dt.normalize()

    mask_dynamic = created_dates == sched_dates
    mask_static = created_dates < sched_dates

    labors_inst_dynamic_df = df[mask_dynamic].copy()
    labors_inst_static_df = df[mask_static].copy()

    # --- Summary per city/day ---
    summary = (
        df.assign(type=np.where(mask_dynamic, "dynamic", "static"))
          .groupby([city_col, "type"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )

    # Add totals row
    total_row = summary.drop(columns=[city_col]).sum().to_dict()
    total_row[city_col] = "ALL"
    total_row[schedule_col] = "ALL"
    summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

    print("\n📊 Static vs Dynamic services per city & day:\n")
    print(summary.to_string(index=False))

    return labors_inst_static_df, labors_inst_dynamic_df

