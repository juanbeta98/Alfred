import pandas as pd

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
