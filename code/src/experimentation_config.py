import pandas as pd
import numpy as np

# Sarario mínimo: https://tickelia.com/co/blog/actualidad/salario-minimo-colombia/#:~:text=Esto%20significa%20que%2C%20aunque%20el,de%20prestaciones%20y%20seguridad%20social.
# Horas extras: https://actualicese.com/horas-extra-y-recargos/?srsltid=AfmBOoox01zXLaHcVGSO28a38fJRnOZ3zLVS9qOXpCZ3ZeGxS8gn8tyu


hyperparameter_selection = {
    'driver_distance':      0.2,
    'driver_extra_time':    0.2,
    'hybrid':               0.2
}

max_iterations = {
    '149': 1000,
        '1': 750,
        '1004': 750,
        '126': 100,
        '150': 100,
        '844': 100,
        '830': 100 ,
}

max_iterations = {city:int(max_iter/50) for city,max_iter in max_iterations.items()}

iterations_nums = {city:[int(p * max_iterations[city]) for p in np.linspace(0.2, 1.0, 4)] for city in max_iterations}

def instance_map(instance_name: str) -> str:
    """
    Returns the instance type ("artif" or "real") based on its naming convention.

    Rules:
        - Instances starting with 'instA' → 'artif' (artificial)
        - Instances starting with 'instR' → 'real' (real)
        - Raises ValueError if pattern not recognized.

    Examples:
        >>> get_instance_type("instAD1")
        'artif'
        >>> get_instance_type("instRS3")
        'real'
    """
    if not instance_name.startswith("inst") or len(instance_name) < 5:
        raise ValueError(f"Invalid instance name: {instance_name}")

    code_letter = instance_name[4].upper()

    if code_letter == "A":
        return "artif"
    elif code_letter == "R":
        return "real"
    else:
        raise ValueError(f"Unrecognized instance type in: {instance_name}")


def fechas_map(instance_name: str) -> str:
    """
    Returns the instance type ("artif" or "real") based on its naming convention.

    Rules:
        - Instances starting with 'instA' → 'artif' (artificial)
        - Instances starting with 'instR' → 'real' (real)
        - Raises ValueError if pattern not recognized.

    Examples:
        >>> get_instance_type("instAD1")
        'artif'
        >>> get_instance_type("instRS3")
        'real'
    """
    if not instance_name.startswith("inst") or len(instance_name) < 5:
        raise ValueError(f"Invalid instance name: {instance_name}")

    code_letter = instance_name[4].upper()

    if code_letter == "A":
        return pd.date_range("2026-01-05", "2026-01-11").strftime("%Y-%m-%d").tolist()
    elif code_letter == "R":
        return pd.date_range("2025-07-21", "2025-07-27").strftime("%Y-%m-%d").tolist()
    else:
        raise ValueError(f"Unrecognized instance type in: {instance_name}")


