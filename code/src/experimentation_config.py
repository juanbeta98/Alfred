import pandas as pd

### Costs
SALARY = 47450
EXTRA_TIME = (SALARY/40) * 1.25
# MOVE_COST = {'149':3193,
#              '1':4543,
#              '1004':4404,
#              ''}

# Sarario mínimo: https://tickelia.com/co/blog/actualidad/salario-minimo-colombia/#:~:text=Esto%20significa%20que%2C%20aunque%20el,de%20prestaciones%20y%20seguridad%20social.
# Horas extras: https://actualicese.com/horas-extra-y-recargos/?srsltid=AfmBOoox01zXLaHcVGSO28a38fJRnOZ3zLVS9qOXpCZ3ZeGxS8gn8tyu


max_drivers = {
    'inst3': {
        '149':[19, 19, 18, 16, 20, 20, 20],
        '1':[7,7,7,7,6,7,6]
        },
    'inst4a': {
        '149':[19, 20, 20, 20, 19, 18, 16],
        '1':[6,7,6,7,7,7,7]}
        ,
    'instAS1': {
        '149':[18, 19, 19, 19, 19, 18, 15],
        '1':[6,7,6,7,7,7,7]
        },
    'instAD1':{
        '149':[18, 19, 19, 19, 19, 18, 15],
        '1':[6,7,6,7,7,7,7]
        },

    'instRS1': {
        '149':[17, 16, 15, 17, 18, 9, 0],
        '1':[5,5,5,6,6,3,0]
        },
    'instRD1':{
        '149':[17, 16, 15, 17, 18, 9, 0],
        '1':[5,5,5,6,6,3,0]
        }
}

instance_map = {
    'inst3':'artif',
    'inst4a':'artif',
    'instAS1':'artif',
    'instAD1':'artif',

    'instRS1':'real',
    'instRD1':'real'
}

instance_map.update({f'{key}-{optimization_variable}':value for key,value in instance_map.items() for optimization_variable in ['driver_move_distance', "driver_extra_time" ,"hybrid"]})

fechas_dict = {
    'inst3':pd.date_range("2025-09-08", "2025-09-14").strftime("%Y-%m-%d").tolist(),
    'inst4a':pd.date_range("2026-01-05", "2026-01-11").strftime("%Y-%m-%d").tolist(),
    'instAS1':pd.date_range("2026-01-05", "2026-01-11").strftime("%Y-%m-%d").tolist(),
    'instAD1':pd.date_range("2026-01-05", "2026-01-11").strftime("%Y-%m-%d").tolist(),
               
    'instRS1':pd.date_range("2025-07-21", "2025-07-27").strftime("%Y-%m-%d").tolist(),
    'instRD1':pd.date_range("2025-07-21", "2025-07-27").strftime("%Y-%m-%d").tolist()
}
