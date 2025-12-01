import json
import os

from src.config.config import REPO_PATH

def define_run_algorithm(decision_time):
    if decision_time[1] == '00:00':
        return 'OFFLINE'
    elif decision_time[1] == '11:00':
        return 'REACT_BUFFER'
    else:
        return 'BUFFER_FIXED'
    

def generate_alfred_parameters(
    decision_time,
    **kwargs):

    parameters = {
        'previous_run': decision_time[0],
        'current_time': decision_time[1],
        'algorithm': decision_time
    }

    save_path = os.path.join(
        REPO_PATH,
        'resultados',
        kwargs['experiment_type'],
        kwargs['dist_method'],
        


    )

    with open(f'ALFRED_params.json', 'w') as fp:
        json.dump(parameters, fp)

