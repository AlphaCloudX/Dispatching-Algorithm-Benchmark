import os

import numpy as np
import yaml


def load_config_file(path: str) -> dict:
    if not os.path.isfile(path):
        return {}

    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def sample_distribution(distribution_config: dict, size=None):
    dist_type = distribution_config['type']

    if dist_type == 'normal':
        mean = distribution_config['mean']
        std = distribution_config['std']
        minimum = distribution_config['min']
        maximum = distribution_config['max']

        result = np.random.normal(loc=mean, scale=std, size=size)
        result = np.clip(result, minimum, maximum)
        return result

    elif dist_type == 'uniform':
        minimum = distribution_config['min']
        maximum = distribution_config['max']
        result = np.random.uniform(low=minimum, high=maximum, size=size)
        return result

    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


from enum import Enum


class CallStatus(Enum):
    # Drv is clear
    IDLE = 0
    EN_ROUTED = 1
    ON_LOCATION = 2
    TOWING = 3
    COMPLETED = 4
