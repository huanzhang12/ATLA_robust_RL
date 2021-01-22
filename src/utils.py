import os
import json
import numpy as np
import itertools

def dict_product(d):
    '''
    Implementing itertools.product for dictionaries.
    E.g. {"a": [1,4],  "b": [2,3]} -> [{"a":1, "b":2}, {"a":1,"b":3} ..]
    Inputs:
    - d, a dictionary {key: [list of possible values]}
    Returns;
    - A list of dictionaries with every possible configuration
    '''
    keys = d.keys()
    vals = d.values()
    prod_values = list(itertools.product(*vals))
    all_dicts = map(lambda x: dict(zip(keys, x)), prod_values)
    return all_dicts

def iwt(start, end, interval, trials):
    return list(np.arange(start, end, interval))*trials

def generate_configs(base_config, params):
    import __main__
    suffix = os.path.splitext(os.path.basename(__main__.__file__))[0]
    config_path = f"agent_configs_{suffix}/"
    agent_path = f"agents_{suffix}/"
    all_configs = [{**base_config, **p} for p in dict_product(params)]
    if os.path.isdir(config_path) or os.path.isdir(agent_path):
        raise ValueError("Please delete the '{}' and '{}' directories".format(config_path, agent_path))
    os.makedirs(config_path)
    os.makedirs(agent_path)

    for i, config in enumerate(all_configs):
        with open(os.path.join(config_path, f"{i}.json"), "w") as f:
            json.dump(config, f)
