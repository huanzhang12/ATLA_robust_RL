import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)





PARAMS = {
    "game": ["Ant-v2"],
    "mode": ["adv_ppo"],
    "out_dir": ["atla_ppo_ant/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "adv_clip_eps": [0.2],
    "ppo_lr_adam": [5e-5],
    "val_lr": [1e-5],
    "entropy_coeff": [3e-4],
    "adv_entropy_coeff": [1e-5],
    "adv_ppo_lr_adam": [3e-5],
    "adv_val_lr": [1e-5],
    "save_iters": [50],
    "train_steps": [2441],
    "robust_ppo_eps": [0.15],  # used for attack
    "value_clipping": [True],
}

generate_configs(BASE_CONFIG, PARAMS)

