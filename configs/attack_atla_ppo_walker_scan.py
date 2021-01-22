import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Walker2d-v2"],
    "mode": ["adv_ppo"],
    "out_dir": ["attack_atla_walker/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True, False],
    "ppo_lr_adam": [0.0],  # this disables policy learning and we run attacks only.
    "adv_clip_eps": [0.2, 0.4],
    "adv_entropy_coeff": [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    "adv_ppo_lr_adam": [3e-4, 1e-3, 3e-3],
    "adv_val_lr": [3e-5, 1e-4, 3e-4],
    "save_iters": [20],
    "train_steps": [488],
    "robust_ppo_eps": [0.075],  # used for attack
    "load_model": ["models/atla_release/ATLA-PPO/model-atla-ppo-walker.model"], # models for attack
    "value_clipping": [True],  
}

generate_configs(BASE_CONFIG, PARAMS)
