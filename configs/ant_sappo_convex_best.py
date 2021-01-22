import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Ant-v2"],
    "mode": ["robust_ppo"],
    "out_dir": ["robust_ppo_ant/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "value_clipping": [True],
    "ppo_lr_adam": [5e-5],
    "val_lr": [1e-5],
    "cpu": [True],
    "advanced_logging": [False],
    "save_iters": [100],
    "robust_ppo_eps": [0.15],
    "robust_ppo_reg": [0.003],
    "train_steps": [4882],
    "robust_ppo_eps_scheduler_opts": ["start=1,length=3662"],
    "robust_ppo_beta": [1.0],
    "robust_ppo_beta_scheduler_opts": ["same"], # Using the same scheduler as eps scheduler
    "robust_ppo_detach_stdev": [False], 
    "robust_ppo_method": ["convex-relax"],
    "adv_clip_eps": [0.4],
    "adv_entropy_coeff": [3e-5],
    "adv_ppo_lr_adam": [3e-5],
    "adv_val_lr": [1e-5],
}

generate_configs(BASE_CONFIG, PARAMS)
