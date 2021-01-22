import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

from ant_atla_ppo_lstm_best import PARAMS as params

PARAMS = {
    "mode": ["adv_sa_ppo"],
    "out_dir": ["robust_atla_ppo_lstm_ant/agents"],
    "robust_ppo_eps": [0.15],
    "robust_ppo_reg": [0.1],
    "robust_ppo_eps_scheduler_opts": ["start=1,length=3662"],
    "robust_ppo_beta": [1.0],
    "robust_ppo_beta_scheduler_opts": ["same"],  # Using the same scheduler as eps scheduler
    "robust_ppo_detach_stdev": [False],
    "robust_ppo_method": ["sgld"],
    "robust_ppo_pgd_steps": [2],
    "adv_clip_eps": [0.2],
    "adv_entropy_coeff": [1e-4],
    "adv_ppo_lr_adam": [1e-3],
    "adv_val_lr": [1e-5],
}

params.update(PARAMS)

generate_configs(BASE_CONFIG, params)
