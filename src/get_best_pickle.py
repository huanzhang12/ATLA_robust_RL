import os
import numpy as np
import argparse
import uuid
from cox.store import Store
import pickle
from policy_gradients.torch_utils import *
import torch as ch

# Avoid HDF5 error
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

def get_alg_name(name):
    name = os.path.basename(os.path.dirname(name))
    if 'trpo' in name:
        return 'trpo'
    elif 'robust_ppo' in name:
        return 'robust_ppo'
    elif 'adv_sa_ppo' in name:
        return 'adv_sa_ppo'
    elif 'adv_ppo' in name:
        return 'adv_ppo'
    elif 'ppo' in name:
        return 'ppo'
    return 'unknown'

def get_env_name(name):
    if 'humanoid' in name:
        return 'humanoid'
    if 'halfcheetah' in name:
        return 'halfcheetah'
    if 'ant' in name:
        return 'ant'
    elif 'hopper' in name:
        return 'hopper'
    elif 'walker' in name:
        return 'walker'
    return 'unknown'

def main(args):
    base_directory = args.base_directory
    exp_id_list = os.listdir(base_directory)
    best_exp_id = None
    all_rew = []
    all_exp_id = []
    train_eps = []
    if args.exp_id == '':
        for exp_id in exp_id_list:
            s = None
            try:
                s = Store(base_directory, exp_id)
                rew = s['final_results'].df['5_rewards'][0]
                # train_eps.append(s['metadata'].df['robust_ppo_eps'][0])
                all_rew.append(rew)
                print(f"rew={rew}")
                all_exp_id.append(exp_id)
                s.close()
            except Exception as e:
                print(f'Load result error for {exp_id}: {e}')
                if s is not None:
                    s.close()
                continue
        n_exps = len(all_rew)
        all_rew = np.array(all_rew)
        all_exp_id = np.array(all_exp_id)
        ind = np.argsort(all_rew)
        for i in range(len(train_eps)):
            if train_eps[i] == 0.075:
                print(all_exp_id[i])
        print(f'Read {n_exps} models. Avg reward is {all_rew.mean()}, median is {all_rew[ind[n_exps//2]]}')
     
    def dump_one_exp_id(best_exp_id):
        print('\n\n>>>selected id', best_exp_id, 'args.best', args.best, '\n\n')
        if best_exp_id is not None:
            env_name = get_env_name(base_directory)
            alg_name = get_alg_name(base_directory)
            store = Store(base_directory, best_exp_id)
            if 'final_results' in store.tables and not args.all_ckpts:
                table_name = 'final_results'
                index_id = 0
            else:
                table_name = 'checkpoints'
                print(f'Warning: final_results table not found for expid {best_exp_id}, using last checkpoints')
                index_id = -1  # use last checkpoint
            ckpts = store[table_name]
            print('loading from exp id:', best_exp_id, ' reward: ', ckpts.df['5_rewards'].iloc[index_id] if '5_rewards' in ckpts.df else "training not finished")
            
            def dump_model(sel_ckpts, sel_index_id, sel_path):
                P = {}
                # mapper = ch.device('cuda:0')
                for name in ['val_model', 'policy_model', 'val_opt', 'policy_opt', 'adversary_policy_model', 'adversary_val_model', 'adversary_policy_opt', 'adversary_val_opt']:
                    if name in sel_ckpts.df:
                        print(f'Saving {name} out of {len(sel_ckpts.df[name])}')
                        P[name] = sel_ckpts.get_state_dict(sel_ckpts.df[name].iloc[sel_index_id])
                P['envs'] = sel_ckpts.get_pickle(sel_ckpts.df['envs'].iloc[sel_index_id])
                
                ch.save(P, sel_path)
                print('\n', sel_path, 'saved.\n')

            if not args.all_ckpts:
                if args.output is None:
                    path = f"best_model-{alg_name}-{env_name}.{best_exp_id[:8]}.model"
                else:
                    path = args.output
                dump_model(ckpts, index_id, path)
            else:
                iters = ckpts.df['iteration']
                
                for i,it in enumerate(iters):
                    if i % args.dump_step != 0:
                        continue
                    path = f"best_model-{alg_name}-{env_name}.{best_exp_id[:8]}.iter{it}.model"
                    if args.output is not None:
                        if not os.path.exists(args.output):
                            os.makedirs(args.output)
                        path = os.path.join(args.output, path)
                    dump_model(ckpts, i, path)
                
            store.close()
        else:
            raise ValueError('no usable exp found! Cannot load.')

    if not args.all_exp:
        if args.best:
            if args.attack:
                sel_exp_id = all_exp_id[ind[0]]
            else:
                sel_exp_id = all_exp_id[ind[-1]]
        else:
            if args.exp_id:
                sel_exp_id = args.exp_id
            else:
                sel_exp_id = all_exp_id[ind[n_exps // 2]]
        dump_one_exp_id(sel_exp_id)
    else:
        for sel_exp_id in all_exp_id:
            dump_one_exp_id(sel_exp_id)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_directory', type=str, help='agent dir containing cox experiments')
    parser.add_argument('--output', type=str, default='', help='output model filename')
    parser.add_argument('--best', action='store_true', help='select best instead of median')
    parser.add_argument('--exp_id', default='', help='specify an exp id to extract')
    parser.add_argument('--all_ckpts', action='store_true', help='dump all checkpoints in training')
    parser.add_argument('--attack', action='store_true', help='this is an attack experiment, select min reward instead of max')
    parser.add_argument('--all_exp', action='store_true', help='dump all exp_id in training')
    parser.add_argument('--dump_step', default=1, type=int, help='training checkpoint to dump every dump_step indices')
    args = parser.parse_args()
    args.base_directory = args.base_directory.rstrip("/")
    uuid_str = os.path.basename(args.base_directory)
    try:
        uuid.UUID(uuid_str)
    except ValueError:
        pass
    else:
        print('input is a path ending with uuid, directly setting --exp_id based on it')
        args.exp_id = uuid_str
        args.base_directory = os.path.dirname(args.base_directory)

    if args.output == '':
        args.output = f'best_model.{args.exp_id[:8]}.model'
    main(args)
