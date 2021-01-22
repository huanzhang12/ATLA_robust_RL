from policy_gradients.agent import Trainer
import git
import pickle
import random
import numpy as np
import os
import argparse
import traceback
from policy_gradients import models
import sys
import json
import torch
from cox.store import Store, schema_from_dict


# Tee object allows for logging to both stdout and to file
class Tee(object):
    def __init__(self, file_path, stream_type, mode='a'):
        assert stream_type in ['stdout', 'stderr']

        self.file = open(file_path, mode)
        self.stream_type = stream_type
        self.errors = 'chill'

        if stream_type == 'stdout':
            self.stream = sys.stdout
            sys.stdout = self
        else:
            self.stream = sys.stderr
            sys.stderr = self

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()

def main(params):
    for k, v in zip(params.keys(), params.values()):
        assert v is not None, f"Value for {k} is None"

    # #
    # Setup logging
    # #
    metadata_schema = schema_from_dict(params)
    base_directory = params['out_dir']
    store = Store(base_directory)

    # redirect stderr, stdout to file
    """
    def make_err_redirector(stream_name):
        tee = Tee(os.path.join(store.path, stream_name + '.txt'), stream_name)
        return tee

    stderr_tee = make_err_redirector('stderr')
    stdout_tee = make_err_redirector('stdout')
    """

    # Store the experiment path and the git commit for this experiment
    metadata_schema.update({
        'store_path': str,
        'git_commit': str
    })

    repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
                    search_parent_directories=True)

    metadata_table = store.add_table('metadata', metadata_schema)
    metadata_table.update_row(params)
    metadata_table.update_row({
        'store_path': store.path,
        'git_commit': repo.head.object.hexsha
    })

    metadata_table.flush_row()

    # Extra items in table when minimax training is enabled.
    if params['mode'] == "adv_ppo" or params['mode'] == 'adv_trpo' or params['mode'] == 'adv_sa_ppo':
        adversary_table_dict = {
            'adversary_policy_model': store.PYTORCH_STATE,
            'adversary_policy_opt': store.PYTORCH_STATE,
            'adversary_val_model': store.PYTORCH_STATE,
            'adversary_val_opt': store.PYTORCH_STATE,
        }
    else:
        adversary_table_dict = {}

    # Table for checkpointing models and envs
    if params['save_iters'] > 0:
        checkpoint_dict = {
            'val_model': store.PYTORCH_STATE,
            'policy_model': store.PYTORCH_STATE,
            'envs': store.PICKLE,
            'policy_opt': store.PYTORCH_STATE,
            'val_opt': store.PYTORCH_STATE,
            'iteration': int,
            '5_rewards': float,
        }
        checkpoint_dict.update(adversary_table_dict)
        store.add_table('checkpoints', checkpoint_dict)

    # The trainer object is in charge of sampling trajectories and
    # taking PPO/TRPO optimization steps

    p = Trainer.agent_from_params(params, store=store)
    if params['initial_std'] != 1.0:
        p.policy_model.log_stdev.data[:] = np.log(params['initial_std'])
    if 'load_model' in params and params['load_model']:
        print('Loading pretrained model', params['load_model'])
        pretrained_model = torch.load(params['load_model'])
        if 'policy_model' in pretrained_model:
            p.policy_model.load_state_dict(pretrained_model['policy_model'])
        if params['deterministic']:
            print('Policy runs in deterministic mode. Ignoring Gaussian noise.')
            p.policy_model.log_stdev.data[:] = -100
        else:
            print('Policy runs in non deterministic mode with Gaussian noise.')
        if 'val_model' in pretrained_model:
            p.val_model.load_state_dict(pretrained_model['val_model'])
        if 'policy_opt' in pretrained_model:
            p.POLICY_ADAM.load_state_dict(pretrained_model['policy_opt'])
        if 'val_opt' in pretrained_model:
            p.val_opt.load_state_dict(pretrained_model['val_opt'])
        # Load adversary models.
        if 'no_load_adv_policy' in params and params['no_load_adv_policy']:
            print('Skipping loading adversary models.')
        else:
            if 'adversary_policy_model' in pretrained_model and hasattr(p, 'adversary_policy_model'):
                p.adversary_policy_model.load_state_dict(pretrained_model['adversary_policy_model'])
            if 'adversary_val_model' in pretrained_model and hasattr(p, 'adversary_val_model'):
                p.adversary_val_model.load_state_dict(pretrained_model['adversary_val_model'])
            if 'adversary_policy_opt' in pretrained_model and hasattr(p, 'adversary_policy_opt'):
                p.adversary_policy_opt.load_state_dict(pretrained_model['adversary_policy_opt'])
            if 'adversary_val_opt' in pretrained_model and hasattr(p, 'adversary_val_opt'):
                p.adversary_val_opt.load_state_dict(pretrained_model['adversary_val_opt'])
        # Load optimizer states.
        # p.POLICY_ADAM.load_state_dict(pretrained_models['policy_opt'])
        # p.val_opt.load_state_dict(pretrained_models['val_opt'])
        # Restore environment parameters, like mean and std.
        if 'envs' in pretrained_model:
            p.envs = pretrained_model['envs']
        for e in p.envs:
            e.setup_visualization(params['show_env'], params['save_frames'], params['save_frames_path'])
    rewards = []

    # Table for final results
    final_dict = {
        'iteration': int,
        '5_rewards': float,
        'terminated_early': bool,
        'val_model': store.PYTORCH_STATE,
        'policy_model': store.PYTORCH_STATE,
        'envs': store.PICKLE,
        'policy_opt': store.PYTORCH_STATE,
        'val_opt': store.PYTORCH_STATE,
    }
    final_dict.update(adversary_table_dict)
    final_table = store.add_table('final_results', final_dict)

    def add_adversary_to_table(p, table_dict):
        if params['mode'] == "adv_ppo" or params['mode'] == 'adv_trpo' or params['mode'] == 'adv_sa_ppo':
            table_dict["adversary_policy_model"] = p.adversary_policy_model.state_dict()
            table_dict["adversary_policy_opt"] = p.ADV_POLICY_ADAM.state_dict()
            table_dict["adversary_val_model"] = p.adversary_val_model.state_dict()
            table_dict["adversary_val_opt"] = p.adversary_val_opt.state_dict()
        return table_dict

    def finalize_table(iteration, terminated_early, rewards):
        final_5_rewards = np.array(rewards)[-5:].mean()
        final_dict = {
            'iteration': iteration,
            '5_rewards': final_5_rewards,
            'terminated_early': terminated_early,
            'val_model': p.val_model.state_dict(),
            'policy_model': p.policy_model.state_dict(),
            'policy_opt': p.POLICY_ADAM.state_dict(),
            'val_opt': p.val_opt.state_dict(),
            'envs': p.envs
        }
        final_dict = add_adversary_to_table(p, final_dict)
        final_table.append_row(final_dict)

    ret = 0
    # Try-except so that we save if the user interrupts the process
    try:
        for i in range(params['train_steps']):
            print('Step %d' % (i,))
            if params['save_iters'] > 0 and i % params['save_iters'] == 0 and i != 0:
                final_5_rewards = np.array(rewards)[-5:].mean()
                print(f'Saving checkpoints to {store.path} with reward {final_5_rewards:.5g}')
                checkpoint_dict = {
                    'iteration': i,
                    'val_model': p.val_model.state_dict(),
                    'policy_model': p.policy_model.state_dict(),
                    'policy_opt': p.POLICY_ADAM.state_dict(),
                    'val_opt': p.val_opt.state_dict(),
                    'envs': p.envs,
                    '5_rewards': final_5_rewards,
                }
                checkpoint_dict = add_adversary_to_table(p, checkpoint_dict)
                store['checkpoints'].append_row(checkpoint_dict)

            mean_reward = p.train_step()
            rewards.append(mean_reward)

            # For debugging and tuning, we can break in the middle.
            if i == params['force_stop_step']:
                print('Terminating early because --force-stop-step is set.')
                raise KeyboardInterrupt

        finalize_table(i, False, rewards)
    except KeyboardInterrupt:
        finalize_table(i, True, rewards)
        ret = 1
    except:
        print("An error occurred during training:")
        traceback.print_exc()
        # Other errors, make sure to finalize the cox store before exiting.
        finalize_table(i, True, rewards)
        ret = -1
    print(f'Models saved to {store.path}')
    store.close()
    return ret

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_common_parser_opts(parser):
    # Basic setup
    parser.add_argument('--game', type=str, help='gym game')
    parser.add_argument('--mode', type=str, choices=['ppo', 'trpo', 'robust_ppo', 'adv_ppo', 'adv_trpo', 'adv_sa_ppo'],
                        help='pg alg')
    parser.add_argument('--out-dir', type=str,
                        help='out dir for store + logging')
    parser.add_argument('--advanced-logging', type=str2bool, const=True, nargs='?')
    parser.add_argument('--kl-approximation-iters', type=int,
                        help='how often to do kl approx exps')
    parser.add_argument('--log-every', type=int)
    parser.add_argument('--policy-net-type', type=str,
                        choices=models.POLICY_NETS.keys())
    parser.add_argument('--value-net-type', type=str,
                        choices=models.VALUE_NETS.keys())
    parser.add_argument('--train-steps', type=int,
                        help='num agent training steps')
    parser.add_argument('--cpu', type=str2bool, const=True, nargs='?')

    # Which value loss to use
    parser.add_argument('--value-calc', type=str,
                        help='which value calculation to use')
    parser.add_argument('--initialization', type=str)

    # General Policy Gradient parameters
    parser.add_argument('--num-actors', type=int, help='num actors (serial)',
                        choices=[1])
    parser.add_argument('--t', type=int,
                        help='num timesteps to run each actor for')
    parser.add_argument('--gamma', type=float, help='discount on reward')
    parser.add_argument('--lambda', type=float, help='GAE hyperparameter')
    parser.add_argument('--val-lr', type=float, help='value fn learning rate')
    parser.add_argument('--val-epochs', type=int, help='value fn epochs')
    parser.add_argument('--initial-std', type=float, help='initial value of std for Gaussian policy. Default is 1.')

    # PPO parameters
    parser.add_argument('--adam-eps', type=float, choices=[0, 1e-5], help='adam eps parameter')

    parser.add_argument('--num-minibatches', type=int,
                        help='num minibatches in ppo per epoch')
    parser.add_argument('--ppo-epochs', type=int)
    parser.add_argument('--ppo-lr', type=float,
                        help='if nonzero, use gradient descent w this lr')
    parser.add_argument('--ppo-lr-adam', type=float,
                        help='if nonzero, use adam with this lr')
    parser.add_argument('--anneal-lr', type=str2bool,
                        help='if we should anneal lr linearly from start to finish')
    parser.add_argument('--clip-eps', type=float, help='ppo clipping')
    parser.add_argument('--clip-val-eps', type=float, help='ppo clipping value')
    parser.add_argument('--entropy-coeff', type=float,
                        help='entropy weight hyperparam')
    parser.add_argument('--value-clipping', type=str2bool,
                        help='should clip values (w/ ppo eps)')
    parser.add_argument('--value-multiplier', type=float,
                        help='coeff for value loss in combined step ppo loss')
    parser.add_argument('--share-weights', type=str2bool,
                        help='share weights in valnet and polnet')
    parser.add_argument('--clip-grad-norm', type=float,
                        help='gradient norm clipping (-1 for no clipping)')
    parser.add_argument('--policy-activation', type=str,
                        help='activation function for countinous policy network')
    
    # TRPO parameters
    parser.add_argument('--max-kl', type=float, help='trpo max kl hparam')
    parser.add_argument('--max-kl-final', type=float, help='trpo max kl final')
    parser.add_argument('--fisher-frac-samples', type=float,
                        help='frac samples to use in fisher vp estimate')
    parser.add_argument('--cg-steps', type=int,
                        help='num cg steps in fisher vp estimate')
    parser.add_argument('--damping', type=float, help='damping to use in cg')
    parser.add_argument('--max-backtrack', type=int, help='max bt steps in fvp')
    parser.add_argument('--trpo-kl-reduce-func', type=str, help='reduce function for KL divergence used in line search. mean or max.')

    # Robust PPO parameters.
    parser.add_argument('--robust-ppo-eps', type=float, help='max eps for robust PPO training')
    parser.add_argument('--robust-ppo-method', type=str, choices=['convex-relax', 'sgld', 'pgd'], help='robustness regularization methods')
    parser.add_argument('--robust-ppo-pgd-steps', type=int, help='number of PGD optimization steps')
    parser.add_argument('--robust-ppo-detach-stdev', type=str2bool, help='detach gradient of standard deviation term')
    parser.add_argument('--robust-ppo-reg', type=float, help='robust PPO regularization')
    parser.add_argument('--robust-ppo-eps-scheduler-opts', type=str, help='options for epsilon scheduler for robust PPO training')
    parser.add_argument('--robust-ppo-beta', type=float, help='max beta (IBP mixing factor) for robust PPO training')
    parser.add_argument('--robust-ppo-beta-scheduler-opts', type=str, help='options for beta scheduler for robust PPO training')

    # Adversarial PPO parameters.
    parser.add_argument('--adv-ppo-lr-adam', type=float,
                        help='if nonzero, use adam for adversary policy with this lr')
    parser.add_argument('--adv-entropy-coeff', type=float,
                        help='entropy weight hyperparam for adversary policy')
    parser.add_argument('--adv-eps', type=float, help='adversary perturbation eps')
    parser.add_argument('--adv-clip-eps', type=float, help='ppo clipping for adversary policy')
    parser.add_argument('--adv-val-lr', type=float, help='value fn learning rate for adversary policy')
    parser.add_argument('--adv-policy-steps', type=float, help='number of policy steps before adversary steps')
    parser.add_argument('--adv-adversary-steps', type=float, help='number of adversary steps before adversary steps')
    parser.add_argument('--adv-adversary-ratio', type=float, help='percentage of frames to attack for the adversary')

    # Adversarial attack parameters.
    parser.add_argument('--attack-method', type=str, choices=["none", "critic", "random", "action", "sarsa", "sarsa+action", "advpolicy", "action+imit"], help='adversarial attack methods.')
    parser.add_argument('--attack-ratio', type=float, help='attack only a ratio of steps.')
    parser.add_argument('--attack-steps', type=int, help='number of PGD optimization steps.')
    parser.add_argument('--attack-eps', type=str, help='epsilon for attack. If set to "same", we will use value of robust-ppo-eps.')
    parser.add_argument('--attack-step-eps', type=str, help='step size for each iteration. If set to "auto", we will use attack-eps / attack-steps')
    parser.add_argument('--attack-sarsa-network', type=str, help='sarsa network to load for attack.')
    parser.add_argument('--attack-sarsa-action-ratio', type=float, help='When set to non-zero, enable sarsa-action attack.')
    parser.add_argument('--attack-advpolicy-network', type=str, help='adversarial policy network to load for attack.')
    parser.add_argument('--collect-perturbed-states', type=str2bool, help='collect perturbed states during training')

    # Normalization parameters
    parser.add_argument('--norm-rewards', type=str, help='type of rewards normalization', 
                        choices=['rewards', 'returns', 'none'])
    parser.add_argument('--norm-states', type=str2bool, help='should norm states')
    parser.add_argument('--clip-rewards', type=float, help='clip rews eps')
    parser.add_argument('--clip-observations', type=float, help='clips obs eps')

    # Sequence training parameters
    parser.add_argument('--history-length', type=int, help='length of history to use for LSTM. If <= 1, we do not use LSTM.')
    parser.add_argument('--use-lstm-val', type=str2bool, help='use a lstm for value function')

    # Saving
    parser.add_argument('--save-iters', type=int, help='how often to save model (0 = no saving)')
    parser.add_argument('--force-stop-step', type=int, help='forcibly terminate after a given number of steps. Useful for debugging and tuning.')

    # Visualization
    parser.add_argument('--show-env', type=str2bool, help='Show environment visualization')
    parser.add_argument('--save-frames', type=str2bool, help='Save environment frames')
    parser.add_argument('--save-frames-path', type=str, help='Path to save environment frames')

    # For grid searches only
    # parser.add_argument('--cox-experiment-path', type=str, default='')
    return parser


def override_json_params(params, json_params, excluding_params):
    # Override the JSON config with the argparse config
    missing_keys = []
    for key in json_params:
        if key not in params:
            missing_keys.append(key)
    assert not missing_keys, "Following keys not in args: " + str(missing_keys)

    missing_keys = []
    for key in params:
        if key not in json_params and key not in excluding_params:
            missing_keys.append(key)
    assert not missing_keys, "Following keys not in JSON: " + str(missing_keys)

    json_params.update({k: params[k] for k in params if params[k] is not None})
    return json_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate experiments to be run.')
    parser.add_argument('--config-path', type=str, required=True,
                        help='json for this config')
    parser.add_argument('--out-dir-prefix', type=str, default="", required=False,
                        help='prefix for output log path')
    parser.add_argument('--load-model', type=str, default=None, required=False, help='load pretrained model and optimizer states before training')
    parser.add_argument('--no-load-adv-policy', action='store_true', required=False, help='Do not load adversary policy and value network from pretrained model.')
    parser.add_argument('--adv-policy-only', action='store_true', required=False, help='Run adversary only, by setting main agent learning rate to 0')
    parser.add_argument('--deterministic', action='store_true', help='disable Gaussian noise in action for --adv-policy-only mode')
    parser.add_argument('--seed', type=int, help='random seed', default=-1)
    parser = add_common_parser_opts(parser)
    
    args = parser.parse_args()

    params = vars(args)
    seed = params['seed']
    json_params = json.load(open(args.config_path))

    extra_params = ['config_path', 'out_dir_prefix', 'load_model', 'no_load_adv_policy', 'adv_policy_only', 'deterministic', 'seed']
    params = override_json_params(params, json_params, extra_params)

    if params['adv_policy_only']:
        if params['adv_ppo_lr_adam'] == 'same':
            params['adv_ppo_lr_adam'] = params['ppo_lr_adam']
            print(f"automatically setting adv_ppo_lr_adam to {params['adv_ppo_lr_adam']}")
        print('disabling policy training (train adversary only)')
        params['ppo_lr_adam'] = 0.0 * params['ppo_lr_adam']
    else:
        # deterministic mode only valid when --adv-policy-only is set
        assert not params['deterministic']

    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
    torch.set_printoptions(threshold=5000, linewidth=120)

    # Append a prefix for output path.
    if args.out_dir_prefix:
        params['out_dir'] = os.path.join(args.out_dir_prefix, params['out_dir'])
        print(f"setting output dir to {params['out_dir']}")
    main(params)

