import torch
import torch as ch
import copy
import tqdm
import sys
import time
import dill
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
import gym
from auto_LiRPA import BoundedModule
from auto_LiRPA.eps_scheduler import LinearScheduler
from auto_LiRPA.bounded_tensor import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from .models import *
from .torch_utils import *
from .steps import value_step, step_with_mode, pack_history
from .logging import *

from multiprocessing import Process, Queue
from .custom_env import Env
from .convex_relaxation import get_kl_bound as get_state_kl_bound

class Trainer():
    '''
    This is a class representing a Policy Gradient trainer, which 
    trains both a deep Policy network and a deep Value network.
    Exposes functions:
    - advantage_and_return
    - multi_actor_step
    - reset_envs
    - run_trajectories
    - train_step
    Trainer also handles all logging, which is done via the "cox"
    library
    '''
    def __init__(self, policy_net_class, value_net_class, params,
                 store, advanced_logging=True, log_every=5):
        '''
        Initializes a new Trainer class.
        Inputs;
        - policy, the class of policy network to use (inheriting from nn.Module)
        - val, the class of value network to use (inheriting from nn.Module)
        - step, a reference to a function to use for the policy step (see steps.py)
        - params, an dictionary with all of the required hyperparameters
        '''
        # Parameter Loading
        self.params = Parameters(params)

        # Whether or not the value network uses the current timestep
        time_in_state = self.VALUE_CALC == "time"

        # Whether to use GPU (as opposed to CPU)
        if not self.CPU:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Environment Loading
        def env_constructor():
            # Whether or not we should add the time to the state
            horizon_to_feed = self.T if time_in_state else None
            return Env(self.GAME, norm_states=self.NORM_STATES,
                       norm_rewards=self.NORM_REWARDS,
                       params=self.params,
                       add_t_with_horizon=horizon_to_feed,
                       clip_obs=self.CLIP_OBSERVATIONS,
                       clip_rew=self.CLIP_REWARDS,
                       show_env=self.SHOW_ENV,
                       save_frames=self.SAVE_FRAMES,
                       save_frames_path=self.SAVE_FRAMES_PATH)

        self.envs = [env_constructor() for _ in range(self.NUM_ACTORS)]
        self.params.AGENT_TYPE = "discrete" if self.envs[0].is_discrete else "continuous"
        self.params.NUM_ACTIONS = self.envs[0].num_actions
        self.params.NUM_FEATURES = self.envs[0].num_features
        self.policy_step = step_with_mode(self.MODE, adversary=False)
        self.adversary_policy_step = step_with_mode(self.MODE, adversary=True)
        self.params.MAX_KL_INCREMENT = (self.params.MAX_KL_FINAL - self.params.MAX_KL) / self.params.TRAIN_STEPS
        self.advanced_logging = advanced_logging
        self.n_steps = 0
        self.log_every = log_every
        self.policy_net_class = policy_net_class

        # Instantiation
        self.policy_model = policy_net_class(self.NUM_FEATURES, self.NUM_ACTIONS,
                                             self.INITIALIZATION,
                                             time_in_state=time_in_state,
                                             activation=self.policy_activation)

        # Instantiate convex relaxation model when mode is 'robust_ppo'
        if self.MODE == 'robust_ppo' or self.MODE == 'adv_sa_ppo':
            self.create_relaxed_model(time_in_state)

        # Minimax training
        if self.MODE == 'adv_ppo' or self.MODE == 'adv_trpo' or self.MODE == 'adv_sa_ppo':
            # Copy parameters if they are set to "same".
            if self.params.ADV_PPO_LR_ADAM == "same":
                self.params.ADV_PPO_LR_ADAM = self.params.PPO_LR_ADAM
            if self.params.ADV_VAL_LR == "same":
                self.params.ADV_VAL_LR = self.params.VAL_LR
            if self.params.ADV_CLIP_EPS == "same":
                self.params.ADV_CLIP_EPS = self.params.CLIP_EPS
            if self.params.ADV_EPS == "same":
                self.params.ADV_EPS = self.params.ROBUST_PPO_EPS
            if self.params.ADV_ENTROPY_COEFF == "same":
                self.params.ADV_ENTROPY_COEFF = self.params.ENTROPY_COEFF
            # The adversary policy has features as input, features as output.
            self.adversary_policy_model = policy_net_class(self.NUM_FEATURES, self.NUM_FEATURES,
                                                 self.INITIALIZATION,
                                                 time_in_state=time_in_state,
                                                 activation=self.policy_activation)
            # Optimizer for adversary
            self.params.ADV_POLICY_ADAM = optim.Adam(self.adversary_policy_model.parameters(), lr=self.ADV_PPO_LR_ADAM, eps=1e-5)

            # Adversary value function.
            self.adversary_val_model = value_net_class(self.NUM_FEATURES, self.INITIALIZATION)
            self.adversary_val_opt = optim.Adam(self.adversary_val_model.parameters(), lr=self.ADV_VAL_LR, eps=1e-5)
            assert self.adversary_policy_model.discrete == (self.AGENT_TYPE == "discrete")

            # Learning rate annealling for adversary.
            if self.ANNEAL_LR:
                adv_lam = lambda f: 1-f/self.TRAIN_STEPS
                adv_ps = optim.lr_scheduler.LambdaLR(self.ADV_POLICY_ADAM, 
                                                        lr_lambda=adv_lam)
                adv_vs = optim.lr_scheduler.LambdaLR(self.adversary_val_opt, lr_lambda=adv_lam)
                self.params.ADV_POLICY_SCHEDULER = adv_ps
                self.params.ADV_VALUE_SCHEDULER = adv_vs

        opts_ok = (self.PPO_LR == -1 or self.PPO_LR_ADAM == -1)
        assert opts_ok, "One of ppo_lr and ppo_lr_adam must be -1 (off)."
        # Whether we should use Adam or simple GD to optimize the policy parameters
        if self.PPO_LR_ADAM != -1:
            kwargs = {
                'lr':self.PPO_LR_ADAM,
            }

            if self.params.ADAM_EPS > 0:
                kwargs['eps'] = self.ADAM_EPS

            self.params.POLICY_ADAM = optim.Adam(self.policy_model.parameters(),
                                                 **kwargs)
        else:
            self.params.POLICY_ADAM = optim.SGD(self.policy_model.parameters(), lr=self.PPO_LR)

        # If using a time dependent value function, add one extra feature
        # for the time ratio t/T
        if time_in_state:
            self.params.NUM_FEATURES = self.NUM_FEATURES + 1

        # Value function optimization
        self.val_model = value_net_class(self.NUM_FEATURES, self.INITIALIZATION)
        self.val_opt = optim.Adam(self.val_model.parameters(), lr=self.VAL_LR, eps=1e-5) 
        assert self.policy_model.discrete == (self.AGENT_TYPE == "discrete")

        # Learning rate annealing
        # From OpenAI hyperparametrs:
        # Set adam learning rate to 3e-4 * alpha, where alpha decays from 1 to 0 over training
        if self.ANNEAL_LR:
            lam = lambda f: 1-f/self.TRAIN_STEPS
            ps = optim.lr_scheduler.LambdaLR(self.POLICY_ADAM, 
                                                    lr_lambda=lam)
            vs = optim.lr_scheduler.LambdaLR(self.val_opt, lr_lambda=lam)
            self.params.POLICY_SCHEDULER = ps
            self.params.VALUE_SCHEDULER = vs

        if store is not None:
            self.setup_stores(store)
        else:
            print("Not saving results to cox store.")

    

    def create_relaxed_model(self, time_in_state=False):
        # Create state perturbation model for robust PPO training.
        if isinstance(self.policy_model, CtsPolicy):
            if self.ROBUST_PPO_METHOD == "convex-relax":
                from .convex_relaxation import RelaxedCtsPolicyForState
                relaxed_policy_model = RelaxedCtsPolicyForState(
                        self.NUM_FEATURES, self.NUM_ACTIONS, time_in_state=time_in_state,
                        activation=self.policy_activation, policy_model=self.policy_model)
                dummy_input1 = torch.randn(1, self.NUM_FEATURES)
                inputs = (dummy_input1, )
                self.relaxed_policy_model = BoundedModule(relaxed_policy_model, inputs)
            else:
                # For SGLD no need to create the relaxed model
                self.relaxed_policy_model = None
            self.robust_eps_scheduler = LinearScheduler(self.params.ROBUST_PPO_EPS, self.params.ROBUST_PPO_EPS_SCHEDULER_OPTS)
            if self.params.ROBUST_PPO_BETA_SCHEDULER_OPTS == "same":
                self.robust_beta_scheduler = LinearScheduler(self.params.ROBUST_PPO_BETA, self.params.ROBUST_PPO_EPS_SCHEDULER_OPTS)
            else:
                self.robust_beta_scheduler = LinearScheduler(self.params.ROBUST_PPO_BETA, self.params.ROBUST_PPO_BETA_SCHEDULER_OPTS)
        else:
            raise NotImplementedError

    """Initialize sarsa training."""
    def setup_sarsa(self, lr_schedule, eps_scheduler, beta_scheduler):
        # Create the Sarsa model, with S and A as the input.
        self.sarsa_model = ValueDenseNet(self.NUM_FEATURES + self.NUM_ACTIONS, self.INITIALIZATION)
        self.sarsa_opt = optim.Adam(self.sarsa_model.parameters(), lr=self.VAL_LR, eps=1e-5)
        self.sarsa_scheduler = optim.lr_scheduler.LambdaLR(self.sarsa_opt, lr_schedule)
        self.sarsa_eps_scheduler = eps_scheduler
        self.sarsa_beta_scheduler = beta_scheduler
        # Convert model with relaxation wrapper.
        dummy_input = torch.randn(1, self.NUM_FEATURES + self.NUM_ACTIONS)
        self.relaxed_sarsa_model = BoundedModule(self.sarsa_model, dummy_input)
    
    """Initialize imitation (snooping) training."""
    def setup_imit(self, train=True, lr=1e-3):
        # Create a same policy network. 
        self.imit_network = self.policy_net_class(self.NUM_FEATURES, self.NUM_ACTIONS,
                                           self.INITIALIZATION,                                                                              time_in_state=self.VALUE_CALC == "time",
                                           activation=self.policy_activation)
        if train:
            if self.PPO_LR_ADAM != -1:
                kwargs = {
                    'lr':lr,
                }

                if self.params.ADAM_EPS > 0:
                    kwargs['eps'] = self.ADAM_EPS
                    self.imit_opt = optim.Adam(self.imit_network.parameters(), **kwargs)
            else:
                self.imit_opt = optim.SGD(self.imit_network.parameters(), lr=lr)

    """Training imitation agent"""
    def imit_steps(self, all_actions, all_states, all_not_dones, num_epochs):
        assert len(all_actions) == len(all_states)
        for e in range(num_epochs):
            total_loss_val = 0.0
            if self.HISTORY_LENGTH > 0:
                loss = 0.0
                batches, alive_masks, time_masks, lengths = pack_history([all_states, all_actions], all_not_dones, max_length=self.HISTORY_LENGTH)
                self.imit_opt.zero_grad()
                hidden = None
                for i, batch in enumerate(batches):
                    batch_states, batch_actions = batch
                    mask = time_masks[i].unsqueeze(2)
                    
                    if hidden is not None:
                        hidden = [h[:, alive_masks[i], :].detach() for h in hidden]
                    mean, std, hidden = self.imit_network.multi_forward(batch_states, hidden=hidden)
                    batch_loss = torch.nn.MSELoss()(mean*mask, batch_actions*mask)
                    loss += batch_loss
                loss.backward()
                self.imit_opt.step()
                total_loss_val = loss.item()

            else:
                state_indices = np.arange(len(all_actions))
                # np.random.shuffle(state_indices)
                splits = np.array_split(state_indices, self.params.NUM_MINIBATCHES)
                np.random.shuffle(splits)
                for selected in splits:
                    def sel(*args):
                        return [v[selected] for v in args]
                    
                    self.imit_opt.zero_grad()
                    sel_states, sel_actions, sel_not_dones = sel(all_states, all_actions, all_not_dones)  
                    act, _ = self.imit_network(sel_states)
                    loss = torch.nn.MSELoss()(sel_actions, act)
            
                    loss.backward()
                    self.imit_opt.step()
                    total_loss_val += loss.item()
            
            print('Epoch [%d/%d] avg loss: %.8f' % (e+1, num_epochs, total_loss_val / len(all_actions)))
                 

    def setup_stores(self, store):
        # Logging setup
        self.store = store
        if self.MODE == 'adv_ppo' or self.MODE == 'adv_trpo' or self.MODE == 'adv_sa_ppo':
            adv_optimization_table = {
                'mean_reward':float,
                'final_value_loss':float,
                'final_policy_loss':float,
                'final_surrogate_loss':float,
                'entropy_bonus':float,
                'mean_std':float
            }
            self.store.add_table('optimization_adv', adv_optimization_table)
        optimization_table = {
            'mean_reward':float,
            'final_value_loss':float,
            'final_policy_loss':float,
            'final_surrogate_loss':float,
            'entropy_bonus':float,
            'mean_std':float,
        }
        self.store.add_table('optimization', optimization_table)

        if self.advanced_logging:
            paper_constraint_cols = {
                'avg_kl':float,
                'max_kl':float,
                'max_ratio':float,
                'opt_step':int
            }

            value_cols = {
                'heldout_gae_loss':float,
                'heldout_returns_loss':float,
                'train_gae_loss':float,
                'train_returns_loss':float
            }

            weight_cols = {}
            for name, _ in self.policy_model.named_parameters():
                name += "."
                for k in ["l1", "l2", "linf", "delta_l1", "delta_l2", "delta_linf"]:
                    weight_cols[name + k] = float

            self.store.add_table('paper_constraints_train',
                                        paper_constraint_cols)
            self.store.add_table('paper_constraints_heldout',
                                        paper_constraint_cols)
            self.store.add_table('value_data', value_cols)
            self.store.add_table('weight_updates', weight_cols)

        if self.params.MODE == 'robust_ppo' or self.params.MODE == 'adv_sa_ppo':
            robust_cols ={
                'eps': float,
                'beta': float,
                'kl': float,
                'surrogate': float,
                'entropy': float,
                'loss': float,
            }
            self.store.add_table('robust_ppo_data', robust_cols)


    def __getattr__(self, x):
        '''
        Allows accessing self.A instead of self.params.A
        '''
        if x == 'params':
            return {}
        try:
            return getattr(self.params, x)
        except KeyError:
            raise AttributeError(x)

    def advantage_and_return(self, rewards, values, not_dones):
        """
        Calculate GAE advantage, discounted returns, and 
        true reward (average reward per trajectory)

        GAE: delta_t^V = r_t + discount * V(s_{t+1}) - V(s_t)
        using formula from John Schulman's code:
        V(s_t+1) = {0 if s_t is terminal
                   {v_s_{t+1} if s_t not terminal and t != T (last step)
                   {v_s if s_t not terminal and t == T
        """
        assert shape_equal_cmp(rewards, values, not_dones)
        
        V_s_tp1 = ch.cat([values[:,1:], values[:, -1:]], 1) * not_dones
        deltas = rewards + self.GAMMA * V_s_tp1 - values

        # now we need to discount each path by gamma * lam
        advantages = ch.zeros_like(rewards)
        returns = ch.zeros_like(rewards)
        indices = get_path_indices(not_dones)
        for agent, start, end in indices:
            advantages[agent, start:end] = discount_path( \
                    deltas[agent, start:end], self.LAMBDA*self.GAMMA)
            returns[agent, start:end] = discount_path( \
                    rewards[agent, start:end], self.GAMMA)

        return advantages.clone().detach(), returns.clone().detach()

    def reset_envs(self, envs):
        '''
        Resets environments and returns initial state with shape:
        (# actors, 1, ... state_shape)
	    '''
        if self.CPU:
            return cpu_tensorize([env.reset() for env in envs]).unsqueeze(1)
        else:
            return cu_tensorize([env.reset() for env in envs]).unsqueeze(1)

    def multi_actor_step(self, actions, envs):
        '''
        Simulate a "step" by several actors on their respective environments
        Inputs:
        - actions, list of actions to take
        - envs, list of the environments in which to take the actions
        Returns:
        - completed_episode_info, a variable-length list of final rewards and episode lengths
            for the actors which have completed
        - rewards, a actors-length tensor with the rewards collected
        - states, a (actors, ... state_shape) tensor with resulting states
        - not_dones, an actors-length tensor with 0 if terminal, 1 otw
        '''
        normed_rewards, states, not_dones = [], [], []
        completed_episode_info = []
        for action, env in zip(actions, envs):
            gym_action = action[0].cpu().numpy()
            new_state, normed_reward, is_done, info = env.step(gym_action)
            if is_done:
                completed_episode_info.append(info['done'])
                new_state = env.reset()

            # Aggregate
            normed_rewards.append([normed_reward])
            not_dones.append([int(not is_done)])
            states.append([new_state])

        tensor_maker = cpu_tensorize if self.CPU else cu_tensorize
        data = list(map(tensor_maker, [normed_rewards, states, not_dones]))
        return [completed_episode_info, *data]

    def run_trajectories(self, num_saps, return_rewards=False, should_tqdm=False,
            collect_adversary_trajectory=False):
        """
        Resets environments, and runs self.T steps in each environment in 
        self.envs. If an environment hits a terminal state, the env is
        restarted and the terminal timestep marked. Each item in the tuple is
        a tensor in which the first coordinate represents the actor, and the
        second coordinate represents the time step. The third+ coordinates, if
        they exist, represent additional information for each time step.
        Inputs: None
        Returns:
        - rewards: (# actors, self.T)
        - not_dones: (# actors, self.T) 1 in timestep if terminal state else 0
        - actions: (# actors, self.T, ) indices of actions
        - action_logprobs: (# actors, self.T, ) log probabilities of each action
        - states: (# actors, self.T, ... state_shape) states
        """
        if collect_adversary_trajectory:
            # The adversary does not change environment normalization.
            # So a trained adversary can be applied to the original policy when it is trained as an optimal attack.
            old_env_read_only_flags = []
            for e in self.envs:
                old_env_read_only_flags.append(e.normalizer_read_only)
                e.normalizer_read_only = True

        # Arrays to be updated with historic info
        envs = self.envs
        initial_states = self.reset_envs(envs)
        self.policy_model.reset()
        self.val_model.reset()

        # Holds information (length and true reward) about completed episodes
        completed_episode_info = []
        traj_length = int(num_saps // self.NUM_ACTORS)

        shape = (self.NUM_ACTORS, traj_length)
        all_zeros = [ch.zeros(shape) for i in range(3)]
        rewards, not_dones, action_log_probs = all_zeros

        if collect_adversary_trajectory:
            # collect adversary trajectory is only valid in minimax training mode.
            assert self.MODE == "adv_ppo" or self.MODE == "adv_trpo" or self.MODE == "adv_sa_ppo"
            # For the adversary, action is a state perturbation.
            actions_shape = shape + (self.NUM_FEATURES,)
        else:
            actions_shape = shape + (self.NUM_ACTIONS,)
        actions = ch.zeros(actions_shape)
        # Mean of the action distribution. Used for avoid unnecessary recomputation.
        action_means = ch.zeros(actions_shape)
        # Log Std of the action distribution.
        # action_stds = ch.zeros(actions_shape)

        states_shape = (self.NUM_ACTORS, traj_length+1) + initial_states.shape[2:]
        states =  ch.zeros(states_shape)
        iterator = range(traj_length) if not should_tqdm else tqdm.trange(traj_length)

        assert self.NUM_ACTORS == 1

        is_advpolicy_training = self.MODE == "adv_ppo" or self.MODE == "adv_trpo" or self.MODE == "adv_sa_ppo"

        collect_perturbed_state = ((is_advpolicy_training and not collect_adversary_trajectory)
                or ((not is_advpolicy_training) and self.COLLECT_PERTURBED_STATES))

        if collect_perturbed_state:
            # States are collected after the perturbation. We cannot set states[:, 0, :] here as we have not started perturbation yet.
            last_states = initial_states.squeeze(1)  # Remove the second dimension (number of actions)
        else:
            # States are collected before the perturbation.
            states[:, 0, :] = initial_states
            last_states = states[:, 0, :]
        for t in iterator:
            # assert shape_equal([self.NUM_ACTORS, self.NUM_FEATURES], last_states)
            # Retrieve probabilities:
            # action_pds: (# actors, # actions), prob dists over actions
            # next_actions: (# actors, 1), indices of actions
            # next_action_probs: (# actors, 1), prob of taken actions

            # The adversary may use the policy or value function, so pause history update.
            self.policy_model.pause_history()
            self.val_model.pause_history()

            if is_advpolicy_training:
                # the new minimax adversarial training.
                # get perturbation density.
                # When collecting trajactory for agent, only run optimal attack when ADV_ADVERSARY_RATIO >= random.random().
                # When collecting trajactory for adversary, always apply the optimal attack.
                if collect_adversary_trajectory or self.params.ADV_ADVERSARY_RATIO >= random.random():
                    # Only attack a portion of steps.
                    adv_perturbation_pds = self.adversary_policy_model(last_states)
                    next_adv_perturbation_means, next_adv_perturbation_stds = adv_perturbation_pds
                    # sample from the density.
                    next_adv_perturbations = self.adversary_policy_model.sample(adv_perturbation_pds)
                    # get log likelyhood for this perturbation.
                    next_adv_perturbation_log_probs = self.adversary_policy_model.get_loglikelihood(adv_perturbation_pds, next_adv_perturbations)
                    # add the perturbation to state (we learn a residual).
                    last_states = last_states + ch.nn.functional.hardtanh(next_adv_perturbations) * self.ADV_EPS
                    # the perturbation itself is the action (similar to the next_actions variable below)
                    next_adv_perturbations = next_adv_perturbations.unsqueeze(1)
            else:
                # (optional) apply naive adversarial training (not optimal attack)
                maybe_attacked_last_states = self.apply_attack(last_states)
                # Note that for naive adversarial training, we use the state under perturbation to get the actions.
                # However in the trajectory we may still save the state without perturbation as the true environment states are not perturbed.
                # (depending on if self.COLLECT_PERTURBED_STATES is set)

                # double check if the attack eps is valid
                max_eps = (maybe_attacked_last_states - last_states).abs().max().item()
                attack_eps = float(self.params.ROBUST_PPO_EPS) if self.params.ATTACK_EPS == "same" else float(self.params.ATTACK_EPS)
                if max_eps > attack_eps + 1e-5:
                    raise RuntimeError(f"{max_eps} > {attack_eps}. Attack implementation has bug and eps is not correctly handled.")
                last_states = maybe_attacked_last_states

            self.policy_model.continue_history()
            self.val_model.continue_history()
            action_pds = self.policy_model(last_states)
            next_action_means, next_action_stds = action_pds
            next_actions = self.policy_model.sample(action_pds)
            next_action_log_probs = self.policy_model.get_loglikelihood(action_pds, next_actions)

            next_action_log_probs = next_action_log_probs.unsqueeze(1)
            # shape_equal([self.NUM_ACTORS, 1], next_action_log_probs)

            # if discrete, next_actions is (# actors, 1) 
            # otw if continuous (# actors, 1, action dim)
            next_actions = next_actions.unsqueeze(1)
            # if self.policy_model.discrete:
            #     assert shape_equal([self.NUM_ACTORS, 1], next_actions)
            # else:
            #     assert shape_equal([self.NUM_ACTORS, 1, self.policy_model.action_dim])

            ret = self.multi_actor_step(next_actions, envs)

            # done_info = List of (length, reward) pairs for each completed trajectory
            # (next_rewards, next_states, next_dones) act like multi-actor env.step()
            done_info, next_rewards, next_states, next_not_dones = ret
            # Reset the policy (if the policy has memory if we are done)
            if next_not_dones.item() == 0:
                self.policy_model.reset()
                self.val_model.reset()
            # assert shape_equal([self.NUM_ACTORS, 1], next_rewards, next_not_dones)
            # assert shape_equal([self.NUM_ACTORS, 1, self.NUM_FEATURES], next_states)

            # If some of the actors finished AND this is not the last step
            # OR some of the actors finished AND we have no episode information
            if len(done_info) > 0 and (t != self.T - 1 or len(completed_episode_info) == 0):
                completed_episode_info.extend(done_info)

            # Update histories
            # each shape: (nact, t, ...) -> (nact, t + 1, ...)

            if collect_adversary_trajectory:
                # negate the reward for minimax training. Collect states before perturbation.
                next_rewards = -next_rewards
                pairs = [
                    (rewards, next_rewards),
                    (not_dones, next_not_dones),
                    (actions, next_adv_perturbations), # The sampled actions, which is perturbations.
                    (action_means, next_adv_perturbation_means), # The Gaussian mean of actions.
                    # (action_stds, next_action_stds), # The Gaussian std of actions, is a constant, no need to save.
                    (action_log_probs, next_adv_perturbation_log_probs),
                    (states, next_states), # we save the true environment state without perturbation.
                ]
            else:
                if collect_perturbed_state:
                    # New adversarial training. We save the perturbed environment state.
                    pairs = [
                        (rewards, next_rewards),
                        (not_dones, next_not_dones),
                        (actions, next_actions), # The sampled actions.
                        (action_means, next_action_means), # The Gaussian mean of actions.
                        # (action_stds, next_action_stds), # The Gaussian std of actions, is a constant, no need to save.
                        (action_log_probs, next_action_log_probs),
                        (states, last_states.unsqueeze(1)), # perturbed environment state.
                    ]
                else:
                    # Previous naive adversarial training. We save the true environment state.
                    pairs = [
                        (rewards, next_rewards),
                        (not_dones, next_not_dones),
                        (actions, next_actions), # The sampled actions.
                        (action_means, next_action_means), # The Gaussian mean of actions.
                        # (action_stds, next_action_stds), # The Gaussian std of actions, is a constant, no need to save.
                        (action_log_probs, next_action_log_probs),
                        (states, next_states), # true environment state.
                    ]

            for total, v in pairs:
                if total is states and not collect_perturbed_state:
                    # Next states, stores in the next position.
                    total[:, t+1] = v
                else:
                    # The current action taken, and reward received.
                    # When perturbed state is collected, we also do not neeed the +1 shift
                    total[:, t] = v
            last_states = next_states[:, 0, :]

        if collect_perturbed_state:
            if is_advpolicy_training:
                # missing the last state; we have not perturb it yet.
                adv_perturbation_pds = self.adversary_policy_model(last_states)
                # sample from the density.
                next_adv_perturbations = self.adversary_policy_model.sample(adv_perturbation_pds)
                # add the perturbation to state (we learn a residual).
                last_states = last_states + ch.nn.functional.hardtanh(next_adv_perturbations) * self.ADV_EPS
            else:
                last_states = self.apply_attack(last_states)
            states[:, -1] = last_states.unsqueeze(1)

        if collect_adversary_trajectory:
            # Finished adversary step. Take new samples for normalizing environment.
            for e, flag in zip(self.envs, old_env_read_only_flags):
                e.normalizer_read_only = flag


        # Calculate the average episode length and true rewards over all the trajectories
        infos = np.array(list(zip(*completed_episode_info)))
        # print(infos)
        if infos.size > 0:
            _, ep_rewards = infos
            avg_episode_length, avg_episode_reward = np.mean(infos, axis=1)
        else:
            ep_rewards = [-1]
            avg_episode_length = -1
            avg_episode_reward = -1

        # Last state is never acted on, discard
        states = states[:,:-1,:]
        trajs = Trajectories(rewards=rewards, 
            action_log_probs=action_log_probs, not_dones=not_dones, 
            actions=actions, states=states, action_means=action_means, action_std=next_action_stds)

        to_ret = (avg_episode_length, avg_episode_reward, trajs)
        if return_rewards:
            to_ret += (ep_rewards,)

        return to_ret

    """Conduct adversarial attack using value network."""
    def apply_attack(self, last_states):
        if self.params.ATTACK_RATIO < random.random():
            # Only attack a portion of steps.
            return last_states
        eps = self.params.ATTACK_EPS
        if eps == "same":
            eps = self.params.ROBUST_PPO_EPS
        else:
            eps = float(eps)
        steps = self.params.ATTACK_STEPS
        if self.params.ATTACK_METHOD == "critic":
            # Find a state that is close the last_states and decreases value most.
            if steps > 0:
                if self.params.ATTACK_STEP_EPS == "auto":
                    step_eps = eps / steps
                else:
                    step_eps = float(self.params.ATTACK_STEP_EPS)
                clamp_min = last_states - eps
                clamp_max = last_states + eps
                # Random start.
                noise = torch.empty_like(last_states).uniform_(-step_eps, step_eps)
                states = last_states + noise
                with torch.enable_grad():
                    for i in range(steps):
                        states = states.clone().detach().requires_grad_()
                        value = self.val_model(states).mean(dim=1)
                        value.backward()
                        update = states.grad.sign() * step_eps
                        # Clamp to +/- eps.
                        states.data = torch.min(torch.max(states.data - update, clamp_min), clamp_max)
                    self.val_model.zero_grad()
                return states.detach()
            else:
                return last_states
        elif self.params.ATTACK_METHOD == "random":
            # Apply an uniform random noise.
            noise = torch.empty_like(last_states).uniform_(-eps, eps)
            return (last_states + noise).detach()
        elif self.params.ATTACK_METHOD == "action" or self.params.ATTACK_METHOD == "action+imit":
            if steps > 0:
                if self.params.ATTACK_STEP_EPS == "auto":
                    step_eps = eps / steps
                else:
                    step_eps = float(self.params.ATTACK_STEP_EPS)
                clamp_min = last_states - eps
                clamp_max = last_states + eps
                # SGLD noise factor. We simply set beta=1.
                noise_factor = np.sqrt(2 * step_eps)
                noise = torch.randn_like(last_states) * noise_factor
                # The first step has gradient zero, so add the noise and projection directly.
                states = last_states + noise.sign() * step_eps
                # Current action at this state.
                if self.params.ATTACK_METHOD == "action+imit":
                    if not hasattr(self, "imit_network") or self.imit_network == None:
                        assert self.params.imit_model_path != None
                        print('\nLoading imitation network for attack: ', self.params.imit_model_path)
                        # Setup imitation network
                        self.setup_imit(train=False)
                        imit_ckpt = torch.load(self.params.imit_model_path)
                        self.imit_network.load_state_dict(imit_ckpt['state_dict'])
                        self.imit_network.reset()
                        self.imit_network.pause_history()
                    old_action, old_stdev = self.imit_network(last_states)
                else:
                    old_action, old_stdev = self.policy_model(last_states)
                # Normalize stdev, avoid numerical issue
                old_stdev /= (old_stdev.mean())
                old_action = old_action.detach()
                with torch.enable_grad():
                    for i in range(steps):
                        states = states.clone().detach().requires_grad_()
                        if self.params.ATTACK_METHOD == "action+imit":
                            action_change = (self.imit_network(states)[0] - old_action) / old_stdev
                        else:
                            action_change = (self.policy_model(states)[0] - old_action) / old_stdev
                        action_change = (action_change * action_change).sum(dim=1)
                        action_change.backward()
                        # Reduce noise at every step.
                        noise_factor = np.sqrt(2 * step_eps) / (i+2)
                        # Project noisy gradient to step boundary.
                        update = (states.grad + noise_factor * torch.randn_like(last_states)).sign() * step_eps
                        # Clamp to +/- eps.
                        states.data = torch.min(torch.max(states.data + update, clamp_min), clamp_max)
                    if self.params.ATTACK_METHOD == "action+imit": 
                        self.imit_network.zero_grad() 
                    self.policy_model.zero_grad()
                return states.detach()
            else:
                return last_states
        elif self.params.ATTACK_METHOD == "sarsa" or self.params.ATTACK_METHOD == "sarsa+action":
            # Attack using a learned value network.
            assert self.params.ATTACK_SARSA_NETWORK is not None
            use_action = self.params.ATTACK_SARSA_ACTION_RATIO > 0 and self.params.ATTACK_METHOD == "sarsa+action"
            action_ratio = self.params.ATTACK_SARSA_ACTION_RATIO
            assert action_ratio >= 0 and action_ratio <= 1
            if not hasattr(self, "sarsa_network"):
                self.sarsa_network = ValueDenseNet(state_dim=self.NUM_FEATURES+self.NUM_ACTIONS, init="normal")
                print("Loading sarsa network", self.params.ATTACK_SARSA_NETWORK)
                sarsa_ckpt = torch.load(self.params.ATTACK_SARSA_NETWORK)
                sarsa_meta = sarsa_ckpt['metadata']
                sarsa_eps = sarsa_meta['sarsa_eps'] if 'sarsa_eps' in sarsa_meta else "unknown"
                sarsa_reg = sarsa_meta['sarsa_reg'] if 'sarsa_reg' in sarsa_meta else "unknown"
                sarsa_steps = sarsa_meta['sarsa_steps'] if 'sarsa_steps' in sarsa_meta else "unknown"
                print(f"Sarsa network was trained with eps={sarsa_eps}, reg={sarsa_reg}, steps={sarsa_steps}")
                if use_action:
                    print(f"objective: {1.0 - action_ratio} * sarsa + {action_ratio} * action_change")
                else:
                    print("Not adding action change objective.")
                self.sarsa_network.load_state_dict(sarsa_ckpt['state_dict'])
            if steps > 0:
                if self.params.ATTACK_STEP_EPS == "auto":
                    step_eps = eps / steps
                else:
                    step_eps = float(self.params.ATTACK_STEP_EPS)
                clamp_min = last_states - eps
                clamp_max = last_states + eps
                # Random start.
                noise = torch.empty_like(last_states).uniform_(-step_eps, step_eps)
                states = last_states + noise
                if use_action:
                    # Current action at this state.
                    old_action, old_stdev = self.policy_model(last_states)
                    old_stdev /= (old_stdev.mean())
                    old_action = old_action.detach()
                with torch.enable_grad():
                    for i in range(steps):
                        states = states.clone().detach().requires_grad_()
                        # This is the mean action...
                        actions = self.policy_model(states)[0]
                        value = self.sarsa_network(torch.cat((last_states, actions), dim=1)).mean(dim=1)
                        if use_action:
                            action_change = (actions - old_action) / old_stdev
                            # We want to maximize the action change, thus the minus sign.
                            action_change = -(action_change * action_change).mean(dim=1)
                            loss = action_ratio * action_change + (1.0 - action_ratio) * value
                        else:
                            action_change = 0.0
                            loss = value
                        loss.backward()
                        update = states.grad.sign() * step_eps
                        # Clamp to +/- eps.
                        states.data = torch.min(torch.max(states.data - update, clamp_min), clamp_max)
                    self.val_model.zero_grad()
                return states.detach()
            else:
                return last_states
        elif self.params.ATTACK_METHOD == "advpolicy":
            # Attack using a learned policy network.
            assert self.params.ATTACK_ADVPOLICY_NETWORK is not None
            if not hasattr(self, "attack_policy_network"):
                self.attack_policy_network = self.policy_net_class(self.NUM_FEATURES, self.NUM_FEATURES,
                                                 self.INITIALIZATION,
                                                 time_in_state=self.VALUE_CALC == "time",
                                                 activation=self.policy_activation)
                print("Loading adversary policy network", self.params.ATTACK_ADVPOLICY_NETWORK)
                advpolicy_ckpt = torch.load(self.params.ATTACK_ADVPOLICY_NETWORK)
                self.attack_policy_network.load_state_dict(advpolicy_ckpt['adversary_policy_model'])
            # Unlike other attacks we don't need step or eps here.
            # We don't sample and use deterministic adversary policy here.
            perturbations_mean, _ = self.attack_policy_network(last_states)
            # Clamp using tanh.
            perturbed_states = last_states + ch.nn.functional.hardtanh(perturbations_mean) * eps
            """
            adv_perturbation_pds = self.attack_policy_network(last_states)
            next_adv_perturbations = self.attack_policy_network.sample(adv_perturbation_pds)
            perturbed_states = last_states + ch.tanh(next_adv_perturbations) * eps
            """
            return perturbed_states.detach()
        elif self.params.ATTACK_METHOD == "none":
            return last_states
        else:
            raise ValueError(f'Unknown attack method {self.params.ATTACK_METHOD}')


    """Run trajectories and return saps and values for each state."""
    def collect_saps(self, num_saps, should_log=True, return_rewards=False,
                     should_tqdm=False, test=False, collect_adversary_trajectory=False):
        table_name_suffix = "_adv" if collect_adversary_trajectory else ""
        with torch.no_grad():
            # Run trajectories, get values, estimate advantage
            output = self.run_trajectories(num_saps,
                                           return_rewards=return_rewards,
                                           should_tqdm=should_tqdm,
                                           collect_adversary_trajectory=collect_adversary_trajectory)

            if not return_rewards:
                avg_ep_length, avg_ep_reward, trajs = output
            else:
                avg_ep_length, avg_ep_reward, trajs, ep_rewards = output

            # No need to compute advantage function for testing.
            if not test:
                # If we are sharing weights between the policy network and 
                # value network, we use the get_value function of the 
                # *policy* to # estimate the value, instead of using the value
                # net
                if not self.SHARE_WEIGHTS:
                    if collect_adversary_trajectory:
                        if self.HISTORY_LENGTH > 0 and self.USE_LSTM_VAL:
                            values = self.adversary_val_model(trajs.states, trajs.not_dones).squeeze(-1)
                        else:
                            values = self.adversary_val_model(trajs.states).squeeze(-1)
                    else:
                        if self.HISTORY_LENGTH > 0 and self.USE_LSTM_VAL:
                            values = self.val_model(trajs.states, trajs.not_dones).squeeze(-1)
                        else:
                            values = self.val_model(trajs.states).squeeze(-1)
                else:
                    assert self.HISTORY_LENGTH < 1
                    if collect_adversary_trajectory:
                        values = self.adversary_policy_model.get_value(trajs.states).squeeze(-1)
                    else:
                        values = self.policy_model.get_value(trajs.states).squeeze(-1)

                # Calculate advantages and returns
                advantages, returns = self.advantage_and_return(trajs.rewards,
                                                values, trajs.not_dones)

                trajs.advantages = advantages
                trajs.returns = returns
                trajs.values = values

                assert shape_equal_cmp(trajs.advantages, 
                                trajs.returns, trajs.values)

            # Logging
            if should_log:
                msg = "Current mean reward: %f | mean episode length: %f"
                print(msg % (avg_ep_reward, avg_ep_length))
                if not test:
                    self.store.log_table_and_tb('optimization'+table_name_suffix, {
                        'mean_reward': avg_ep_reward
                    })

            # Unroll the trajectories (actors, T, ...) -> (actors*T, ...)
            saps = trajs.unroll()

        to_ret = (saps, avg_ep_reward, avg_ep_length)
        if return_rewards:
            to_ret += (ep_rewards,)

        return to_ret


    def sarsa_steps(self, saps):
        # Begin advanged logging code
        assert saps.unrolled
        loss = torch.nn.SmoothL1Loss()
        action_std = torch.exp(self.policy_model.log_stdev).detach().requires_grad_(False)  # Avoid backprop twice.
        # We treat all value epochs as one epoch.
        self.sarsa_eps_scheduler.set_epoch_length(self.params.VAL_EPOCHS * self.params.NUM_MINIBATCHES)
        self.sarsa_beta_scheduler.set_epoch_length(self.params.VAL_EPOCHS * self.params.NUM_MINIBATCHES)
        # We count from 1.
        self.sarsa_eps_scheduler.step_epoch()
        self.sarsa_beta_scheduler.step_epoch()
        # saps contains state->action->reward and not_done.
        for i in range(self.params.VAL_EPOCHS):
            # Create minibatches with shuffuling
            state_indices = np.arange(saps.rewards.nelement())
            np.random.shuffle(state_indices)
            splits = np.array_split(state_indices, self.params.NUM_MINIBATCHES)

            # Minibatch SGD
            for selected in splits:
                def sel(*args):
                    return [v[selected] for v in args]

                self.sarsa_opt.zero_grad()
                sel_states, sel_actions, sel_rewards, sel_not_dones = sel(saps.states, saps.actions, saps.rewards, saps.not_dones)
                self.sarsa_eps_scheduler.step_batch()
                self.sarsa_beta_scheduler.step_batch()
                
                inputs = torch.cat((sel_states, sel_actions), dim=1)
                # action_diff = self.sarsa_eps_scheduler.get_eps() * action_std
                # inputs_lb = torch.cat((sel_states, sel_actions - action_diff), dim=1).detach().requires_grad_(False)
                # inputs_ub = torch.cat((sel_states, sel_actions + action_diff), dim=1).detach().requires_grad_(False)
                # bounded_inputs = BoundedTensor(inputs, ptb=PerturbationLpNorm(norm=np.inf, eps=None, x_L=inputs_lb, x_U=inputs_ub))
                bounded_inputs = BoundedTensor(inputs, ptb=PerturbationLpNorm(norm=np.inf, eps=self.sarsa_eps_scheduler.get_eps()))

                q = self.relaxed_sarsa_model(bounded_inputs).squeeze(-1)
                q_old = q[:-1]
                q_next = q[1:] * self.GAMMA * sel_not_dones[:-1] + sel_rewards[:-1]
                q_next = q_next.detach()
                # q_loss = (q_old - q_next).pow(2).sum(dim=-1).mean()
                q_loss = loss(q_old, q_next)
                # Compute the robustness regularization.
                if self.sarsa_eps_scheduler.get_eps() > 0 and self.params.SARSA_REG > 0:
                    beta = self.sarsa_beta_scheduler.get_eps()
                    ilb, iub = self.relaxed_sarsa_model.compute_bounds(IBP=True, method=None)
                    if beta < 1:
                        clb, cub = self.relaxed_sarsa_model.compute_bounds(IBP=False, method='backward')
                        lb = beta * ilb + (1 - beta) * clb
                        ub = beta * iub + (1 - beta) * cub
                    else:
                        lb = ilb
                        ub = iub
                    # Output dimension is 1. Remove the extra dimension and keep only the batch dimension.
                    lb = lb.squeeze(-1)
                    ub = ub.squeeze(-1)
                    diff = torch.max(ub - q, q - lb)
                    reg_loss = self.params.SARSA_REG * (diff * diff).mean()
                    sarsa_loss = q_loss + reg_loss
                    reg_loss = reg_loss.item()
                else:
                    reg_loss = 0.0
                    sarsa_loss = q_loss
                sarsa_loss.backward()
                self.sarsa_opt.step()
            print(f'q_loss={q_loss.item():.6g}, reg_loss={reg_loss:.6g}, sarsa_loss={sarsa_loss.item():.6g}')

        if self.ANNEAL_LR:
            self.sarsa_scheduler.step()
        # print('value:', self.val_model(saps.states).mean().item())

        return q_loss, q.mean()


    def take_steps(self, saps, logging=True, value_only=False, adversary_step=False, increment_scheduler=True):
        if adversary_step:
            # collect adversary trajectory is only valid in minimax training mode.
            assert self.MODE == "adv_ppo" or self.MODE == "adv_trpo" or self.MODE == "adv_sa_ppo"

        # Begin advanged logging code
        assert saps.unrolled
        should_adv_log = self.advanced_logging and \
                     self.n_steps % self.log_every == 0 and logging

        self.params.SHOULD_LOG_KL = self.advanced_logging and \
                        self.KL_APPROXIMATION_ITERS != -1 and \
                        self.n_steps % self.KL_APPROXIMATION_ITERS == 0
        store_to_pass = self.store if should_adv_log else None
        # End logging code

        if adversary_step:
            policy_model = self.adversary_policy_model
            if self.ANNEAL_LR:
                policy_scheduler = self.ADV_POLICY_SCHEDULER
                val_scheduler = self.ADV_VALUE_SCHEDULER
            policy_params = Parameters(self.params.copy())
            # In policy_step(), some hard coded attributes will be accessed. We override them.
            policy_params.PPO_LR = self.ADV_PPO_LR_ADAM
            policy_params.PPO_LR_ADAM = self.ADV_PPO_LR_ADAM
            policy_params.POLICY_ADAM = self.ADV_POLICY_ADAM
            policy_params.CLIP_EPS = policy_params.ADV_CLIP_EPS
            policy_params.ENTROPY_COEFF = policy_params.ADV_ENTROPY_COEFF
            val_model = self.adversary_val_model
            val_opt = self.adversary_val_opt
            table_name_suffix = '_adv'
        else:
            policy_model = self.policy_model
            if self.ANNEAL_LR:
                policy_scheduler = self.POLICY_SCHEDULER
                val_scheduler = self.VALUE_SCHEDULER
            policy_params = self.params
            val_model = self.val_model
            val_opt = self.val_opt
            table_name_suffix = ''

        if should_adv_log:
            # collect some extra trajactory for validation of KL and max KL.
            num_saps = saps.advantages.shape[0]
            val_saps = self.collect_saps(num_saps, should_log=False, collect_adversary_trajectory=adversary_step)[0]

            out_train = policy_model(saps.states)
            out_val = policy_model(val_saps.states)

            old_pds = select_prob_dists(out_train, detach=True)
            val_old_pds = select_prob_dists(out_val, detach=True)

        # Update the value function before unrolling the trajectories
        # Pass the logging data into the function if applicable
        val_loss = ch.tensor(0.0)
        if not self.SHARE_WEIGHTS:
            val_loss = value_step(saps.states, saps.returns, 
                saps.advantages, saps.not_dones, val_model,
                val_opt, self.params, store_to_pass,
                old_vs=saps.values.detach()).mean()

        if self.ANNEAL_LR and increment_scheduler:
            val_scheduler.step()

        if value_only:
            # Run the value iteration only. Return now.
            return val_loss

        if logging:
            self.store.log_table_and_tb('optimization'+table_name_suffix, {
                'final_value_loss': val_loss
            })

        if (self.MODE == 'robust_ppo' or self.MODE == 'adv_sa_ppo') and not adversary_step and logging:
            # Logging Robust PPO KL, entropy, etc.
            store_to_pass = self.store

        # Take optimizer steps
        args = [saps.states, saps.actions, saps.action_log_probs,
                saps.rewards, saps.returns, saps.not_dones, 
                saps.advantages, policy_model, policy_params, 
                store_to_pass, self.n_steps]

        if (self.MODE == 'robust_ppo' or self.MODE == 'adv_sa_ppo') and isinstance(self.policy_model, CtsPolicy) and not adversary_step:
            args += [self.relaxed_policy_model, self.robust_eps_scheduler, self.robust_beta_scheduler]

        self.MAX_KL += self.MAX_KL_INCREMENT
        if should_adv_log:
            # Save old parameter to investigate weight updates.
            old_parameter = copy.deepcopy(self.policy_model.state_dict())

        # Policy optimization step
        if adversary_step:
            policy_loss, surr_loss, entropy_bonus = self.adversary_policy_step(*args)
        else:
            policy_loss, surr_loss, entropy_bonus = self.policy_step(*args)

        # If the anneal_lr option is set, then we decrease the 
        # learning rate at each training step
        if self.ANNEAL_LR and increment_scheduler:
            policy_scheduler.step()

        if should_adv_log and not adversary_step:
            log_value_losses(self, val_saps, 'heldout')
            log_value_losses(self, saps, 'train')
            old_pds = saps.action_means, saps.action_std
            paper_constraints_logging(self, saps, old_pds,
                            table='paper_constraints_train')
            paper_constraints_logging(self, val_saps, val_old_pds,
                            table='paper_constraints_heldout')
            log_weight_updates(self, old_parameter, self.policy_model.state_dict())

            self.store['paper_constraints_train'].flush_row()
            self.store['paper_constraints_heldout'].flush_row()
            self.store['value_data'].flush_row()
            self.store['weight_updates'].flush_row()
        if (self.params.MODE == 'robust_ppo' or self.params.MODE == 'adv_sa_ppo') and not adversary_step:
            self.store['robust_ppo_data'].flush_row()

        if self.ANNEAL_LR:
            print(f'val lr: {val_scheduler.get_last_lr()}, policy lr: {policy_scheduler.get_last_lr()}')
        val_loss = val_loss.mean().item()
        return policy_loss, surr_loss, entropy_bonus, val_loss

    def train_step(self):
        if self.MODE == "adv_ppo" or self.MODE == "adv_trpo" or self.MODE == "adv_sa_ppo":
            avg_ep_reward = 0.0
            if self.PPO_LR_ADAM != 0.0:
                for i in range(int(self.ADV_POLICY_STEPS)):
                    avg_ep_reward = self.train_step_impl(adversary_step = False, increment_scheduler = (i==self.ADV_POLICY_STEPS-1))
                for i in range(int(self.ADV_ADVERSARY_STEPS)):
                    self.train_step_impl(adversary_step = True, increment_scheduler = (i==self.ADV_ADVERSARY_STEPS-1))
            else:
                print('skipping policy training because learning rate is 0. adv_policy_steps and adv_adversary_steps ignored.')
                avg_ep_reward = self.train_step_impl(adversary_step = True)
        else:
            avg_ep_reward = self.train_step_impl(adversary_step = False)

        self.n_steps += 1
        print()
        return avg_ep_reward

    def train_step_impl(self, adversary_step=False, increment_scheduler=True):
        '''
        Take a training step, by first collecting rollouts, then 
        calculating advantages, then taking a policy gradient step, and 
        finally taking a value function step.

        Inputs: None
        Returns: 
        - The current reward from the policy (per actor)
        '''
        start_time = time.time()

        table_name_suffix = "_adv" if adversary_step else ""

        if adversary_step:
            print('++++++++ Adversary training ++++++++++')
            policy_model = self.adversary_policy_model
        else:
            print('++++++++ Policy training ++++++++++')
            policy_model = self.policy_model

        num_saps = self.T * self.NUM_ACTORS
        saps, avg_ep_reward, avg_ep_length = self.collect_saps(num_saps, collect_adversary_trajectory=adversary_step)
        policy_loss, surr_loss, entropy_bonus, val_loss = self.take_steps(saps, adversary_step=adversary_step, increment_scheduler=increment_scheduler)
        # Logging code
        print(f"Policy Loss: {policy_loss:.5g}, | Entropy Bonus: {entropy_bonus:.5g}, | Value Loss: {val_loss:.5g}")
        print("Time elapsed (s):", time.time() - start_time)
        if not policy_model.discrete:
            mean_std = ch.exp(policy_model.log_stdev).mean()
            print("Agent stdevs: %s" % mean_std.detach().cpu().numpy())
            self.store.log_table_and_tb('optimization'+table_name_suffix, {
                'mean_std': mean_std,
                'final_policy_loss' : policy_loss,
                'final_surrogate_loss': surr_loss,
                'entropy_bonus': entropy_bonus,
            })
        else:
            self.store['optimization'+table_name_suffix].update_row({
                'mean_std': np.nan,
                'final_policy_loss' : policy_loss,
                'final_surrogate_loss': surr_loss,
                'entropy_bonus': entropy_bonus,
            })

        self.store['optimization'+table_name_suffix].flush_row()
        print("-" * 80)
        sys.stdout.flush()
        sys.stderr.flush()
        # End logging code

        return avg_ep_reward

    def sarsa_step(self):
        '''
        Take a training step, by first collecting rollouts, and 
        taking a value function step.

        Inputs: None
        Returns: 
        - The current reward from the policy (per actor)
        '''
        print("-" * 80)
        start_time = time.time()

        num_saps = self.T * self.NUM_ACTORS
        saps, avg_ep_reward, avg_ep_length = self.collect_saps(num_saps, should_log=True, test=True)
         
        sarsa_loss, q = self.sarsa_steps(saps)
        print("Sarsa Loss:", sarsa_loss.item())
        print("Q:", q.item())
        print("Time elapsed (s):", time.time() - start_time)
        sys.stdout.flush()
        sys.stderr.flush()

        self.n_steps += 1
        return avg_ep_reward

    def run_test(self, max_len=2048, compute_bounds=False, use_full_backward=False, original_stdev=None):
        print("-" * 80)
        start_time = time.time()
        if compute_bounds and not hasattr(self, "relaxed_policy_model"):
            self.create_relaxed_model()
        #saps, avg_ep_reward, avg_ep_length = self.collect_saps(num_saps=None, should_log=True, test=True, num_episodes=num_episodes)
        with torch.no_grad():
            output = self.run_test_trajectories(max_len=max_len)
            ep_length, ep_reward, actions, action_means, states = output
            msg = "Episode reward: %f | episode length: %f"
            print(msg % (ep_reward, ep_length))
            if compute_bounds:
                if original_stdev is None:
                    kl_stdev = torch.exp(self.policy_model.log_stdev)
                else:
                    kl_stdev = torch.exp(original_stdev)
                eps = float(self.params.ROBUST_PPO_EPS) if self.params.ATTACK_EPS == "same" else float(self.params.ATTACK_EPS)
                kl_upper_bound = get_state_kl_bound(self.relaxed_policy_model, states, action_means,
                        eps=eps, beta=0.0,
                        stdev=kl_stdev, use_full_backward=use_full_backward).mean()
                kl_upper_bound = kl_upper_bound.item()
            else:
                kl_upper_bound = float("nan")
            # Unroll the trajectories (actors, T, ...) -> (actors*T, ...)
        return ep_length, ep_reward, actions.cpu().numpy(), action_means.cpu().numpy(), states.cpu().numpy(), kl_upper_bound

    def run_test_trajectories(self, max_len, should_tqdm=False):
        # Arrays to be updated with historic info
        envs = self.envs
        initial_states = self.reset_envs(envs)
        if hasattr(self, "imit_network"):
            self.imit_network.reset()
        self.policy_model.reset()
        self.val_model.reset()

        # Holds information (length and true reward) about completed episodes
        completed_episode_info = []

        shape = (1, max_len)
        rewards = ch.zeros(shape)

        actions_shape = shape + (self.NUM_ACTIONS,)
        actions = ch.zeros(actions_shape)
        # Mean of the action distribution. Used for avoid unnecessary recomputation.
        action_means = ch.zeros(actions_shape)

        states_shape = (1, max_len+1) + initial_states.shape[2:]
        states =  ch.zeros(states_shape)

        iterator = range(max_len) if not should_tqdm else tqdm.trange(max_len)


        states[:, 0, :] = initial_states
        last_states = states[:, 0, :]
        
        for t in iterator:
            if (t+1) % 100 == 0:
                print('Step {} '.format(t+1))
            # assert shape_equal([self.NUM_ACTORS, self.NUM_FEATURES], last_states)
            # Retrieve probabilities 
            # action_pds: (# actors, # actions), prob dists over actions
            # next_actions: (# actors, 1), indices of actions
            
            # pause updating hidden state because the attack may inference the model.
            self.policy_model.pause_history()
            self.val_model.pause_history()
            if hasattr(self, "imit_network"):
                self.imit_network.pause_history()
            
            maybe_attacked_last_states = self.apply_attack(last_states)
            
            self.policy_model.continue_history()
            self.val_model.continue_history()
            if hasattr(self, "imit_network"):
                self.imit_network.continue_history()

            action_pds = self.policy_model(maybe_attacked_last_states)
            if hasattr(self, "imit_network"):
                _ = self.imit_network(maybe_attacked_last_states) 
            
            next_action_means, next_action_stds = action_pds
            # Double check if the attack is within eps range.
            if self.params.ATTACK_METHOD != "none":
                max_eps = (maybe_attacked_last_states - last_states).abs().max()
                attack_eps = float(self.params.ROBUST_PPO_EPS) if self.params.ATTACK_EPS == "same" else float(self.params.ATTACK_EPS)
                if max_eps > attack_eps + 1e-5:
                    raise RuntimeError(f"{max_eps} > {attack_eps}. Attack implementation has bug and eps is not correctly handled.")
            next_actions = self.policy_model.sample(action_pds)


            # if discrete, next_actions is (# actors, 1) 
            # otw if continuous (# actors, 1, action dim)
            next_actions = next_actions.unsqueeze(1)

            ret = self.multi_actor_step(next_actions, envs)

            # done_info = List of (length, reward) pairs for each completed trajectory
            # (next_rewards, next_states, next_dones) act like multi-actor env.step()
            done_info, next_rewards, next_states, next_not_dones = ret
            # Reset the policy (if the policy has memory if we are done)
            if next_not_dones.item() == 0:
                self.policy_model.reset()
                self.val_model.reset()

            # Update histories
            # each shape: (nact, t, ...) -> (nact, t + 1, ...)

            pairs = [
                (rewards, next_rewards),
                (actions, next_actions), # The sampled actions.
                (action_means, next_action_means), # The sampled actions.
                (states, next_states),
            ]

            last_states = next_states[:, 0, :]
            for total, v in pairs:
                if total is states:
                    # Next states, stores in the next position.
                    total[:, t+1] = v
                else:
                    # The current action taken, and reward received.
                    total[:, t] = v
            
            # If some of the actors finished AND this is not the last step
            # OR some of the actors finished AND we have no episode information
            if len(done_info) > 0:
                completed_episode_info.extend(done_info)
                break

        if len(completed_episode_info) > 0:
            ep_length, ep_reward = completed_episode_info[0]
        else:
            ep_length = np.nan
            ep_reward = np.nan

        actions = actions[0][:t+1]
        action_means = action_means[0][:t+1]
        states = states[0][:t+1]

        to_ret = (ep_length, ep_reward, actions, action_means, states)
        
        
        return to_ret

    @staticmethod
    def agent_from_data(store, row, cpu, extra_params=None, override_params=None, excluded_params=None):
        '''
        Initializes an agent from serialized data (via cox)
        Inputs:
        - store, the name of the store where everything is logged
        - row, the exact row containing the desired data for this agent
        - cpu, True/False whether to use the CPU (otherwise sends to GPU)
        - extra_params, a dictionary of extra agent parameters. Only used
          when a key does not exist from the loaded cox store.
        - override_params, a dictionary of agent parameters that will override
          current agent parameters.
        - excluded_params, a dictionary of parameters that we do not copy or
          override.
        Outputs:
        - agent, a constructed agent with the desired initialization and
              parameters
        - agent_params, the parameters that the agent was constructed with
        '''

        ckpts = store['final_results']

        get_item = lambda x: list(row[x])[0]

        items = ['val_model', 'policy_model', 'val_opt', 'policy_opt']
        names = {i: get_item(i) for i in items}

        param_keys = list(store['metadata'].df.columns)
        param_values = list(store['metadata'].df.iloc[0,:])

        def process_item(v):
            try:
                return v.item()
            except:
                return v

        param_values = [process_item(v) for v in param_values]
        agent_params = {k:v for k, v in zip(param_keys, param_values)}

        if 'adam_eps' not in agent_params: 
            agent_params['adam_eps'] = 1e-5
        if 'cpu' not in agent_params:
            agent_params['cpu'] = cpu

        # Update extra params if they do not exist in current parameters.
        if extra_params is not None:
            for k in extra_params.keys():
                if k not in agent_params and k not in excluded_params:
                    print(f'adding key {k}={extra_params[k]}')
                    agent_params[k] = extra_params[k]
        if override_params is not None:
            for k in override_params.keys():
                if k not in excluded_params and override_params[k] is not None and override_params[k] != agent_params[k]:
                    print(f'overwriting key {k}: old={agent_params[k]}, new={override_params[k]}')
                    agent_params[k] = override_params[k]

        agent = Trainer.agent_from_params(agent_params)

        def load_state_dict(model, ckpt_name):
            mapper = ch.device('cuda:0') if not cpu else ch.device('cpu')
            state_dict = ckpts.get_state_dict(ckpt_name, map_location=mapper)
            model.load_state_dict(state_dict)

        load_state_dict(agent.policy_model, names['policy_model'])
        load_state_dict(agent.val_model, names['val_model'])
        if agent.ANNEAL_LR:
            agent.POLICY_SCHEDULER.last_epoch = get_item('iteration')
            agent.VALUE_SCHEDULER.last_epoch = get_item('iteration')
        load_state_dict(agent.POLICY_ADAM, names['policy_opt'])
        load_state_dict(agent.val_opt, names['val_opt'])
        agent.envs = ckpts.get_pickle(get_item('envs'))

        return agent, agent_params

    @staticmethod
    def agent_from_params(params, store=None):
        '''
        Construct a trainer object given a dictionary of hyperparameters.
        Trainer is in charge of sampling trajectories, updating policy network,
        updating value network, and logging.
        Inputs:
        - params, dictionary of required hyperparameters
        - store, a cox.Store object if logging is enabled
        Outputs:
        - A Trainer object for training a PPO/TRPO agent
        '''
        if params['history_length'] > 0:
            agent_policy = CtsLSTMPolicy
            if params['use_lstm_val']:
                agent_value = ValueLSTMNet
            else:
                agent_value = value_net_with_name(params['value_net_type'])
        else:
            agent_policy = policy_net_with_name(params['policy_net_type'])
            agent_value = value_net_with_name(params['value_net_type'])

        advanced_logging = params['advanced_logging'] and store is not None
        log_every = params['log_every'] if store is not None else 0

        if params['cpu']:
            torch.set_num_threads(1)
        p = Trainer(agent_policy, agent_value, params, store, log_every=log_every,
                    advanced_logging=advanced_logging)

        return p

