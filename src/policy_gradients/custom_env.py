import os
import numpy as np
from PIL import Image
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box as Continuous
import gym
import random
from .torch_utils import RunningStat, ZFilter, Identity, StateWithTime, RewardFilter

class Env:
    '''
    A wrapper around the OpenAI gym environment that adds support for the following:
    - Rewards normalization
    - State normalization
    - Adding timestep as a feature with a particular horizon T
    Also provides utility functions/properties for:
    - Whether the env is discrete or continuous
    - Size of feature space
    - Size of action space
    Provides the same API (init, step, reset) as the OpenAI gym
    '''
    def __init__(self, game, norm_states, norm_rewards, params, add_t_with_horizon=None, clip_obs=None, clip_rew=None, 
            show_env=False, save_frames=False, save_frames_path=""):
        self.env = gym.make(game)
        clip_obs = None if clip_obs < 0 else clip_obs
        clip_rew = None if clip_rew < 0 else clip_rew

        # Environment type
        self.is_discrete = type(self.env.action_space) == Discrete
        assert self.is_discrete or type(self.env.action_space) == Continuous

        # Number of actions
        action_shape = self.env.action_space.shape
        assert len(action_shape) <= 1 # scalar or vector actions
        self.num_actions = self.env.action_space.n if self.is_discrete else 0 \
                            if len(action_shape) == 0 else action_shape[0]
        
        # Number of features
        assert len(self.env.observation_space.shape) == 1
        self.num_features = self.env.reset().shape[0]

        # Support for state normalization or using time as a feature
        self.state_filter = Identity()
        if norm_states:
            self.state_filter = ZFilter(self.state_filter, shape=[self.num_features], \
                                            clip=clip_obs)
        if add_t_with_horizon is not None:
            self.state_filter = StateWithTime(self.state_filter, horizon=add_t_with_horizon)
        
        # Support for rewards normalization
        self.reward_filter = Identity()
        if norm_rewards == "rewards":
            self.reward_filter = ZFilter(self.reward_filter, shape=(), center=False, clip=clip_rew)
        elif norm_rewards == "returns":
            self.reward_filter = RewardFilter(self.reward_filter, shape=(), gamma=params.GAMMA, clip=clip_rew)

        # Running total reward (set to 0.0 at resets)
        self.total_true_reward = 0.0

        # Set normalizers to read-write mode by default.
        self._read_only = False

        self.setup_visualization(show_env, save_frames, save_frames_path)

    # For environments that are created from a picked object.
    def setup_visualization(self, show_env, save_frames, save_frames_path):
        self.save_frames = save_frames
        self.show_env = show_env
        self.save_frames_path = save_frames_path
        self.episode_counter = 0
        self.frame_counter = 0
        if self.save_frames:
            print(f'We will save frames to {self.save_frames_path}!')
            os.makedirs(os.path.join(self.save_frames_path, "000"), exist_ok=True)
    
    @property
    def normalizer_read_only(self):
        return self._read_only

    @normalizer_read_only.setter
    def normalizer_read_only(self, value):
        self._read_only = bool(value)
        if isinstance(self.state_filter, ZFilter):
            if not hasattr(self.state_filter, 'read_only') and value:
                print('Warning: requested to set state_filter.read_only=True but the underlying ZFilter does not support it.')
            elif hasattr(self.state_filter, 'read_only'):
                self.state_filter.read_only = self._read_only
        if isinstance(self.reward_filter, ZFilter) or isinstance(self.reward_filter, RewardFilter):
            if not hasattr(self.reward_filter, 'read_only') and value:
                print('Warning: requested to set reward_filter.read_only=True but the underlying ZFilter does not support it.')
            elif hasattr(self.reward_filter, 'read_only'):
                self.reward_filter.read_only = self._read_only
    

    def reset(self):
        # Set a deterministic random seed for reproduicability
        self.env.seed(random.getrandbits(31))
        # Reset the state, and the running total reward
        start_state = self.env.reset()
        self.total_true_reward = 0.0
        self.counter = 0.0
        self.episode_counter += 1
        if self.save_frames:
            os.makedirs(os.path.join(self.save_frames_path, f"{self.episode_counter:03d}"), exist_ok=True)
            self.frame_counter = 0
        self.state_filter.reset()
        self.reward_filter.reset()
        return self.state_filter(start_state, reset=True)

    def step(self, action):
        state, reward, is_done, info = self.env.step(action)
        if self.show_env:
            self.env.render()
        # Frameskip (every 6 frames, will be rendered at 25 fps)
        if self.save_frames and int(self.counter) % 6 == 0:
            image = self.env.render(mode='rgb_array')
            path = os.path.join(self.save_frames_path, f"{self.episode_counter:03d}", f"{self.frame_counter+1:04d}.bmp")
            image = Image.fromarray(image)
            image.save(path)
            self.frame_counter += 1
        state = self.state_filter(state)
        self.total_true_reward += reward
        self.counter += 1
        _reward = self.reward_filter(reward)
        if is_done:
            info['done'] = (self.counter, self.total_true_reward)
        return state, _reward, is_done, info

    
