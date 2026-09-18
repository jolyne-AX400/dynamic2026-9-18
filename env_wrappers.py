import time
import numpy as np
import torch
from gym import spaces
from inspect import getfullargspec

class GymWrapper(object):
    '''
    for multi-agent
    '''
    def __init__(self, env):
        self.env = env
        self.last_step_info = dict()

    @property
    def observation_dim(self):
        '''
        for multi-agent, this is the obs per agent
        '''

        # tuple space
        if hasattr(self.env.observation_space, 'spaces'):
            total_obs_dim = 0
            for space in self.env.observation_space.spaces:
                if hasattr(self.env.action_space, 'shape'):
                    total_obs_dim += int(np.prod(space.shape))
                else: # Discrete
                    total_obs_dim += 1
            return total_obs_dim
        else:
            return int(np.prod(self.env.observation_space.shape))

    @property
    def num_actions(self):
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return int(self.env.action_space.nvec[0])
        elif hasattr(self.env.action_space, 'n'):
            # Discrete
            return self.env.action_space.n

    @property
    def dim_actions(self):
        # for multi-agent, this is the number of action per agent
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return self.env.action_space.shape[0]
            # return len(self.env.action_space.shape)
        elif hasattr(self.env.action_space, 'n'):
            # Discrete => only 1 action takes place at a time.
            return 1

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, epoch):
        reset_args = getfullargspec(self.env.reset).args
        if 'epoch' in reset_args:
            obs = self.env.reset(epoch)
        else:
            obs = self.env.reset()

        self.last_step_info = self.get_comm_info()
        obs = self._flatten_obs(obs)
        return obs

    def display(self):
        self.env.render()
        time.sleep(0.5)

    def end_display(self):
        self.env.exit_render()

    def step(self, action):
        # TODO: Modify all environments to take list of action
        # instead of doing this
        if self.dim_actions == 1:
            action = action[0]
        obs, r, done, info = self.env.step(action)
        obs = self._flatten_obs(obs)
        info = self.get_comm_info(info)
        self.last_step_info = info
        return (obs, r, done, info)

    def get_comm_info(self, info=None):
        merged = dict(info) if isinstance(info, dict) else dict()

        if hasattr(self.env, 'car_loc') and 'positions' not in merged:
            merged['positions'] = np.copy(self.env.car_loc)
        if hasattr(self.env, 'predator_loc') and 'positions' not in merged:
            merged['positions'] = np.copy(self.env.predator_loc)

        if hasattr(self.env, 'alive_mask') and 'alive_mask' not in merged:
            merged['alive_mask'] = np.copy(self.env.alive_mask)

        if 'positions' not in merged and 'left_team' in merged:
            positions = []
            left_team = np.asarray(merged['left_team'])
            left_count = getattr(self.env, 'num_controlled_lagents', 0)
            if left_team.ndim >= 2 and left_count > 0 and left_team.shape[0] >= left_count:
                positions.append(left_team[:left_count, :2])

            right_team = np.asarray(merged['right_team']) if 'right_team' in merged else None
            right_count = getattr(self.env, 'num_controlled_ragents', 0)
            if right_team is not None and right_team.ndim >= 2 and right_count > 0 and right_team.shape[0] >= right_count:
                positions.append(right_team[:right_count, :2])

            if positions:
                merged['positions'] = np.concatenate(positions, axis=0)

        return merged

    def reward_terminal(self):
        if hasattr(self.env, 'reward_terminal'):
            return self.env.reward_terminal()
        else:
            return np.zeros(1)

    def _flatten_obs(self, obs):
        if isinstance(obs, tuple):
            _obs=[]
            for agent in obs: #list/tuple of observations.
                ag_obs = []
                for obs_kind in agent:
                    ag_obs.append(np.array(obs_kind).flatten())
                _obs.append(np.concatenate(ag_obs))
            obs = np.stack(_obs)

        obs = obs.reshape(1, -1, self.observation_dim)
        obs = torch.from_numpy(obs).double()
        return obs

    def get_stat(self):
        if hasattr(self.env, 'stat'):
            self.env.stat.pop('steps_taken', None)
            return self.env.stat
        else:
            return dict()
