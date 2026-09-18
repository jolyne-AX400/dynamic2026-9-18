"""Compatibility wrapper between lb-foraging 1.1.1 and the MAGIC trainer.

The original project wrapper was written for IC3Net environments.  LBF uses a
Tuple observation/action space and returns one done flag per agent, so it needs
slightly different shape and termination handling.
"""

import time

import numpy as np
import torch


class LBFWrapper(object):
    """Expose an LBF Gym environment through the interface used by Trainer."""

    def __init__(self, env):
        self.env = env
        self.last_step_info = dict()
        self._episode_reward = np.zeros(self.nagents, dtype=np.float64)
        self._initial_food_count = 0
        self._food_count = 0

    @property
    def _base_env(self):
        return getattr(self.env, "unwrapped", self.env)

    @property
    def nagents(self):
        action_space = self.env.action_space
        if hasattr(action_space, "spaces"):
            return len(action_space.spaces)
        players = getattr(self._base_env, "players", None)
        if players is not None:
            return len(players)
        raise RuntimeError("Unable to determine the number of LBF agents")

    @property
    def observation_dim(self):
        """Flattened observation dimension for one agent, not all agents."""
        observation_space = self.env.observation_space
        if hasattr(observation_space, "spaces"):
            spaces = observation_space.spaces
            if len(spaces) != self.nagents:
                raise RuntimeError("Unexpected LBF observation-space layout")
            return int(np.prod(spaces[0].shape))
        return int(np.prod(observation_space.shape))

    @property
    def num_actions(self):
        """Number of discrete actions available to each agent."""
        action_space = self.env.action_space
        if hasattr(action_space, "spaces"):
            counts = [int(space.n) for space in action_space.spaces]
            if any(count != counts[0] for count in counts):
                raise RuntimeError("All LBF agents must use the same action space")
            return counts[0]
        if hasattr(action_space, "n"):
            return int(action_space.n)
        raise RuntimeError("LBF requires a discrete action space")

    @property
    def dim_actions(self):
        # One categorical action (none/up/down/left/right/load) per agent.
        return 1

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, epoch=None):
        del epoch
        result = self.env.reset()
        # Keep compatibility with newer Gym-style reset while targeting Gym 0.19.
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            obs = result[0]
        else:
            obs = result

        self._episode_reward = np.zeros(self.nagents, dtype=np.float64)
        self._initial_food_count = self._count_food()
        self._food_count = self._initial_food_count
        self.last_step_info = self.get_comm_info()
        return self._flatten_obs(obs)

    def step(self, action):
        env_action = self._format_action(action)
        result = self.env.step(env_action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = np.logical_or(terminated, truncated)
        else:
            obs, reward, done, info = result

        reward = np.asarray(reward, dtype=np.float64).reshape(self.nagents)
        self._episode_reward += reward
        self._food_count = self._count_food()

        if isinstance(done, (list, tuple, np.ndarray)):
            done = bool(np.all(np.asarray(done, dtype=np.bool_)))
        else:
            done = bool(done)

        info = self.get_comm_info(info)
        self.last_step_info = info
        return self._flatten_obs(obs), reward, done, info

    def _format_action(self, action):
        """Convert MAGIC's one-action-head output to Gym's Tuple action."""
        if isinstance(action, (list, tuple)) and len(action) == 1:
            action = action[0]
        action = np.asarray(action).reshape(-1)
        if action.size != self.nagents:
            raise ValueError(
                "Expected {} LBF actions, received {}".format(
                    self.nagents, action.size
                )
            )
        return tuple(int(value) for value in action)

    def _flatten_obs(self, obs):
        if not isinstance(obs, (list, tuple)):
            obs = np.asarray(obs)
            if obs.shape[0] != self.nagents:
                raise ValueError("LBF observation does not contain one row per agent")
            flat = obs.reshape(self.nagents, -1)
        else:
            flat = np.stack(
                [np.asarray(agent_obs).reshape(-1) for agent_obs in obs], axis=0
            )

        if flat.shape != (self.nagents, self.observation_dim):
            raise ValueError(
                "Unexpected LBF observation shape {}; expected ({}, {})".format(
                    flat.shape, self.nagents, self.observation_dim
                )
            )
        return torch.from_numpy(flat.astype(np.float64, copy=False)).view(
            1, self.nagents, self.observation_dim
        )

    def _player_positions(self):
        players = getattr(self._base_env, "players", None)
        if players is None or len(players) != self.nagents:
            return None

        positions = []
        for player in players:
            position = getattr(player, "position", None)
            if position is None:
                return None
            position = np.asarray(position, dtype=np.float64).reshape(-1)
            if position.size < 2:
                return None
            positions.append(position[:2])
        return np.stack(positions, axis=0)

    def get_comm_info(self, info=None):
        merged = dict(info) if isinstance(info, dict) else dict()
        positions = self._player_positions()
        if positions is not None:
            # Raw grid coordinates are intentional: dynamic_comm_range uses cells.
            merged["positions"] = positions
        merged["alive_mask"] = np.ones(self.nagents, dtype=np.float64)
        return merged

    def _count_food(self):
        field = getattr(self._base_env, "field", None)
        if field is None:
            return 0
        return int(np.count_nonzero(np.asarray(field) > 0))

    def reward_terminal(self):
        return np.zeros(self.nagents, dtype=np.float64)

    def get_stat(self):
        collected = max(0, self._initial_food_count - self._food_count)
        if self._initial_food_count > 0:
            completion = float(collected) / float(self._initial_food_count)
        else:
            completion = 1.0
        return {
            "success": float(self._food_count == 0),
            "food_collected": float(collected),
            "food_completion": completion,
        }

    def display(self):
        self.env.render()
        time.sleep(0.1)

    def end_display(self):
        if hasattr(self.env, "close"):
            self.env.close()
