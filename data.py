import sys

import gym
import ic3net_envs

from env_wrappers import *
from lbf_wrapper import LBFWrapper


def init(env_name, args, final_init=True, seed_offset=0):
    if env_name == 'predator_prey':
        env = gym.make('PredatorPrey-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'traffic_junction':
        env = gym.make('TrafficJunction-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'grf':
        env = gym.make('GRFWrapper-v0')
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'lbf':
        # Importing lbforaging registers all Foraging-* Gym environments.
        import lbforaging  # noqa: F401

        env_id = getattr(
            args,
            'lbf_env_id',
            'Foraging-2s-8x8-2p-2f-coop-v2',
        )
        raw_env = gym.make(env_id)
        base_seed = int(getattr(args, 'seed', -1))
        if base_seed >= 0 and hasattr(raw_env, 'seed'):
            raw_env.seed(base_seed + int(seed_offset))
        env = LBFWrapper(raw_env)

        if int(args.nagents) != env.nagents:
            raise ValueError(
                "--nagents={} does not match the {} agents encoded in {}".format(
                    args.nagents, env.nagents, env_id
                )
            )
    else:
        raise RuntimeError("wrong env name: {}".format(env_name))

    return env
