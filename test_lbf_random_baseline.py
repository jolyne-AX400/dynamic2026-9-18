"""Measure the uniform-random baseline on the current LBF task.

Compatible with Python 3.6, Gym 0.19 and lb-foraging 1.1.1.
Place this file in the dynamic_lbf_fixed project root and run it there.
"""

from __future__ import print_function

import random

import gym
import numpy as np

import lbforaging  # noqa: F401 - importing registers Foraging environments
from lbf_wrapper import LBFWrapper


ENV_ID = "Foraging-2s-10x10-5p-5f-v2"
NUM_EPISODES = 500
MAX_STEPS = 50
SEED = 0


def seed_environment(raw_env, seed):
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(raw_env, "seed"):
        raw_env.seed(seed)
    if hasattr(raw_env.action_space, "seed"):
        raw_env.action_space.seed(seed)


def main():
    raw_env = gym.make(ENV_ID)
    seed_environment(raw_env, SEED)
    env = LBFWrapper(raw_env)
    rng = np.random.RandomState(SEED)

    completion_sum = 0.0
    collected_sum = 0.0
    success_sum = 0.0
    steps_sum = 0.0

    for episode in range(1, NUM_EPISODES + 1):
        env.reset()
        steps = 0

        for _ in range(MAX_STEPS):
            actions = rng.randint(0, env.num_actions, size=env.nagents)
            _, _, done, _ = env.step(actions)
            steps += 1
            if done:
                break

        stat = env.get_stat()
        completion_sum += stat["food_completion"]
        collected_sum += stat["food_collected"]
        success_sum += stat["success"]
        steps_sum += steps

        if episode % 100 == 0:
            print(
                "Episode {:3d} | completion {:.4f} | success {:.4f}".format(
                    episode,
                    completion_sum / episode,
                    success_sum / episode,
                )
            )

    print("=" * 60)
    print("Environment:       {}".format(ENV_ID))
    print("Episodes:          {}".format(NUM_EPISODES))
    print("Random completion: {:.4f}".format(completion_sum / NUM_EPISODES))
    print("Random collected:  {:.4f}".format(collected_sum / NUM_EPISODES))
    print("Random success:    {:.4f}".format(success_sum / NUM_EPISODES))
    print("Average steps:     {:.2f}".format(steps_sum / NUM_EPISODES))
    env.end_display()


if __name__ == "__main__":
    main()
