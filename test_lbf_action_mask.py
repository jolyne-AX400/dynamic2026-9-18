"""Smoke test for the LBF legal-action mask used by MAGIC."""

from __future__ import print_function

import gym
import numpy as np

import lbforaging  # noqa: F401
from lbf_wrapper import LBFWrapper


ENV_ID = "Foraging-2s-10x10-5p-5f-v2"


def main():
    raw_env = gym.make(ENV_ID)
    if hasattr(raw_env, "seed"):
        raw_env.seed(0)
    env = LBFWrapper(raw_env)
    env.reset()

    checked_steps = 0
    saw_invalid_action = False
    saw_legal_load = False
    saw_illegal_load = False

    for _ in range(10):
        env.reset()
        for _ in range(50):
            info = env.get_comm_info()
            mask = np.asarray(info["action_mask"])
            assert mask.shape == (env.nagents, env.num_actions)
            assert np.all(mask[:, 0] == 1), "NONE must always be legal"
            assert np.all(mask.sum(axis=1) >= 1), "every agent needs a legal action"

            saw_invalid_action = saw_invalid_action or bool(np.any(mask == 0))
            saw_legal_load = saw_legal_load or bool(np.any(mask[:, 5] == 1))
            saw_illegal_load = saw_illegal_load or bool(np.any(mask[:, 5] == 0))

            actions = []
            for agent_index in range(env.nagents):
                legal = np.flatnonzero(mask[agent_index] > 0)
                actions.append(int(np.random.choice(legal)))
                assert mask[agent_index, actions[-1]] == 1

            _, _, done, _ = env.step(actions)
            checked_steps += 1
            if done:
                break

    assert saw_invalid_action, "test never observed an invalid action"
    assert saw_legal_load, "test never observed a state where LOAD was legal"
    assert saw_illegal_load, "test never observed a state where LOAD was illegal"
    print("LBF action-mask test passed")
    print("Checked steps:", checked_steps)
    print("Mask shape:", (env.nagents, env.num_actions))
    env.end_display()


if __name__ == "__main__":
    main()
