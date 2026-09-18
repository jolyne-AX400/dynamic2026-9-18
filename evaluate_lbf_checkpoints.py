"""Evaluate a trained Dynamic communication policy on LBF without training.

Compatible with Python 3.6, PyTorch 1.5.1, Gym 0.19 and LBF 1.1.1.
Place this file in the dynamic_lbf_fixed project root.
"""

from __future__ import print_function

import argparse
import csv
import os
import random

import gym
import numpy as np
import torch

import lbforaging  # noqa: F401 - registers the Foraging environments
from lbf_wrapper import LBFWrapper
from magic import MAGIC


ENV_ID = "Foraging-2s-10x10-5p-5f-v2"
ACTION_NAMES = ("none", "up", "down", "left", "right", "load")


class Args(object):
    pass


def make_model_args(env):
    """Recreate the architecture used by the saved seed-0 checkpoint."""
    args = Args()
    args.nagents = env.nagents
    args.batch_size = 1
    args.hid_size = 128
    args.obs_size = env.observation_dim
    args.num_actions = [env.num_actions]
    args.dim_actions = 1
    args.naction_heads = [env.num_actions]
    args.continuous = False

    args.comm_arch = "dynamic"
    args.comm_init = "uniform"
    args.comm_mask_zero = False
    args.directed = True
    args.first_graph_complete = False
    args.second_graph_complete = False
    args.learn_second_graph = False
    args.use_gat_encoder = False
    args.gat_encoder_out_size = 64
    args.ge_num_heads = 4
    args.gat_encoder_normalize = False
    args.first_gat_normalize = False
    args.second_gat_normalize = False
    args.gat_num_heads = 4
    args.gat_num_heads_out = 1
    args.gat_hid_size = 32
    args.self_loop_type1 = 1
    args.self_loop_type2 = 1
    args.message_encoder = False
    args.message_decoder = True

    args.dynamic_hidden = 64
    args.dynamic_comm_range = 6.0
    args.dynamic_top_k = 2
    args.dynamic_phys_soft_tau = 3.0
    args.dynamic_phys_hard_factor = 1.0
    args.dynamic_phys_eps = 0.0
    args.dynamic_phys_prior_mix = 0.35
    args.dynamic_use_phys_candidate_mask = 1
    args.dynamic_gate_min = 0.2
    args.dynamic_adj_threshold = 0.4
    args.dynamic_use_soft_adj = True
    args.dynamic_straight_through = 1
    args.dynamic_temporal_smooth = 0.10
    args.dynamic_candidate_residual = 0.35
    args.dynamic_min_edge_density = 0.0
    args.dynamic_candidate_soft_floor = 0.35
    args.dynamic_use_info_controller = 1
    args.dynamic_use_learned_adj = 1
    args.dynamic_rebuild_second_graph = False
    args.dynamic_use_ude = 0
    args.dynamic_use_magic_importance = 1
    args.dynamic_magic_gate_mix = 0.5
    args.dynamic_sparsity_coeff = 0.0
    args.dynamic_smoothness_coeff = 0.0
    args.dynamic_locality_coeff = 0.0
    return args


def make_env(seed):
    raw_env = gym.make(ENV_ID)
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(raw_env, "seed"):
        raw_env.seed(seed)
    if hasattr(raw_env.action_space, "seed"):
        raw_env.action_space.seed(seed)
    return LBFWrapper(raw_env)


def choose_actions(log_probs, mode):
    probs = log_probs.exp().view(-1, log_probs.size(-1))
    if mode == "greedy":
        return probs.max(dim=1)[1].cpu().numpy()
    return torch.multinomial(probs, 1).view(-1).cpu().numpy()


def evaluate(policy, episodes, max_steps, seed, mode):
    env = make_env(seed)
    torch.manual_seed(seed)

    totals = {
        "completion": 0.0,
        "collected": 0.0,
        "success": 0.0,
        "steps": 0.0,
        "reward": 0.0,
        "edge_density": 0.0,
        "hard_edge_density": 0.0,
        "avg_degree": 0.0,
        "graph_steps": 0,
    }
    action_counts = np.zeros(env.num_actions, dtype=np.int64)
    episode_metrics = {
        "completion": [],
        "collected": [],
        "success": [],
        "steps": [],
        "reward": [],
    }

    policy.eval()
    with torch.no_grad():
        for episode in range(1, episodes + 1):
            state = env.reset()
            hidden = policy.init_hidden(batch_size=1)
            episode_reward = 0.0
            steps = 0

            for _ in range(max_steps):
                info = env.get_comm_info()
                info["epoch"] = 100
                action_out, _, hidden = policy([state, hidden], info)
                actions = choose_actions(action_out[0], mode)
                action_counts += np.bincount(actions, minlength=env.num_actions)

                state, reward, done, _ = env.step(actions)
                episode_reward += float(np.asarray(reward).sum())
                steps += 1

                graph_stats = getattr(policy, "last_graph_stats", {}) or {}
                if graph_stats:
                    totals["edge_density"] += graph_stats.get("dynamic_edge_density", 0.0)
                    totals["hard_edge_density"] += graph_stats.get("dynamic_hard_edge_density", 0.0)
                    totals["avg_degree"] += graph_stats.get("dynamic_avg_degree", 0.0)
                    totals["graph_steps"] += 1

                if done:
                    break

            stat = env.get_stat()
            totals["completion"] += stat["food_completion"]
            totals["collected"] += stat["food_collected"]
            totals["success"] += stat["success"]
            totals["steps"] += steps
            totals["reward"] += episode_reward
            episode_metrics["completion"].append(float(stat["food_completion"]))
            episode_metrics["collected"].append(float(stat["food_collected"]))
            episode_metrics["success"].append(float(stat["success"]))
            episode_metrics["steps"].append(float(steps))
            episode_metrics["reward"].append(float(episode_reward))

            if episode % 100 == 0:
                print(
                    "{} episode {:3d} | completion {:.4f} | success {:.4f}".format(
                        mode,
                        episode,
                        totals["completion"] / episode,
                        totals["success"] / episode,
                    )
                )

    env.end_display()
    graph_steps = max(1, totals["graph_steps"])
    action_total = max(1, int(action_counts.sum()))
    result = {
        "completion": totals["completion"] / episodes,
        "collected": totals["collected"] / episodes,
        "success": totals["success"] / episodes,
        "steps": totals["steps"] / episodes,
        "reward": totals["reward"] / episodes,
        "edge_density": totals["edge_density"] / graph_steps,
        "hard_edge_density": totals["hard_edge_density"] / graph_steps,
        "avg_degree": totals["avg_degree"] / graph_steps,
        "action_distribution": action_counts.astype(np.float64) / action_total,
    }
    for metric_name, values in episode_metrics.items():
        values = np.asarray(values, dtype=np.float64)
        if values.size > 1:
            standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
        else:
            standard_error = 0.0
        result[metric_name + "_ci95"] = 1.96 * standard_error
    return result


def print_result(mode, result):
    print("=" * 64)
    print("Mode:              {}".format(mode))
    print("Food completion:   {:.4f}".format(result["completion"]))
    print("Food collected:    {:.4f}".format(result["collected"]))
    print("Success rate:      {:.4f}".format(result["success"]))
    print("Average steps:     {:.2f}".format(result["steps"]))
    print("Team reward:       {:.4f}".format(result["reward"]))
    print("Edge density:      {:.4f}".format(result["edge_density"]))
    print("Hard-edge density: {:.4f}".format(result["hard_edge_density"]))
    print("Average degree:    {:.4f}".format(result["avg_degree"]))
    print("95% CI success:    +/- {:.4f}".format(result["success_ci95"]))
    print("95% CI completion: +/- {:.4f}".format(result["completion_ci95"]))
    print("95% CI steps:      +/- {:.2f}".format(result["steps_ci95"]))
    for name, probability in zip(ACTION_NAMES, result["action_distribution"]):
        print("Action {:>5s}:      {:.4f}".format(name, probability))


def result_to_row(label, checkpoint_path, mode, episodes, seed, result):
    row = {
        "label": label,
        "checkpoint": checkpoint_path,
        "mode": mode,
        "episodes": episodes,
        "evaluation_seed": seed,
    }
    for key in (
        "success", "success_ci95", "completion", "completion_ci95",
        "collected", "collected_ci95", "steps", "steps_ci95",
        "reward", "reward_ci95", "edge_density", "hard_edge_density",
        "avg_degree",
    ):
        row[key] = result[key]
    for index, name in enumerate(ACTION_NAMES):
        row["action_{}".format(name)] = result["action_distribution"][index]
    return row


def write_csv(path, rows):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved CSV: {}".format(os.path.abspath(path)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="", help="single checkpoint path")
    parser.add_argument(
        "--checkpoints", nargs="+", default=None,
        help="one or more checkpoint paths evaluated in order",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="optional labels corresponding one-to-one with checkpoints",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("greedy", "sampled"),
        default=("greedy", "sampled"),
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output_csv", default="evaluation_results.csv")
    cli = parser.parse_args()

    checkpoint_paths = cli.checkpoints
    if checkpoint_paths is None:
        checkpoint_paths = [cli.checkpoint] if cli.checkpoint else []
    if not checkpoint_paths:
        parser.error("provide --checkpoint PATH or --checkpoints PATH [PATH ...]")
    missing = [path for path in checkpoint_paths if not os.path.isfile(path)]
    if missing:
        parser.error("checkpoint not found: {}".format(", ".join(missing)))
    if cli.labels is not None and len(cli.labels) != len(checkpoint_paths):
        parser.error("--labels must contain exactly one label per checkpoint")
    labels = cli.labels or [os.path.basename(os.path.dirname(path)) + ":" + os.path.basename(path)
                            for path in checkpoint_paths]

    torch.set_default_tensor_type("torch.DoubleTensor")
    build_env = make_env(cli.seed)
    model_args = make_model_args(build_env)
    build_env.end_display()

    rows = []
    for label, checkpoint_path in zip(labels, checkpoint_paths):
        policy = MAGIC(model_args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        policy.load_state_dict(checkpoint["policy_net"])
        print("Loaded checkpoint [{}]: {}".format(label, checkpoint_path))

        for mode in cli.modes:
            result = evaluate(policy, cli.episodes, cli.max_steps, cli.seed, mode)
            print_result(mode, result)
            rows.append(result_to_row(
                label, checkpoint_path, mode, cli.episodes, cli.seed, result
            ))

    write_csv(cli.output_csv, rows)


if __name__ == "__main__":
    main()
