import sys
import time
import signal
import argparse
import os
import csv
import multiprocessing
from pathlib import Path

import numpy as np
import torch
import visdom
import data
from magic import MAGIC
from utils import *
from action_utils import parse_action_args
from trainer import Trainer
from multi_processing import MultiProcessTrainer
import gym

gym.logger.set_level(40)

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='Multi-Agent Graph Attention Communication')

# training
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=16,
                    help='How many processes to run')

# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--directed', action='store_true', default=False,
                    help='whether the communication graph is directed')
parser.add_argument('--self_loop_type1', default=2, type=int,
                    help='self loop type in the first gat layer (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)')
parser.add_argument('--self_loop_type2', default=2, type=int,
                    help='self loop type in the second gat layer (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)')
parser.add_argument('--gat_num_heads', default=1, type=int,
                    help='number of heads in gat layers except the last one')
parser.add_argument('--gat_num_heads_out', default=1, type=int,
                    help='number of heads in output gat layer')
parser.add_argument('--gat_hid_size', default=64, type=int,
                    help='hidden size of one head in gat')
parser.add_argument('--ge_num_heads', default=4, type=int,
                    help='number of heads in the gat encoder')
parser.add_argument('--first_gat_normalize', action='store_true', default=False,
                    help='whether normalize the coefficients in the first gat layer of the message processor')
parser.add_argument('--second_gat_normalize', action='store_true', default=False,
                    help='whether normilize the coefficients in the second gat layer of the message proccessor')
parser.add_argument('--gat_encoder_normalize', action='store_true', default=False,
                    help='whether normilize the coefficients in the gat encoder (they have been normalized if the input graph is complete)')
parser.add_argument('--use_gat_encoder', action='store_true', default=False,
                    help='whether use the gat encoder before learning the first graph')
parser.add_argument('--gat_encoder_out_size', default=64, type=int,
                    help='hidden size of output of the gat encoder')
parser.add_argument('--first_graph_complete', action='store_true', default=False,
                    help='whether the first communication graph is set to a complete graph')
parser.add_argument('--second_graph_complete', action='store_true', default=False,
                    help='whether the second communication graph is set to a complete graph')
parser.add_argument('--learn_second_graph', action='store_true', default=False,
                    help='whether learn a new communication graph at the second round of communication')
parser.add_argument('--message_encoder', action='store_true', default=False,
                    help='whether use the message encoder')
parser.add_argument('--message_decoder', action='store_true', default=False,
                    help='whether use the message decoder')
parser.add_argument('--nagents', type=int, default=1,
                    help="number of agents")
parser.add_argument('--mean_ratio', default=0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="whether block the communication")
parser.add_argument('--comm_arch', type=str, default='magic', choices=['magic', 'dynamic'],
                    help='communication architecture to use')
parser.add_argument('--dynamic_hidden', default=64, type=int,
                    help='hidden size inside the dynamic graph adjacency modules')
parser.add_argument('--dynamic_comm_range', default=6.0, type=float,
                    help='physical communication range used by dynamic graph')
parser.add_argument('--dynamic_top_k', default=0, type=int,
                    help='maximum number of outgoing neighbors per agent in dynamic graph')
parser.add_argument('--dynamic_phys_soft_tau', default=8.0, type=float,
                    help='distance decay temperature for dynamic graph physical weights')
parser.add_argument('--dynamic_phys_hard_factor', default=0.0, type=float,
                    help='hard cutoff multiplier relative to communication range')
parser.add_argument('--dynamic_phys_eps', default=0.0, type=float,
                    help='minimum physical edge weight kept in dynamic graph')
parser.add_argument('--dynamic_phys_prior_mix', default=0.35, type=float,
                    help='strength of physical priors when blending them into learned dynamic adjacency')
parser.add_argument('--dynamic_use_phys_candidate_mask', default=0, type=int,
                    help='whether to use physical connectivity as a hard candidate mask for learned adjacency')
parser.add_argument('--dynamic_gate_min', default=0.2, type=float,
                    help='minimum retained strength for node communication gates')
parser.add_argument('--dynamic_adj_threshold', default=0.4, type=float,
                    help='threshold used when converting dynamic adjacency probabilities to hard graph')
parser.add_argument('--dynamic_use_soft_adj', action='store_true', default=False,
                    help='feed soft adjacency weights into the existing GAT processors')
parser.add_argument('--dynamic_straight_through', default=1, type=int,
                    help='preserve graph-scorer gradients when using hard adjacency')
parser.add_argument('--dynamic_temporal_smooth', default=0.0, type=float,
                    help='temporal smoothing factor for dynamic adjacency probabilities')
parser.add_argument('--dynamic_candidate_residual', default=0.35, type=float,
                    help='residual blend from candidate graph into final dynamic adjacency to avoid over-pruning')
parser.add_argument('--dynamic_min_edge_density', default=0.28, type=float,
                    help='minimum average edge density target used to prevent overly sparse dynamic graphs')
parser.add_argument('--dynamic_candidate_soft_floor', default=0.35, type=float,
                    help='soft floor that keeps part of physical candidate edges active outside the top-k mask')
parser.add_argument('--dynamic_use_info_controller', default=1, type=int,
                    help='enable node-level communication gating in dynamic graph mode')
parser.add_argument('--dynamic_use_learned_adj', default=1, type=int,
                    help='enable learned adjacency prediction in dynamic graph mode')
parser.add_argument('--dynamic_rebuild_second_graph', action='store_true', default=False,
                    help='rebuild adjacency before the second communication round')
parser.add_argument('--dynamic_use_ude', default=0, type=int,
                    help='enable ODE-based adjacency dynamics in dynamic graph mode')
parser.add_argument('--dynamic_use_magic_importance', default=1, type=int,
                    help='inject MAGIC-style task importance into dynamic node gating')
parser.add_argument('--dynamic_magic_gate_mix', default=0.5, type=float,
                    help='mix ratio for MAGIC-style task importance inside dynamic node gating')
parser.add_argument('--dynamic_sparsity_coeff', default=0.0, type=float,
                    help='regularization strength encouraging sparse dynamic graphs')
parser.add_argument('--dynamic_smoothness_coeff', default=0.0, type=float,
                    help='regularization strength discouraging abrupt graph changes across steps')
parser.add_argument('--dynamic_locality_coeff', default=0.0, type=float,
                    help='regularization strength discouraging long-range edges outside the candidate graph')
parser.add_argument('--dynamic_graph_reg_coeff', default=1.0, type=float,
                    help='global coefficient applied to the sum of dynamic graph regularizers')

# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed') 
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coefficient for value loss term')
parser.add_argument('--max_grad_norm', type=float, default=0.0,
                    help='clip gradient norm to this value; disabled when set to 0')

# environment
parser.add_argument('--env_name', default="grf",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')

# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--plot_port', default='8097', type=str,
                    help='plot port')
parser.add_argument('--save', action="store_true", default=False,
                    help='save the model after training')
parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--metrics_save_every', default=100, type=int,
                    help='save selected training metrics after every n_th epoch; disabled when set to 0')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='display environment state')
parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")


def format_metric_line(name, value, fmt='{:.4f}'):
    if isinstance(value, np.ndarray):
        return '{}: {}'.format(name, value)
    if isinstance(value, (int, float, np.floating, np.integer)):
        return '{}: {}'.format(name, fmt.format(value))
    return '{}: {}'.format(name, value)


def print_training_panel(stat, epoch, num_episodes, epoch_time):
    print('Epoch {}'.format(epoch))
    print('Episode: {}'.format(num_episodes))
    print('Time: {:.2f}s'.format(epoch_time))

    print('[Task]')
    print(format_metric_line('Reward', stat.get('reward', 0), '{}'))
    if 'enemy_reward' in stat:
        print(format_metric_line('Enemy-Reward', stat.get('enemy_reward', 0), '{}'))
    if 'success' in stat:
        print(format_metric_line('Success', stat.get('success', 0.0)))
    if 'steps_taken' in stat:
        print(format_metric_line('Steps-Taken', stat.get('steps_taken', 0.0), '{:.2f}'))
    if 'add_rate' in stat:
        print(format_metric_line('Add-Rate', stat.get('add_rate', 0.0), '{:.2f}'))

    print('[Optimization]')
    if 'action_loss' in stat:
        print(format_metric_line('Action-Loss', stat.get('action_loss', 0.0), '{:.6f}'))
    if 'value_loss' in stat:
        print(format_metric_line('Value-Loss', stat.get('value_loss', 0.0), '{:.6f}'))
    if 'entropy' in stat:
        print(format_metric_line('Entropy', stat.get('entropy', 0.0), '{:.6f}'))
    if 'graph_reg_loss' in stat:
        print(format_metric_line('Graph-Reg-Loss', stat.get('graph_reg_loss', 0.0), '{:.6f}'))

    print('[Graph]')
    if 'dynamic_edge_density' in stat:
        print(format_metric_line('Dynamic-Edge-Density', stat.get('dynamic_edge_density', 0.0)))
    if 'dynamic_candidate_density' in stat:
        print(format_metric_line('Dynamic-Candidate-Density', stat.get('dynamic_candidate_density', 0.0)))
    if 'dynamic_graph_change' in stat:
        print(format_metric_line('Dynamic-Graph-Change', stat.get('dynamic_graph_change', 0.0)))
    if 'dynamic_locality_gap' in stat:
        print(format_metric_line('Dynamic-Locality-Gap', stat.get('dynamic_locality_gap', 0.0)))
    if 'dynamic_hard_edge_density' in stat:
        print(format_metric_line('Dynamic-Hard-Edge-Density', stat.get('dynamic_hard_edge_density', 0.0)))
    if 'dynamic_avg_degree' in stat:
        print(format_metric_line('Dynamic-Avg-Degree', stat.get('dynamic_avg_degree', 0.0)))
    if 'dynamic_active_agents' in stat:
        print(format_metric_line('Dynamic-Active-Agents', stat.get('dynamic_active_agents', 0.0)))
    if 'dynamic_graph_grad_norm' in stat:
        print(format_metric_line('Dynamic-Graph-Grad-Norm', stat.get('dynamic_graph_grad_norm', 0.0), '{:.6f}'))

    if 'comm_action' in stat or 'enemy_comm' in stat:
        print('[Communication]')
        if 'comm_action' in stat:
            print(format_metric_line('Comm-Action', stat.get('comm_action', 0), '{}'))
        if 'enemy_comm' in stat:
            print(format_metric_line('Enemy-Comm', stat.get('enemy_comm', 0), '{}'))


def main():
    init_args_for_env(parser)
    args = parser.parse_args()

    if args.env_name == 'traffic_junction' and args.comm_arch == 'dynamic':
        if args.dynamic_top_k <= 0:
            print('[Warning] dynamic_top_k <= 0: communication is not budget-limited.')
        if args.dynamic_phys_hard_factor <= 0 and args.dynamic_phys_eps <= 0:
            print('[Warning] no hard physical cutoff is active; all live cars are candidates.')

    args.nfriendly = args.nagents
    if hasattr(args, 'enemy_comm') and args.enemy_comm:
        if hasattr(args, 'nenemies'):
            args.nagents += args.nenemies
        else:
            raise RuntimeError("Env. needs to pass argument 'nenemy'.")

    if args.env_name == 'grf':
        render = args.render
        args.render = False
    env = data.init(args.env_name, args, False)

    args.obs_size = env.observation_dim
    args.num_actions = env.num_actions

    if not isinstance(args.num_actions, (list, tuple)):
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions

    parse_action_args(args)

    if args.seed == -1:
        args.seed = np.random.randint(0,10000)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(args)

    policy_net = MAGIC(args)

    if not args.display:
        display_models([policy_net])

    for p in policy_net.parameters():
        p.data.share_memory_()

    disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
    disp_trainer.display = True

    def disp():
        x = disp_trainer.get_episode()
        return x

    if args.env_name == 'grf':
        args.render = render
    if args.nprocesses > 1:
            trainer = MultiProcessTrainer(args, policy_net)
    else:
        trainer = Trainer(args, policy_net, data.init(args.env_name, args))

    log = dict()
    log['epoch'] = LogField(list(), False, None, None)
    log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
    log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
    log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
    log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
    log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
    log['graph_reg_loss'] = LogField(list(), True, 'epoch', 'num_steps')
    log['dynamic_edge_density'] = LogField(list(), True, 'epoch', 'num_steps')
    log['dynamic_candidate_density'] = LogField(list(), True, 'epoch', 'num_steps')
    log['dynamic_graph_change'] = LogField(list(), True, 'epoch', 'num_steps')
    log['dynamic_locality_gap'] = LogField(list(), True, 'epoch', 'num_steps')
    log['dynamic_hard_edge_density'] = LogField(list(), True, 'epoch', 'num_steps')
    log['dynamic_avg_degree'] = LogField(list(), True, 'epoch', 'num_steps')
    log['dynamic_active_agents'] = LogField(list(), True, 'epoch', 'num_steps')
    log['dynamic_graph_grad_norm'] = LogField(list(), True, 'epoch', 'num_grad_updates')

    vis = None
    if args.plot:
        vis = visdom.Visdom(env=args.plot_env, port=args.plot_port)

    model_dir = Path('./saved') / args.env_name
    if args.env_name == 'grf':
        model_dir = model_dir / args.scenario
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run

    def save(final, epoch=0):
        d = dict()
        d['policy_net'] = policy_net.state_dict()
        d['log'] = log
        d['trainer'] = trainer.state_dict()
        if final:
            torch.save(d, run_dir / 'model.pt')
        else:
            torch.save(d, run_dir / ('model_ep%i.pt' % (epoch)))

    def load(path):
        d = torch.load(path)
        policy_net.load_state_dict(d['policy_net'])
        log.update(d['log'])
        trainer.load_state_dict(d['trainer'])

    def export_metric_window(start_epoch, end_epoch):
        if end_epoch < start_epoch:
            return

        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / 'metrics_epochs_{}_{}.csv'.format(start_epoch, end_epoch)
        fieldnames = [
            'epoch', 'reward', 'success', 'steps_taken',
            'dynamic_edge_density', 'dynamic_hard_edge_density',
            'dynamic_avg_degree', 'dynamic_active_agents',
            'dynamic_candidate_density', 'dynamic_graph_change',
            'dynamic_locality_gap', 'graph_reg_loss',
            'dynamic_graph_grad_norm',
        ]
        metric_names = tuple(fieldnames[1:])
        metric_rows = []

        for epoch_value in range(start_epoch, end_epoch + 1):
            idx = epoch_value - 1
            row = {'epoch': epoch_value}
            for metric_name in metric_names:
                metric_data = log[metric_name].data
                row[metric_name] = metric_data[idx] if idx < len(metric_data) else 0
            metric_rows.append(row)

        with metrics_path.open('w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metric_rows)

    def export_metric_snapshots(interval):
        if interval <= 0:
            return

        total_epochs = len(log['epoch'].data)
        if total_epochs == 0:
            return

        completed_windows = total_epochs // interval
        for window_idx in range(completed_windows):
            start_epoch = window_idx * interval + 1
            end_epoch = start_epoch + interval - 1
            export_metric_window(start_epoch, end_epoch)

        remaining_start = completed_windows * interval + 1
        if remaining_start <= total_epochs:
            export_metric_window(remaining_start, total_epochs)

    def signal_handler(_signal, _frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        sys.exit(0)

    def run(num_epochs):
        num_episodes = 0
        if args.save:
            os.makedirs(run_dir)
        for ep in range(num_epochs):
            epoch_begin_time = time.time()
            stat = dict()
            for n in range(args.epoch_size):
                if n == args.epoch_size - 1 and args.display:
                    trainer.display = True
                s = trainer.train_batch(ep)
                print('batch: ', n)
                merge_stat(s, stat)
                trainer.display = False

            epoch_time = time.time() - epoch_begin_time
            epoch = len(log['epoch'].data) + 1
            num_episodes += stat['num_episodes']
            for k, v in log.items():
                if k == 'epoch':
                    v.data.append(epoch)
                else:
                    if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                        stat[k] = stat[k] / stat[v.divide_by]
                    v.data.append(stat.get(k, 0))

            np.set_printoptions(precision=2)

            print_training_panel(stat, epoch, num_episodes, epoch_time)

            if args.plot:
                for k, v in log.items():
                    if v.plot and len(v.data) > 0:
                        vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                                 win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

            if args.metrics_save_every and epoch % args.metrics_save_every == 0:
                export_metric_snapshots(args.metrics_save_every)

            if args.save_every and ep and args.save and (ep + 1) % args.save_every == 0:
                save(final=False, epoch=ep + 1)

            if args.save:
                save(final=True)

    signal.signal(signal.SIGINT, signal_handler)

    if args.load != '':
        load(args.load)

    run(args.num_epochs)
    if args.display:
        env.end_display()

    if args.save:
        save(final=True)

    if args.metrics_save_every:
        export_metric_snapshots(args.metrics_save_every)

    if sys.flags.interactive == 0 and args.nprocesses > 1:
        trainer.quit()
        os._exit(0)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

