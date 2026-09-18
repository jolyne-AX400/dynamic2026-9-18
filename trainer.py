from collections import namedtuple
from inspect import getfullargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
import itertools

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state', 'reward', 'misc'))


class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]


    def get_episode(self, epoch):
        episode = []
        reset_args = getfullargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = self.env.get_comm_info() if hasattr(self.env, 'get_comm_info') else dict()
        if not isinstance(info, dict):
            info = dict()
        info['epoch'] = epoch
        switch_t = -1

        prev_hid = None

        for t in range(self.args.max_steps):
            misc = dict()

            if not isinstance(info, dict):
                info = dict()
            info['epoch'] = epoch

            if t == 0:
                prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])
            x = [state, prev_hid]
            policy_out = self.policy_net(x, info)
            if len(policy_out) == 4:
                action_out, value, prev_hid, attn_weights = policy_out
            elif len(policy_out) == 3:
                action_out, value, prev_hid = policy_out
                attn_weights = None
            else:
                raise ValueError('Unexpected policy output length: {}'.format(len(policy_out)))
            graph_reg_loss = getattr(self.policy_net, 'last_graph_reg_loss', None)
            graph_stats = dict(getattr(self.policy_net, 'last_graph_stats', {}) or {})

            # The policy above acted using the current state's mask.  Save that
            # mask with the transition before env.step() advances to the next
            # state; otherwise a vehicle completing its route loses the
            # gradient for its final valid action.
            current_alive_mask = np.asarray(
                info.get('alive_mask', np.ones(self.args.nagents)),
                dtype=np.float64,
            ).reshape(-1)[:self.args.nagents].copy()

            if (t + 1) % self.args.detach_gap == 0:
                if prev_hid is not None:
                    prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            next_state, reward, done, info = self.env.step(actual)

            misc['alive_mask'] = current_alive_mask.reshape(reward.shape)
            if graph_reg_loss is not None:
                misc['graph_reg_loss'] = graph_reg_loss
            for key, graph_stat_value in graph_stats.items():
                misc[key] = graph_stat_value

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat)


    def compute_grad(self, batch):
        stat = dict()
        # num_actions: number of discrete actions in the action space
        num_actions = self.args.num_actions
        # dim_actions: number of action heads
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        # rewards: [batch_size * n]
        rewards = torch.Tensor(batch.reward)
        # episode_mask: [batch_size * n]
        episode_masks = torch.Tensor(batch.episode_mask)
        # episode_mini_mask: [batch_size * n]
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        # actions: [batch_size * n * dim_actions] have been detached
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        # alive_masks: [batch_size * n]
        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)

        coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
        deltas = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            active_advantages = advantages.view(-1)[alive_masks > 0]
            if active_advantages.numel() > 1:
                active_mean = active_advantages.mean()
                active_std = active_advantages.std().clamp(min=1e-8)
                advantages = (advantages - active_mean) / active_std
            
        # element of log_p_a: [(batch_size*n) * num_actions[i]]
        log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
        # actions: [(batch_size*n) * dim_actions]
        actions = actions.contiguous().view(-1, dim_actions)

        if self.args.advantages_per_action:
            # log_prob: [(batch_size*n) * dim_actions]
            log_prob = multinomials_log_densities(actions, log_p_a)
            # the log prob of each action head is multiplied by the advantage
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            # log_prob: [(batch_size*n) * 1]
            log_prob = multinomials_log_density(actions, log_p_a)
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        graph_reg_loss = None
        graph_reg_coeff = float(getattr(self.args, 'dynamic_graph_reg_coeff', 1.0))
        if graph_reg_coeff > 0.0:
            graph_losses = [item.get('graph_reg_loss') for item in batch.misc if isinstance(item, dict) and item.get('graph_reg_loss') is not None]
            if graph_losses:
                graph_reg_loss = torch.stack(graph_losses).sum()
                loss = loss + (graph_reg_coeff * graph_reg_loss)
                stat['graph_reg_loss'] = graph_reg_loss.item()
            else:
                stat['graph_reg_loss'] = 0.0
        else:
            stat['graph_reg_loss'] = 0.0

        # entropy regularization term
        entropy = 0
        for i in range(len(log_p_a)):
            entropy_per_agent = -(log_p_a[i] * log_p_a[i].exp()).sum(dim=1)
            entropy += (entropy_per_agent * alive_masks).sum()
        stat['entropy'] = entropy.item()
        if self.args.entr > 0:
            loss -= self.args.entr * entropy

        graph_stat_keys = (
            'dynamic_edge_density',
            'dynamic_candidate_density',
            'dynamic_graph_change',
            'dynamic_locality_gap',
            'dynamic_hard_edge_density',
            'dynamic_avg_degree',
            'dynamic_active_agents',
        )
        for key in graph_stat_keys:
            values = [item.get(key) for item in batch.misc if isinstance(item, dict) and key in item]
            if values:
                stat[key] = float(sum(values))
            else:
                stat[key] = 0.0

        loss.backward()

        graph_grad_sq = 0.0
        for name, parameter in self.policy_net.named_parameters():
            if name.startswith('dynamic_graph.') and parameter.grad is not None:
                graph_grad_sq += float(parameter.grad.detach().pow(2).sum().item())
        stat['dynamic_graph_grad_norm'] = graph_grad_sq ** 0.5
        stat['num_grad_updates'] = 1

        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
#         for name, param in self.policy_net.named_parameters():
#             print(name)
#             print(param.grad)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        max_grad_norm = float(getattr(self.args, 'max_grad_norm', 0.0))
        if max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.params, max_grad_norm)
        self.optimizer.step()

        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
