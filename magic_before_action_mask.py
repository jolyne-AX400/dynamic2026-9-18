import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from action_utils import select_action, translate_action
from gnn_layers import GraphAttention
from dynamic_graph_core import DynamicGraphComposer, DynamicGraphConfig

class MAGIC(nn.Module):
    """
    The communication protocol of Multi-Agent Graph AttentIon Communication (MAGIC)
    """
    def __init__(self, args):
        super(MAGIC, self).__init__()
        """
        Initialization method for the MAGIC communication protocol (2 rounds of communication)

        Arguements:
            args (Namespace): Parse arguments
        """

        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_arch = getattr(args, 'comm_arch', 'magic')
        self.use_dynamic_graph = self.comm_arch == 'dynamic'
        
        dropout = 0
        negative_slope = 0.2

        # initialize sub-processors
        self.sub_processor1 = GraphAttention(args.hid_size, args.gat_hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads, self_loop_type=args.self_loop_type1, average=False, normalize=args.first_gat_normalize)
        self.sub_processor2 = GraphAttention(args.gat_hid_size*args.gat_num_heads, args.hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads_out, self_loop_type=args.self_loop_type2, average=True, normalize=args.second_gat_normalize)
        # initialize the gat encoder for the Scheduler
        if args.use_gat_encoder:
            self.gat_encoder = GraphAttention(args.hid_size, args.gat_encoder_out_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.ge_num_heads, self_loop_type=1, average=True, normalize=args.gat_encoder_normalize)

        self.dynamic_graph = None
        if self.use_dynamic_graph:
            dynamic_hidden = getattr(args, 'dynamic_hidden', args.hid_size)
            self.dynamic_graph = DynamicGraphComposer(
                node_feat_dim=args.hid_size,
                comm_feat_dim=args.hid_size,
                hidden=dynamic_hidden,
                use_learned_adj=getattr(args, 'dynamic_use_learned_adj', True),
                use_info_controller=getattr(args, 'dynamic_use_info_controller', True),
                use_ude=getattr(args, 'dynamic_use_ude', False),
                use_magic_importance=getattr(args, 'dynamic_use_magic_importance', True),
                magic_gate_mix=getattr(args, 'dynamic_magic_gate_mix', 0.5),
            )
            self.prev_dynamic_adj1 = None
            self.prev_dynamic_adj2 = None
            self.last_graph_reg_loss = None
            self.last_graph_stats = {}
            self.last_dynamic_graph = None

        self.obs_encoder = nn.Linear(args.obs_size, args.hid_size)

        self.init_hidden(args.batch_size)
        self.lstm_cell= nn.LSTMCell(args.hid_size, args.hid_size)

        # initialize mlp layers for the sub-schedulers
        if not args.first_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))
                
        if args.learn_second_graph and not args.second_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))

        if args.message_encoder:
            self.message_encoder = nn.Linear(args.hid_size, args.hid_size)
        if args.message_decoder:
            self.message_decoder = nn.Linear(args.hid_size, args.hid_size)

        # initialize weights as 0
        if args.comm_init == 'zeros':
            if args.message_encoder:
                self.message_encoder.weight.data.zero_()
            if args.message_decoder:
                self.message_decoder.weight.data.zero_()
            if not args.first_graph_complete:
                self.sub_scheduler_mlp1.apply(self.init_linear)
            if args.learn_second_graph and not args.second_graph_complete:
                self.sub_scheduler_mlp2.apply(self.init_linear)
                   
        # initialize the action head (in practice, one action head is used)
        self.action_heads = nn.ModuleList([nn.Linear(2*args.hid_size, o)
                                        for o in args.naction_heads])
        # initialize the value head
        self.value_head = nn.Linear(2 * self.hid_size, 1)


    def forward(self, x, info={}):
        """
        Forward function of MAGIC (two rounds of communication)

        Arguments:
            x (list): a list for the input of the communication protocol [observations, (previous hidden states, previous cell states)]
            observations (tensor): the observations for all agents [1 (batch_size) * n * obs_size]
            previous hidden/cell states (tensor): the hidden/cell states from the previous time steps [n * hid_size]

        Returns:
            action_out (list): a list of tensors of size [1 (batch_size) * n * num_actions] that represent output policy distributions
            value_head (tensor): estimated values [n * 1]
            next hidden/cell states (tensor): next hidden/cell states [n * hid_size]
        """

        # n: number of agents

        obs, extras = x

        # encoded_obs: [1 (batch_size) * n * hid_size]
        encoded_obs = self.obs_encoder(obs)
        hidden_state, cell_state = extras

        batch_size = encoded_obs.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        agent_mask = agent_mask.to(device=encoded_obs.device, dtype=encoded_obs.dtype)

        # if self.args.comm_mask_zero == True, block the communiction (can also comment out the protocol to make training faster)
        if self.args.comm_mask_zero:
            agent_mask *= torch.zeros(n, 1)

        hidden_state, cell_state = self.lstm_cell(encoded_obs.squeeze(), (hidden_state, cell_state))

        # comm: [n * hid_size]
        comm = hidden_state
        if self.args.message_encoder:
            comm = self.message_encoder(comm)
            
        # mask communcation from dead agents (only effective in Traffic Junction)
        comm = comm * agent_mask
        comm_ori = comm.clone()

        # sub-scheduler 1
        # if args.first_graph_complete == True, sub-scheduler 1 will be disabled
        positions = self.extract_positions(info) if self.use_dynamic_graph else None

        if self.comm_arch == 'dynamic':
            adj1, dynamic_meta1 = self.build_dynamic_adjacency(
                comm_ori,
                positions,
                agent_mask,
                previous_probs=self.prev_dynamic_adj1,
            )
            encoded_state1 = None
        elif not self.args.first_graph_complete:
            if self.args.use_gat_encoder:
                adj_complete = self.get_complete_graph(agent_mask)
                encoded_state1 = self.gat_encoder(comm, adj_complete)
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, encoded_state1, agent_mask, self.args.directed)
            else:
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask, self.args.directed)
        else:
            adj1 = self.get_complete_graph(agent_mask)

        # sub-processor 1
        comm = F.elu(self.sub_processor1(comm, adj1))
        
        # sub-scheduler 2
        if self.comm_arch == 'dynamic':
            if getattr(self.args, 'dynamic_rebuild_second_graph', False):
                adj2, dynamic_meta2 = self.build_dynamic_adjacency(
                    comm,
                    positions,
                    agent_mask,
                    previous_probs=self.prev_dynamic_adj2,
                )
            else:
                adj2 = adj1
                dynamic_meta2 = dynamic_meta1
        elif self.args.learn_second_graph and not self.args.second_graph_complete:
            if self.args.use_gat_encoder:
                if self.args.first_graph_complete:
                    adj_complete = self.get_complete_graph(agent_mask)
                    encoded_state2 = self.gat_encoder(comm_ori, adj_complete)
                else:
                    encoded_state2 = encoded_state1
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, encoded_state2, agent_mask, self.args.directed)
            else:
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, comm_ori, agent_mask, self.args.directed)
        elif not self.args.learn_second_graph and not self.args.second_graph_complete:
            adj2 = adj1
        else:
            adj2 = self.get_complete_graph(agent_mask)
            
        # sub-processor 2
        comm = self.sub_processor2(comm, adj2)
        if self.use_dynamic_graph:
            self.prev_dynamic_adj1 = dynamic_meta1.get('adj_probs').detach() if dynamic_meta1.get('adj_probs') is not None else None
            self.prev_dynamic_adj2 = dynamic_meta2.get('adj_probs').detach() if dynamic_meta2.get('adj_probs') is not None else None
            self.last_graph_reg_loss, self.last_graph_stats = self.compute_dynamic_graph_loss(dynamic_meta1, dynamic_meta2)
            self.last_dynamic_graph = {
                'adj1': adj1.detach(),
                'adj2': adj2.detach(),
                'meta1': dynamic_meta1,
                'meta2': dynamic_meta2,
            }
        
        # mask communication to dead agents (only effective in Traffic Junction)
        comm = comm * agent_mask
        
        if self.args.message_decoder:
            comm = self.message_decoder(comm)

        value_head = self.value_head(torch.cat((hidden_state, comm), dim=-1))
        h = hidden_state.view(batch_size, n, self.hid_size)
        c = comm.view(batch_size, n, self.hid_size)

        action_out = [F.log_softmax(action_head(torch.cat((h, c), dim=-1)), dim=-1) for action_head in self.action_heads]

        return action_out, value_head, (hidden_state.clone(), cell_state.clone())

    def get_agent_mask(self, batch_size, info):
        """
        Function to generate agent mask to mask out inactive agents (only effective in Traffic Junction)

        Returns:
            num_agents_alive (int): number of active agents
            agent_mask (tensor): [n, 1]
        """

        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()

        return num_agents_alive, agent_mask

    def extract_positions(self, info):
        if not isinstance(info, dict):
            return None

        if 'positions' in info and info['positions'] is not None:
            return info['positions'][:self.nagents]

        if 'car_loc' in info and info['car_loc'] is not None:
            return info['car_loc'][:self.nagents]

        if 'predator_locs' in info and info['predator_locs'] is not None:
            return info['predator_locs'][:self.nagents]

        if 'left_team' in info:
            positions = []
            left_count = min(getattr(self.args, 'num_controlled_lagents', self.nagents), self.nagents)
            left_team = np.asarray(info['left_team'])
            if left_team.ndim >= 2 and left_team.shape[0] >= left_count:
                positions.append(left_team[:left_count, :2])

            right_count = getattr(self.args, 'num_controlled_ragents', 0)
            if right_count > 0 and 'right_team' in info:
                right_team = np.asarray(info['right_team'])
                if right_team.ndim >= 2 and right_team.shape[0] >= right_count:
                    positions.append(right_team[:right_count, :2])

            if positions:
                return np.concatenate(positions, axis=0)[:self.nagents]

        return None

    def build_dynamic_adjacency(self, comm_feats, positions, agent_mask, previous_probs=None):
        task_gate = self.get_complete_graph(agent_mask)
        config = DynamicGraphConfig(
            comm_range=getattr(self.args, 'dynamic_comm_range', float(self.nagents)),
            top_k=getattr(self.args, 'dynamic_top_k', 4),
            phys_soft_tau=getattr(self.args, 'dynamic_phys_soft_tau', 8.0),
            phys_hard_factor=getattr(self.args, 'dynamic_phys_hard_factor', 2.0),
            phys_eps=getattr(self.args, 'dynamic_phys_eps', 1e-2),
            phys_prior_mix=getattr(self.args, 'dynamic_phys_prior_mix', 0.35),
            gate_min=getattr(self.args, 'dynamic_gate_min', 0.2),
            adj_threshold=getattr(self.args, 'dynamic_adj_threshold', 0.4),
            use_soft_actor_adj=getattr(self.args, 'dynamic_use_soft_adj', False),
            use_phys_candidate_mask=getattr(self.args, 'dynamic_use_phys_candidate_mask', False),
            temporal_smooth=getattr(self.args, 'dynamic_temporal_smooth', 0.0),
            candidate_residual=getattr(self.args, 'dynamic_candidate_residual', 0.35),
            min_edge_density=getattr(self.args, 'dynamic_min_edge_density', 0.28),
            candidate_soft_floor=getattr(self.args, 'dynamic_candidate_soft_floor', 0.35),
            straight_through_hard=getattr(self.args, 'dynamic_straight_through', True),
        )
        adj, meta = self.dynamic_graph.build_actor_adjacency(
            node_feats=comm_feats,
            comm_feats=comm_feats,
            positions=positions,
            config=config,
            task_gate=task_gate,
            previous_adj=previous_probs,
        )
        if not self.args.directed:
            adj = torch.max(adj, adj.transpose(0, 1))
            if meta.get('adj_probs') is not None:
                meta['adj_probs'] = torch.max(meta['adj_probs'], meta['adj_probs'].transpose(0, 1))
            if meta.get('candidate_mask') is not None:
                meta['candidate_mask'] = torch.max(meta['candidate_mask'], meta['candidate_mask'].transpose(0, 1))
            if meta.get('candidate_weight') is not None:
                meta['candidate_weight'] = torch.max(meta['candidate_weight'], meta['candidate_weight'].transpose(0, 1))
            if meta.get('temporal_delta') is not None:
                meta['temporal_delta'] = torch.max(meta['temporal_delta'], meta['temporal_delta'].transpose(0, 1))
            if meta.get('valid_edge_mask') is not None:
                meta['valid_edge_mask'] = torch.max(meta['valid_edge_mask'], meta['valid_edge_mask'].transpose(0, 1))
            if meta.get('topk_mask') is not None:
                meta['topk_mask'] = torch.max(meta['topk_mask'], meta['topk_mask'].transpose(0, 1))
        return adj, meta

    def offdiag_mean(self, matrix, valid_mask=None):
        if matrix is None:
            return None
        count = matrix.size(0)
        if count <= 1:
            return matrix.new_tensor(0.0)
        offdiag = 1.0 - torch.eye(count, device=matrix.device, dtype=matrix.dtype)
        if valid_mask is not None:
            offdiag = offdiag * valid_mask.to(device=matrix.device, dtype=matrix.dtype)
        denom = offdiag.sum().clamp(min=1.0)
        return (matrix * offdiag).sum() / denom

    def compute_dynamic_graph_loss(self, *metas):
        total_loss = None
        stats = {
            'dynamic_edge_density': 0.0,
            'dynamic_candidate_density': 0.0,
            'dynamic_graph_change': 0.0,
            'dynamic_locality_gap': 0.0,
            'dynamic_hard_edge_density': 0.0,
            'dynamic_avg_degree': 0.0,
            'dynamic_active_agents': 0.0,
        }
        valid_count = 0

        sparsity_coeff = float(getattr(self.args, 'dynamic_sparsity_coeff', 0.0))
        smoothness_coeff = float(getattr(self.args, 'dynamic_smoothness_coeff', 0.0))
        locality_coeff = float(getattr(self.args, 'dynamic_locality_coeff', 0.0))

        seen_metas = set()
        for meta in metas:
            if not meta:
                continue
            # When the same graph is reused for both communication rounds,
            # count its regularizer and statistics only once.
            meta_id = id(meta)
            if meta_id in seen_metas:
                continue
            seen_metas.add(meta_id)
            adj_probs = meta.get('adj_probs')
            if adj_probs is None:
                continue
            valid_count += 1

            valid_mask = meta.get('valid_edge_mask')
            density = self.offdiag_mean(adj_probs, valid_mask)
            candidate_density = self.offdiag_mean(meta.get('candidate_mask'), valid_mask)
            temporal_delta = self.offdiag_mean(meta.get('temporal_delta'), valid_mask)
            locality_gap = None
            candidate_weight = meta.get('candidate_weight')
            if candidate_weight is not None:
                locality_gap = self.offdiag_mean(adj_probs * (1.0 - candidate_weight), valid_mask)

            hard_adj = meta.get('adj_input')
            if hard_adj is None:
                hard_adj = (adj_probs > float(getattr(self.args, 'dynamic_adj_threshold', 0.4))).to(adj_probs.dtype)
            hard_density = self.offdiag_mean((hard_adj > 0).to(adj_probs.dtype), valid_mask)
            active_node_mask = meta.get('active_node_mask')
            if active_node_mask is not None:
                active_nodes = active_node_mask.to(device=adj_probs.device, dtype=adj_probs.dtype).sum()
            elif valid_mask is not None:
                active_nodes = (valid_mask.sum(dim=1) > 0).to(adj_probs.dtype).sum()
            else:
                active_nodes = adj_probs.new_tensor(float(adj_probs.size(0)))
            avg_degree = ((hard_adj > 0).to(adj_probs.dtype) * (valid_mask if valid_mask is not None else 1.0)).sum()
            avg_degree = avg_degree / active_nodes.clamp(min=1.0)

            stats['dynamic_edge_density'] += float(density.item()) if density is not None else 0.0
            stats['dynamic_candidate_density'] += float(candidate_density.item()) if candidate_density is not None else 0.0
            stats['dynamic_graph_change'] += float(temporal_delta.item()) if temporal_delta is not None else 0.0
            stats['dynamic_locality_gap'] += float(locality_gap.item()) if locality_gap is not None else 0.0
            stats['dynamic_hard_edge_density'] += float(hard_density.item()) if hard_density is not None else 0.0
            stats['dynamic_avg_degree'] += float(avg_degree.item())
            stats['dynamic_active_agents'] += float(active_nodes.item())

            loss_terms = []
            if sparsity_coeff > 0.0 and density is not None:
                loss_terms.append(sparsity_coeff * density)
            if smoothness_coeff > 0.0 and temporal_delta is not None:
                loss_terms.append(smoothness_coeff * temporal_delta)
            if locality_coeff > 0.0 and locality_gap is not None:
                loss_terms.append(locality_coeff * locality_gap)

            if loss_terms:
                meta_loss = sum(loss_terms)
                total_loss = meta_loss if total_loss is None else total_loss + meta_loss

        if valid_count > 0:
            for key in stats:
                stats[key] /= float(valid_count)

        if total_loss is None:
            total_loss = next(self.parameters()).new_tensor(0.0)

        return total_loss, stats

    def build_magic_adjacency(self, comm, fallback_comm, agent_mask, first_round, cached_encoded_state=None, previous_adj=None):
        if first_round:
            if not self.args.first_graph_complete:
                if self.args.use_gat_encoder:
                    adj_complete = self.get_complete_graph(agent_mask)
                    encoded_state = self.gat_encoder(comm, adj_complete)
                    adj = self.sub_scheduler(self.sub_scheduler_mlp1, encoded_state, agent_mask, self.args.directed)
                else:
                    encoded_state = None
                    adj = self.sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask, self.args.directed)
            else:
                encoded_state = None
                adj = self.get_complete_graph(agent_mask)
            return adj, encoded_state

        if self.args.learn_second_graph and not self.args.second_graph_complete:
            if self.args.use_gat_encoder:
                if self.args.first_graph_complete:
                    adj_complete = self.get_complete_graph(agent_mask)
                    encoded_state2 = self.gat_encoder(fallback_comm, adj_complete)
                else:
                    encoded_state2 = cached_encoded_state
                adj = self.sub_scheduler(self.sub_scheduler_mlp2, encoded_state2, agent_mask, self.args.directed)
            else:
                adj = self.sub_scheduler(self.sub_scheduler_mlp2, fallback_comm, agent_mask, self.args.directed)
        elif not self.args.learn_second_graph and not self.args.second_graph_complete:
            adj = previous_adj
        else:
            adj = self.get_complete_graph(agent_mask)
        return adj, cached_encoded_state

    def init_linear(self, m):
        """
        Function to initialize the parameters in nn.Linear as o 
        """
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)
        
    def init_hidden(self, batch_size):
        """
        Function to initialize the hidden states and cell states
        """
        self.prev_dynamic_adj1 = None
        self.prev_dynamic_adj2 = None
        self.last_graph_reg_loss = None
        self.last_graph_stats = {}
        self.last_dynamic_graph = None
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))
    
    
    def sub_scheduler(self, sub_scheduler_mlp, hidden_state, agent_mask, directed=True):
        """
        Function to perform a sub-scheduler

        Arguments: 
            sub_scheduler_mlp (nn.Sequential): the MLP layers in a sub-scheduler
            hidden_state (tensor): the encoded messages input to the sub-scheduler [n * hid_size]
            agent_mask (tensor): [n * 1]
            directed (bool): decide if generate directed graphs

        Return:
            adj (tensor): a adjacency matrix which is the communication graph [n * n]  
        """

        # hidden_state: [n * hid_size]
        n = self.args.nagents
        hid_size = hidden_state.size(-1)
        # hard_attn_input: [n * n * (2*hid_size)]
        hard_attn_input = torch.cat([hidden_state.repeat(1, n).view(n * n, -1), hidden_state.repeat(n, 1)], dim=1).view(n, -1, 2 * hid_size)
        # hard_attn_output: [n * n * 2]
        if directed:
            hard_attn_output = F.gumbel_softmax(sub_scheduler_mlp(hard_attn_input), hard=True)
        else:
            hard_attn_output = F.gumbel_softmax(0.5*sub_scheduler_mlp(hard_attn_input)+0.5*sub_scheduler_mlp(hard_attn_input.permute(1,0,2)), hard=True)
        # hard_attn_output: [n * n * 1]
        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
        # agent_mask and agent_mask_transpose: [n * n]
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        # adj: [n * n]
        adj = hard_attn_output.squeeze() * agent_mask * agent_mask_transpose
        
        return adj
    
    def get_complete_graph(self, agent_mask):
        """
        Function to generate a complete graph, and mask it with agent_mask
        """
        n = self.args.nagents
        adj = torch.ones(n, n, device=agent_mask.device, dtype=agent_mask.dtype)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = adj * agent_mask * agent_mask_transpose
        
        return adj
