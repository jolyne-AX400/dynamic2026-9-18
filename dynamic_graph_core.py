import importlib
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGraphConfig:
    """Runtime configuration for dynamic graph construction."""

    def __init__(
        self,
        comm_range: float,
        top_k: Optional[int] = 4,
        phys_soft_tau: float = 8.0,
        phys_hard_factor: float = 2.0,
        phys_eps: float = 1e-2,
        phys_prior_mix: float = 0.35,
        gate_min: float = 0.2,
        adj_threshold: float = 0.4,
        use_soft_actor_adj: bool = False,
        use_phys_candidate_mask: bool = False,
        temporal_smooth: float = 0.0,
        candidate_residual: float = 0.0,
        min_edge_density: float = 0.0,
        candidate_soft_floor: float = 0.0,
        straight_through_hard: bool = True,
    ):
        self.comm_range = float(comm_range)
        self.top_k = top_k
        self.phys_soft_tau = float(phys_soft_tau)
        self.phys_hard_factor = float(phys_hard_factor)
        self.phys_eps = float(phys_eps)
        self.phys_prior_mix = float(phys_prior_mix)
        self.gate_min = float(gate_min)
        self.adj_threshold = float(adj_threshold)
        self.use_soft_actor_adj = bool(use_soft_actor_adj)
        self.use_phys_candidate_mask = bool(use_phys_candidate_mask)
        self.temporal_smooth = float(temporal_smooth)
        self.candidate_residual = float(candidate_residual)
        self.min_edge_density = float(min_edge_density)
        self.candidate_soft_floor = float(candidate_soft_floor)
        self.straight_through_hard = bool(straight_through_hard)

    def clamped_gate_min(self) -> float:
        return min(max(float(self.gate_min), 0.0), 0.95)

    def clamped_phys_prior_mix(self) -> float:
        return min(max(float(self.phys_prior_mix), 0.0), 1.0)

    def clamped_temporal_smooth(self) -> float:
        return min(max(float(self.temporal_smooth), 0.0), 0.99)

    def clamped_candidate_residual(self) -> float:
        return min(max(float(self.candidate_residual), 0.0), 1.0)

    def clamped_min_edge_density(self) -> float:
        return min(max(float(self.min_edge_density), 0.0), 0.95)

    def clamped_candidate_soft_floor(self) -> float:
        return min(max(float(self.candidate_soft_floor), 0.0), 1.0)


def _make_offdiag_mask(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.ones((size, size), device=device, dtype=dtype)
    idx = torch.arange(size, device=device)
    mask[idx, idx] = 0.0
    return mask


def _rowwise_topk_mask(
    scores: torch.Tensor,
    top_k: Optional[int],
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_nodes = scores.size(0)
    offdiag_mask = _make_offdiag_mask(num_nodes, scores.device, scores.dtype)
    valid_mask = offdiag_mask
    if candidate_mask is not None:
        valid_mask = valid_mask * candidate_mask.to(device=scores.device, dtype=scores.dtype)

    if top_k is None or top_k <= 0 or num_nodes <= 1:
        return valid_mask

    k = min(int(top_k), num_nodes - 1)
    masked_scores = scores.masked_fill(valid_mask <= 0, -1e9)
    _, indices = torch.topk(masked_scores, k=k, dim=1)
    topk_mask = torch.zeros_like(scores)
    topk_mask.scatter_(1, indices, 1.0)
    return topk_mask * valid_mask


def _offdiag_mean(matrix: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if matrix is None:
        return None
    count = matrix.size(0)
    if count <= 1:
        return matrix.new_tensor(0.0)
    offdiag_mask = _make_offdiag_mask(count, matrix.device, matrix.dtype)
    denom = offdiag_mask.sum().clamp(min=1.0)
    return (matrix * offdiag_mask).sum() / denom


def _masked_offdiag_mean(
    matrix: Optional[torch.Tensor],
    valid_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Mean over valid, non-self edges only.

    Traffic Junction keeps a fixed number of vehicle slots although many of
    them are inactive.  Averaging over all slots makes graph density depend on
    the arrival process rather than on the learned communication policy.
    """
    if matrix is None:
        return None
    mask = _make_offdiag_mask(matrix.size(0), matrix.device, matrix.dtype)
    if valid_mask is not None:
        mask = mask * valid_mask.to(device=matrix.device, dtype=matrix.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (matrix * mask).sum() / denom


def _as_tensor(
    x: Any,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype or x.dtype)
    return torch.tensor(x, dtype=dtype or torch.float32, device=device)


def _resolve_num_nodes(
    positions: Any,
    fallback_num_nodes: Optional[int] = None,
) -> int:
    if positions is not None:
        if torch.is_tensor(positions):
            return int(positions.size(0))
        return int(len(positions))
    if fallback_num_nodes is None:
        raise ValueError("num_nodes must be provided when positions is None")
    return int(fallback_num_nodes)


def build_hard_physical_adjacency(
    positions: Any,
    comm_range: float,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    num_nodes: Optional[int] = None,
) -> torch.Tensor:
    """Binary physical adjacency based on pairwise Euclidean distance."""
    if positions is None:
        count = _resolve_num_nodes(None, fallback_num_nodes=num_nodes)
        adj = torch.ones((count, count), device=device, dtype=dtype or torch.float32)
        idx = torch.arange(count, device=adj.device)
        adj[idx, idx] = 0.0
        return adj

    pos = _as_tensor(positions, device=device, dtype=dtype or torch.float32)
    dist = torch.cdist(pos, pos, p=2)
    adj = (dist <= float(comm_range)).to(dtype=pos.dtype)
    idx = torch.arange(adj.size(0), device=adj.device)
    adj[idx, idx] = 0.0
    return adj


def build_soft_physical_weights(
    positions: Any,
    comm_range: float,
    tau: float = 4.0,
    hard_factor: float = 2.0,
    eps: float = 1e-2,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    num_nodes: Optional[int] = None,
) -> torch.Tensor:
    """Soft communication weights with exponential distance decay."""
    if positions is None:
        count = _resolve_num_nodes(None, fallback_num_nodes=num_nodes)
        weights = torch.ones((count, count), device=device, dtype=dtype or torch.float32)
        idx = torch.arange(count, device=weights.device)
        weights[idx, idx] = 0.0
        return weights

    pos = _as_tensor(positions, device=device, dtype=dtype or torch.float32)
    dist = torch.cdist(pos, pos, p=2)
    tau = max(float(tau), 1e-6)
    weights = torch.exp(-dist / tau)

    if hard_factor is not None and float(hard_factor) > 0:
        hard_range = float(comm_range) * float(hard_factor)
        weights = weights * (dist <= hard_range).to(dtype=weights.dtype)

    if eps is not None and float(eps) > 0:
        weights = torch.where(weights >= float(eps), weights, torch.zeros_like(weights))

    idx = torch.arange(weights.size(0), device=weights.device)
    weights[idx, idx] = 0.0
    return weights


def pairwise_gate_from_node_gates(node_gates: torch.Tensor, gate_min: float = 0.2) -> torch.Tensor:
    """Convert node-level gates into a symmetric pairwise gate matrix."""
    gate_min = min(max(float(gate_min), 0.0), 0.95)
    gates = gate_min + (1.0 - gate_min) * node_gates.clamp(0.0, 1.0)
    return gates.unsqueeze(1) * gates.unsqueeze(0)


def to_actor_adjacency(
    adj_probs: torch.Tensor,
    threshold: float = 0.4,
    use_soft: bool = False,
    straight_through: bool = True,
) -> torch.Tensor:
    """Convert adjacency probabilities to the form consumed by a graph policy."""
    if use_soft:
        return adj_probs.clamp(0.0, 1.0)
    hard_adj = (adj_probs > float(threshold)).to(dtype=adj_probs.dtype)
    if straight_through:
        # Hard edges in the forward pass, identity gradient in the backward
        # pass.  A plain comparison would completely freeze the graph scorer.
        return hard_adj.detach() - adj_probs.detach() + adj_probs
    return hard_adj


class SimpleGATLayer(nn.Module):
    """Single-head graph attention layer supporting hard or soft adjacency."""

    def __init__(self, in_dim: int, out_dim: int, leaky_relu_neg_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(out_dim * 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_neg_slope)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor]) -> torch.Tensor:
        Wh = self.W(x)
        num_nodes = Wh.size(0)

        Wh_i = Wh.unsqueeze(1).expand(-1, num_nodes, -1)
        Wh_j = Wh.unsqueeze(0).expand(num_nodes, -1, -1)
        cat = torch.cat([Wh_i, Wh_j], dim=-1)
        e = self.leaky_relu(self.a(cat).squeeze(-1))

        if adj is None:
            mask = torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=Wh.device)
            soft_w = None
        else:
            if adj.dtype.is_floating_point:
                soft_w = adj.clamp(min=0.0, max=1.0)
                mask = soft_w > 0
            else:
                soft_w = None
                mask = adj > 0
            eye = torch.eye(num_nodes, dtype=torch.bool, device=Wh.device)
            mask = mask | eye

        if soft_w is not None:
            e = e + torch.where(mask, torch.log(soft_w + 1e-6), torch.zeros_like(e))

        e = torch.where(mask, e, torch.full_like(e, -9e15))
        alpha = torch.softmax(e, dim=1)
        return torch.matmul(alpha, Wh)


class GraphPolicyActor(nn.Module):
    """Two-layer GAT actor for continuous control on dynamic graphs."""

    def __init__(
        self,
        node_in_dim: int,
        gat_hidden: int = 64,
        action_dim: int = 3,
        std_min: float = 0.05,
        std_max: float = 0.6,
    ):
        super().__init__()
        self.std_min = float(std_min)
        self.std_max = float(std_max)
        if self.std_min <= 0.0:
            raise ValueError("std_min must be positive")
        if self.std_max <= self.std_min:
            raise ValueError("std_max must be greater than std_min")

        self.gat1 = SimpleGATLayer(node_in_dim, gat_hidden)
        self.gat2 = SimpleGATLayer(gat_hidden, gat_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(gat_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(self, node_feats: torch.Tensor, adj: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.elu(self.gat1(node_feats, adj))
        x = F.elu(self.gat2(x, adj))
        mean = self.mlp(x)
        bounded_std = self.std_min + (self.std_max - self.std_min) * torch.sigmoid(self.log_std)
        std = bounded_std.unsqueeze(0).expand_as(mean)
        return mean, std


class SingleLayerGraphPolicyActor(nn.Module):
    """A lighter single-layer GAT actor for ablation or low-budget setups."""

    def __init__(
        self,
        node_in_dim: int,
        gat_hidden: int = 64,
        action_dim: int = 3,
        std_min: float = 0.05,
        std_max: float = 0.6,
    ):
        super().__init__()
        self.std_min = float(std_min)
        self.std_max = float(std_max)
        if self.std_min <= 0.0:
            raise ValueError("std_min must be positive")
        if self.std_max <= self.std_min:
            raise ValueError("std_max must be greater than std_min")

        self.gat = SimpleGATLayer(node_in_dim, gat_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(gat_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(self, node_feats: torch.Tensor, adj: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.elu(self.gat(node_feats, adj))
        mean = self.mlp(x)
        bounded_std = self.std_min + (self.std_max - self.std_min) * torch.sigmoid(self.log_std)
        std = bounded_std.unsqueeze(0).expand_as(mean)
        return mean, std


class AdjPredictor(nn.Module):
    """Learnable adjacency predictor operating on communication features."""

    def __init__(self, node_in_dim: int, hidden: int = 64):
        super().__init__()
        self.node_enc = nn.Linear(node_in_dim, hidden)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        top_k: Optional[int] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = node_feats.size(0)
        h = F.relu(self.node_enc(node_feats))
        hi = h.unsqueeze(1).expand(-1, num_nodes, -1)
        hj = h.unsqueeze(0).expand(num_nodes, -1, -1)
        cat = torch.cat([hi, hj], dim=-1)
        logits = self.pair_mlp(cat).squeeze(-1)

        idx = torch.arange(num_nodes, device=logits.device)
        logits[idx, idx] = -9e15
        probs = torch.sigmoid(logits)

        if top_k is None or top_k <= 0:
            return probs, logits

        if candidate_mask is not None:
            candidate_mask = candidate_mask.to(device=probs.device, dtype=probs.dtype)
            scores = probs.masked_fill(candidate_mask <= 0, -1.0)
        else:
            scores = probs

        _, indices = torch.topk(scores, k=min(int(top_k), num_nodes - 1), dim=1)
        hard_mask = torch.zeros_like(probs)
        hard_mask.scatter_(1, indices, 1.0)
        if candidate_mask is not None:
            hard_mask = hard_mask * candidate_mask

        probs_st = hard_mask.detach() - probs.detach() + probs
        return probs_st, logits

#边重要性打分部分
class EdgeUtilityPredictor(nn.Module):
    """Pairwise edge scorer that mixes node state with relative position cues."""

    def __init__(self, node_in_dim: int, hidden: int = 64):
        super().__init__()
        hidden_mid = max(16, hidden // 2)
        self.node_enc = nn.Linear(node_in_dim, hidden)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden_mid),
            nn.ReLU(),
            nn.Linear(hidden_mid, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = node_feats.size(0)
        h = F.relu(self.node_enc(node_feats))
        hi = h.unsqueeze(1).expand(-1, num_nodes, -1)
        hj = h.unsqueeze(0).expand(num_nodes, -1, -1)

        if positions is None:
            rel_features = torch.zeros((num_nodes, num_nodes, 3), device=node_feats.device, dtype=node_feats.dtype)
        else:
            rel = positions.unsqueeze(1) - positions.unsqueeze(0)
            dist = torch.norm(rel, dim=-1, keepdim=True)
            rel_features = torch.cat([rel, dist], dim=-1)

        logits = self.pair_mlp(torch.cat([hi, hj, rel_features], dim=-1)).squeeze(-1)
        idx = torch.arange(num_nodes, device=logits.device)
        logits[idx, idx] = -9e15

        if candidate_mask is not None:
            logits = logits.masked_fill(candidate_mask <= 0, -9e15)

        probs = torch.sigmoid(logits)
        offdiag_mask = _make_offdiag_mask(num_nodes, probs.device, probs.dtype)
        probs = probs * offdiag_mask
        return probs, logits

#节点门控部分
class InfoController(nn.Module):
    """Node-level communication gate that scores how much each node should speak."""

    def __init__(self, node_in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        return self.net(node_feats).squeeze(-1)

#任务重要性部分
class TaskImportanceScorer(nn.Module):

    def __init__(self, node_in_dim: int, hidden: int = 64):
        super().__init__()
        self.node_enc = nn.Linear(node_in_dim, hidden)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = node_feats.size(0)
        h = F.relu(self.node_enc(node_feats))
        hi = h.unsqueeze(1).expand(-1, num_nodes, -1)
        hj = h.unsqueeze(0).expand(num_nodes, -1, -1)
        cat = torch.cat([hi, hj], dim=-1)
        logits = self.pair_mlp(cat).squeeze(-1)

        idx = torch.arange(num_nodes, device=logits.device)
        logits[idx, idx] = -9e15
        pair_scores = torch.sigmoid(logits)
        node_scores = pair_scores.mean(dim=1)
        return node_scores, pair_scores, logits


class UDEAdjDynamics(nn.Module):
    """Optional ODE-based adjacency dynamics module."""

    def __init__(self, node_feat_dim: int, hidden: int = 64):
        super().__init__()
        self.node_enc = nn.Linear(node_feat_dim, hidden)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _odefunc(self, node_feats: torch.Tensor, num_nodes: int):
        h = F.relu(self.node_enc(node_feats))

        def f(t, a_flat):
            _ = t
            _ = a_flat.view(num_nodes, num_nodes)
            hi = h.unsqueeze(1).expand(-1, num_nodes, -1)
            hj = h.unsqueeze(0).expand(num_nodes, -1, -1)
            cat = torch.cat([hi, hj], dim=-1)
            dA = self.pair_mlp(cat).squeeze(-1)
            idx = torch.arange(num_nodes, device=dA.device)
            dA[idx, idx] = 0.0
            dA = torch.tanh(dA)
            return dA.view(-1)

        return f

    def integrate(
        self,
        A0: torch.Tensor,
        node_feats: torch.Tensor,
        t_span: Tuple[float, float] = (0.0, 1.0),
        method: str = "rk4",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            odeint = importlib.import_module("torchdiffeq").odeint
        except Exception as exc:
            raise RuntimeError("UDEAdjDynamics requires torchdiffeq. Install it with: pip install torchdiffeq") from exc

        num_nodes = A0.size(0)
        f = self._odefunc(node_feats, num_nodes)
        t = torch.tensor([float(t_span[0]), float(t_span[1])], device=A0.device, dtype=A0.dtype)
        traj = odeint(f, A0.reshape(-1), t, method=method)
        a_end = traj[-1].view(num_nodes, num_nodes)
        probs = torch.sigmoid(a_end)
        offdiag_mask = _make_offdiag_mask(num_nodes, probs.device, probs.dtype)
        probs = probs * offdiag_mask
        return probs, a_end


class DynamicGraphComposer(nn.Module):
    """
    Reusable dynamic graph stack.

    Inputs expected from an external project:
    1. node_feats: features consumed by the graph policy or UDE branch.
    2. comm_feats: features used to predict topology and node gates.
    3. positions: node coordinates used to impose physical communication constraints.
    4. task_gate: optional external gate matrix from any task logic.
    """

    def __init__(
        self,
        node_feat_dim: int,
        comm_feat_dim: int,
        hidden: int = 64,
        use_learned_adj: bool = True,
        use_info_controller: bool = True,
        use_ude: bool = False,
        use_magic_importance: bool = True,
        magic_gate_mix: float = 0.5,
    ):
        super().__init__()
        self.use_learned_adj = bool(use_learned_adj)
        self.use_info_controller = bool(use_info_controller)
        self.use_ude = bool(use_ude)
        self.use_magic_importance = bool(use_magic_importance)
        self.magic_gate_mix = min(max(float(magic_gate_mix), 0.0), 1.0)

        self.adj_predictor = EdgeUtilityPredictor(comm_feat_dim, hidden=hidden) if self.use_learned_adj else None
        self.info_controller = None
        if self.use_learned_adj and self.use_info_controller:
            self.info_controller = InfoController(comm_feat_dim, hidden=max(16, hidden // 2))
        self.task_importance = None
        if self.use_learned_adj and self.use_info_controller and self.use_magic_importance:
            self.task_importance = TaskImportanceScorer(comm_feat_dim, hidden=hidden)
        self.ude_model = UDEAdjDynamics(node_feat_dim=node_feat_dim, hidden=hidden) if self.use_ude else None

    def build_adjacency(
        self,
        node_feats: torch.Tensor,
        comm_feats: torch.Tensor,
        positions: Any,
        config: DynamicGraphConfig,
        task_gate: Optional[torch.Tensor] = None,
        previous_adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        device = node_feats.device
        pos = None if positions is None else _as_tensor(positions, device=device, dtype=node_feats.dtype)
        task_gate_tensor = None
        if task_gate is not None:
            task_gate_tensor = _as_tensor(task_gate, device=device, dtype=node_feats.dtype)

        meta: Dict[str, Optional[torch.Tensor]] = {
            "phys_weight": None,
            "phys_mask": None,
            "candidate_weight": None,
            "candidate_mask": None,
            "raw_adj": None,
            "raw_logits": None,
            "utility_adj": None,
            "utility_logits": None,
            "base_node_gates": None,
            "node_gates": None,
            "task_node_importance": None,
            "task_pair_importance": None,
            "task_pair_logits": None,
            "pair_gates": None,
            "task_gate": task_gate_tensor,
            "previous_adj": previous_adj,
            "pre_smooth_adj_probs": None,
            "adj_probs": None,
            "temporal_delta": None,
            "density_boost": None,
            "valid_edge_mask": None,
            "topk_mask": None,
            "active_node_mask": None,
        }
        candidate_weight = None
        valid_edge_mask = _make_offdiag_mask(node_feats.size(0), device, node_feats.dtype)
        if task_gate_tensor is not None:
            valid_edge_mask = valid_edge_mask * task_gate_tensor
            active_node_mask = torch.diagonal(task_gate_tensor).clamp(0.0, 1.0)
        else:
            active_node_mask = torch.ones(node_feats.size(0), device=device, dtype=node_feats.dtype)
        meta["valid_edge_mask"] = valid_edge_mask
        meta["active_node_mask"] = active_node_mask

        if self.ude_model is not None:
            phys_mask = build_hard_physical_adjacency(
                pos,
                config.comm_range,
                device=device,
                dtype=node_feats.dtype,
                num_nodes=node_feats.size(0),
            )
            phys_mask = phys_mask * valid_edge_mask
            meta["phys_mask"] = phys_mask
            meta["candidate_mask"] = phys_mask
            meta["candidate_weight"] = phys_mask
            candidate_weight = phys_mask

            if self.use_learned_adj and self.adj_predictor is not None:
                _, raw_logits = self.adj_predictor(comm_feats, positions=pos, candidate_mask=phys_mask)
                a0_logits_masked = raw_logits.to(device) * phys_mask
                adj_probs, final_logits = self.ude_model.integrate(a0_logits_masked, node_feats, t_span=(0.0, 1.0))
                meta["raw_logits"] = raw_logits
                meta["raw_adj"] = torch.sigmoid(raw_logits)
                meta["final_logits"] = final_logits
            else:
                adj_probs = phys_mask
        else:
            phys_weight = build_soft_physical_weights(
                pos,
                comm_range=config.comm_range,
                tau=config.phys_soft_tau,
                hard_factor=config.phys_hard_factor,
                eps=config.phys_eps,
                device=device,
                dtype=node_feats.dtype,
                num_nodes=node_feats.size(0),
            )
            phys_weight = phys_weight * valid_edge_mask
            phys_mask = (phys_weight > 0).to(dtype=node_feats.dtype) * valid_edge_mask
            meta["phys_weight"] = phys_weight
            meta["phys_mask"] = phys_mask
            candidate_mask = phys_mask
            candidate_soft_floor = config.clamped_candidate_soft_floor()
            candidate_support = ((1.0 - candidate_soft_floor) * candidate_mask) + (candidate_soft_floor * phys_mask)
            candidate_weight = phys_weight * candidate_support
            meta["candidate_mask"] = candidate_mask
            meta["candidate_weight"] = candidate_weight

            if self.use_learned_adj and self.adj_predictor is not None:
                edge_mask = phys_mask if config.use_phys_candidate_mask else None
                utility_adj, utility_logits = self.adj_predictor(comm_feats, positions=pos, candidate_mask=edge_mask)
                meta["raw_adj"] = utility_adj
                meta["raw_logits"] = utility_logits
                meta["utility_adj"] = utility_adj
                meta["utility_logits"] = utility_logits

                if self.info_controller is None:
                    pair_gates = torch.ones_like(utility_adj)
                else:
                    base_node_gates = self.info_controller(comm_feats)
                    node_gates = base_node_gates
                    meta["base_node_gates"] = base_node_gates

                    if self.task_importance is not None:
                        task_node_importance, task_pair_importance, task_pair_logits = self.task_importance(comm_feats)
                        mix = self.magic_gate_mix
                        node_gates = ((1.0 - mix) * base_node_gates) + (mix * task_node_importance)
                        node_gates = node_gates.clamp(0.0, 1.0)
                        meta["task_node_importance"] = task_node_importance
                        meta["task_pair_importance"] = task_pair_importance
                        meta["task_pair_logits"] = task_pair_logits

                    pair_gates = pairwise_gate_from_node_gates(node_gates, gate_min=config.clamped_gate_min())
                    meta["node_gates"] = node_gates
                    meta["pair_gates"] = pair_gates

                phys_prior_mix = config.clamped_phys_prior_mix()
                phys_prior = (1.0 - phys_prior_mix) + (phys_prior_mix * phys_weight)
                meta["phys_prior"] = phys_prior
                utility_mix = utility_adj
                if meta["task_pair_importance"] is not None:
                    mix = self.magic_gate_mix
                    utility_mix = ((1.0 - mix) * utility_adj) + (mix * meta["task_pair_importance"])
                edge_context = 0.75 + (0.25 * pair_gates)
                adj_probs = candidate_weight * utility_mix * edge_context * phys_prior
            else:
                adj_probs = candidate_weight

            candidate_residual = config.clamped_candidate_residual()
            if candidate_residual > 0.0:
                adj_probs = ((1.0 - candidate_residual) * adj_probs) + (candidate_residual * candidate_weight)

        if task_gate_tensor is not None:
            adj_probs = adj_probs * task_gate_tensor

        min_edge_density = config.clamped_min_edge_density()
        current_density = _masked_offdiag_mean(adj_probs, valid_edge_mask)
        if current_density is not None and candidate_weight is not None and min_edge_density > 0.0 and current_density.item() < min_edge_density:
            current_value = max(float(current_density.item()), 1e-6)
            boost = min(max((min_edge_density - current_value) / max(min_edge_density, 1e-6), 0.0), 1.0)
            support = candidate_weight
            if task_gate_tensor is not None:
                support = support * task_gate_tensor
            adj_probs = adj_probs + (boost * support * (1.0 - adj_probs))
            meta["density_boost"] = adj_probs.new_tensor(boost)

        meta["pre_smooth_adj_probs"] = adj_probs
        temporal_smooth = config.clamped_temporal_smooth()
        if previous_adj is not None and temporal_smooth > 0.0:
            prev = previous_adj.to(device=adj_probs.device, dtype=adj_probs.dtype)
            prev = prev * valid_edge_mask
            adj_probs = (temporal_smooth * prev) + ((1.0 - temporal_smooth) * adj_probs)
            meta["temporal_delta"] = torch.abs(adj_probs - prev)

        # Enforce the actual per-agent communication budget.  The old code
        # parsed dynamic_top_k but never used it in the EdgeUtilityPredictor
        # path, so the reported top-k experiments were not top-k at all.
        topk_candidates = meta.get("candidate_mask")
        if topk_candidates is None:
            topk_candidates = valid_edge_mask
        topk_mask = _rowwise_topk_mask(
            adj_probs.detach(),
            config.top_k,
            candidate_mask=topk_candidates,
        )
        adj_probs = adj_probs * topk_mask
        meta["topk_mask"] = topk_mask

        adj_probs = adj_probs.clamp(min=0.0, max=1.0)
        meta["adj_probs"] = adj_probs
        return adj_probs, meta

    def build_actor_adjacency(
        self,
        node_feats: torch.Tensor,
        comm_feats: torch.Tensor,
        positions: Any,
        config: DynamicGraphConfig,
        task_gate: Optional[torch.Tensor] = None,
        previous_adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        adj_probs, meta = self.build_adjacency(
            node_feats=node_feats,
            comm_feats=comm_feats,
            positions=positions,
            config=config,
            task_gate=task_gate,
            previous_adj=previous_adj,
        )
        adj_input = to_actor_adjacency(
            adj_probs,
            threshold=config.adj_threshold,
            use_soft=config.use_soft_actor_adj,
            straight_through=config.straight_through_hard,
        )
        meta["adj_input"] = adj_input
        return adj_input, meta


ActorGAT = GraphPolicyActor
ActorGATSingle = SingleLayerGraphPolicyActor


__all__ = [
    "DynamicGraphConfig",
    "build_hard_physical_adjacency",
    "build_soft_physical_weights",
    "pairwise_gate_from_node_gates",
    "to_actor_adjacency",
    "SimpleGATLayer",
    "GraphPolicyActor",
    "SingleLayerGraphPolicyActor",
    "ActorGAT",
    "ActorGATSingle",
    "AdjPredictor",
    "EdgeUtilityPredictor",
    "InfoController",
    "TaskImportanceScorer",
    "UDEAdjDynamics",
    "DynamicGraphComposer",
]
