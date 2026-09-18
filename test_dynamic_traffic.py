import torch

from dynamic_graph_core import (
    DynamicGraphComposer,
    DynamicGraphConfig,
    to_actor_adjacency,
)
from gnn_layers import GraphAttention


def test_hard_adjacency_keeps_gradient():
    probs = torch.tensor([[0.0, 0.2], [0.8, 0.0]], requires_grad=True)
    hard = to_actor_adjacency(probs, threshold=0.4, use_soft=False, straight_through=True)
    assert torch.equal(hard.detach(), torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    hard.sum().backward()
    assert probs.grad is not None
    assert torch.all(probs.grad == 1)


def test_topk_alive_mask_and_graph_gradient():
    torch.manual_seed(0)
    node_feats = torch.randn(5, 8, dtype=torch.double, requires_grad=True)
    positions = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [-1.0, -1.0]],
        dtype=torch.double,
    )
    alive = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.double)
    task_gate = alive.unsqueeze(1) * alive.unsqueeze(0)

    model = DynamicGraphComposer(8, 8, hidden=16).double()
    config = DynamicGraphConfig(
        comm_range=3.0,
        top_k=2,
        phys_soft_tau=3.0,
        phys_hard_factor=1.0,
        use_phys_candidate_mask=True,
        use_soft_actor_adj=True,
        candidate_residual=0.15,
        min_edge_density=0.0,
    )
    adj, meta = model.build_actor_adjacency(
        node_feats=node_feats,
        comm_feats=node_feats,
        positions=positions,
        config=config,
        task_gate=task_gate,
    )

    assert torch.all((adj > 0).sum(dim=1) <= 2)
    assert torch.all(adj[4] == 0) and torch.all(adj[:, 4] == 0)
    assert meta['valid_edge_mask'].sum().item() == 12

    adj.sum().backward()
    grad_norm = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert grad_norm > 0.0


def test_sparse_gat_is_finite():
    layer = GraphAttention(
        4, 4, dropout=0.0, negative_slope=0.2,
        num_heads=1, self_loop_type=1, average=True,
    ).double()
    features = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
    adj = torch.zeros(5, 5, dtype=torch.double)
    adj[0, 1] = 0.8
    output = layer(features, adj)
    assert output.shape == (5, 4)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


if __name__ == '__main__':
    test_hard_adjacency_keeps_gradient()
    test_topk_alive_mask_and_graph_gradient()
    test_sparse_gat_is_finite()
    print('dynamic traffic tests passed')
