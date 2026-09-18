import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
    
class GraphAttention(nn.Module):
    """
    Graph-Attentional layer used in MAGIC that can process differentiable communication graphs
    """

    def __init__(self, in_features, out_features, dropout, negative_slope, num_heads=1, bias=True, self_loop_type=2, average=False, normalize=False):
        super(GraphAttention, self).__init__()
        """
        Initialization method for the graph-attentional layer

        Arguments:
            in_features (int): number of features in each input node
            out_features (int): number of features in each output node
            dropout (int/float): dropout probability for the coefficients
            negative_slope (int/float): control the angle of the negative slope in leakyrelu
            number_heads (int): number of heads of attention
            bias (bool): if adding bias to the output
            self_loop_type (int): 0 -- force no self-loop; 1 -- force self-loop; other values (2)-- keep the input adjacency matrix unchanged
            average (bool): if averaging all attention heads
            normalize (bool): if normalizing the coefficients after zeroing out weights using the communication graph
        """

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_heads = num_heads
        self.self_loop_type = self_loop_type
        self.average = average
        self.normalize = normalize

        self.W = nn.Parameter(torch.zeros(size=(in_features, num_heads * out_features)))
        self.a_i = nn.Parameter(torch.zeros(size=(num_heads, out_features, 1)))
        self.a_j = nn.Parameter(torch.zeros(size=(num_heads, out_features, 1)))
        if bias:
            if average:
                self.bias = nn.Parameter(torch.DoubleTensor(out_features))
            else:
                self.bias = nn.Parameter(torch.DoubleTensor(num_heads * out_features))
        else:
            self.register_parameter('bias', None)
        self.leakyrelu = nn.LeakyReLU(self.negative_slope)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialization for the parameters of the graph-attentional layer
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W.data, gain=gain)
        nn.init.xavier_normal_(self.a_i.data, gain=gain)
        nn.init.xavier_normal_(self.a_j.data, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, input, adj):
        """
        Forward function for the graph attention layer used in MAGIC

        Arguments:
            input (tensor): input of the graph attention layer [N * in_features, N: number of agents]
            adj (tensor): the learned communication graph (adjancy matrix) by the sub-scheduler [N * N]

        Return:
            the output of the graph attention layer
        """

        # perform linear transformation on the input, and generate multiple heads
        # self.W: [in_features * (num_heads*out_features)]
        # h (tensor): the matrix after performing the linear transformation [N * num_heads * out_features]
        h = torch.mm(input, self.W).view(-1, self.num_heads, self.out_features)
        N = h.size()[0]
    
        eye = torch.eye(N, device=adj.device, dtype=adj.dtype)
        ones = torch.ones_like(adj)

        # force the self-loop not to happen
        if self.self_loop_type == 0:
            adj = adj * (ones - eye)
        # force the self-loop to happen
        elif self.self_loop_type == 1:
            adj = eye + adj * (ones - eye)
        # the self-loop will be decided by the sub-scheduler
        else:
            pass
        
        e = []

        # compute the unnormalized coefficients
        # a_i, a_j (tensors): weight vectors to compute the unnormalized coefficients [num_heads * out_features * 1]
        for head in range(self.num_heads):
            # coeff_i, coeff_j (tensors): intermediate matrices to calculate unnormalized coefficients [N * 1]
            coeff_i = torch.mm(h[:, head, :], self.a_i[head, :, :])
            coeff_j = torch.mm(h[:, head, :], self.a_j[head, :, :])
            # coeff (tensor): the matrix of unnormalized coefficients for each head [N * N * 1]
            coeff = coeff_i.expand(N, N) + coeff_j.transpose(0, 1).expand(N, N)
            coeff = coeff.unsqueeze(-1)
            
            e.append(coeff)
            
        # e (tensor): the matrix of unnormalized coefficients for all heads [N * N * num_heads]
        # sometimes the unnormalized coefficients can be large, so regularization might be used 
        # to limit the large unnormalized coefficient values (TODO)
        e = self.leakyrelu(torch.cat(e, dim=-1)) 
            
        # A zero edge must be removed before softmax.  The previous
        # implementation replaced its logit by zero, so every non-neighbor
        # still consumed probability mass; this severely attenuated messages
        # in sparse 20-car Traffic Junction graphs.
        edge_weight = adj.clamp(min=0.0, max=1.0)
        edge_mask = edge_weight > 0
        empty_rows = ~edge_mask.any(dim=1)
        if empty_rows.any():
            fallback = torch.diag(empty_rows.to(dtype=edge_weight.dtype))
            edge_weight = torch.max(edge_weight, fallback)
            edge_mask = edge_weight > 0

        expanded_mask = edge_mask.unsqueeze(-1).expand(N, N, self.num_heads)
        expanded_weight = edge_weight.unsqueeze(-1).expand(N, N, self.num_heads)
        masked_logits = e.masked_fill(~expanded_mask, -1e9)
        attention = F.softmax(masked_logits, dim=1) * expanded_weight

        # Renormalizing after multiplying by a soft adjacency makes the graph
        # weights a proper differentiable prior.  For binary graphs this is
        # equivalent to standard masked graph attention.
        denom = attention.sum(dim=1, keepdim=True).clamp(min=1e-12)
        attention = attention / denom
        # dropout on the coefficients  
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # output (tensor): the matrix of output of the gat layer [N * (num_heads*out_features)]
        output = []
        for head in range(self.num_heads):
            h_prime = torch.matmul(attention[:, :, head], h[:, head, :])
            output.append(h_prime)
        if self.average:
            output = torch.mean(torch.stack(output, dim=-1), dim=-1)
        else:
            output = torch.cat(output, dim=-1)
        
        if self.bias is not None:
            output += self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(in_features={}, out_features={})'.format(self.in_features, self.out_features)
    
