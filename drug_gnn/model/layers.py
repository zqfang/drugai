from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(MessagePassing):
    def __init__(self, args):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.batch_norm = nn.BatchNorm1d(args.hidden_size)


    def forward(self, x, edge_index, edge_attr):

        # no edge updates
        x = self.linear(x)

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

        x = x_new + F.relu(x)

        return self.batch_norm(x)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)


class GINEConv(MessagePassing):
    def __init__(self, args):
        super(GINEConv, self).__init__(aggr="add")
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.mlp = nn.Sequential(nn.Linear(args.hidden_size, 2 * args.hidden_size),
                                 nn.BatchNorm1d(2 * args.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(2 * args.hidden_size, args.hidden_size))
        self.batch_norm = nn.BatchNorm1d(args.hidden_size)


    def forward(self, x, edge_index, edge_attr):
        # no edge updates
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        x = self.mlp((1 + self.eps) * x + x_new)
        return self.batch_norm(x)

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)


class DMPNNConv(MessagePassing):
    def __init__(self, args):
        super(DMPNNConv, self).__init__(aggr='add')
        self.atom_messages = args.atom_messages
        input_size = args.hidden_size
        if self.atom_messages: input_size += args.num_edge_features
        self.mlp = nn.Sequential(nn.Linear(input_size, args.hidden_size),
                                  nn.BatchNorm1d(args.hidden_size),
                                  nn.ReLU(),
                                  nn.Dropout(args.dropout))         
    def forward(self, x, edge_index, edge_attr):
        """
        """
        message = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return message

    def message(self, x_j, edge_attr):
        if self.atom_messages:
            # good thing is we concat fwd, rev edge together adjacently. b2a, a2b is esasy to handle then
            return torch.cat([x_j, edge_attr], dim=1) # concat to node_features
        return edge_attr 

    def update(self, aggr_out, edge_index, edge_attr):
        # if self.atom_messages:
        if not self.atom_messages:
            row, col = edge_index
            # b2a, _ = edge_index
            ## filp -> Reverse the order of a n-D tensor along given axis in dims
            ## then reshape to (num_bonds, hidden)
            rev_message = edge_attr.view(edge_attr.size(0) // 2, 2, -1).flip(dims=[1]).view(edge_attr.size(0), -1)
            aggr_out = aggr_out.index_select(0, row) - rev_message
        # if atom_message: aggr_out size: [num_nodes, embed_size], 
        # else: [num_edges, embed_size]
        return self.mlp(aggr_out)


class DMPNNConv2(MessagePassing):
    """Aggregate edge/nodes to nodes, then concat bond_features
    """
    def __init__(self, args, edge2node=True):
        """edge2node: aggregate edge messages to node only if both edge2node and atom_message are true
        """
        super(DMPNNConv2, self).__init__(aggr='add')
        self.node_messages = edge2node
        self.atom_messages = args.atom_messages 
        self.mlp = nn.Sequential(nn.Linear(args.num_node_features + args.hidden_size, args.hidden_size),
                                  nn.BatchNorm1d(args.hidden_size),
                                  nn.ReLU(),
                                  #nn.Dropout(args.dropout)
                                  )        
    def forward(self, x, edge_index, edge_attr, x0):
        """
        x0: raw bond features
        """
        a_message = self.propagate(edge_index, x=x, edge_attr=edge_attr, x0=x0)
        return a_message

    def message(self, x_j, edge_attr):
        if self.atom_messages and self.node_messages:
            return x_j
        return edge_attr 

    def update(self, aggr_out, x0):
        aggr_out2 = torch.cat([x0, aggr_out], dim=1)  # num_atoms x (atom_fdim + hidden)
        return self.mlp(aggr_out2)