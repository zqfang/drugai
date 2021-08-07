from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tetra import get_tetra_update


class GCNConv(MessagePassing):
    def __init__(self, args):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.batch_norm = nn.BatchNorm1d(args.hidden_size)
        self.tetra = args.tetra
        if self.tetra:
            self.tetra_update = get_tetra_update(args)

    def forward(self, x, edge_index, edge_attr, parity_atoms):

        # no edge updates
        x = self.linear(x)

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

        if self.tetra:
            tetra_ids = parity_atoms.nonzero().squeeze(1)
            if tetra_ids.nelement() != 0:
                x_new[tetra_ids] = self.tetra_message(x, edge_index, edge_attr, tetra_ids, parity_atoms)
        x = x_new + F.relu(x)

        return self.batch_norm(x), edge_attr

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def tetra_message(self, x, edge_index, edge_attr, tetra_ids, parity_atoms):

        row, col = edge_index
        tetra_nei_ids = torch.cat([row[col == i].unsqueeze(0) for i in range(x.size(0)) if i in tetra_ids])

        # calculate pseudo tetra degree aligned with GCN method
        deg = degree(col, x.size(0), dtype=x.dtype)
        t_deg = deg[tetra_nei_ids]
        t_deg_inv_sqrt = t_deg.pow(-0.5)
        t_norm = 0.5 * t_deg_inv_sqrt.mean(dim=1)

        # switch entries for -1 rdkit labels
        ccw_mask = parity_atoms[tetra_ids] == -1
        tetra_nei_ids[ccw_mask] = tetra_nei_ids.clone()[ccw_mask][:, [1, 0, 2, 3]]

        # calculate reps
        edge_ids = torch.cat([tetra_nei_ids.view(1, -1), tetra_ids.repeat_interleave(4).unsqueeze(0)], dim=0)
        # dense_edge_attr = to_dense_adj(edge_index, batch=None, edge_attr=edge_attr).squeeze(0)
        # edge_reps = dense_edge_attr[edge_ids[0], edge_ids[1], :].view(tetra_nei_ids.size(0), 4, -1)
        attr_ids = [torch.where((a == edge_index.t()).all(dim=1))[0] for a in edge_ids.t()]
        edge_reps = edge_attr[attr_ids, :].view(tetra_nei_ids.size(0), 4, -1)
        reps = x[tetra_nei_ids] + edge_reps

        return t_norm.unsqueeze(-1) * self.tetra_update(reps)


class GINEConv(MessagePassing):
    def __init__(self, args):
        super(GINEConv, self).__init__(aggr="add")
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.mlp = nn.Sequential(nn.Linear(args.hidden_size, 2 * args.hidden_size),
                                 nn.BatchNorm1d(2 * args.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(2 * args.hidden_size, args.hidden_size))
        self.batch_norm = nn.BatchNorm1d(args.hidden_size)
        self.tetra = args.tetra
        if self.tetra:
            self.tetra_update = get_tetra_update(args)

    def forward(self, x, edge_index, edge_attr, parity_atoms):
        # no edge updates
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if self.tetra:
            tetra_ids = parity_atoms.nonzero().squeeze(1)
            if tetra_ids.nelement() != 0:
                x_new[tetra_ids] = self.tetra_message(x, edge_index, edge_attr, tetra_ids, parity_atoms)

        x = self.mlp((1 + self.eps) * x + x_new)
        return self.batch_norm(x), edge_attr

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def tetra_message(self, x, edge_index, edge_attr, tetra_ids, parity_atoms):

        row, col = edge_index
        tetra_nei_ids = torch.cat([row[col == i].unsqueeze(0) for i in range(x.size(0)) if i in tetra_ids])

        # switch entries for -1 rdkit labels
        ccw_mask = parity_atoms[tetra_ids] == -1
        tetra_nei_ids[ccw_mask] = tetra_nei_ids.clone()[ccw_mask][:, [1, 0, 2, 3]]

        # calculate reps
        edge_ids = torch.cat([tetra_nei_ids.view(1, -1), tetra_ids.repeat_interleave(4).unsqueeze(0)], dim=0)
        # dense_edge_attr = to_dense_adj(edge_index, batch=None, edge_attr=edge_attr).squeeze(0)
        # edge_reps = dense_edge_attr[edge_ids[0], edge_ids[1], :].view(tetra_nei_ids.size(0), 4, -1)
        attr_ids = [torch.where((a == edge_index.t()).all(dim=1))[0] for a in edge_ids.t()]
        edge_reps = edge_attr[attr_ids, :].view(tetra_nei_ids.size(0), 4, -1)
        reps = x[tetra_nei_ids] + edge_reps

        return self.tetra_update(reps)


class DMPNNConv(MessagePassing):
    def __init__(self, args, atom_messages):
        super(DMPNNConv, self).__init__(aggr='add')
        self.atom_messages = atom_messages
        inp_size = args.hidden_size
        if atom_messages: inp_size += args.b_fdim # FIXME
        
        self.mlp1 = nn.Sequential(nn.Linear(inp_size, args.hidden_size),
                                  nn.BatchNorm1d(args.hidden_size),
                                  nn.ReLU(),
                                  nn.Dropout(0.0))

        self.mlp2 = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                 nn.BatchNorm1d(args.hidden_size),
                                 nn.ReLU(),
                                 nn.Dropout(0.0))
        self.tetra = args.tetra
        if self.tetra:
            self.tetra_update = get_tetra_update(args)

    def forward(self, x, edge_index, edge_attr, parity_atoms):
        row, col = edge_index # source node, target node

        if self.atom_messages: 
            # atom_message passing
            a_message = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            # FIXME
            if self.tetra:
                tetra_ids = parity_atoms.nonzero().squeeze(1)
                if tetra_ids.nelement() != 0:
                    a_message[tetra_ids] = self.tetra_message(x, edge_index, edge_attr, tetra_ids, parity_atoms)
        else: 
            # edge message passing
            # x: line_graph's node_features (node_graph's edge_attr), set_edge_attr to None 
            a_message = self.propagate(edge_index, x=edge_attr, edge_attr=None,)
            
        return a_message, self.mlp2(a_message)

    def message(self, x_j, x, edge_attr):
        if self.atom_messages:
            # good thing is we concat fwd, rev edge together adjacently. b2a, a2b is esasy to handle then
            # edge_index must be node_graph, not line_graph
            x_j = torch.cat([x_j, edge_attr], dim=1) # concat to node_features
            return self.mlp1(x_j)
        return self.mlp1(x) # return line_graph's directed_node_attr.

    def update(self, aggr_out):
        # if self.atom_messages:
        if not self.atom_messages:
            ## filp -> Reverse the order of a n-D tensor along given axis in dims
            ## then reshape to (num_bonds, hidden)
            rev_message = aggr_out.view(aggr_out.size(0) // 2, 2, -1).flip(dims=[1]).view(aggr_out.size(0), -1)
            aggr_out -= rev_message
        # if atom_message: aggr_out size: [num_nodes, embed_size], 
        # line_graph: [num_edges, embed_size]
        return aggr_out # self.mpl2(aggr_out) 


    def tetra_message(self, x, edge_index, edge_attr, tetra_ids, parity_atoms):

        row, col = edge_index
        tetra_nei_ids = torch.cat([row[col == i].unsqueeze(0) for i in range(x.size(0)) if i in tetra_ids])

        # switch entries for -1 rdkit labels
        ccw_mask = parity_atoms[tetra_ids] == -1
        tetra_nei_ids[ccw_mask] = tetra_nei_ids.clone()[ccw_mask][:, [1, 0, 2, 3]]

        # calculate reps
        edge_ids = torch.cat([tetra_nei_ids.view(1, -1), tetra_ids.repeat_interleave(4).unsqueeze(0)], dim=0)
        # dense_edge_attr = to_dense_adj(edge_index, batch=None, edge_attr=edge_attr).squeeze(0)
        # edge_reps = dense_edge_attr[edge_ids[0], edge_ids[1], :].view(tetra_nei_ids.size(0), 4, -1)
        attr_ids = [torch.where((a == edge_index.t()).all(dim=1))[0] for a in edge_ids.t()]
        edge_reps = edge_attr[attr_ids, :].view(tetra_nei_ids.size(0), 4, -1)

        return self.tetra_update(edge_reps)




