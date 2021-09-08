import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from model.layers import GCNConv, GINEConv, DMPNNConv, DMPNNConv2


class GNN(nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()

        self.depth = args.depth
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.gnn_type = args.gnn_type
        self.graph_pool = args.graph_pool
        self.task = args.task
        self.output_size = args.output_size
        self.atom_messages = args.atom_messages
        self.ffn_hidden_size = args.ffn_hidden_size

        if self.gnn_type == 'dmpnn':
            if self.atom_messages:
                self.node_init = nn.Linear(args.num_node_features, self.hidden_size)             
            else:
                self.edge_init = nn.Linear(args.num_node_features + args.num_edge_features, self.hidden_size)
            # last layer before ffn
            # self.aggr_nodes = DMPNNConv2(args) # atom center messages
        else:
            self.node_init = nn.Linear(args.num_node_features, self.hidden_size)
            self.edge_init = nn.Linear(args.num_edge_features, self.hidden_size)

        # layers
        self.convs = torch.nn.ModuleList()

        for layer in range(self.depth):
            if self.gnn_type == 'gin':
                self.convs.append(GINEConv(args))
            elif self.gnn_type == 'gcn':
                self.convs.append(GCNConv(args))
            elif self.gnn_type == 'dmpnn':
                if layer == (self.depth -1):
                    self.convs.append(DMPNNConv2(args)) # atom center messages
                else:
                    self.convs.append(DMPNNConv(args)) # conv on edges
            else:
                ValueError('Undefined GNN type called {}'.format(self.gnn_type))
        # graph pooling
        if self.graph_pool == "sum":
            self.pool = global_add_pool
        elif self.graph_pool == "mean":
            self.pool = global_mean_pool
        elif self.graph_pool == "max":
            self.pool = global_max_pool
        elif self.graph_pool == "attn":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 2 * self.hidden_size),
                                            torch.nn.BatchNorm1d(2 * self.hidden_size),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(2 * self.hidden_size, 1)))
        elif self.graph_pool == "set2set":
            self.pool = Set2Set(self.hidden_size, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        # ffn
        self.mult = 2 if self.graph_pool == "set2set" else 1
        self.ffn = torch.nn.Sequential(nn.Linear(self.mult * self.hidden_size, self.ffn_hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.ffn_hidden_size, self.output_size))
    def forward(self, data):
        x0, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = x0.clone()
        if self.gnn_type == 'dmpnn':
            row, col = edge_index
            if self.atom_messages:
                x = F.relu(self.node_init(x))
            else:
                edge_attr = torch.cat([x[row], edge_attr], dim=1) # we did not concat features in the Data object
                edge_attr = F.relu(self.edge_init(edge_attr))
        else:
            x = F.relu(self.node_init(x))
            edge_attr = F.relu(self.edge_init(edge_attr))

        x_list = [x]
        edge_attr_list = [edge_attr]

        # convolutions
        for l in range(self.depth):
            # last layer
            if (self.gnn_type == 'dmpnn') and ( l == (self.depth -1)):
                if self.atom_messages:
                    h = self.convs[l](x_list[-1], edge_index, edge_attr_list[-1], x0) # h: num_nodes X hidden_size
                else:
                    h = self.convs[l](x_list[-1], edge_index, edge_attr_list[-1], x0) # h: num_edges X hidden_size
            else: 
                # previous layer
                h = self.convs[l](x_list[-1], edge_index, edge_attr_list[-1])

            if (self.gnn_type == 'dmpnn') and (not self.atom_messages):
                edge_attr_list.append(h)
            else:
                x_list.append(h)

        # move above
        # if self.gnn_type == 'dmpnn': 
        #     if self.atom_messages:
        #         h = self.aggr_nodes(h, edge_index, edge_attr_list[-1], x0) # h: num_nodes X hidden_size
        #     else:
        #         h = self.aggr_nodes(x_list[-1], edge_index, h, x0) # h: num_edges X hidden_size

        # batch => which assigns each node to a specific molecule
        # self.pool, aggreagate node to graph representatiotion
        return torch.clamp(self.ffn(self.pool(h, batch)).squeeze(-1), min=0) # expresssion value must >=0


