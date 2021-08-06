import torch
from torch_sparse import coalesce
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops


class myLineGraph(object):
    r"""Converts a graph to its corresponding line-graph:

    .. math::
        L(\mathcal{G}) &= (\mathcal{V}^{\prime}, \mathcal{E}^{\prime})

        \mathcal{V}^{\prime} &= \mathcal{E}

        \mathcal{E}^{\prime} &= \{ (e_1, e_2) : e_1 \cap e_2 \neq \emptyset \}

    Line-graph node indices are equal to indices in the original graph's
    coalesced :obj:`edge_index`.
    For undirected graphs, the maximum line-graph node index is
    :obj:`(data.edge_index.size(1) // 2) - 1`.

    New node features are given by old edge attributes.
    For undirected graphs, edge attributes for reciprocal edges
    :obj:`(row, col)` and :obj:`(col, row)` get summed together.

    Args:
        force_directed (bool, optional): If set to :obj:`True`, the graph will
            be always treated as a directed graph. (default: :obj:`False`)
    """
    def __init__(self, force_directed=False):
        self.force_directed = force_directed

    def __call__(self, data):
        ## FIXME: need to keep record of new_nodes to old nodes
        N = data.num_nodes
        edge_index, edge_attr = data.edge_index, data.edge_attr
        
        # Row-wise sorts :obj:`value` and removes duplicate entries. Duplicate
        # entries are removed by scattering them together.
        # row: source, col: targets
        (row, col), edge_attr = coalesce(edge_index, edge_attr, N, N) # (row, col) pair is new_nodes, edge_attr is new_nodes' attr
        
        if self.force_directed or data.is_directed():
            # Compute new edge indices according to `i`.
            i = torch.arange(row.size(0), dtype=torch.long, device=row.device)
            
            # count: num of atoms's bonds. tensor
            count = scatter_add(torch.ones_like(row), row, dim=0, dim_size=data.num_nodes)
            
            # size of cumsum = count.size() + 1, it's just insert a 0 at top of queqe
            # generate indice of new edges connected by nodes (soruce, target)
            cumsum = torch.cat([count.new_zeros(1), count.cumsum(0)], dim=0)

            ## cols is a list of tensors, col is a adjcant list
            ## index is target nodes, values is neigbors (index)
            ## e.g. cols[0]:  tensor([1, 2, 3, 4]), cols[1]: tensor([0])
            cols = [ i[cumsum[col[j]]:cumsum[col[j] + 1]] for j in range(col.size(0)) ]
            
#             cols = []
#             for j in range(col.size(0)):
#                 target = col[j]
#                 fr = cumsum[target]
#                 to = cumsum[target + 1]
#                 new_edges = i[fr:to]
#                 cols.append(cols)
            
            # similar to cols
            # now is source nodes
            rows = [row.new_full((c.numel(), ), j) for j, c in enumerate(cols)]
            
#            rows = []
#             for j, c in enumerate(cols):
#                 # new_full  Returns a Tensor of size size filled with fill_value
#                 r = row.new_full((c.numel(), ), j)
#                 rows.append(r)
                
            # concate => adjcant list to new edge_index, 
            # which means, node is old_edge, edge is new_edge 
            row, col = torch.cat(rows, dim=0), torch.cat(cols, dim=0)
            # assign new edge_index
            data.edge_index = torch.stack([row, col], dim=0)
            # assign new node_attr
            data.x = data.edge_attr
            # update num_nodes
            data.num_nodes = edge_index.size(1)
            # now, LineGraph is done, use GCNConv to do the real  `Convolute on edges`  
        else:
            # Compute node indices.
            mask = row < col
            row, col = row[mask], col[mask]
            i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

            (row, col), i = coalesce(
                torch.stack([
                    torch.cat([row, col], dim=0),
                    torch.cat([col, row], dim=0)
                ], dim=0), torch.cat([i, i], dim=0), N, N)

            # Compute new edge indices according to `i`.
            count = scatter_add(torch.ones_like(row), row, dim=0,
                                dim_size=data.num_nodes)
            joints = torch.split(i, count.tolist())

            def generate_grid(x):
                row = x.view(-1, 1).repeat(1, x.numel()).view(-1)
                col = x.repeat(x.numel())
                return torch.stack([row, col], dim=0)

            joints = [generate_grid(joint) for joint in joints]
            joints = torch.cat(joints, dim=1)
            joints, _ = remove_self_loops(joints)
            N = row.size(0) // 2
            joints, _ = coalesce(joints, None, N, N)

            if edge_attr is not None:
                data.x = scatter_add(edge_attr, i, dim=0, dim_size=N)
            data.edge_index = joints
            data.num_nodes = edge_index.size(1) // 2

        data.edge_attr = None
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


## donvert node_graph data object to line_graph data obj
# newdata = myLineGraph(force_directed=True)(data.clone())enumerate



# m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
# message ---> a_message = sum(nei_a_message) - rev_message
# e.g. neigbor(a1) = { a0, a2}
# only aggreate message with with same direction
# that's why rev_message need to be drop