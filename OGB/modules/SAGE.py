from modules.utils import get_pool

from typing import Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul
import torch
from torch_geometric.nn import global_add_pool

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, OptTensor
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class SAGEMolConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(SAGEMolConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.bond_encoder = BondEncoder(out_channels)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        edge_emb = self.bond_encoder(edge_attr)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_emb, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr) -> Tensor:
        return torch.relu(x_j + edge_attr)

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)


class SAGEMolNode(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, drop_ratio=0.5,
        JK='last', residual=False
    ):
        super(SAGEMolNode, self).__init__()
        self.emb_dim = emb_dim
        self.num_layer = num_layer
        self.JK = JK
        self.residual = residual
        self.atom_encoder = AtomEncoder(emb_dim)
        if drop_ratio > 0 and drop_ratio < 1:
            self.dropout_layer = torch.nn.Dropout(drop_ratio)
        else:
            self.dropout_layer = torch.nn.Sequential()

        self.convs = torch.nn.ModuleList()
        self.BNs = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            self.convs.append(SAGEMolConv(emb_dim, emb_dim))
            self.BNs.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr):
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.BNs[layer](h)
            if layer == self.num_layer - 1:
                h = self.dropout_layer(h)
            else:
                h = self.dropout_layer(torch.relu(h))
            if self.residual:
                h += h_list[layer]
            h_list.append(h)
        if self.JK == 'last':
            node_repr = h_list[-1]
        elif self.JK == 'sum':
            node_repr = 0
            for layer in range(self.num_layer + 1):
                node_repr += h_list[layer]
        else:
            raise ValueError('JK should be "last" or "sum"')
        return node_repr

class VirtSAGEMolNode(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, drop_ratio=0.5,
        JK='last', residual=False
    ):
        super(VirtSAGEMolNode, self).__init__()
        self.emb_dim = emb_dim
        self.num_layer = num_layer
        self.JK = JK
        self.residual = residual
        self.atom_encoder = AtomEncoder(emb_dim)
        if drop_ratio > 0 and drop_ratio < 1:
            self.dropout_layer = torch.nn.Dropout(drop_ratio)
        else:
            self.dropout_layer = torch.nn.Sequential()

        self.convs = torch.nn.ModuleList()
        self.BNs = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            self.convs.append(SAGEMolConv(emb_dim, emb_dim))
            self.BNs.append(torch.nn.BatchNorm1d(emb_dim))
        self.vir_mlp = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim * 2),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * emb_dim, emb_dim),
                torch.nn.BatchNorm1d(emb_dim),
                torch.nn.ReLU()
            ) for _ in range(self.num_layer - 1)
        ])
        self.virtual_emb = torch.nn.Embedding(1, emb_dim)

    def forward(self, batched_data):
        x, batch = batched_data.x, batched_data.batch
        edge_index, edge_attr = batched_data.edge_index, batched_data.edge_attr
        virt_emb = torch.zeros(batch[-1].item() + 1).to(edge_index)
        virt_emb = self.virtual_emb(virt_emb)
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            h_list[layer] = h_list[layer] + virt_emb[batch]
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.BNs[layer](h)
            if layer == self.num_layer - 1:
                h = self.dropout_layer(h)
            else:
                h = self.dropout_layer(torch.relu(h))
            if self.residual:
                h += h_list[layer]
            h_list.append(h)
            if layer < self.num_layer - 1:
                virt_emb_temp = global_add_pool(h_list[layer], batch) + virt_emb
                if self.residual:
                    virt_emb += self.dropout_layer(
                        self.vir_mlp[layer](virt_emb_temp))
                else:
                    virt_emb = self.dropout_layer(
                        self.vir_mlp[layer](virt_emb_temp))
        if self.JK == 'last':
            node_repr = h_list[-1]
        elif self.JK == 'sum':
            node_repr = 0
            for layer in range(self.num_layer + 1):
                node_repr += h_list[layer]
        else:
            raise ValueError('JK should be "last" or "sum"')
        return node_repr


class SAGEMolGraph(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, drop_ratio=0.5,
        JK='last', residual=False, pooling='mean'
    ):
        super(SAGEMolGraph, self).__init__()
        self.pool_method = pooling
        if pooling not in ['attention', 'set2set']:
            self.pool = get_pool(pooling)
        elif pooling == 'attention':
            gate = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim * 2, 1)
            )
            self.pool = torch_geometric.nn.GlobalAttention(gate_nn=gate)
        else:
            self.pool = torch_geometric.nn.Set2Set(emb_dim, processing_steps=2)

        self.model = SAGEMolNode(emb_dim, num_layer, drop_ratio, JK, residual)

    def forward(self, batched_data):
        x, batch = batched_data.x, batched_data.batch
        edge_index, edge_attr = batched_data.edge_index, batched_data.edge_attr
        node_feat = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph_feat = self.pool(node_feat, batch)
        return graph_feat


class VirtSAGEMolGraph(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, drop_ratio=0.5,
        JK='last', residual=False, pooling='mean'
    ):
        super(VirtSAGEMolGraph, self).__init__()
        self.pool_method = pooling
        if pooling not in ['attention', 'set2set']:
            self.pool = get_pool(pooling)
        elif pooling == 'attention':
            gate = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim * 2, 1)
            )
            self.pool = torch_geometric.nn.GlobalAttention(gate_nn=gate)
        else:
            self.pool = torch_geometric.nn.Set2Set(emb_dim, processing_steps=2)

        self.model = VirtSAGEMolNode(
            emb_dim, num_layer, drop_ratio, JK, residual
        )

    def forward(self, batched_data):
        node_feat = self.model(batched_data)
        graph_feat = self.pool(node_feat, batched_data.batch)
        return graph_feat


class SAGEMol(torch.nn.Module):
    def __init__(
        self, emb_dim, num_tasks, num_layer, drop_ratio=0.5,
        JK='last', residual=False, pooling='mean', virtual=False
    ):
        super(SAGEMol, self).__init__()
        if virtual:
            self.model = VirtSAGEMolGraph(
                emb_dim=emb_dim, num_layer=num_layer, drop_ratio=drop_ratio,
                JK=JK, residual=residual, pooling=pooling
            )
        else:
            self.model = SAGEMolGraph(
                emb_dim=emb_dim, num_layer=num_layer, drop_ratio=drop_ratio,
                JK=JK, residual=residual, pooling=pooling
            )
        if pooling == 'set2set':
            self.predictor = torch.nn.Linear(2 * emb_dim, num_tasks)
        else:
            self.predictor = torch.nn.Linear(emb_dim, num_tasks)

    def forward(self, batched_data):
        graph_feat = self.model(batched_data)
        return self.predictor(graph_feat)
