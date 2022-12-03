import torch
from .utils import collect_batch_substrure_graphs
from drugood.core import move_to_device
import math

__all__ = ['Framework', 'ConditionalGnn', 'DomainClassifier']


class AttentionAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, Mdim):
        super(AttentionAgger, self).__init__()
        self.model_dim = Mdim
        self.WQ = torch.nn.Linear(Qdim, Mdim)
        self.WK = torch.nn.Linear(Qdim, Mdim)

    def forward(self, Q, K, V, mask=None):
        Q, K = self.WQ(Q), self.WK(K)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)
        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))
        Attn = torch.softmax(Attn, dim=-1)
        return torch.matmul(Attn, V)


class Framework(torch.nn.Module):
    def __init__(self, base, sub, num_class, base_dim, sub_dim, dropout=0.5):
        super(Framework, self).__init__()
        self.base_model, self.sub_model = base, sub
        self.base_dim, self.sub_dim = base_dim, sub_dim
        hidden_dim = max(self.base_dim, self.sub_dim)
        self.aggr = AttentionAgger(base_dim, sub_dim, hidden_dim)

        predictor_layers = [torch.nn.Linear(sub_dim, sub_dim)]
        if dropout < 1 and dropout > 0:
            predictor_layers.append(torch.nn.Dropout(dropout))
        predictor_layers.append(torch.nn.ReLU())
        predictor_layers.append(torch.nn.Linear(sub_dim, num_class))
        self.predictor = torch.nn.Sequential(*predictor_layers)

    def substruct_feats(self, subs, device, return_mask=True):
        graphs, mask = collect_batch_substrure_graphs(subs, True)
        graphs = move_to_device(graphs, device)
        graph_feats = self.sub_model(graphs)
        return (graph_feats, mask) if return_mask else graph_feats

    def forward(self, graphs, subs):
        main_feat = self.base_model(graphs)
        subs_feat, mask = self.substruct_feats(subs, graphs.device, True)
        mask = torch.from_numpy(mask).to(graphs.device)
        attn_mask = torch.logical_not(mask)
        molecule_feat = self.aggr(main_feat, subs_feat, subs_feat, attn_mask)
        return self.predictor(molecule_feat)


class ConditionalGnn(torch.nn.Module):
    def __init__(self, emb_dim, backend_dim, backend, num_domain, num_class):
        super(ConditionalGnn, self).__init__()
        self.emb_dim = emb_dim
        self.class_emb = torch.nn.Parameter(
            torch.zeros(num_domain, emb_dim)
        )
        self.backend = backend
        self.predictor = torch.nn.Linear(backend_dim + emb_dim, num_class)

    def forward(self, batched_data, domains):
        domain_feat = torch.index_select(self.class_emb, 0, domains)
        graph_feat = self.backend(batched_data)
        result = self.predictor(torch.cat([graph_feat, domain_feat], dim=1))
        return result


class DomainClassifier(torch.nn.Module):
    def __init__(self, backend_dim, backend, num_domain, num_task):
        super(DomainClassifier, self).__init__()
        self.backend = backend
        self.num_task = num_task
        self.predictor = torch.nn.Linear(backend_dim + num_task, num_domain)

    def forward(self, batched_data):
        graph_feat = self.backend(batched_data['input'])
        y_part = torch.nan_to_num(batched_data['gt_label']).float()
        y_part = y_part.reshape(len(y_part), self.num_task)
        return self.predictor(torch.cat([graph_feat, y_part], dim=-1))
