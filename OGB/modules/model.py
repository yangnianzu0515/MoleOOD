import torch
import math
import torch_geometric
from .ChemistryProcess import graph_from_substructure
from .utils import split_into_groups
from torch.distributions.normal import Normal

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
    def __init__(
        self, base_model, sub_model, num_tasks,
        base_dim, sub_dim, dropout=0.5,
    ):
        super(Framework, self).__init__()
        self.base_model = base_model
        self.sub_model = sub_model
        self.attenaggr = AttentionAgger(
            base_dim, sub_dim, max(base_dim, sub_dim)
        )
        predictor_layers = [torch.nn.Linear(sub_dim, sub_dim)]
        if dropout < 1 and dropout > 0:
            predictor_layers.append(torch.nn.Dropout(dropout))
        predictor_layers.append(torch.nn.ReLU())
        predictor_layers.append(torch.nn.Linear(sub_dim, num_tasks))
        self.predictor = torch.nn.Sequential(*predictor_layers)

    def sub_feature_from_graphs(self, subs, device, return_mask=False):
        substructure_graph, mask = graph_from_substructure(subs, True, 'pyg')
        substructure_graph = substructure_graph.to(device)
        substructure_feat = self.sub_model(substructure_graph)
        return (substructure_feat, mask) if return_mask else substructure_feat

    def forward(self, substructures, batched_data):
        graph_feat = self.base_model(batched_data)
        substructure_feat, mask = self.sub_feature_from_graphs(
            subs=substructures, device=batched_data.x.device,
            return_mask=True
        )
        mask = torch.from_numpy(mask).to(batched_data.x.device)
        Attn_mask = torch.logical_not(mask)
        molecule_feat = self.attenaggr(
            Q=graph_feat, K=substructure_feat,
            V=substructure_feat, mask=Attn_mask
        )
        result = self.predictor(molecule_feat)
        return result


class MeanLoss(torch.nn.Module):
    def __init__(self, base_loss):
        super(MeanLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, pred, gt, domain):
        _, group_indices, _ = split_into_groups(domain)
        total_loss, total_cnt = 0, 0
        for i_group in group_indices:
            pred_group, gt_group = pred[i_group], gt[i_group]
            islabeled = gt_group == gt_group
            total_loss += self.base_loss(
                pred_group[islabeled], gt_group[islabeled]
            )
            total_cnt += 1
        return total_loss / total_cnt


class DeviationLoss(torch.nn.Module):
    def __init__(self, activation, reduction='mean'):
        super(DeviationLoss, self).__init__()
        assert activation in ['relu', 'abs', 'none'],\
            'Invaild activation function'
        assert reduction in ['mean', 'sum'], \
            'Invalid reduction method'

        self.activation = activation
        self.reduction = reduction

    def forward(self, pred, condition_pred_mean):
        if self.activation == 'relu':
            loss = torch.relu(pred - condition_pred_mean)
        elif self.activation == 'abs':
            loss = torch.abs(pred - condition_pred_mean)
        else:
            loss = pred - condition_pred_mean

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class ConditionalGnn(torch.nn.Module):
    def __init__(self, emb_dim, backend_dim, backend, num_domain, num_classes):
        super(ConditionalGnn, self).__init__()
        self.emb_dim = emb_dim
        self.class_emb = torch.nn.Parameter(
            torch.zeros(num_domain, emb_dim)
        )
        self.backend = backend
        self.predictor = torch.nn.Linear(backend_dim + emb_dim, num_classes)

    def forward(self, batched_data, domains):
        domain_feat = torch.index_select(self.class_emb, 0, domains)
        graph_feat = self.backend(batched_data)
        result = self.predictor(torch.cat([graph_feat, domain_feat], dim=1))
        return result


class DomainClassifier(torch.nn.Module):
    def __init__(self, backend_dim, backend, num_domains, num_tasks):
        super(DomainClassifier, self).__init__()
        self.backend = backend
        self.predictor = torch.nn.Linear(backend_dim + num_tasks, num_domains)

    def forward(self, batched_data):
        graph_feat = self.backend(batched_data)
        y_part = torch.nan_to_num(batched_data.y).float()
        return self.predictor(torch.cat([graph_feat, y_part], dim=-1))


def KLDist(p, q, eps=1e-8):
    log_p, log_q = torch.log(p + eps), torch.log(q + eps)
    return torch.sum(p * (log_p - log_q))


def bce_log(pred, gt, eps=1e-8):
    prob = torch.sigmoid(pred)
    return -(gt * torch.log(prob + eps) + (1 - gt) * torch.log(1 - prob + eps))


def discrete_gaussian(nums, std=1):
    Dist = Normal(loc=0, scale=1)
    plen, halflen = std * 6 / nums, std * 3 / nums
    posx = torch.arange(-3 * std + halflen, 3 * std, plen)
    result = Dist.cdf(posx + halflen) - Dist.cdf(posx - halflen)
    return result / result.sum()