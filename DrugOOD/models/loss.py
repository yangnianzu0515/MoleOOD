import torch
from .utils import split_into_groups
from torch.distributions.normal import Normal

__all__ = ['KLDist', 'bce_log', 'MeanLoss', 'DeviationLoss', 'discrete_gaussian']


def KLDist(p, q, eps=1e-8):
    log_p, log_q = torch.log(p + eps), torch.log(q + eps)
    return torch.sum(p * (log_p - log_q))


def bce_log(pred, gt, eps=1e-8):
    prob = torch.sigmoid(pred)
    return -(gt * torch.log(prob + eps) + (1 - gt) * torch.log(1 - prob + eps))


class MeanLoss(torch.nn.Module):
    def __init__(self, base_loss):
        super(MeanLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, pred, gt, domain):
        _, group_indices, _ = split_into_groups(domain)
        total_loss, total_cnt = 0, 0
        for i_group in group_indices:
            total_loss += self.base_loss(pred[i_group], gt[i_group])
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


def discrete_gaussian(nums, std=1):
    Dist = Normal(loc=0, scale=1)
    plen, halflen = std * 6 / nums, std * 3 / nums
    posx = torch.arange(-3 * std + halflen, 3 * std, plen)
    result = Dist.cdf(posx + halflen) - Dist.cdf(posx - halflen)
    return result / result.sum()
