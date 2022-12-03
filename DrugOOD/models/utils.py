import torch
import numpy as np
import dgl
from drugood.utils import smile2graph
from functools import reduce
from sklearn.metrics import roc_auc_score


def split_into_groups(g):
    unique_groups, unique_counts = torch.unique(
        g, sorted=False, return_counts=True
    )
    group_indices = [
        torch.nonzero(g == group, as_tuple=True)[0]
        for group in unique_groups
    ]
    return unique_groups, group_indices, unique_counts


def collect_batch_substrure_graphs(subs, return_mask=True):
    sub_struct_list = list(reduce(lambda x, y: x.update(y) or x, subs, set()))
    sub_to_idx = {x: idx for idx, x in enumerate(sub_struct_list)}
    mask = np.zeros([len(subs), len(sub_struct_list)], dtype=bool)
    for idx, sub in enumerate(subs):
        mask[idx][list(sub_to_idx[t] for t in sub)] = True

    graphs = [smile2graph(x) for x in sub_struct_list]
    batch_data = dgl.batch(graphs)
    return (batch_data, mask) if return_mask else batch_data


def evaluate(pred, gt, metric='auc'):
    if isinstance(metric, str):
        metric = [metric]
    allowed_metric = ['auc', 'accuracy']
    invalid_metric = set(metric) - set(allowed_metric)
    if len(invalid_metric) != 0:
        raise ValueError(f'Invalid Value {invalid_metric}')
    result = {}
    for M in metric:
        if M == 'auc':
            all_prob = pred[:, 0] + pred[:, 1]
            assert torch.all(torch.abs(all_prob - 1) < 1e-2), \
                "Input should be a binary distribution"
            score = pred[:, 1]
            result[M] = roc_auc_score(gt, score)
        else:
            pred = pred.argmax(dim=-1)
            total, correct = len(pred), torch.sum(pred.long() == gt.long())
            result[M] = (correct / total).item()
    return result
