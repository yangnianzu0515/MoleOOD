import json
import time
import os
import numpy as np
import torch
import argparse

from copy import deepcopy
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from modules.GNNs import GNNGraph
from modules.SAGE import SAGEMolGraph, VirtSAGEMolGraph
from modules.DataLoading import pyg_molsubdataset
from modules.model import Framework
from modules.utils import get_device, split_into_groups
from ogb.graphproppred import Evaluator


def build_backend_from_config(config):
    model_type = config['type']
    if model_type == 'gin':
        model = GNNGraph(gnn_type='gin', virtual_node=False, **config['paras'])
    elif model_type == 'gin_virtual':
        model = GNNGraph(gnn_type='gin', virtual_node=True, **config['paras'])
    elif model_type == 'gcn':
        model = GNNGraph(gnn_type='gcn', virtual_node=False, **config['paras'])
    elif model_type == 'gcn-virtual':
        model = GNNGraph(gnn_type='gcn', virtual_node=True, **config['paras'])
    elif model_type == 'gat':
        model = GATMolGraph(**config['paras'])
    elif model_type == 'gat_virtual':
        model = VirtGATMolGraph(**config['paras'])
    elif model_type == 'sage':
        model = SAGEMolGraph(**config['paras'])
    elif model_type == 'sage-virtual':
        model = VirtSAGEMolGraph(**config['paras'])
    else:
        raise ValueError(f'Invalid model type called {model_type}')
    return model


def init_args():
    parser = argparse.ArgumentParser('Parser For Experiment on OGB')
    parser.add_argument(
        '--base_backend', type=str, required=True,
        help='the path of gnn backend config of base model'
    )
    parser.add_argument(
        '--sub_backend', type=str, required=True,
        help='the path of gnn backend config of substructure encoder'
    )
    parser.add_argument(
        '--dataset', type=str, default='ogbg-molbace',
        help='the dataset to run experiment'
    )
    parser.add_argument(
        '--device', type=int, default=-1,
        help='the gpu id for training, negative for cpu'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='path of well trained model paras'
    )

    parser.add_argument(
        '--decomp_method', choices=['brics', 'recap'], default='brics',
        help='the method to decompose the molecules into substructures'
    )
    parser.add_argument(
        '--drop_ratio', type=float, default=0.5,
        help='the dropout ratio of base model while training'
    )

    args = parser.parse_args()
    return args


def eval_one_epoch(loader, evaluator, model, device, verbose=False):
    model = model.eval()
    y_pred, y_gt = [], []
    iterx = tqdm(loader) if verbose else loader
    for batch_sub, batch_graph in iterx:
        # subs: a list of string, eval(string) can
        # get the substructures of corresponding molecule
        batch_sub = [eval(x) for x in batch_sub]
        batch_graph = batch_graph.to(device)
        with torch.no_grad():
            pred = model(batch_sub, batch_graph)
        y_pred.append(pred.detach().cpu())
        y_gt.append(batch_graph.y.reshape(pred.shape).detach().cpu())
    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_gt = torch.cat(y_gt, dim=0).numpy()
    return evaluator.eval({'y_true': y_gt, 'y_pred': y_pred})


if __name__ == '__main__':
    args = init_args()
    print(args)

    with open(args.base_backend) as Fin:
        base_backend_config = json.load(Fin)
    with open(args.sub_backend) as Fin:
        sub_backend_config = json.load(Fin)

    total_smiles, total_subs, dataset = pyg_molsubdataset(
        args.dataset, args.decomp_method
    )
    device = get_device(args.device)

    base_backend = build_backend_from_config(base_backend_config)
    sub_backend = build_backend_from_config(sub_backend_config)
    main_model = Framework(
        base_model=base_backend, sub_model=sub_backend,
        base_dim=base_backend_config['result_dim'],
        sub_dim=sub_backend_config['result_dim'],
        num_tasks=dataset.num_tasks, dropout=args.drop_ratio
    ).to(device)

    evaluator = Evaluator(args.dataset)
    data_split_idx = dataset.get_idx_split()
    train_idx = data_split_idx['train']
    valid_idx = data_split_idx['valid']
    test_idx = data_split_idx['test']

    train_dataset = dataset[train_idx]
    valid_dataset = dataset[valid_idx]
    test_dataset = dataset[test_idx]
    train_subs = [str(total_subs[x.item()]) for x in train_idx]
    valid_subs = [str(total_subs[x.item()]) for x in valid_idx]
    test_subs = [str(total_subs[x.item()]) for x in test_idx]

    train_loader = DataLoader(
        list(zip(train_subs, train_dataset)),
        batch_size=64, shuffle=False
    )
    valid_loader = DataLoader(
        list(zip(valid_subs, valid_dataset)),
        batch_size=64, shuffle=False
    )
    test_loader = DataLoader(
        list(zip(test_subs, test_dataset)),
        batch_size=64, shuffle=False
    )

    best_model_para = torch.load(args.model_path, map_location=device)
    main_model.load_state_dict(best_model_para['main'])
    main_model = main_model.eval()
    train_perf = eval_one_epoch(
        train_loader, evaluator, main_model,
        device, verbose=True
    )
    valid_perf = eval_one_epoch(
        valid_loader, evaluator, main_model,
        device, verbose=True
    )
    test_perf = eval_one_epoch(
        test_loader, evaluator, main_model,
        device, verbose=True
    )
    print('train: {}'.format(train_perf[dataset.eval_metric]))
    print("valid: {}".format(valid_perf[dataset.eval_metric]))
    print("test: {}".format(test_perf[dataset.eval_metric]))
