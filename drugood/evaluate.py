import time
import os
import argparse
import torch
import torch_scatter
import models
import json
import numpy as np

from tqdm import tqdm
from drugood.datasets import build_dataset, build_dataloader
from drugood.models import build_backbone
from mmcv import Config
from models import Framework, evaluate
from torch.optim import AdamW
from copy import deepcopy
from torch_geometric.seed import seed_everything


def init_args():
    parser = argparse.ArgumentParser('Experiment for Drugood Dataset')
    parser.add_argument(
        '--data_config', type=str,
        help='the config for building dataset',
        default='configs/data_assay.py'
    )
    parser.add_argument(
        '--model_config', type=str,
        help='the config for building models',
        default='configs/GIN_0.1_mean.py'
    )
    parser.add_argument(
        '--device', default=-1, type=int,
        help='the gpu id for training, minus number for cpu'
    )
    parser.add_argument(
        '--seed', default=2022, type=int,
        help='the seed of training'
    )
    parser.add_argument(
        '--model_path', required=True, type=str,
        help='path of well trained model paras'
    )
    args = parser.parse_args()
    return args


def build_models_from_cfg(cfg, device, num_class):
    base_backbone = build_backbone(cfg.model.main)
    sub_backbone = build_backbone(cfg.model.sub)
    main_model = Framework(
        base=base_backbone, sub=sub_backbone, dropout=cfg.dropout,
        base_dim=cfg.model.main.emb_dim, sub_dim=cfg.model.sub.emb_dim,
        num_class=num_class
    ).to(device)
    return main_model


def eval_one_epoch(model, loader, device, verbose=True):
    model = model.eval()
    result_all, gt_all = [], []
    for data in (tqdm(loader) if verbose else loader):
        with torch.no_grad():
            subs = [eval(x) for x in data['subs']]
            data['input'] = data['input'].to(device)
            result = torch.softmax(model(data['input'], subs), dim=-1)
            result_all.append(result.detach().cpu())
            gt_all.append(data['gt_label'].cpu())
    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    return evaluate(pred=result_all, gt=gt_all, metric=['auc', 'accuracy'])


if __name__ == '__main__':
    args = init_args()
    print(args)
    seed_everything(args.seed)
    dataset_config = Config.fromfile(args.data_config)
    model_config = Config.fromfile(args.model_config)
    print(dataset_config.pretty_text)
    print(model_config.pretty_text)

    device = torch.device('cpu') if args.device < 0 \
        else torch.device(f'cuda:{args.device}')

    valid_set = build_dataset(dataset_config.data.ood_val)
    test_set = build_dataset(dataset_config.data.ood_test)
    dataset_config.data.ood_val.test_mode = True
    dataset_config.data.ood_test.test_mode = True
    valid_loader = build_dataloader(
        valid_set, dataset_config.data.samples_per_gpu,
        dataset_config.data.workers_per_gpu, num_gpus=1,
        dist=False, round_up=True, seed=args.seed, shuffle=False
    )
    test_loader = build_dataloader(
        test_set, dataset_config.data.samples_per_gpu,
        dataset_config.data.workers_per_gpu, num_gpus=1,
        dist=False, round_up=True, seed=args.seed, shuffle=False
    )

    main_model = build_models_from_cfg(
        model_config, device,
        dataset_config.data.num_class
    )
    model_weight = torch.load(args.model_path, map_location=device)
    main_model.load_state_dict(model_weight['main'])
    main_model = main_model.eval()
    val_perf = eval_one_epoch(main_model, valid_loader, device)
    test_perf = eval_one_epoch(main_model, test_loader, device)
    print('valid set:', val_perf)
    print('test set:', test_perf)
