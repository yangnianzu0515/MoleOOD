from copy import deepcopy
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
import json
import time
import os
from modules.GNNs import GNNGraph
from modules.SAGE import SAGEMolGraph, VirtSAGEMolGraph
from modules.DataLoading import pyg_molsubdataset
from torch.optim import Adam
from modules.model import Framework, ConditionalGnn, DomainClassifier
from modules.model import bce_log, KLDist, MeanLoss, DeviationLoss
from modules.model import discrete_gaussian
import argparse
from modules.utils import get_device, split_into_groups
from ogb.graphproppred import Evaluator
from torch.nn import BCEWithLogitsLoss
import torch_scatter
import torch


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
        '--seed', default=2022, type=int,
        help='the random seed for running experiment'
    )
    parser.add_argument(
        '--base_backend', type=str, required=True,
        help='the path of gnn backend config of base model'
    )
    parser.add_argument(
        '--sub_backend', type=str, required=True,
        help='the path of gnn backend config of substructure encoder'
    )
    parser.add_argument(
        '--domain_backend', type=str, required=True,
        help='the path of gnn backend config of domain classifier'
    )
    parser.add_argument(
        '--conditional_backend', type=str, required=True,
        help='the path of gnn_backend config of conditional gnn'
    )
    parser.add_argument(
        '--drop_ratio', type=float, default=0.5,
        help='the dropout ratio of base model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='the batch size of training'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='the learning rate of training'
    )
    parser.add_argument(
        '--exp_name', type=str, default='',
        help='the name of logging file'
    )
    parser.add_argument(
        '--epoch_main', type=int, default=100,
        help='the number of training epoch for main model'
    )
    parser.add_argument(
        '--epoch_ast', type=int, default=100,
        help='the number of training epoch for assistant model'
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
        '--num_domain', type=int, default=10,
        help='the number of domain for training'
    )
    parser.add_argument(
        '--lambda_loss', type=float, default=0.1,
        help='the weight of mean term in loss'
    )

    parser.add_argument(
        '--decomp_method', choices=['brics', 'recap'], default='brics',
        help='the method to decompose the molecules into substructures'
    )

    parser.add_argument(
        '--prior', default='uniform', choices=['uniform', 'gaussian'],
        type=str, help='the prior distribution of ELBO'
    )

    args = parser.parse_args()
    args.work_dir = get_work_dir(args)
    if args.exp_name == '':
        args.exp_name, args.time = get_file_name(args)
    return args


def get_basename(file_name):
    ans = os.path.basename(file_name)
    if '.' in ans:
        ans = ans.split('.')
        ans = '.'.join(ans[:-1])
    return ans


def get_work_dir(args):
    file_dir = [f'base_{get_basename(args.base_backend)}']
    file_dir.append(f'sub_{get_basename(args.sub_backend)}')
    file_dir.append(f'domain_{get_basename(args.domain_backend)}')
    file_dir.append(f'cond_{get_basename(args.conditional_backend)}')
    return os.path.join(args.dataset, '-'.join(file_dir))


def get_file_name(args):
    file_name = [(f'bs_{args.batch_size}')]
    file_name.append(f'lr_{args.lr}')
    file_name.append(f'dp_{args.drop_ratio}')
    file_name.append(f'dom_{args.num_domain}')
    file_name.append(f'ep_ast_{args.epoch_ast}')
    file_name.append(f'ep_main_{args.epoch_main}')
    file_name.append(f'lambda_{args.lambda_loss}')
    # file_name.append(f'dataset_{args.dataset}')
    file_name.append(f'prior_{args.prior}')
    current_time = time.time()
    file_name.append(f'{current_time}')
    return '-'.join(file_name) + '.json', current_time


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


def get_prior(num_domain, dtype='uniform'):
    assert dtype in ['uniform', 'gaussian'], 'Invalid distribution type'
    if dtype == 'uniform':
        prior = torch.ones(num_domain) / num_domain
    else:
        prior = discrete_gaussian(num_domain)
    return prior


if __name__ == '__main__':
    args = init_args()
    print(args)
    if not os.path.exists('log'):
        os.mkdir('log')
    if not os.path.exists(os.path.join('log', args.work_dir)):
        os.makedirs(os.path.join('log', args.work_dir))
    seed_everything(args.seed)

    with open(args.base_backend) as Fin:
        base_backend_config = json.load(Fin)
    with open(args.sub_backend) as Fin:
        sub_backend_config = json.load(Fin)
    with open(args.domain_backend) as Fin:
        domain_backend_config = json.load(Fin)
    with open(args.conditional_backend) as Fin:
        conditioanl_backend_config = json.load(Fin)

    total_smiles, total_subs, dataset = pyg_molsubdataset(
        args.dataset, args.decomp_method
    )
    device = get_device(args.device)
    prior = get_prior(args.num_domain, args.prior).to(device)

    base_backend = build_backend_from_config(base_backend_config)
    sub_backend = build_backend_from_config(sub_backend_config)
    main_model = Framework(
        base_model=base_backend, sub_model=sub_backend,
        base_dim=base_backend_config['result_dim'],
        sub_dim=sub_backend_config['result_dim'],
        num_tasks=dataset.num_tasks, dropout=args.drop_ratio
    ).to(device)

    domain_backend = build_backend_from_config(domain_backend_config)
    domain_classifier = DomainClassifier(
        backend_dim=domain_backend_config['result_dim'],
        backend=domain_backend, num_domains=args.num_domain,
        num_tasks=dataset.num_tasks
    ).to(device)

    conditional_backend = build_backend_from_config(conditioanl_backend_config)
    conditional_gnn = ConditionalGnn(
        emb_dim=conditioanl_backend_config['result_dim'],
        backend_dim=conditioanl_backend_config['result_dim'],
        backend=conditional_backend, num_classes=dataset.num_tasks,
        num_domain=args.num_domain
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
        batch_size=args.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        list(zip(valid_subs, valid_dataset)),
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        list(zip(test_subs, test_dataset)),
        batch_size=args.batch_size, shuffle=False
    )
    optimizer_main = Adam(main_model.parameters(), lr=args.lr)
    optimizer_dom = Adam(domain_classifier.parameters(), lr=args.lr)
    optimizer_con = Adam(conditional_gnn.parameters(), lr=args.lr)
    CLSLoss = BCEWithLogitsLoss()
    mean_loss = MeanLoss(CLSLoss)
    dev_loss = DeviationLoss(activation='abs', reduction='mean')

    train_curv, valid_curv, test_curv = [], [], []
    min_loss, best_ep, best_para, best_model_para = None, None, None, None
    best_valid = 0

    for ep in range(args.epoch_ast):
        print(f'[INFO] training on assistant models on {ep} epoch')
        conditional_gnn = conditional_gnn.train()
        domain_classifier = domain_classifier.train()
        Eqs, ELs = [], []
        for subs, graphs in tqdm(train_loader):
            # subs: a list of string, eval(string) can
            # get the substructures of corresponding molecule
            graphs = graphs.to(device)
            q_e = torch.softmax(domain_classifier(graphs), dim=-1)
            losses, batch_size = [], len(subs)

            for dom in range(args.num_domain):
                domain_info = torch.ones(batch_size).long().to(device)
                p_ye = conditional_gnn(graphs, domain_info * dom)
                labeled = graphs.y == graphs.y
                # there are nan in the labels so use this to mask them
                # and this is a multitask binary classification
                data_belong = torch.arange(batch_size).long()
                data_belong = data_belong.unsqueeze(dim=-1).to(device)
                data_belong = data_belong.repeat(1, dataset.num_tasks)
                # [batch_size, num_tasks] same as p_ye

                loss = bce_log(p_ye[labeled], graphs.y[labeled].float())
                # shape: [numbers of not nan gts]
                batch_loss = torch_scatter.scatter(
                    loss, dim=0, index=data_belong[labeled],
                    reduce='mean'
                )  # [batch_size]
                # considering the dataset is a multitask binary
                # classification task, the process above is to
                # get a average loss among all the tasks,
                # when there is only one task, it's equilvant to
                # bce_with_logit without reduction
                losses.append(batch_loss)

            losses = torch.stack(losses, dim=1)  # [batch_size, num_domain]
            Eq = torch.mean(torch.sum(q_e * losses, dim=-1))
            ELBO = Eq + KLDist(q_e, prior)

            ELBO.backward()
            optimizer_con.step()
            optimizer_dom.step()
            optimizer_dom.zero_grad()
            optimizer_con.zero_grad()

            Eqs.append(Eq.item())
            ELs.append(ELBO.item())
        print(f'[INFO] Eq: {np.mean(Eqs)}, ELBO: {np.mean(ELs)}')
        if best_ep is None or np.mean(ELs) < min_loss:
            min_loss, best_ep = np.mean(ELs), ep
            best_para = {
                'con': deepcopy(conditional_gnn.state_dict()),
                'dom': deepcopy(domain_classifier.state_dict())
            }

    print(f'[INFO] Using the best model in {best_ep} epoch')
    domain_classifier.load_state_dict(best_para['dom'])
    conditional_gnn.load_state_dict(best_para['con'])

    domain_classifier = domain_classifier.eval()
    conditional_gnn = conditional_gnn.eval()
    for ep in range(args.epoch_main):
        loss_in_epoch, num_in_epoch = {}, {}
        print(f'[INFO] Start training main model on {ep} epoch')
        main_model = main_model.train()
        for subs, graphs in tqdm(train_loader):
            subs = [eval(x) for x in subs]
            graphs = graphs.to(device)
            batch_size = len(subs)
            data_belong = torch.arange(batch_size).long()
            data_belong = data_belong.unsqueeze(dim=-1).to(device)
            data_belong = data_belong.repeat(1, dataset.num_tasks)
            labeled = graphs.y == graphs.y
            with torch.no_grad():
                cond_result = []
                for dom in range(args.num_domain):
                    domain_info = torch.ones(batch_size).long()
                    domain_info = (domain_info * dom).to(device)
                    cond_term = bce_log(
                        conditional_gnn(graphs, domain_info)[labeled],
                        graphs.y.float()[labeled]
                    )
                    cond_term = torch_scatter.scatter(
                        cond_term, dim=0, index=data_belong[labeled],
                        reduce='mean'
                    )
                    cond_result.append(cond_term)
                cond_result = torch.stack(cond_result, dim=0)
                # [num_domain, batch_size]
                cond_result = torch.matmul(prior, cond_result)
                # cond_result = torch.mean(cond_result, dim=0)
                # [batch_size]

            with torch.no_grad():
                p_e = domain_classifier(graphs)
                group = torch.argmax(p_e, dim=-1)
                # [batch_size]

            pred = main_model(subs, graphs)
            mean_term = mean_loss(pred, graphs.y.float(), group)
            this_loss = bce_log(pred[labeled], graphs.y.float()[labeled])
            this_loss = torch_scatter.scatter(
                this_loss, dim=0, index=data_belong[labeled],
                reduce='mean'
            )
            # print(this_loss.shape, cond_result.shape)
            dev_term = dev_loss(this_loss, cond_result)
            loss = args.lambda_loss * mean_term + dev_term
            loss.backward()
            optimizer_main.step()
            optimizer_main.zero_grad()

        print('[INFO] Evaluating the models')
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

        if valid_perf[dataset.eval_metric] > best_valid:
            best_valid = valid_perf[dataset.eval_metric]
            best_model_para = {
                'con': best_para['con'],
                'dom': best_para['dom'],
                'main': deepcopy(main_model.state_dict())
            }
        train_curv.append(train_perf[dataset.eval_metric])
        valid_curv.append(valid_perf[dataset.eval_metric])
        test_curv.append(test_perf[dataset.eval_metric])
        print({'Train': train_perf, 'Valid': valid_perf, 'Test': test_perf})

    # save model, input your own savepath
    # torch.save(best_model_para, your_path)

    best_val_epoch = np.argmax(np.array(valid_curv))
    print('Finished training!')
    print('Best epoch: {}'.format(best_val_epoch))
    print('Train score: {}'.format(train_curv[best_val_epoch]))
    print('Best validation score: {}'.format(valid_curv[best_val_epoch]))
    print('Test score: {}'.format(test_curv[best_val_epoch]))
    with open(os.path.join('log', args.work_dir, args.exp_name), 'w') as Fout:
        json.dump({
            'ast_best': best_ep,
            'config': args.__dict__,
            'train': train_curv,
            'valid': valid_curv,
            'test': test_curv,
            'best': [
                int(best_val_epoch),
                train_curv[best_val_epoch],
                valid_curv[best_val_epoch],
                test_curv[best_val_epoch]
            ],
        }, Fout)
