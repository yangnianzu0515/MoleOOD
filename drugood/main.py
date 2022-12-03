from tqdm import tqdm
import time
import os
import argparse
from drugood.datasets import build_dataset, build_dataloader
from drugood.models import build_backbone
from mmcv import Config
import models
from models import Framework, ConditionalGnn, DomainClassifier, evaluate
from models import KLDist, MeanLoss, DeviationLoss, discrete_gaussian
import torch
import torch_scatter
from torch.optim import AdamW
import numpy as np
from torch.nn.functional import cross_entropy
from copy import deepcopy
import json
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
        '--lr', default=1e-3, type=float,
        help='the learning rate for training'
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
        '--num_domain', default=20, type=int,
        help='the number of domains for training'
    )
    parser.add_argument(
        '--epoch_main', default=50, type=int,
        help='the number of epochs of training main model'
    )
    parser.add_argument(
        '--epoch_ast', default=20, type=int,
        help='the number of epochs of training assistant models'
    )
    parser.add_argument(
        '--lambda_loss', default=1, type=float,
        help='the lambda for dev term in loss'
    )
    parser.add_argument(
        '--dist', default='uniform', type=str,
        help='the prior distribution of ELBO'
    )
    args = parser.parse_args()
    return args


def make_log(args):
    dataset_name = os.path.basename(args.data_config)
    dataset_name = dataset_name[:-3]
    model_name = os.path.basename(args.model_config)
    model_name = model_name[:-3]
    log_dir = os.path.join('log', dataset_name, model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fname = [
        f'lr_{args.lr}', f'domain_{args.num_domain}', f'seed_{args.seed}',
        f'ast_{args.epoch_ast}', f'main_{args.epoch_main}',
        f'lambda_{args.lambda_loss}', f'dist_{args.dist}', str(time.time()),
    ]
    fname = '-'.join(fname) + '.json'
    return log_dir, fname


def build_models_from_cfg(cfg, device, num_domain, num_class):
    base_backbone = build_backbone(cfg.model.main)
    sub_backbone = build_backbone(cfg.model.sub)
    cond_backbone = build_backbone(cfg.model.conditional)
    domain_backbone = build_backbone(cfg.model.domain)
    main_model = Framework(
        base=base_backbone, sub=sub_backbone, dropout=cfg.dropout,
        base_dim=cfg.model.main.emb_dim, sub_dim=cfg.model.sub.emb_dim,
        num_class=num_class
    ).to(device)
    conditional_gnn = ConditionalGnn(
        emb_dim=cfg.model.conditional.emb_dim,
        backend_dim=cfg.model.conditional.emb_dim,
        backend=cond_backbone, num_domain=num_domain,
        num_class=num_class
    ).to(device)
    domain_classifier = DomainClassifier(
        backend_dim=cfg.model.domain.emb_dim, num_task=1,
        backend=domain_backbone, num_domain=num_domain
    ).to(device)
    return main_model, conditional_gnn, domain_classifier


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
    seed_everything(args.seed)
    dataset_config = Config.fromfile(args.data_config)
    model_config = Config.fromfile(args.model_config)
    print(dataset_config.pretty_text)
    print(model_config.pretty_text)

    log_dir, log_name = make_log(args)

    device = torch.device('cpu') if args.device < 0 \
        else torch.device(f'cuda:{args.device}')

    train_set = build_dataset(dataset_config.data.train)
    valid_set = build_dataset(dataset_config.data.ood_val)
    test_set = build_dataset(dataset_config.data.ood_test)
    dataset_config.data.ood_val.test_mode = True
    dataset_config.data.ood_test.test_mode = True
    train_loader = build_dataloader(
        train_set, dataset_config.data.samples_per_gpu,
        dataset_config.data.workers_per_gpu, num_gpus=1,
        dist=False, round_up=True, seed=args.seed, shuffle=True
    )
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

    main_model, conditional_gnn, domain_classifier = \
        build_models_from_cfg(
            model_config, device, args.num_domain,
            dataset_config.data.num_class
        )

    optimizer_main = AdamW(main_model.parameters(), lr=args.lr)
    optimizer_dom = AdamW(domain_classifier.parameters(), lr=args.lr)
    optimizer_con = AdamW(conditional_gnn.parameters(), lr=args.lr)
    CLSLoss = torch.nn.CrossEntropyLoss()
    mean_loss = MeanLoss(CLSLoss)
    dev_loss = DeviationLoss(activation='abs', reduction='mean')
    prior = get_prior(args.num_domain, args.dist).to(device)

    loss_curv, min_loss, best_ep, best_para = [], None, None, {}
    for ep in range(args.epoch_ast):
        print(f'[INFO] training on assistant models on {ep} epoch')
        conditional_gnn = conditional_gnn.train()
        domain_classifier = domain_classifier.train()
        Eqs, ELs = [], []
        for data in tqdm(train_loader):
            data['input'] = data['input'].to(device)
            data['gt_label'] = data['gt_label'].to(device)
            q_e = torch.softmax(domain_classifier(data), dim=-1)
            losses, batch_size = [], len(data['subs'])
            for dom in range(args.num_domain):
                domain_info = torch.ones(batch_size).long().to(device)
                p_ye = conditional_gnn(data['input'], domain_info * dom)
                loss = cross_entropy(
                    p_ye, data['gt_label'].long(), reduction='none'
                )
                losses.append(loss)
            losses = torch.stack(losses, dim=1)
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
        loss_curv.append((np.mean(Eqs), np.mean(ELs)))
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

    with open(os.path.join(log_dir, log_name), 'w') as Fout:
        json.dump({
            "config": args.__dict__,
            "loss": loss_curv,
            'best_ep': best_ep
        }, Fout, indent=4)

    valid_curv, test_curv, max_valid_auc = {}, {}, None
    model_path = os.path.join(log_dir, f'model-{time.time()}.pth')
    for ep in range(args.epoch_main):
        print(f'[INFO] Start training main model on {ep} epoch')
        main_model = main_model.train()
        for data in tqdm(train_loader):
            subs = [eval(x) for x in data['subs']]
            data['input'] = data['input'].to(device)
            data['gt_label'] = data['gt_label'].to(device)
            batch_size, cond_result = len(subs), []
            with torch.no_grad():
                for dom in range(args.num_domain):
                    domain_info = torch.ones(batch_size).long()
                    domain_info = (domain_info * dom).to(device)
                    cond_term = cross_entropy(
                        conditional_gnn(data['input'], domain_info),
                        data['gt_label'].long(), reduction='none'
                    )
                    cond_result.append(cond_term)
                cond_result = torch.stack(cond_result, dim=0)
                # cond_result = torch.mean(cond_result, dim=0)
                cond_result = torch.matmul(prior, cond_result)

                p_e = domain_classifier(data)
                group = torch.argmax(p_e, dim=-1)

            pred = main_model(data['input'], subs)
            mean_term = mean_loss(pred, data['gt_label'].long(), group)
            this_loss = cross_entropy(
                pred, data['gt_label'].long(), reduction='none'
            )
            assert this_loss.shape == cond_result.shape, \
                "The Shape Should be the Same Size"
            # print(this_loss.shape, cond_result.shape)
            dev_term = dev_loss(this_loss, cond_result)
            loss = args.lambda_loss * mean_term + dev_term
            loss.backward()
            optimizer_main.step()
            optimizer_main.zero_grad()
        print('[INFO] evaluating the model')

        val_perf = eval_one_epoch(main_model, valid_loader, device)
        test_perf = eval_one_epoch(main_model, test_loader, device)
        for k, v in val_perf.items():
            if k not in valid_curv:
                valid_curv[k], test_curv[k] = [], []
            valid_curv[k].append(val_perf[k])
            test_curv[k].append(test_perf[k])

        print('[INFO] valid: {}'.format(val_perf))
        print('[INFO] test: {}'.format(test_perf))

        with open(os.path.join(log_dir, log_name), 'w') as Fout:
            json.dump({
                'ast_best': best_ep,
                'config': args.__dict__,
                'valid': valid_curv,
                'test': test_curv,
                'loss': loss_curv
            }, Fout, indent=4)
        if max_valid_auc is None or val_perf['auc'] > max_valid_auc:
            torch.save({
                'main': main_model.state_dict(),
                'dom': best_para['dom'],
                'con': best_para['con']
            }, model_path)
            max_valid_auc = val_perf['auc']

    best_result = {}
    for k, v in valid_curv.items():
        pos = int(np.argmax(v))
        best_result[k] = [pos, v[pos], test_curv[k][pos]]
    print('[INFO] best results:', best_result)
    with open(os.path.join(log_dir, log_name), 'w') as Fout:
        json.dump({
            'ast_best': int(best_ep),
            'config': args.__dict__,
            'valid': valid_curv,
            'test': test_curv,
            'loss': loss_curv,
            'best': best_result
        }, Fout, indent=4)
