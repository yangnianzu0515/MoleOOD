import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from modules.GNNs import GNN
from modules.SAGE import SAGEMol
from modules.utils import get_device
import torch_geometric
from tqdm import tqdm
import argparse
import os
import numpy as np

# importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            # ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(
                    pred.to(torch.float32)[is_labeled],
                    batch.y.to(torch.float32)[is_labeled]
                )
            else:
                loss = reg_criterion(
                    pred.to(torch.float32)[is_labeled],
                    batch.y.to(torch.float32)[is_labeled]
                )
            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        'GNN baselines on ogbgmol* data with Pytorch Geometrics'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='which gpu to use if any (default: 0), negative for cpu'
    )
    parser.add_argument(
        '--gnn', type=str, default='gin-virtual',
        help='GNN gin, gin-virtual, gat, gat-virtual, sage, sage-virtual'
        ', gcn, or gcn-virtual (default: gin-virtual)'
    )
    parser.add_argument(
        '--drop_ratio', type=float, default=0.5,
        help='dropout ratio (default: 0.5)'
    )
    parser.add_argument(
        '--num_layer', type=int, default=5,
        help='number of GNN message passing layers (default: 5)'
    )
    parser.add_argument(
        '--emb_dim', type=int, default=256,
        help='dimensionality of hidden units in GNNs (default: 256)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='input batch size for training (default: 32)'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='number of epochs to train (default: 100)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='number of workers (default: 0)'
    )
    parser.add_argument(
        '--dataset', type=str, default="ogbg-molhiv",
        help='dataset name (default: ogbg-molhiv)'
    )
    parser.add_argument(
        '--seed', default=2022, type=int,
        help='the random seed for experiment'
    )

    parser.add_argument(
        '--feature', type=str, default="full",
        help='full feature or simple feature'
    )
    parser.add_argument(
        '--filename', type=str, default="",
        help='filename to output result (default: )'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help="learning rate"
    )

    parser.add_argument(
        '--heads', type=int, default=1, help='heads for gat'
    )

    args = parser.parse_args()

    torch_geometric.seed.seed_everything(args.seed)


    device = get_device(args.device)

    # automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = dataset.get_idx_split()

    # automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(
        dataset[split_idx["train"]], batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]], batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]], batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    if args.gnn == 'gin':
        model = GNN(
            gnn_type='gin', num_tasks=dataset.num_tasks,
            num_layer=args.num_layer, emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio, virtual_node=False
        ).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(
            gnn_type='gin', num_tasks=dataset.num_tasks,
            num_layer=args.num_layer, emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio, virtual_node=True
        ).to(device)
    elif args.gnn == 'gcn':
        model = GNN(
            gnn_type='gcn', num_tasks=dataset.num_tasks,
            num_layer=args.num_layer, emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio, virtual_node=False
        ).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(
            gnn_type='gcn', num_tasks=dataset.num_tasks,
            num_layer=args.num_layer, emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio, virtual_node=True
        ).to(device)
    elif args.gnn == 'sage':
        model = SAGEMol(
            emb_dim=args.emb_dim, num_layer=args.num_layer,
            drop_ratio=args.drop_ratio,
            num_tasks=dataset.num_tasks, virtual=False
        ).to(device)
    elif args.gnn == 'sage-virtual':
        model = SAGEMol(
            emb_dim=args.emb_dim, num_layer=args.num_layer,
            drop_ratio=args.drop_ratio,
            num_tasks=dataset.num_tasks, virtual=True
        ).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({
            'Val': valid_curve[best_val_epoch],
            'Test': test_curve[best_val_epoch],
            'Train': train_curve[best_val_epoch],
            'BestTrain': best_train
        }, args.filename)


if __name__ == "__main__":
    main()
