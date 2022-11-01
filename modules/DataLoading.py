import pickle
import os
import pandas


def pyg_loader(d_name, bs, shuffle=True, datainfo=None):
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.loader import DataLoader

    work_dir = os.path.abspath(os.path.dirname(__file__))
    # work_dir = os.path.dirname(work_dir)
    data_dir = os.path.join(work_dir, '../dataset')

    dataset = PygGraphPropPredDataset(d_name, root=data_dir)
    split_idx = dataset.get_idx_split()
    opt_train = {'batch_size': bs, 'shuffle': True}
    opt_other = {'batch_size': bs, 'shuffle': False}

    # x_mol = ['123'] * len(split_idx['train'])
    # train_loader = DataLoader(list(zip(x_mol, dataset[split_idx['train']])), **opt_train)

    train_loader = DataLoader(dataset[split_idx['train']], **opt_train)
    valid_loader = DataLoader(dataset[split_idx['valid']], **opt_other)
    test_loader = DataLoader(dataset[split_idx['test']], **opt_other)

    if datainfo is not None:
        result = {x: getattr(dataset, x) for x in datainfo}
        return train_loader, valid_loader, test_loader, result
    else:
        return train_loader, valid_loader, test_loader


def pyg_molloader(d_name, bs, shuffle=True, datainfo=None):
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.loader import DataLoader
    work_dir = os.path.abspath(os.path.dirname(__file__))
    # work_dir = os.path.dirname(work_dir)
    data_dir = os.path.join(work_dir, '../dataset')
    # print(data_dir)

    dataset = PygGraphPropPredDataset(d_name, root=data_dir)
    split_idx = dataset.get_idx_split()

    data_name = d_name.replace('-', '_')
    if not os.path.exists(os.path.join(work_dir, '../dataset', data_name)):
        raise IOError('No such dataset')

    dataset_dir = os.path.join(work_dir, '../dataset', data_name)

    original_data = pandas.read_csv(
        os.path.join(dataset_dir, 'mapping', 'mol.csv.gz'),
        compression='gzip'
    )
    train_smiles = [original_data.smiles[x.item()] for x in split_idx['train']]
    valid_smiles = [original_data.smiles[x.item()] for x in split_idx['valid']]
    test_smiles = [original_data.smiles[x.item()] for x in split_idx['test']]

    opt_train = {'batch_size': bs, 'shuffle': True}
    opt_other = {'batch_size': bs, 'shuffle': False}

    train_data = list(zip(train_smiles, dataset[split_idx['train']]))
    valid_data = list(zip(valid_smiles, dataset[split_idx['valid']]))
    test_data = list(zip(test_smiles, dataset[split_idx['test']]))

    train_loader = DataLoader(train_data, **opt_train)
    valid_loader = DataLoader(valid_data, **opt_other)
    test_loader = DataLoader(test_data, **opt_other)

    if datainfo is not None:
        result = {x: getattr(dataset, x) for x in datainfo}
        return train_loader, valid_loader, test_loader, result
    else:
        return train_loader, valid_loader, test_loader


def pyg_dataset(d_name):
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.loader import DataLoader

    work_dir = os.path.abspath(os.path.dirname(__file__))
    # work_dir = os.path.dirname(work_dir)
    data_dir = os.path.join(work_dir, '../dataset')

    dataset = PygGraphPropPredDataset(d_name, root=data_dir)
    return dataset


def pyg_moldataset(d_name):
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.loader import DataLoader

    work_dir = os.path.abspath(os.path.dirname(__file__))
    # work_dir = os.path.dirname(work_dir)
    data_dir = os.path.join(work_dir, '../dataset')

    dataset = PygGraphPropPredDataset(d_name, root=data_dir)
    data_name = d_name.replace('-', '_')
    dataset_dir = os.path.join(work_dir, '../dataset', data_name)
    original_data = pandas.read_csv(
        os.path.join(dataset_dir, 'mapping', 'mol.csv.gz'),
        compression='gzip'
    )
    smiles = original_data.smiles
    return smiles, dataset


def pyg_molsubdataset(d_name, preprocess_method='brics'):
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.loader import DataLoader

    work_dir = os.path.abspath(os.path.dirname(__file__))
    # work_dir = os.path.dirname(work_dir)
    data_dir = os.path.join(work_dir, '../dataset')

    dataset = PygGraphPropPredDataset(d_name, root=data_dir)
    data_name = d_name.replace('-', '_')
    dataset_dir = os.path.join(work_dir, '../dataset', data_name)
    original_data = pandas.read_csv(
        os.path.join(dataset_dir, 'mapping', 'mol.csv.gz'),
        compression='gzip'
    )
    smiles = original_data.smiles

    pre_name = os.path.join(work_dir, '../preprocess', data_name)
    pre_file = os.path.join(pre_name, 'substructures.pkl') \
        if preprocess_method == 'brics' else \
        os.path.join(pre_name, 'substructures_recap.pkl')
    
    if not os.path.exists(pre_file):
        raise IOError('please run preprocess script for dataset')
    with open(pre_file, 'rb') as Fin:
        substructures = pickle.load(Fin)

    return smiles, substructures, dataset


def split_loader(dataset, bs, shuffle=True):
    from torch_geometric.loader import DataLoader
    split_idx = dataset.get_idx_split()
    opt_train = {'batch_size': bs, 'shuffle': True}
    opt_other = {'batch_size': bs, 'shuffle': False}
    train_loader = DataLoader(dataset[split_idx['train']], **opt_train)
    valid_loader = DataLoader(dataset[split_idx['valid']], **opt_other)
    test_loader = DataLoader(dataset[split_idx['test']], **opt_other)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    dataset_opt = {'d_name': 'ogbg-molhiv', 'bs': 16, 'shuffle': True}
    train_loader, valid_loader, test_loader = pyg_molloader(**dataset_opt)
    for idx, (smiles, graph) in enumerate(test_loader):
        print(len(smiles), graph)
        print(graph.batch, graph.batch.shape)
        exit()
