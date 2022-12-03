import torch
import torch_geometric.nn as Gnn


def split_into_groups(g):
    unique_groups, unique_counts = torch.unique(
        g, sorted=False, return_counts=True
    )
    group_indices = [
        torch.nonzero(g == group, as_tuple=True)[0]
        for group in unique_groups
    ]
    return unique_groups, group_indices, unique_counts


def get_pool(method):
    result = None
    if method == 'add':
        result = Gnn.global_add_pool
    elif method == 'mean':
        result = Gnn.global_mean_pool
    elif method == 'max':
        result = Gnn.global_max_pool

    else:
        raise ValueError("Invalid graph pooling type")

    return result


def get_device(number):
    if number < 0 or not torch.cuda.is_available():
        return torch.device('cpu')
    else:
        return torch.device(f'cuda:{number}')


def get_final_dim(aggr, dim1, dim2):
    if aggr == 'only':
        return dim1
    elif aggr == 'concat':
        return dim1 + dim2
    elif aggr == 'dot':
        return max(dim1, dim2)
    else:
        raise ValueError('Error Aggr Value')


def get_config_name(name):
    name = name.strip().split('/')[-1]
    name = name.split('\\')[-1]
    name = name.split('.')[-2]
    return name


if __name__ == '__main__':
    pool = get_pool('add')
    print(pool)
