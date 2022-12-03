import rdkit
from rdkit.Chem import BRICS, Recap
from rdkit import Chem
from rdkit.Chem import Draw
import os
import numpy as np
from ogb.utils import smiles2graph
import torch
from functools import reduce
from torch_geometric.data import Data


def get_substructure(mol=None, smile=None, decomp='brics'):
    assert mol is not None or smile is not None,\
        'need at least one info of mol'
    assert decomp in ['brics', 'recap'], 'Invalid decomposition method'
    if mol is None:
        mol = Chem.MolFromSmiles(smile)

    if decomp == 'brics':
        substructures = BRICS.BRICSDecompose(mol)
    else:
        recap_tree = Recap.RecapDecompose(mol)
        leaves = recap_tree.GetLeaves()
        substructures = set(leaves.keys())
    return substructures


def substructure_batch(smiles, return_mask, return_type='numpy'):
    sub_structures = [get_substructure(smile=x) for x in smiles]
    return graph_from_substructure(sub_structures, return_mask, return_type)


def graph_from_substructure(subs, return_mask=False, return_type='numpy'):
    sub_struct_list = list(reduce(lambda x, y: x.update(y) or x, subs, set()))
    sub_to_idx = {x: idx for idx, x in enumerate(sub_struct_list)}
    mask = np.zeros([len(subs), len(sub_struct_list)], dtype=bool)
    sub_graph = [smiles2graph(x) for x in sub_struct_list]
    for idx, sub in enumerate(subs):
        mask[idx][list(sub_to_idx[t] for t in sub)] = True

    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    for idx, graph in enumerate(sub_graph):
        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        # 'node_feat': np.concatenate(node_feats, axis=0),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }

    assert return_type in ['numpy', 'torch', 'pyg'], 'Invaild return type'
    if return_type in ['torch', 'pyg']:
        for k, v in result.items():
            result[k] = torch.from_numpy(v)

    result['num_nodes'] = lstnode

    if return_type == 'pyg':
        result = Data(**result)

    if return_mask:
        return result, mask
    else:
        return result


if __name__ == '__main__':
    if not os.path.exists('test'):
        os.mkdir('test')
    smile_test = 'CC(=O)Oc1ccccc1C(=O)O'
    smile_test2 = 'O([H])C1=C([H])C([H])=C([H])C([H])=C1C(=O)O[H]'
    import networkx
    import matplotlib.pyplot as plt
    from torch_geometric.utils import to_networkx
    whole_mol = Chem.MolFromSmiles(smile_test2)
    sub_structures = BRICS.BRICSDecompose(whole_mol)
    whole_mol_img = Draw.MolToImage(whole_mol)
    whole_mol_img.save('test/whole_mol2.jpg')

    for idx, sub in enumerate(sub_structures):
        sub_fig = Draw.MolToImage(Chem.MolFromSmiles(sub))
        sub_fig.save(f'test/substructure2_{idx}.jpg')
        print(smiles2graph(sub))

    # m = Chem.MolFromSmiles('O=C(NCc1cc(OC)c(O)cc1)Cc1cocc1CC')
    # core = MurckoScaffold.GetScaffoldForMol(m)
    # print(Chem.MolToSmiles(core))
    # m_core = [m, core]
    # scaffold_mol = Draw.MolsToGridImage(m_core, subImgSize=(250, 250))
    # scaffold_mol.save(f'test/mol_scaffold.jpg')

    graph_data, mask = substructure_batch(
        [smile_test, smile_test2], return_mask=True,
        return_type='torch'
    )
    print(mask)
    print(graph_data)
    gdt = Data(**graph_data)
    plt.clf()
    graph_to_draw = to_networkx(gdt)
    networkx.draw(graph_to_draw)
    plt.savefig('test/subpart.png')
