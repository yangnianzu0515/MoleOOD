import os
import json
from tqdm import tqdm


def process(bmethod):
    result = {}
    print(f'[INFO] Loading substructures from {bmethod}')
    prefix = 'substructure' if bmethod == 'brics' else 'substructure_recap'
    for file in os.listdir(prefix):
        if not file.endswith(('.json', '.txt')):
            continue
        if file.endswith('.json'):
            with open(os.path.join(prefix, file)) as Fin:
                INFO = json.load(Fin)
            for k, v in tqdm(INFO.items()):
                assert type(eval(v)) == set, f'Invalid value1 {v}'
                result[k] = v
        else:
            with open(os.path.join(prefix, file)) as Fin:
                for lin in tqdm(Fin):
                    if len(lin) <= 1:
                        continue
                    lin = lin.strip().split('\t')
                    assert len(lin) == 2, f'Invalid Line {lin}'
                    assert type(eval(lin[1])) == set, \
                        'Invalid value1 {}'.format(lin[1])
                    result[lin[0]] = lin[1]

    with open('all_smiles.json') as Fin:
        all_smiles = json.load(Fin)

    new_result, empty_num = {}, 0
    for k, v in result.items():
        v = eval(v)
        if len(v) == 0:
            print(f'[INFO] {k} have no substruct, use it self as substruct')
            new_result[k], empty_num = set([k]), empty_num + 1
        else:
            new_result[k] = v

    print(f'[INFO] there are {empty_num} mols have no substruct')

    result = {k: str(v) for k, v in new_result.items()}

    for method in ['assay', 'scaffold', 'size']:
        print(f'[INFO] Processing method {method}')
        with open(f'lbap_core_ec50_{method}.json') as Fin:
            INFO = json.load(Fin)
        for k, v in INFO['split'].items():
            new_mol_list = []
            for mol in tqdm(v):
                mol['substructure'] = result[mol['smiles']]
                new_mol_list.append(mol)
            INFO['split'][k] = new_mol_list
        with open(f'lbap_core_ec50_{method}_{bmethod}.json', 'w') as Fout:
            json.dump(INFO, Fout, indent=4)


if __name__ == '__main__':
    if os.path.exists('substructure'):
        process('brics')
    if os.path.exists('substructure_recap'):
        process('recap')
