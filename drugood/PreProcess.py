import time
import subprocess
import argparse
import json
import os
from tqdm import tqdm


def get_all_smiles(prefix='data', ds='ic50'):
    with open(os.path.join(prefix, f'lbap_core_{ds}_assay.json')) as Fin:
        info = json.load(Fin)
    parts = ['train', 'iid_test', 'ood_test', 'iid_val', 'ood_val']
    all_smiles = []
    for part in parts:
        all_smiles += [x['smiles'] for x in info['split'][part]]
    with open(os.path.join(prefix, 'all_smiles.json'), 'w') as Fout:
        json.dump(all_smiles, Fout, indent=4)

    return all_smiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser('substructure decomposition for drugood')
    parser.add_argument(
        '--start', default=0, type=int,
        help='start idx for decomposition'
    )
    parser.add_argument(
        '--num', default=5000, type=int,
        help='the number of mols for decomposition'
    )
    parser.add_argument(
        '--timeout', default=120, type=int,
        help='the timeout of processing one mol'
    )
    parser.add_argument(
        '--method', choices=['brics', 'recap'], default='brics',
        help='the method for substructure decomposition'
    )
    parser.add_argument(
        '--dataset', choices=['ic50', 'ec50'], default='ic50',
        help='the dataset to make substructure decomposition'
    )
    args = parser.parse_args()

    PREFIX = os.path.join('data', args.dataset)
    if not os.path.exists(os.path.join(PREFIX, 'all_smiles.json')):
        all_smiles = get_all_smiles(PREFIX, args.dataset)
    else:
        with open(os.path.join(PREFIX, 'all_smiles.json')) as Fin:
            all_smiles = json.load(Fin)

    if args.method == 'brics':
        output_dir = os.path.join(PREFIX, 'substructure')
    else:
        output_dir = os.path.join(PREFIX, 'substructure_recap')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f'[INFO] there are {len(all_smiles)} mols in total')
    assert args.start < len(all_smiles), 'start_idx too large'
    print(f'[INFO] decomposition from {args.start} to {args.start + args.num - 1}')

    out_file_name = f'{args.start}-{args.start + args.num - 1}.txt'
    Fout = open(os.path.join(output_dir, out_file_name), 'w')
    escapes = []
    for idx, smile in enumerate(tqdm(all_smiles[args.start: args.start + args.num])):
        try:
            subprocess.run(
                ['python', 'GetSubStruct.py', '--smile',
                    smile, '--method', args.method],
                check=True, timeout=args.timeout, stdout=Fout
            )
        except subprocess.TimeoutExpired:
            escapes.append(idx)
            Fout.write(f'{smile}\t{str(set([smile]))}\n')

    Fout.close()
    print(f'[INFO] there are {len(escapes)} mols escapes')
    print(escapes)
