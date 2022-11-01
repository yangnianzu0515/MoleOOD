from rdkit.Chem import BRICS
from DataLoading import pyg_moldataset
import argparse
import os
import pickle
import subprocess
from tqdm import tqdm


def get_result_dir():
    work_dir = os.path.abspath(os.path.dirname(__file__))
    result_dir = os.path.join(work_dir, '../preprocess')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocessing For Dataset')
    parser.add_argument(
        '--dataset', default='ogbg-molhiv', type=str,
        help='the dataset to preprocess'
    )
    parser.add_argument(
        '--timeout', default=120, type=int,
        help='maximal time to process a single molecule, count int seconds'
    )
    parser.add_argument(
        '--method', choices=['recap', 'brics'], default='brics',
        help='the method to decompose the molecule, brics or recap'
    )
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    print(args)

    result_dir = get_result_dir()
    data_name = args.dataset.replace('-', '_')
    if not os.path.exists(os.path.join(result_dir, data_name)):
        os.mkdir(os.path.join(result_dir, data_name))
    smiles, dataset = pyg_moldataset(args.dataset)
    file_name = 'substructures.pkl' if args.method == 'brics' \
        else 'substructures_recap.pkl'
    file_name = os.path.join(result_dir, data_name, file_name)
    escapes = []
    with open(file_name, 'w') as Fout:
        for idx, smile in enumerate(tqdm(smiles)):
            try:
                subprocess.run([
                    'python', 'modules/GetSubStruct.py', '--smile',
                    smile, '--method', args.method
                ], check=True, timeout=args.timeout, stdout=Fout)
            except subprocess.TimeoutExpired:
                escapes.append(idx)
                Fout.write(f'{smile}\t{str(set([smile]))}\n')

    if len(escapes) > 0:
        print('[INFO] the following molecules are processed unsuccessfully:')
        [print(smiles[x]) for x in escapes]

    substruct_list = []

    with open(file_name) as Fin:
        for lin in Fin:
            if len(lin) <= 1:
                continue
            lin = lin.strip().split('\t')
            assert len(lin) == 2, f'Invalid Line {lin}'
            assert type(eval(lin[1])) == set, f'Invalid value1 {lin[1]}'
            if len(eval(lin[1])) == 0:
                print(
                    f'[INFO] empty substruct find for {lin[0]},'
                    'consider itself as a substructure'
                )
                substruct_list.append(set([lin[0]]))
            else:
                substruct_list.append(eval(lin[1]))

    with open(file_name, 'wb') as Fout:
        pickle.dump(substruct_list, Fout)
