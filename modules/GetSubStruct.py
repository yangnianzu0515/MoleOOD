from ChemistryProcess import get_substructure
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Function Aux')
    parser.add_argument('--smile', type=str, required=True)
    parser.add_argument(
        '--method', choices=['brics', 'recap'], default='brics'
    )
    args = parser.parse_args()

    tx = get_substructure(smile=args.smile, decomp=args.method)
    print(f'{args.smile}\t{str(tx)}')
