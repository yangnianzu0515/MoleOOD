import rdkit
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Recap

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
