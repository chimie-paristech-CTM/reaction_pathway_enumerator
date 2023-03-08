#from Molecule import Molecule
#from generate_products import enumerate_reaction_possibilites, generate_products
import argparse

from enumerator.Molecule import Molecule
from enumerator.generate_products import enumerate_reaction_possibilites, generate_products


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', action='store', type=str)
    parser.add_argument('--idx-list', action='store', type=list, default=None)
    parser.add_argument('--n-bonding-systems', action='store', type=int, default=4)
    return parser.parse_args()


def enumerate():
    args = get_args()
    mol = Molecule(args.smiles)
    if args.idx_list:
        products = generate_products(mol, args.idx_list)
    elif args.n_bonding_systems:
        products = enumerate_reaction_possibilites(mol, args.n_bonding_systems)
    
    print(products)
