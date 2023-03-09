#from Molecule import Molecule
#from generate_products import enumerate_reaction_possibilites, generate_products
import argparse
from tqdm import tqdm

from enumerator.Molecule import Molecule
from enumerator.generate_products import enumerate_reaction_possibilites, generate_products
#from enumerator.get_energies import AimnetCalculator
from enumerator.get_energies import get_system_energy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', action='store', type=str)
    parser.add_argument('--idx-list', action='store', type=list, default=None)
    parser.add_argument('--n-bonding-systems', action='store', type=int, default=4)
    return parser.parse_args()


def enumerate_reaction_possibilities():
    args = get_args()
    mol = Molecule(args.smiles)
    if args.idx_list:
        products = generate_products(mol, args.idx_list)
    elif args.n_bonding_systems:
        products = enumerate_reaction_possibilites(mol, args.n_bonding_systems)
    
    return products

def get_energies(products):
    energy_dict = {}
    #calculator = AimnetCalculator()
    for product in tqdm(products, total=len(products)):
        energy_dict[product] = get_system_energy(product)

    print(energy_dict)