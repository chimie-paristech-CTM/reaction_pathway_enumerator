import argparse
from tqdm import tqdm
import logging

from enumerator.Molecule import Molecule

HARTREE_TO_EV = 27.2114


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", action="store", type=str)
    parser.add_argument("--idx-list", nargs="+", default=None)
    parser.add_argument("--solvent", action="store", default=None)
    parser.add_argument("--n-bonding-systems", action="store", type=int, default=4)
    
    return parser.parse_args()


def get_thermodynamically_feasible_products():
    """Returns a list of feasible product molecules based on the SMILES input."""
    args = get_args()
    logging.basicConfig(
        filename=f"test.log", encoding="utf-8", level=logging.DEBUG
    )
    products = enumerate_potential_products(
        args.smiles, args.idx_list, args.n_bonding_systems
    )
    #product_energies_dict = get_energy_dict(args.smiles, products, args.solvent)

    #logging.info(product_energies_dict)
    #logging.info(len(product_energies_dict))
    #feasible_products_dict = dict(
    #    (k, product_energies_dict[k])
    #    for k in product_energies_dict.keys()
    #    if product_energies_dict[k] < 0
    #)

    #print(feasible_products_dict)
    #print(len(feasible_products_dict))

    #print(len(product_energies_dict))
    #print(product_energies_dict)


def enumerate_potential_products(smiles, idx_list=None, n_bonding_systems=4):
    """Enumerates all the potential products based on either an index list or a number of bonding systems.

    Args:
        smiles (str): A SMILES string
        idx_list (list, optional): A list of bonding system indices. Defaults to None.
        n_bonding_systems (int, optional): The maximum number of active bonding systems. Defaults to 4.

    Returns:
        list: A list of product SMILES.
    """
    mol = Molecule(smiles)
    paths = mol.construct_reaction_paths(2)
    mol.modify_reaction_paths(paths)
    
    #generate_promotion_states(mol, 3)



    #if idx_list:
    #    products = generate_products(mol, list(map(int, idx_list)))
    #elif n_bonding_systems:
    #    products = enumerate_reaction_possibilities(mol, n_bonding_systems)

    #products = list(set(products))

    #return products
