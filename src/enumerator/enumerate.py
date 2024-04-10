import argparse
from tqdm import tqdm
import logging

from enumerator.reacting_system import ReactingSystem
from enumerator.get_energies import get_system_energy 

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
    print(products)
    print(len(products))
    product_energies_dict = get_energy_dict(args.smiles, products, args.solvent)

    logging.info(product_energies_dict)
    logging.info(len(product_energies_dict))
    feasible_products_dict = dict(
        (k, product_energies_dict[k])
        for k in product_energies_dict.keys()
        if product_energies_dict[k] < 0
    )

    print(feasible_products_dict)
    print(len(feasible_products_dict))

    print(len(product_energies_dict))


def enumerate_potential_products(smiles, idx_list=None, n_bonding_systems=4):
    """Enumerates all the potential products based on either an index list or a number of bonding systems.

    Args:
        smiles (str): A SMILES string
        idx_list (list, optional): A list of bonding system indices. Defaults to None.
        n_bonding_systems (int, optional): The maximum number of active bonding systems. Defaults to 4.

    Returns:
        list: A list of product SMILES.
    """
    reacting_system = ReactingSystem(smiles)
    original_paths = reacting_system.generate_reaction_paths()
    products = reacting_system.generate_products(original_paths)

    return products


def get_energy_dict(reactants, products, solvent):
    """Obtains a dictionary of relative product energies.

    Args:
        reactants (str): SMILES string corresponding to the reactants.
        products (str): SMILES string corresponding to the products.
        solvent (str): SMILES string corresponding to the solvent.

    Returns:
        dict: a dictionary of SMILES and their corresponding energies.
    """
    energy_dict = {}
    reactant_energy = get_system_energy(reactants, solvent=solvent)
    for product in tqdm(products, total=len(products)):
        try:
            energy_dict[product] = (get_system_energy(product, solvent=solvent) - reactant_energy) * HARTREE_TO_EV
        except TypeError:
            continue

    return energy_dict
