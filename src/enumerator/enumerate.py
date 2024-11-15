import argparse
from tqdm import tqdm

from enumerator.reacting_system import ReactingSystem
from enumerator.get_energies import get_system_energy
from enumerator.utils import create_logger
from rdkit.Chem.Draw import MolsToGridImage, rdMolDraw2D
from rdkit import Chem

HARTREE_TO_EV = 27.2114


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", action="store", type=str)
    parser.add_argument("--idx-list", nargs="+", default=None, type=int)
    parser.add_argument("--solvent", action="store", default=None)
    parser.add_argument("--max-length", action="store", type=int, default=2)
    parser.add_argument("--allow-zwitterions", action="store_true", default=False)
    parser.add_argument("--print-configuration", action="store_true", default=False)
    parser.add_argument("--nbo", action="store_true", default=False)
    parser.add_argument("--nbo-dir", action="store", default=None)
    parser.add_argument("--threshold-sec-interaction", action="store", default=11.5)
    parser.add_argument("--threshold-strong-sec-interaction", action="store", type=float, default=85.0)
    parser.add_argument("--ts-tools", action="store_true", default=False)

    return parser.parse_args()


def get_thermodynamically_feasible_products():
    """Returns a list of feasible product molecules based on the SMILES input."""
    args = get_args()
    logger = create_logger(name='output')

    if args.print_configuration:
        reacting_system = ReactingSystem(args.smiles, args.nbo, args.nbo_dir, args.threshold_strong_sec_interaction)
        for orbital_system in reacting_system.localized_configuration.active_orbital_systems_list:
            print(orbital_system)
    else:
        products, smiles_reactants = enumerate_potential_products(
            args.smiles, args.idx_list, args.max_length, args.allow_zwitterions, args.nbo, args.nbo_dir,
            args.threshold_strong_sec_interaction
        )
        #print(products)
        #print(len(products))
        product_energies_dict = get_energy_dict(args.smiles, products, args.solvent)

        for k in product_energies_dict.keys():
            logger.info(f"{k}  {product_energies_dict[k]}")
        logger.info(len(product_energies_dict))
        feasible_products_dict = dict(
            (k, product_energies_dict[k])
            for k in product_energies_dict.keys()
            if product_energies_dict[k] < 0
        )
        print_rdkit_mol(product_energies_dict)

        #print(feasible_products_dict)
        print(len(feasible_products_dict))

        print(len(product_energies_dict))

        if args.ts_tools:
            print_input_ts_tools(smiles_reactants, products)



def enumerate_potential_products(smiles, idx_list, max_length=2, allow_zwitterions=True, nbo=False, nbo_dir=None,
                                 threshold_strong_sec_interaction=85.0):
    """Enumerates all the potential products based on either an index list or a number of bonding systems.

    Args:
        smiles (str): A SMILES string.
        idx_list (list, optional): A list of bonding system indices. Defaults to None.
        max_length (int, optional): The maximum number of orbital systems in a single fragment.
        allow_zwitterions (bool, optional): Whether or not to allow the generation of zwitterions
        nbo (bool, optional): Use NBO
        nbo_dir (str, optional): Directory with NBO output
        threshold_strong_sec_interaction (float, optional): Threshold for strong secondary interaction

    Returns:
        list: A list of product SMILES.
    """
    reacting_system = ReactingSystem(smiles, nbo, nbo_dir, threshold_strong_sec_interaction)
    original_paths = reacting_system.generate_reaction_paths(idx_list=idx_list, max_length=max_length)
    products = reacting_system.generate_products(original_paths, allow_zwitterions=allow_zwitterions)

    return products, reacting_system.numbered_smiles


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


def print_rdkit_mol(products):

    smiles_list = []
    legend_list = []

    for key in products.keys():
        smiles_list.append(key)
        legend_list.append(f"{products[key]:.3f}")
        smiles_mol = [Chem.MolFromSmiles(smile) for smile in smiles_list]

    img = MolsToGridImage(mols=smiles_mol, legends=legend_list, molsPerRow=5)
    img.save('output.png')


def print_input_ts_tools(smiles_reactants, products):

    with open('reactions_input.txt', 'w') as file:
        for idx, prod in enumerate(products):
            rxn_smiles = f"{smiles_reactants}>>{prod}"
            file.write(f"R{idx}  {rxn_smiles}\n")
