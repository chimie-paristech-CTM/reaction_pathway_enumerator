from rdkit import Chem
import copy
import numpy as np
from Molecule import Molecule, BondingSystem, Atom, ValenceOrbital
from generate_smiles import generate_smiles
import itertools
import timeit

    
def determine_reaction_type(bonding_system_init):
    if len(bonding_system_init) == 2:
        return 'concerted' # TODO: is this correct? Can't you have simultaneous heterolytic dissociation and reaction? Maybe based on a EN-diff threshold?
    elif len(bonding_system_init) == 1:
        if bonding_system_init.vos[0].num_electrons == 2:
            return 'electrophilic'
        elif bonding_system_init.vos[0].num_electrons == 1:
            return 'radical'
        elif bonding_system_init.vos[0].num_electrons == 0:
                return 'nucleophilic'
    else:
        print('Not yet implemented!')


def set_polarization_bonding_systems(bonding_systems, reaction_type):
    if reaction_type == 'electrophilic':
        for bonding_system in bonding_systems:
            if len(bonding_system) > 1:
                bonding_system.set_polarity()
                if bonding_system.polarity is not None:
                    bonding_system.vos = [bonding_system.polarity['pos_pole'], bonding_system.polarity['neg_pole']]
    elif reaction_type == 'nucleophilic':
        for bonding_system in bonding_systems:
            if len(bonding_system) > 1:
                bonding_system.set_polarity()
                if bonding_system.polarity is not None:
                    bonding_system.vos = [bonding_system.polarity['neg_pole'], bonding_system.polarity['pos_pole']]
    


def construct_new_bonding_system(vo1, vo2, idx=-1):
    """ Auxiliary function to construct a new bonding system from 2 existing vos.

    Args:
        vo1 (ValenceOrbital): first valence orbital object.
        vo2 (ValenceOrbital): second valence orbital object.
        idx (int): the index of the bonding system to be formed.

    Returns:
        BondingSystem: the new bonding system.
    """
    new_bonding_system = BondingSystem(idx)
    vo1.set_population(1)
    vo2.set_population(1)
    new_bonding_system.add_vo(vo1)
    new_bonding_system.add_vo(vo2)

    return new_bonding_system


def enumerate_reaction_possibilites(molecule: 'Molecule', max_length: int):
    # TODO: These initial tests should be integrated in BondingSystem class???
    active_bonding_system_ids = []
    # heuristics to reduce number of possibilities -> only one lone pair and one X-H bond per atom 
    # + only one interatomic bonding system if multiple
    atoms_with_lone_pairs_covered = []
    atoms_with_xh_covered = []
    bonds_covered = []
    
    # filter out non-active bonding systems
    for bonding_system in molecule.bonding_systems:
        print(bonding_system)
        info = bonding_system.get_atom_info()
        if len(bonding_system) == 1 and bonding_system.num_electrons == 2:
            if info['atom_ids'] in atoms_with_lone_pairs_covered:
                continue
            else:
                active_bonding_system_ids.append(bonding_system.idx)
                atoms_with_lone_pairs_covered.append(info['atom_ids'])
        elif 'H' in info['atom_types']:
            if min(info['atom_ids']) in atoms_with_xh_covered: # H atoms have higher indices
                continue
            else:
                active_bonding_system_ids.append(bonding_system.idx)
                atoms_with_xh_covered.append(info['atom_ids'][0])
        elif info['atom_ids'] in bonds_covered:
            continue
        else:
            active_bonding_system_ids.append(bonding_system.idx)
            bonds_covered.append(info['atom_ids'])

    all_products = []
    for L in range(2, max_length + 1):
        for idx_comb in itertools.combinations(active_bonding_system_ids, L):
            for idx_perm in itertools.permutations(idx_comb):
                print(idx_perm)
                generated_products = generate_products(molecule, list(idx_perm))
                if None not in generated_products:
                    all_products += generated_products

    return all_products


# TODO treat exception when you have a single bonding_system -> should abort when single vo; if two vos, you should distinguish homo- and heterolytic dissociation
def generate_products(molecule: 'Molecule', idx_list: list):
    """ Generate products based on permutation of a subset of the bonding systems.

    Args:
        molecule (Molecule): a molecule object.
        idx_list (list): the list of bonding systems that need to be permutated.

    Returns:
        products (list): list of product SMILES.
    """

    products = []
    
    # save a copy of the bonding systems being modified
    old_bonding_systems = [copy.deepcopy(molecule.bonding_systems[idx]) for idx in idx_list]

    # you don't want a lone pair or empty orbital halfway a sequence and you also don't want lone pairs in radical/concerted sequence
    # TODO charge transfer!!!
    if any([len(bonding_system) == 1 for bonding_system in old_bonding_systems[1:-1]]):
        return None, None 

    # you don't want two vos on the same atom to be part of a single reactive event, except for lone pairs/empty orbitals at edges of reaction path 
    atom_list = []
    for bonding_system in old_bonding_systems:
        atom_list += [vo.atom_idx for vo in bonding_system.vos]
    if len(atom_list[1:-1]) != len(set(atom_list[1:-1])):
        return None, None

    bonding_system_init = molecule.bonding_systems[idx_list[0]]
    reaction_type = determine_reaction_type(bonding_system_init)

    # get plausible arrangments; for polar reaction there is a preferential ordering; for radical/concerted ones you need to take all combinations into account
    bonding_system_arrangments = []
    if reaction_type == 'electrophilic' or reaction_type == 'nucleophilic':
        set_polarization_bonding_systems([molecule.bonding_systems[idx] for idx in idx_list], reaction_type)

    bonding_system_arrangments.append([copy.deepcopy(molecule.bonding_systems[idx_list[0]])])
    if len(molecule.bonding_systems[idx_list[0]]) == 2:
        molecule.bonding_systems[idx_list[0]].reverse_vo_order()
        bonding_system_arrangments.append([copy.deepcopy(molecule.bonding_systems[idx_list[0]])])

    for idx in idx_list[1:]:
        if len(molecule.bonding_systems[idx]) == 2:
            if (reaction_type == 'electrophilic' or reaction_type == 'nucleophilic') and molecule.bonding_systems[idx].polarity != None:
                for i in range(len(bonding_system_arrangments)):
                    bonding_system_arrangments[i].append(molecule.bonding_systems[idx])
            else:
                bonding_system_arrangments = [copy.deepcopy(bonding_system_arrangment) for bonding_system_arrangment in bonding_system_arrangments for _ in (0, 1)] 
                # duplicate the number of arrangments everytime there is a choice
                for i in range(0, len(bonding_system_arrangments), 2):
                    bonding_system_arrangments[i].append(copy.deepcopy(molecule.bonding_systems[idx]))
                    molecule.bonding_systems[idx].reverse_vo_order()
                    bonding_system_arrangments[i+1].append(copy.deepcopy(molecule.bonding_systems[idx]))
        else:
            for i in range(len(bonding_system_arrangments)):
                bonding_system_arrangments[i].append(molecule.bonding_systems[idx])

    # for every arrangment, get a candidate product
    for arrangment in bonding_system_arrangments:
        new_bonding_systems = []
        reaction_path = []

        reaction_path += [arrangment[0].vos[0]]
        for bonding_system in arrangment[1:]:
            reaction_path += [vo for vo in bonding_system.vos]
        if len(arrangment[0]) == 2:
            reaction_path += [arrangment[0].vos[1]]

        num_electrons = sum([vo.num_electrons for vo in reaction_path])

        if len(reaction_path) % 2 == 0:
            for i in range(0, len(reaction_path) - 1, 2):
                new_bonding_system = construct_new_bonding_system(reaction_path[i], reaction_path[i+1])
                new_bonding_systems.append(new_bonding_system)
        elif len(reaction_path) % 2 == 1:
            if len(arrangment[0]) == 1:
                for i in range(0,len(reaction_path) - 2, 2):
                    new_bonding_system = construct_new_bonding_system(reaction_path[i], reaction_path[i+1]) 
                    new_bonding_systems.append(new_bonding_system)
                new_bonding_system = BondingSystem(-1)
                reaction_path[-1].set_population(num_electrons - (len(reaction_path) - 1))
                new_bonding_system.add_vo(reaction_path[-1])
                new_bonding_systems.append(new_bonding_system)
            if len(arrangment[-1]) == 1:
                return None, None
                #for i in range(1, len(reaction_path), 2):
                #    new_bonding_system = construct_new_bonding_system(reaction_path[i], reaction_path[i+1])
                #    new_bonding_systems.append(new_bonding_system)
                #new_bonding_system = BondingSystem(-1)
                #reaction_path[0].set_population(num_electrons - (len(reaction_path) - 1))
                #new_bonding_system.add_vo(reaction_path[0])
                #new_bonding_systems.append(new_bonding_system)

        products.append(generate_smiles(molecule.orig_molecule, old_bonding_systems, new_bonding_systems))
                    
    return products


if __name__ == '__main__':
    #mol = Molecule('COCO.C#N')
    mol = Molecule('C.C#N')
    products = generate_products(mol, [8, 1])
    print(mol.smi, products)
    raise KeyError()
    starttime = timeit.default_timer()
    all_products = enumerate_reaction_possibilites(mol, 3)
    print("The time difference is :", timeit.default_timer() - starttime)
    print(list(set(all_products)))
