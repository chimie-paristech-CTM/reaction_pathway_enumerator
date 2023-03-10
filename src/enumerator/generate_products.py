import copy
from enumerator.Molecule import BondingSystem
from enumerator.generate_smiles import generate_smiles
import itertools
from tqdm import tqdm
import logging


def determine_reaction_type(bonding_system_init):
    """
    Determines the reaction type based on the nature of the initial bonding system.

    Args:
        bonding_system_init (BondingSystem): The initial bonding system of the reaction sequence

    Returns:
        str: the reaction type
    """
    if len(bonding_system_init) == 2:
        return "concerted"  # TODO: rename this type???
    elif len(bonding_system_init) == 1:
        if bonding_system_init.vos[0].num_electrons == 0:
            return "electrophilic"
        elif bonding_system_init.vos[0].num_electrons == 1:
            return "radical"
        elif bonding_system_init.vos[0].num_electrons == 2:
            return "nucleophilic"
    else:
        print("Not yet implemented!")


def set_polarization_bonding_systems(bonding_systems):
    """
    Sets an ordering of the bonding systems.

    Args:
        bonding_systems (list): list of bonding systems.
    """
    for bonding_system in bonding_systems:
        if len(bonding_system) > 1:
            bonding_system.set_polarity()


def construct_new_bonding_system(vo1, vo2, idx=-1):
    """Auxiliary function to construct a new bonding system from 2 existing vos.

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


def enumerate_reaction_possibilities(molecule: "Molecule", max_length: int):
    active_bonding_system_ids = []
    # heuristics to reduce number of possibilities -> only one lone pair and one X-H bond per atom
    # + only one interatomic bonding system if multiple
    atoms_with_lone_pairs_covered = []
    atoms_with_xh_covered = []
    bonds_covered = []

    # filter out non-active bonding systems
    for bonding_system in molecule.bonding_systems:
        logging.info(bonding_system)
        if bonding_system.is_lone_pair():
            if bonding_system.get_heavy_atoms()[0] in atoms_with_lone_pairs_covered:
                continue
            else:
                active_bonding_system_ids.append(bonding_system.idx)
                atoms_with_lone_pairs_covered.append(
                    bonding_system.get_heavy_atoms()[0]
                )
        elif bonding_system.is_xh_bond():
            if bonding_system.get_heavy_atoms()[0] in atoms_with_xh_covered:
                continue
            else:
                active_bonding_system_ids.append(bonding_system.idx)
                atoms_with_xh_covered.append(bonding_system.get_heavy_atoms()[0])
        elif bonding_system.get_heavy_atoms() in bonds_covered:
            continue
        else:
            active_bonding_system_ids.append(bonding_system.idx)
            bonds_covered.append(bonding_system.get_heavy_atoms())

    logging.info(len(active_bonding_system_ids))

    all_products = []
    for L in range(1, max_length + 1):
        print(
            f"Generating all products for combinations of up to {L} bonding systems..."
        )
        for idx_comb in tqdm(
            itertools.combinations(active_bonding_system_ids, L),
            total=len(list(itertools.combinations(active_bonding_system_ids, L))),
        ):
            for idx_perm in itertools.permutations(idx_comb):
                generated_products = generate_products(molecule, list(idx_perm))
                if generated_products != None:
                    all_products += generated_products

    return all_products


def split_single_bonding_system(old_bonding_system, num_electrons, population_first_vo):
    """Splits a single bonding system in two.

    Args:
        old_bonding_system (BondingSystem): the single bonding system to be split
        num_electrons (int): number of electrons present in the bonding system
        population_first_vo (int): number of electrons to be placed on the first vo

    Returns:
        list: a list of (two) bonding systems
    """
    new_bonding_systems = []

    new_bonding_system = BondingSystem(-1)
    old_bonding_system.vos[0].set_population(population_first_vo)
    new_bonding_system.add_vo(old_bonding_system.vos[0])
    new_bonding_systems.append(new_bonding_system)

    new_bonding_system = BondingSystem(-1)
    old_bonding_system.vos[1].set_population(num_electrons - population_first_vo)
    new_bonding_system.add_vo(old_bonding_system.vos[1])
    new_bonding_systems.append(new_bonding_system)

    return new_bonding_systems


def generate_products_for_single_bonding_system(molecule, old_bonding_systems):
    """Generates homolytic products for a single bonding system; if polarization is set, then heterolytic
    bond dissociation is also considered.

    Args:
        molecule (Molecule): a molecule object.
        old_bonding_systems (list): the list of old bonding systems (will have length 1).

    Returns:
        list: product list.
    """
    products = []
    # if there is only a single vo, there is nothing to do!
    if len(old_bonding_systems[0]) == 1:
        return None

    elif len(old_bonding_systems[0]) == 2:
        # homolytic splitting
        new_bonding_systems = split_single_bonding_system(
            old_bonding_systems[0], old_bonding_systems[0].num_electrons, 1
        )
        products.append(
            generate_smiles(
                molecule.orig_molecule, old_bonding_systems, new_bonding_systems
            )
        )
        # heterolytic splitting
        if old_bonding_systems[0].polarity_set:
            new_bonding_systems = split_single_bonding_system(
                old_bonding_systems[0], old_bonding_systems[0].num_electrons, 2
            )
            products.append(
                generate_smiles(
                    molecule.orig_molecule, old_bonding_systems, new_bonding_systems
                )
            )

    if None in products:
        return None
    else:
        return products


def get_bonding_system_arrangments(molecule, idx_list, reaction_type):
    """
    Gets bonding system arrangments.

    Args:
        molecule (Molecule): a molecule object.
        idx_list (list): list of bond system indices.
        reaction_type (str): the reaction type

    Returns:
        list: list of arrangments
    """
    bonding_system_arrangments = []

    bonding_system_arrangments.append(
        [copy.deepcopy(molecule.bonding_systems[idx_list[0]])]
    )
    if len(molecule.bonding_systems[idx_list[0]]) == 2:
        molecule.bonding_systems[idx_list[0]].reverse_vo_order()
        bonding_system_arrangments.append(
            [copy.deepcopy(molecule.bonding_systems[idx_list[0]])]
        )

    for idx in idx_list[1:]:
        if len(molecule.bonding_systems[idx]) == 2:
            if (
                reaction_type == "electrophilic"
                and molecule.bonding_systems[idx].polarity_set == True
            ):
                for i in range(len(bonding_system_arrangments)):
                    bonding_system_arrangments[i].append(
                        copy.deepcopy(molecule.bonding_systems[idx])
                    )
            elif (
                reaction_type == "nucleophilic"
                and molecule.bonding_systems[idx].polarity_set == True
            ):
                for i in range(len(bonding_system_arrangments)):
                    bonding_system = copy.deepcopy(molecule.bonding_systems[idx])
                    bonding_system.reverse_vo_order()
                    bonding_system_arrangments[i].append(bonding_system)
            else:
                bonding_system_arrangments = [
                    copy.deepcopy(bonding_system_arrangment)
                    for bonding_system_arrangment in bonding_system_arrangments
                    for _ in (0, 1)
                ]
                # duplicate the number of arrangments everytime there is a choice
                for i in range(0, len(bonding_system_arrangments), 2):
                    bonding_system_arrangments[i].append(
                        copy.deepcopy(molecule.bonding_systems[idx])
                    )
                    molecule.bonding_systems[idx].reverse_vo_order()
                    bonding_system_arrangments[i + 1].append(
                        copy.deepcopy(molecule.bonding_systems[idx])
                    )
        else:
            for i in range(len(bonding_system_arrangments)):
                bonding_system_arrangments[i].append(molecule.bonding_systems[idx])

    return bonding_system_arrangments


def generate_products(molecule: "Molecule", idx_list: list):
    """
    Generates products based on permutation of a subset of the bonding systems.

    Args:
        molecule (Molecule): a molecule object.
        idx_list (list): the list of bonding systems that need to be permutated.

    Returns:
        products (list): list of product SMILES.
    """
    products = []

    # set polarity of all the bonding systems
    set_polarization_bonding_systems(
        [molecule.bonding_systems[idx] for idx in idx_list]
    )

    # save a copy of the bonding systems being modified
    old_bonding_systems = [
        copy.deepcopy(molecule.bonding_systems[idx]) for idx in idx_list
    ]

    if len(old_bonding_systems) == 1:
        return generate_products_for_single_bonding_system(
            molecule, old_bonding_systems
        )

    # you don't want a lone pair or empty orbital halfway a sequence and you also don't want lone pairs in radical/concerted sequence
    if any([len(bonding_system) == 1 for bonding_system in old_bonding_systems[1:-1]]):
        return None

    # you don't want two vos on the same atom to be part of a single reactive event, except for lone pairs/empty orbitals at edges of reaction path
    atom_list = []
    for bonding_system in old_bonding_systems:
        atom_list += [vo.atom_idx for vo in bonding_system.vos]
    if len(old_bonding_systems[0]) == 2 and len(old_bonding_systems[-1]) == 2:
        if len(atom_list) != len(set(atom_list)):
            return None
    else:
        if len(atom_list[-1:1]) != len(set(atom_list[-1:1])):
            return None

    bonding_system_init = molecule.bonding_systems[idx_list[0]]
    reaction_type = determine_reaction_type(bonding_system_init)

    # get plausible arrangments; for polar reaction there is a preferential ordering; for radical/concerted ones you need to take all combinations into account
    bonding_system_arrangments = get_bonding_system_arrangments(
        molecule, idx_list, reaction_type
    )

    # for every arrangment, get a candidate product
    for arrangment in bonding_system_arrangments:
        new_bonding_systems = modify_bonding_systems(arrangment)
        if new_bonding_systems != None:
            products.append(
                generate_smiles(
                    molecule.orig_molecule, old_bonding_systems, new_bonding_systems
                )
            )
            if reaction_type == "concerted":
                # also take non-concerted pathway into account
                new_bonding_systems_mod = copy.deepcopy(new_bonding_systems)
                # diradical
                final_bonding_systems_diradical = split_single_bonding_system(
                    new_bonding_systems_mod[-1], 2, 1
                )
                new_bonding_systems_mod_diradical = (
                    new_bonding_systems_mod[:-1] + final_bonding_systems_diradical
                )
                smiles = generate_smiles(
                    molecule.orig_molecule,
                    old_bonding_systems,
                    new_bonding_systems_mod_diradical,
                )
                if smiles != None:
                    products.append(
                        generate_smiles(
                            molecule.orig_molecule,
                            old_bonding_systems,
                            new_bonding_systems_mod_diradical,
                        )
                    )
                # zwitter (this will not always make sense -> if a single molecule you can't distinguish between zwitter and diradical)
                new_bonding_systems_mod[-1].set_polarity()
                final_bonding_systems_zwitter = split_single_bonding_system(
                    new_bonding_systems_mod[-1], 2, 2
                )
                new_bonding_systems_mod_zwitter = (
                    new_bonding_systems_mod[:-1] + final_bonding_systems_zwitter
                )
                smiles = generate_smiles(
                    molecule.orig_molecule,
                    old_bonding_systems,
                    new_bonding_systems_mod_zwitter,
                )
                if smiles != None:
                    products.append(
                        generate_smiles(
                            molecule.orig_molecule,
                            old_bonding_systems,
                            new_bonding_systems_mod_zwitter,
                        )
                    )

    return products


def modify_bonding_systems(arrangment):
    """
    If sequence of vos is even in length, pair up the first vo with the second and so forth.
    If the sequence is odd in length, verify that it is the first bonding system which has only one vo,
    subsequently pair all the vos until the last one, which forms its own terminal bonding system.

    Args:
        arrangment (list): a list of bonding systems to be modified

    Returns:
        list: a list of modified bonding systems
    """
    reaction_path = construct_reaction_path(arrangment)
    num_electrons = sum([vo.num_electrons for vo in reaction_path])
    new_bonding_systems = []

    if len(reaction_path) % 2 == 0:
        # abort if num_electrons != len(reaction_path) -> would mean either 2 lone pairs, two empty orbitals or either of those and a radical
        if num_electrons != len(reaction_path):
            return None
        for i in range(0, len(reaction_path) - 1, 2):
            new_bonding_system = construct_new_bonding_system(
                copy.deepcopy(reaction_path[i]), copy.deepcopy(reaction_path[i + 1])
            )
            new_bonding_systems.append(new_bonding_system)

    elif len(reaction_path) % 2 == 1:
        if len(arrangment[-1]) == 1:
            return None
        if len(arrangment[0]) == 1:
            for i in range(0, len(reaction_path) - 2, 2):
                new_bonding_system = construct_new_bonding_system(
                    copy.deepcopy(reaction_path[i]), copy.deepcopy(reaction_path[i + 1])
                )
                new_bonding_systems.append(new_bonding_system)
            terminal_bonding_system = construct_terminal_bonding_system(
                num_electrons, reaction_path
            )
            new_bonding_systems.append(terminal_bonding_system)

    # abort if bond between two vos on same atom (can happen when first and last old bonding system contained vos on same atom, but then
    # the new bonding system will be of length 1)
    for bonding_system in [
        bonding_system
        for bonding_system in new_bonding_systems
        if len(bonding_system) > 1
    ]:
        if bonding_system.vos[0].atom_idx == bonding_system.vos[1].atom_idx:
            return None

    return new_bonding_systems


def construct_reaction_path(arrangment):
    """
    Construct reaction path by concatenating all the vos in the bonding systems arrangment.
    If the first bonding system consists of 2 vos, then put the second vo at the end of the sequence.

    Args:
        arrangement (list): a list of bonding systems

    Returns:
        list: a list of vos
    """
    reaction_path = []
    reaction_path += [arrangment[0].vos[0]]
    for bonding_system in arrangment[1:]:
        reaction_path += [vo for vo in bonding_system.vos]
    if len(arrangment[0]) == 2:
        reaction_path += [arrangment[0].vos[1]]

    return reaction_path


def construct_terminal_bonding_system(num_electrons, reaction_path):
    """Constructs a terminal bonding system for reaction paths of odd length.

    Args:
        num_electrons (int): number of electrons present in the reaction path
        reaction_path (list): sequence of vos

    Returns:
        BondingSystem: the terminal bonding system constructed
    """
    terminal_bonding_system = BondingSystem(-1)
    reaction_path[-1].set_population(num_electrons - (len(reaction_path) - 1))
    terminal_bonding_system.add_vo(reaction_path[-1])

    return terminal_bonding_system
