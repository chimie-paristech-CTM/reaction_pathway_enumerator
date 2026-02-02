import itertools

from rdkit import Chem
from rdkit.Chem import EnumerateStereoisomers
import re
from itertools import permutations, product
from tqdm import tqdm
from typing import Dict, List

from enumerator.utils import fix_radical_counts_at_endpoints_path, increase_bond_order, decrease_bond_order, fix_bonding_hypervalent_compound
from enumerator.utils import clear_numbering, get_neighbors_dict, ordering_smiles, generate_stereoisomers
from enumerator.orbital_systems import DelocalizedOrbitalSystem
from enumerator.localized_configuration import Atom, LocalizedConfiguration
from enumerator.localized_configuration_NBO import AtomNBO, LocalizedConfigurationNBO
from enumerator.utils_nbo import get_nbo, read_from_chk, extract_electrons_based_bond_matrix

from copy import deepcopy


ps = Chem.SmilesParserParams()
ps.removeHs = False

metal_symbols = ["Si", "Fe", "Cu", "Au", "Ag",  "Zn", "Ni",  "Sn",  "Pb",  "Pt",  "Hg",  "Ti", "Co",
                 "Cr",  "Mg",  "Mn",  "W",   "Bi",  "Sb",  "Cd",  "V",   "U",   "Pd",  "Rh",  "Ru"]

upper_3rd_row_symbols = ["P", "S", "Cl", "As", "Se",  "Br", "Sb",  "Te",  "I"]

class Reaction:

    def __init__(self, orig_mol, vo_list, existing_interactions, strong_secondary_interactions, organometallic, conventional_path=True):
        self.orig_mol = orig_mol
        self.orig_path = vo_list
        self.modified_path = deepcopy(vo_list)
        self.existing_interactions = existing_interactions
        self.strong_secondary_interactions = strong_secondary_interactions
        self.reduction_process_metal = False
        self.organometallic = organometallic

        self.ignore_first_vo = False
        self.ignore_final_vo = False

        if not conventional_path:
            self.connect_end_vos = False
            self.invert_vo_populations_to_complete_modified_path()
        else:
            self.connect_end_vos = True

        self.adjust_vo_populations_along_modified_path()

    def invert_vo_populations_to_complete_modified_path(self):
        """
        This method inverses VO populations if VOs not at the endpoints of a path, but still in the first or last orbital system, 
        contain a number of electrons differing from 1 (reflecting the inherent delocalization present in these systems). 
        In case of 2c3e/2c1e, the first/last vo is set inactive, since the actual reorganization of the path will need to 
        start from the second (to last) vo.
        """
        # 3c situations
        if len(self.modified_path) > 3:
            if self.modified_path[0].num_electrons == 1 and self.modified_path[2].num_electrons != 1:
                self.invert_vo_populations(self.modified_path[0], self.modified_path[2])
            if self.modified_path[-1].num_electrons == 1 and self.modified_path[-3].num_electrons != 1:
                self.invert_vo_populations(self.modified_path[-1], self.modified_path[-3])
        # 2c3e or 2c1e situations
        if len(self.modified_path) > 2:
            if self.modified_path[0].num_electrons == 1 and self.modified_path[1].num_electrons != 1:
                self.invert_vo_populations(self.modified_path[0], self.modified_path[1])
                self.ignore_first_vo == True
            if self.modified_path[-1].num_electrons == 1 and self.modified_path[-2].num_electrons != 1:
                self.invert_vo_populations(self.modified_path[-1], self.modified_path[-2])
                self.ignore_final_vo == True

    def adjust_vo_populations_along_modified_path(self):
        """
        This method redistributes the electrons along the modified path. No distinction between 1/2/3c orbital systems is needed here since 
        bonding anomalies should already have been resolved through application of the previous method.
        """
        start_idx, end_idx = self.get_start_end_idx()

        if self.orig_path[start_idx].num_electrons == 1 and self.orig_path[end_idx].num_electrons == 1:
            pass # no CT, simply a covalent rearrangement
        elif self.orig_path[start_idx].num_electrons + self.orig_path[end_idx].num_electrons == 2:
            # this means that either both endpoints carry 1 electron, or 1 carries 2 and the other none 
            # -> optimally, you maximize the number of interactions in the modified path
            self.modified_path[start_idx].num_electrons = 1
            self.modified_path[end_idx].num_electrons = 1
        elif self.modified_path[start_idx].num_electrons == 2 and self.modified_path[
            start_idx + 1].num_electrons == 0 and self.modified_path[start_idx + 1].atom_type in metal_symbols:
            # possible reduction product
            self.modified_path[start_idx].num_electrons = 1
            self.modified_path[start_idx + 1].num_electrons = 1
            self.reduction_process_metal = True
        elif self.modified_path[start_idx].num_electrons != 1 and self.modified_path[end_idx].num_electrons == 1:
            # pass on excess/shortage of electrons from one side of the path to the other
            self.modified_path[end_idx].num_electrons = self.orig_path[start_idx].num_electrons
            self.modified_path[start_idx].num_electrons = 1
        elif self.modified_path[start_idx].num_electrons == 1 and self.modified_path[end_idx].num_electrons != 1:
            # reverse from above
            self.modified_path[start_idx].num_electrons = self.orig_path[end_idx].num_electrons
            self.modified_path[end_idx].num_electrons = 1

    # TODO: for 3 center systems, you will still need to add a bond at the edges because you are breaking up the bonding system completely.
    # For now you can ignore this however since this is inherently problematic with SMILES.
    def generate_smiles(self, allow_zwitterions=True):
        """
        Generate an output SMILES string.

        Returns:
            str: the output SMILES
        """
        editable_mol = Chem.RWMol(self.orig_mol)  # editable version of the molecule
        start_idx, end_idx = self.get_start_end_idx()

        if self.reduction_process_metal:
            pass
        else:
            # modify atom properties
            for vo, modified_vo in zip(self.orig_path, self.modified_path):
                if vo.num_electrons != modified_vo.num_electrons:
                    init_charge = editable_mol.GetAtomWithIdx(vo.atom_idx - 1).GetFormalCharge()
                    new_charge = init_charge - (modified_vo.num_electrons - vo.num_electrons)
                    editable_mol.GetAtomWithIdx(vo.atom_idx - 1).SetFormalCharge(new_charge)

        # modify bonding situation
        for i, vo in enumerate(self.orig_path[start_idx:end_idx]):
            if self.orig_path[i+1].atom_idx == vo.atom_idx:
                continue
            if self.orig_path[i+1] in self.existing_interactions[vo.identifier] or \
                (self.orig_path[i+1] in self.strong_secondary_interactions[vo.identifier] and self.organometallic):
                editable_mol = decrease_bond_order(editable_mol, vo, self.orig_path[i+1])
            else:
                editable_mol = increase_bond_order(editable_mol, vo, self.orig_path[i+1])

        # take care of the terminal vos of the path -> connect or leave radical site
        if self.connect_end_vos and self.orig_path[start_idx].atom_idx == self.orig_path[end_idx].atom_idx:
            pass
        elif self.connect_end_vos and self.orig_path[start_idx].is_paired() and self.orig_path[end_idx].is_paired():
            editable_mol = increase_bond_order(editable_mol, self.orig_path[0], self.orig_path[-1]) # finish covalent path
        elif (not self.orig_path[start_idx].is_paired() and self.orig_path[end_idx].is_paired()) and \
                self.orig_path[start_idx].num_electrons == 1 and self.modified_path[end_idx].num_electrons == 1: # fix radical sites
                editable_mol = fix_radical_counts_at_endpoints_path(editable_mol, self.orig_path[start_idx], self.orig_path[end_idx])
        elif (not self.orig_path[end_idx].is_paired() and self.orig_path[start_idx].is_paired()) and \
            self.orig_path[end_idx].num_electrons == 1 and self.modified_path[start_idx].num_electrons == 1:
                editable_mol = fix_radical_counts_at_endpoints_path(editable_mol, self.orig_path[end_idx], self.orig_path[start_idx])

        # if 1 atom carries both a lone pair and an empty orbital, sanitization will add Hs -> you don't want that!
        try:
            if len(editable_mol.GetAtoms()) != len(Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(editable_mol))).GetAtoms()):
                return None
        except Exception as e:
            print(e)
            print(self.orig_path, self.modified_path)
            print(Chem.MolToSmiles(editable_mol))

        if self.possible_generation_hypervalent_compound():
            editable_mol = fix_bonding_hypervalent_compound(editable_mol)

        final_smiles = Chem.MolToSmiles(editable_mol)

        # final filter in case you don't want zwitterions
        if not allow_zwitterions:
            init_smiles = Chem.MolToSmiles(self.orig_mol)
            if final_smiles.count('+') > init_smiles.count('+'):
                return None

        return final_smiles

    def invert_vo_populations(self, vo1, vo2):
        """
        Switch the electronic populations of two VOs.
        """
        tmp = vo1.num_electrons
        vo1.set_population(vo2.num_electrons)
        vo2.set_population(tmp)

    def get_start_end_idx(self):
        """
        Determine the start and end idx based on whether the first and final VO need to be ignored.
        """
        if self.ignore_first_vo:
            start_idx = 1
        else:
            start_idx = 0
        if self.ignore_final_vo:
            end_idx = -2
        else:
            end_idx = -1

        return start_idx, end_idx

    def possible_generation_hypervalent_compound(self):

        for vo in self.orig_path:
            if vo.atom_type in upper_3rd_row_symbols:
                return True

        return False


class OrbitalGraph:
    """ 
    A class corresponding to an abstract graph, where nodes correspond to VOs, 
    and edges correspond to existing and potential intrafragment interactions. 
    """
    def __init__(self, localized_configuration, numbered_smiles, orig_mol, nbo, organometallic):
        self.localized_configuration = localized_configuration
        self.nbo = nbo
        self.organometallic = organometallic

        if self.nbo and not self.organometallic:
            self.orig_mol = self.localized_configuration.nbo_mol
            self.numbered_smiles = Chem.MolToSmiles(self.orig_mol)
        else:
            self.numbered_smiles = numbered_smiles
            self.orig_mol = orig_mol

        self.atom_to_fragment_dict = self.get_atom_to_fragment_dict(self.numbered_smiles)
        self.dipole_idxs = self.check_dipole()

        self.existing_interactions = {}
        self.potential_intrafragment_interactions = {}
        self.secondary_interactions = {}
        self.strong_secondary_interactions = {}

        self.add_vos_to_graph()
        self.add_existing_interactions()
        self.add_potential_interactions()
        self.add_strong_secondary_interactions()

        self.delocalized_orbital_systems = self.construct_delocalized_systems()
        self.vo_to_deloc_orbital_systems_dict = self.get_vo_to_deloc_orbital_systems_dict()

    def get_atom_to_fragment_dict(self, numbered_smiles):
        """
        Get a dictionary mapping atom indices to fragment indices based on numbered SMILES.

        Args:
            numbered_smiles (str): A string representing the numbered SMILES of a molecule.

        Returns:
            dict: A dictionary where keys are atom indices and values are the indices of the 
            fragments to which the atoms belong.
        """
        atom_to_fragment_dict = {}
        fragment_smiles_list = numbered_smiles.split('.')
        for i, fragment_smiles in enumerate(fragment_smiles_list):
            atom_numbers = re.findall(r':(\d+)\]', fragment_smiles)
            for atom_number in atom_numbers:
                atom_to_fragment_dict[int(atom_number)] = i

        return atom_to_fragment_dict

    def add_vos_to_graph(self):
        """
        Add valence orbitals (VOs) to the graph.
        """
        for vo in self.localized_configuration.get_vos():
            if vo not in self.existing_interactions:
                self.existing_interactions[vo.identifier] = set()
            if vo not in self.potential_intrafragment_interactions:
                self.potential_intrafragment_interactions[vo.identifier] = set()
            if vo not in self.strong_secondary_interactions:
                self.strong_secondary_interactions[vo.identifier] = set()
            if vo not in self.secondary_interactions:
                self.secondary_interactions[vo.identifier] = set()

    def add_existing_interactions(self):
        """
        Add existing interactions between valence orbitals within the same orbital system.
        """
        for orbital_system in self.localized_configuration.active_orbital_systems_list:
            if len(orbital_system.vos) > 1:
                for i, vo in enumerate(orbital_system.vos[:-1]):
                    self.existing_interactions[vo.identifier].add(orbital_system.vos[i+1])
                    self.existing_interactions[orbital_system.vos[i+1].identifier].add(vo)

    def add_strong_secondary_interactions(self):
        """
        Add strong secondary interactions between valence orbitals within the same orbital system.
        """
        if self.nbo:
            for orbital_system in self.localized_configuration.strong_sec_int_orbital_systems_list:
                if len(orbital_system.vos) > 1:
                    for i, vo in enumerate(orbital_system.vos[:-1]):
                        self.strong_secondary_interactions[vo.identifier].add(orbital_system.vos[i+1])
                        self.strong_secondary_interactions[orbital_system.vos[i+1].identifier].add(vo)

    def add_potential_interactions(self):
        """
        Add potential interactions between valence orbitals within and between orbital systems.

        Notes:
            This method iterates through valence orbitals (VOs) in the localized configuration of the molecule
            and determines potential interactions between them. It considers various conditions to exclude certain
            interactions, such as existing interactions, orbitals on the same atom, and orbitals already involved
            in existing interactions with another orbital on the same atom. It distinguishes between intrafragment
            and interfragment interactions based on the owning fragments of the VOs and adds them to the appropriate
            sets -- thus completing the orbital graph with edge information.
        """
        neighbors_dict = get_neighbors_dict(self.orig_mol)
        for vo1 in self.localized_configuration.get_vos():
            owning_fragment = self.atom_to_fragment_dict[vo1.atom_idx]
            for vo2 in self.localized_configuration.get_vos():
                # don't include existing interactions
                if vo2.identifier in self.existing_interactions[vo1.identifier]:
                    continue
                # don't include orbitals on the same atom if is not a metal
                elif vo2.atom_idx == vo1.atom_idx:
                    if vo2.atom_type in metal_symbols:
                        self.add_potential_intrafragment_interaction(vo1, vo2)
                    else:
                        continue
                # don't include orbital pairs when one of them is already involved an existing interaction with another orbital on the same atom => results in no productive repairing
                elif vo2.atom_idx in [vo.atom_idx for vo in self.existing_interactions[vo1.identifier]]:
                    continue
                # if intrafragment interactions, we only consider atoms that are adjacent
                if owning_fragment == self.atom_to_fragment_dict[vo2.atom_idx]:
                    if vo2.atom_idx in neighbors_dict[vo1.atom_idx]: # if the idx in the candidate partner is in neighbor set => adjacent orbital systems
                        self.add_potential_intrafragment_interaction(vo1, vo2)
                    else:
                        continue

    def add_potential_intrafragment_interaction(self, source, destination):
        """
        Add a potential intrafragment interaction between valence orbitals.

        Args:
            source: The source valence orbital.
            destination: The destination valence orbital.
        """
        if source.identifier in self.potential_intrafragment_interactions and destination.identifier in self.potential_intrafragment_interactions \
            and destination not in self.potential_intrafragment_interactions[source.identifier]:
            self.potential_intrafragment_interactions[source.identifier].add(destination)

    # TODO: complete this once you have integrated NBO read-in/-out
    def add_secondary_interaction(self, source, destination):
        """
        Add a secondary interaction between valence orbitals.

        Args:
            source: The source valence orbital.
            destination: The destination valence orbital.
        """
        pass

    # TODO: complete this once you have integrated NBO read-in/-out
    def get_secondary_interactions(self, vo):
        return list(self.secondary_interactions.get(vo.identifier, {}))

    def get_interacting_orbitals(self, vo):
        return list(self.existing_interactions.get(vo.identifier, {}))

    def get_empty_orbitals(self, vo):

        atom_idx = vo.atom_idx
        empty_orbital = None
        for partner_vo in self.localized_configuration.get_vos():
            if partner_vo.num_electrons == 0 and partner_vo.atom_idx == atom_idx:
                    empty_orbital = partner_vo
        return empty_orbital

    def get_intrafragment_neighbors(self, vo):
        return self.potential_intrafragment_interactions.get(vo.identifier, [])

    def construct_delocalized_systems(self):
        """
        Construct delocalized orbital systems from individual VOs.

        Returns:
            list: A list of delocalized orbital systems.

        Notes:
            This method constructs delocalized orbital systems from individual VOs.
            It iterates through each valence orbital (VO) in the localized configuration and 
            constructs a delocalized orbital system starting from that VO. It adds interacting
            VOs and VOs connected through secondary interactions recursively to the delocalized orbital system,
            ensuring that no VO is included in multiple delocalized systems. The resulting list
            contains all constructed delocalized orbital systems.
        """
        vo_list = deepcopy(self.localized_configuration.get_vos())
        delocalized_orbital_systems = []
        # iterate through VOs 
        for vo in vo_list:
            delocalized_orbital_system = DelocalizedOrbitalSystem(len(delocalized_orbital_systems), vo)
            partners = self.get_interacting_orbitals(vo) + self.get_secondary_interactions(vo)
            # recursively add partners
            while len(partners) != 0:
                new_partners = []
                for partner in partners:
                    # remove VOs already included in the delocalized system from the VO list to avoid double-counting
                    if partner in vo_list:
                        vo_list.remove(partner)
                    delocalized_orbital_system.add_vo(partner)
                    partners_of_partner = self.get_interacting_orbitals(partner) + self.get_secondary_interactions(partner)
                    new_partners += [vo for vo in partners_of_partner if vo not in delocalized_orbital_system.get_vos()]
                partners = new_partners
            delocalized_orbital_systems.append(delocalized_orbital_system)

        return delocalized_orbital_systems

    def get_vo_to_deloc_orbital_systems_dict(self):
        """
        Get a dictionary mapping valence orbitals to their corresponding delocalized orbital systems.

        Returns:
            dict: A dictionary where keys are valence orbitals (VOs) and values are the indices of 
            the delocalized orbital systems to which the VOs belong.
        """
        vo_to_deloc_orbital_systems_dict = {}
        for delocalized_orbital_system in self.delocalized_orbital_systems:
            vo_list = delocalized_orbital_system.get_vos()
            for vo in vo_list:
                vo_to_deloc_orbital_systems_dict[vo.identifier] = delocalized_orbital_system.idx

        return vo_to_deloc_orbital_systems_dict

    def get_number_of_delocalized_orbital_systems(self, path):
        """
        Get the number of unique delocalized orbital systems present in a given path.

        Args:
            path (list): A list of valence orbitals forming a path.
        """
        return len(set([self.vo_to_deloc_orbital_systems_dict[vo.identifier] for vo in path]))

    def get_intrafragment_paths(self, max_length=2):
        """
        Get all possible intrafragment paths within the molecule.

        Args:
            max_length (int, optional): The maximum length of paths to consider. Defaults to 2.

        Returns:
            list: A list of lists, where each inner list represents a set of valence orbitals forming 
            an intrafragment path within a fragment.

        Notes:
            This method computes all possible intrafragment paths within the molecule. It starts by 
            initializing paths for each valence orbital (VO) in the localized configuration. Then, it 
            iteratively extends these paths by adding neighboring VOs until the maximum path length 
            is reached. The method considers various conditions to avoid premature endings of paths, 
            such as excluding orbitals already present in the path and ensuring no crossings occur 
            within the path. The resulting list contains all computed intrafragment paths.
        """
        all_intrafragment_paths = [[] for _ in self.numbered_smiles.split('.')]

        # initialize
        for vo in self.localized_configuration.get_vos():
            new_path = []
            partner_vos = self.get_interacting_orbitals(vo)
            if len(partner_vos) == 0:
                new_path.append(vo)
                if vo.num_electrons == 2 and vo.atom_type in metal_symbols: # for oxidative addition we need a lone pair and empty orbital
                    alternative_path = []
                    alternative_partner_vos = self.get_empty_orbitals(vo)
                    if alternative_partner_vos:
                        alternative_path.append(vo)
                        alternative_path.append(alternative_partner_vos)
                        all_intrafragment_paths[self.atom_to_fragment_dict[vo.atom_idx]].append(alternative_path)
            elif len(partner_vos) == 1:
                if vo.num_electrons == 1 and partner_vos[0].num_electrons == 1:
                    alternative_path = []
                    alternative_path.append(partner_vos[0])
                    alternative_path.append(vo)
                    all_intrafragment_paths[self.atom_to_fragment_dict[vo.atom_idx]].append(alternative_path)
                    new_path.append(vo)
                    new_path.append(partner_vos[0])
                elif vo.num_electrons == 1 and partner_vos[0].num_electrons != 1: # 2c1e/2c3e
                    new_path.append(partner_vos[0]) # put the other vo first so that you can leave this out during re-pairing (it will be an end-point)
                    new_path.append(vo)
                elif vo.num_electrons == 2 and partner_vos[0].num_electrons == 0: # empty orbital + lone pair, bond description in metals
                    new_path.append(vo)
                    new_path.append(partner_vos[0])

                if len(self.get_interacting_orbitals(partner_vos[0])) == 2: # 3c systems
                    remaining_vo = [vo3 for vo3 in self.get_interacting_orbitals(partner_vos[0]) if vo3 != vo]
                    new_path.append(remaining_vo[0])
            else:
                if vo.atom_type in metal_symbols and vo.num_electrons == 1:
                    for partner_vo in partner_vos:
                        if partner_vo.num_electrons == 1:
                            new_path.append(vo)
                            new_path.append(partner_vo)
                            break
                else:
                    continue # in a 3c system, starting from either of the two extremes is enough to capture all possibilities;
                         # in 2c3e or 2c1e you do not need to start from the empty/doubly filled VO (this is already treated later on)
            all_intrafragment_paths[self.atom_to_fragment_dict[vo.atom_idx]].append(new_path)

        # iterate through the intrafragment paths
        for fragment_paths in all_intrafragment_paths:
            # initialize -- we consider plausible extensions in every fragment (make a shallow copy so that take a static snapshot of the fragment paths at first)
            paths_to_extend = fragment_paths.copy()

            while len(paths_to_extend) > 0:
                new_paths_to_extend = []
                for path in paths_to_extend:
                    if len(path) >= max_length:
                        continue
                    for neighbor in self.get_intrafragment_neighbors(path[-1]):
                        partners_of_neighbor = self.get_interacting_orbitals(neighbor)
                        if len(partners_of_neighbor) == 0 and neighbor.num_electrons == 2:
                            new_path = path.copy()
                            new_path.append(neighbor)
                            fragment_paths.append(new_path)
                            if self.get_number_of_delocalized_orbital_systems(new_path) < max_length:
                                new_paths_to_extend.append(new_path)
                            else:
                                continue
                        if len(partners_of_neighbor) == 0 or len(partners_of_neighbor) == 2:
                            continue # you don't want to prematurely end paths
                        elif len(partners_of_neighbor) == 1:
                            if neighbor.num_electrons != 1 or partners_of_neighbor[0].num_electrons != 1:
                                continue
                            # 2c1e/2c3e bonds should not be in the middle of the path either since this would also end the path
                            elif neighbor.num_electrons + partners_of_neighbor[0].num_electrons != 2:
                                continue
                            elif neighbor not in path and partners_of_neighbor[0].atom_idx not in [vo.atom_idx for vo in path]:
                                new_path = path.copy() # only a shallow copy is needed -> don't duplicate the VOs
                                new_path.append(neighbor)
                                new_path.append(partners_of_neighbor[0])
                                fragment_paths.append(new_path)
                                if self.get_number_of_delocalized_orbital_systems(new_path) < max_length:
                                    new_paths_to_extend.append(new_path)
                                else:
                                    continue
                            else:
                                continue # no crossings within path
                paths_to_extend = new_paths_to_extend

        if self.nbo:
            # adding the delocalized vos that were obtained with NBO and secondary interaction analysis
            for secondary_vos in self.localized_configuration.secondary_interaction_vos_systems:
                vo = secondary_vos[0]
                all_intrafragment_paths[self.atom_to_fragment_dict[vo.atom_idx]].append(secondary_vos)
                if self.dipole_idxs and len(secondary_vos) == 3:
                    dipole_path = []
                    idx_plus, idx_minus = self.dipole_idxs
                    for i in range(3):
                        for vo in secondary_vos:
                            if vo.atom_idx == idx_minus and i == 0:
                                dipole_path.append(vo)
                            if vo.atom_idx == idx_plus and i == 2:
                                dipole_path.append(vo)
                            if i == 1 and vo.atom_idx != idx_plus and vo.atom_idx != idx_minus:
                                dipole_path.append(vo)
                    all_intrafragment_paths[self.atom_to_fragment_dict[vo.atom_idx]].append(dipole_path)

        return all_intrafragment_paths

    def get_interfragment_paths(self, all_intrafragment_paths):
        """    
        Get all possible interfragment paths by permuting fragment order and combining intrafragment paths.

        Args:
            all_intrafragment_paths (list): A list of lists, where each inner list represents a set of 
            valence orbitals forming an intrafragment path within a fragment.

        Returns:
            list: A list of lists, where each inner list represents a set of valence orbitals forming 
            an interfragment path across multiple fragments.
        """
        all_interfragment_paths = []
        # first permutate the fragment order
        potential_fragment_arrangements = list(permutations(list(range(len(all_intrafragment_paths))), len(all_intrafragment_paths)))
        for arrangement in potential_fragment_arrangements:
            intrafragment_paths_reordered = [all_intrafragment_paths[i].copy() for i in arrangement]
            # now make all the possible combinations within the selected fragment order
            all_combinations_list = list(product(*intrafragment_paths_reordered))
            for combination in all_combinations_list:

                lp_empty_vo = False
                new_interfragment_path = []

                if len(combination[0]) > 1:
                    if combination[0][0].num_electrons == 2 and combination[0][1].num_electrons == 0:
                        lp_empty_vo = True

                if lp_empty_vo:
                    new_interfragment_path.append(combination[0][0])
                else:
                    new_interfragment_path += combination[0]
                for fragment_path in combination[1:-1]:
                    # only add the fragment if it enables a continuous path
                    if (fragment_path[0].num_electrons == 1 and fragment_path[0].is_paired()) \
                        and (fragment_path[-1].num_electrons == 1 and fragment_path[-1].is_paired()):
                        new_interfragment_path += fragment_path
                    else:
                        continue

                # inverse the final fragment_path before attachment (so that lone pairs etc. end up at the very end)
                terminal_path = combination[-1].copy()

                # because you are inserting a double bond in this way ... vo(Me) + vo(NoMe) + vo(Me)
                if lp_empty_vo and len(terminal_path) != 2:
                    continue

                if self.dipole_idxs:
                    if new_interfragment_path[0].atom_idx == self.dipole_idxs[1] and new_interfragment_path[-1].atom_idx == self.dipole_idxs[0]:
                        new_interfragment_path[1:1] = terminal_path[::-1]
                    else:
                        new_interfragment_path += terminal_path[::-1]
                else:
                    new_interfragment_path += terminal_path[::-1]

                if lp_empty_vo:
                    new_interfragment_path.append(combination[0][1])
                # remove invalid paths -- you should only keep continuous paths, i.e., when the bridging vos had an interacting orbital to start with
                # only start looking from the second vo, because you want to retain paths in which there is a 3c orbital system at the end
                if any([len(self.get_interacting_orbitals(vo)) != 1 for vo in new_interfragment_path[2:-2]]):
                    if len(new_interfragment_path[2:-2]) != 1:
                        continue
                    else:
                        if new_interfragment_path[2:-2][0].num_electrons != 2:
                            continue

                # 2 electron deficient or two electron exessive endpoints -> abort
                if new_interfragment_path[0].num_electrons + new_interfragment_path[-1].num_electrons == 0 or \
                   new_interfragment_path[0].num_electrons + new_interfragment_path[-1].num_electrons == 4:
                    if len(new_interfragment_path) != 2:
                        continue
                    # addition to metals ... like addition of C#O to Co
                    if not (new_interfragment_path[0].atom_type in metal_symbols or
                            new_interfragment_path[1].atom_type in metal_symbols):
                        continue
                all_interfragment_paths.append(new_interfragment_path)

        return all_interfragment_paths

    def get_intramolecular_paths(self, max_length=2):
        """
        If only a single fragment in the reacting system, generate intramolecular reactions by determining extra long fragments, 
        potentially adding terminal fragments (if path started with a VO that cannot be reconnected).

        Args:
            max_length (int, optional): The maximum length of paths to consider. Defaults to 2.

        Returns:
            list: A list of lists, where each inner list represents a set of valence orbitals forming 
            a full intramolecular path.
        """
        intrafragment_paths = self.get_intrafragment_paths(max_length=max_length)[0]
        terminal_fragment_paths = [path.copy()[::-1] for path in intrafragment_paths if path[0].num_electrons != 1]
        intramolecular_paths = intrafragment_paths.copy()

        # For now, we only consider combinations of up to 2 intrafragment paths
        for _ in range(1):
            extra_paths = []
            for current_path in intramolecular_paths:
                for extension in intrafragment_paths:
                    if len(extension) % 2 != 0:
                        # check about possible recombination of diradical
                        if len(current_path) == 1 and len(extension) == 1:
                            if not current_path[0].num_electrons == extension[0].num_electrons == 1:
                                continue

                    extended_path = current_path + extension

                    atom_indices = [vo.atom_idx for vo in extended_path]
                    atom_types = [vo.atom_type for vo in extended_path]

                    # only keep non-intercrossing paths
                    if len(set(atom_indices)) == len(atom_indices):
                        extra_paths.append(extended_path)
                    else:
                        duplicates = [atom_idx for atom_idx in atom_indices if atom_indices.count(atom_idx) > 1]

                        if any(extended_path[i].atom_type in metal_symbols for i, atom_idx in enumerate(atom_indices)
                               if atom_idx in duplicates) and (len(atom_indices)-len(set(atom_indices))) == 1:
                            extra_paths.append(extended_path)
                        else:
                            continue
            intramolecular_paths += extra_paths

        extra_paths = []
        for path in intramolecular_paths:
            if path[0].num_electrons != 1:
                for terminal_fragment_path in terminal_fragment_paths:
                    if path[0].num_electrons + terminal_fragment_path[-1].num_electrons == 2 and \
                        (len(set([vo.atom_idx for vo in extended_path])) == len([vo.atom_idx for vo in extended_path])):
                        extra_paths.append(path + terminal_fragment_path)
        intramolecular_paths += extra_paths
        intramolecular_paths_filtered = [path for path in intramolecular_paths if len(path) > 1]
        return intramolecular_paths_filtered

    def generate_products(self, all_paths, allow_zwitterions):
        """
        Generate unique product SMILES representations from a list of reaction paths.

        Args:
            all_paths (list): A list of reaction paths.
            allow_zwitterions (bool): A bool indicating whether zwitterionic SMILES are considered

        Returns:
            list: A list containing unique product SMILES representations.

        This method iterates through each reaction path provided, creates a Reaction object
        based on the path, and generates the SMILES representation of the product. Only unique
        products are stored in the result list, ensuring that duplicate products are excluded. 
        """

        products = []
        unique_products = set()
        unique_products_enum = set()
        products_with_paths: Dict[str, List[str]] = dict()
        map_between_smiles = dict()
        for path in tqdm(all_paths):

            if (self.localized_configuration.vo_to_orbital_system_dict[path[0].identifier].is_conventional() and
                    self.localized_configuration.vo_to_orbital_system_dict[path[-1].identifier].is_conventional()):
                reaction = Reaction(self.orig_mol, path, self.existing_interactions, self.strong_secondary_interactions, self.organometallic, conventional_path=True)
            else:
                reaction = Reaction(self.orig_mol, path, self.existing_interactions, self.strong_secondary_interactions, self.organometallic, conventional_path=False)

            smiles = reaction.generate_smiles(allow_zwitterions=allow_zwitterions)

            if smiles != None:
                smiles_without_numbering = clear_numbering(smiles)
                if smiles_without_numbering:
                    if smiles_without_numbering not in unique_products:
                        unique_products.add(smiles_without_numbering)
                        products.append(smiles)
                    if smiles not in unique_products_enum:
                        unique_products_enum.add(smiles)
                        products_with_paths[smiles_without_numbering] = products_with_paths.get(smiles_without_numbering,
                                                                                                []) + [smiles]
                        map_between_smiles[smiles] = map_between_smiles.get(smiles, '') + smiles_without_numbering

        return products, products_with_paths, map_between_smiles

    def check_dipole(self):

        if self.numbered_smiles.count('+') == self.numbered_smiles.count('-') == 1:
            elements = re.findall(r'\[[^\]]+\]', self.numbered_smiles)
            for elem in elements:
                if '-' in elem:
                    idx_minus = int(elem[elem.index(':')+1:-1])
                if '+' in elem:
                    idx_plus = int(elem[elem.index(':')+1:-1])
            return (idx_plus, idx_minus)
        else:
            return None

    def __str__(self) -> str:
        return f'existing: {self.existing_interactions}; secondary: {self.secondary_interactions}: intra: {self.potential_intrafragment_interactions}'


class ReactingSystem:
    """A class corresponding to reacting systems (can consist of multiple molecules)."""

    def __init__(self, smiles: str, nbo: bool = False, nbo_dir: str = None,
                 threshold_strong_secondary_interaction: float = 85.0, nproc: int = 4,
                 threshold_secondary_interaction: float = 12.0, mult: int = -1):
        self.orig_mol, self.numbered_smiles, self.orig_atom_idxs = self.parse_smiles(smiles)
        self.organometallic = self.check_if_reaction_organometallic()
        self.radicalic = self.check_if_reaction_radicalic()
        self.nbo = nbo
        self.threshold_ssi = threshold_strong_secondary_interaction
        self.threshold_si = threshold_secondary_interaction
        print(self.numbered_smiles)

        self.num_atoms = self.orig_mol.GetNumAtoms()

        if self.nbo:
            if nbo_dir:
                self.nbo_lines = read_from_chk(self.numbered_smiles, nbo_dir)
            else:
                self.nbo_lines = get_nbo(self.numbered_smiles, mult, nproc)
            self.atoms = self.set_up_atoms_NBO()
            self.localized_configuration = self.set_up_localized_configuration_nbo()
            self.orbital_graph = self.set_up_orbital_graph()
        else:
            self.atoms = self.set_up_atoms()
            self.localized_configuration = self.set_up_localized_configuration()
            self.orbital_graph = self.set_up_orbital_graph()

    def parse_smiles(self, smiles):
        """Get mol object with hydrogens, fully numbered and kekulized """
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)  # always add H's to make bonding correct
        Chem.Kekulize(mol) # change to kekulized smiles to remove aromatic bonds
        [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in mol.GetAtoms()]
        numbered_smiles = Chem.MolToSmiles(mol)
        orig_atom_idxs = list(map(int, mol.GetProp("_smilesAtomOutputOrder")[1:-1].split(",")))

        return mol, numbered_smiles, orig_atom_idxs

    def check_if_reaction_organometallic(self):
        """
        Check if the reacting system is organometallic.

        Returns:
        - bool: True if organometallic, False otherwise.

        Notes:
        - The function examines the atomic symbols of atoms in the reactant molecule.
        - It checks if any of the symbols match those in the 'metal_symbols'.
        """

        for atom in self.orig_mol.GetAtoms():
            if atom.GetSymbol() in metal_symbols:
                return True
        return False

    def check_if_reaction_radicalic(self):
        """
        Check if the reacting system is radicalic.

        Returns:
        - bool: True if radicalic, False otherwise.
        """

        for atom in self.orig_mol.GetAtoms():
            if atom.GetNumRadicalElectrons() and not self.organometallic:
                return True
        return False

    def set_up_atoms(self):
        """Process rdkit_atoms, add them to the editable version of the molecule, and create Atom objects."""
        atoms = []

        rd_periodic_table = Chem.GetPeriodicTable()
        for atom in self.orig_mol.GetAtoms():
            atom_is_metal = (atom.GetSymbol() in metal_symbols)
            atom.SetIsAromatic(False)  # remove aromaticity properties
            num_valence_electrons = (
                rd_periodic_table.GetNOuterElecs(atom.GetSymbol())
                - atom.GetFormalCharge()
            )
            atoms.append(
                Atom(
                    molecule=self,
                    atom_type=atom.GetSymbol(),
                    idx=atom.GetAtomMapNum(),
                    num_valence_electrons=num_valence_electrons,
                    metal=atom_is_metal,
                )
            )

        return atoms

    def set_up_atoms_NBO(self):
        """Process NBO atoms, add them to the editable version of the molecule, and create Atom objects."""
        atoms = []
        smiles_list = self.numbered_smiles.split('.')
        num_electrons, lp_per_atom = extract_electrons_based_bond_matrix(self.nbo_lines, smiles_list, self.organometallic)

        for atom in self.orig_mol.GetAtoms():
            atom_idx = atom.GetAtomMapNum()
            atom_is_metal = (atom.GetSymbol() in metal_symbols)
            atom.SetIsAromatic(False)  # remove aromaticity properties
            num_valence_electrons = int(num_electrons[atom_idx])
            if atom_is_metal and atom_idx in lp_per_atom:
                n_doubly_occ = lp_per_atom[atom_idx]
            else:
                n_doubly_occ = None
            atoms.append(
                AtomNBO(
                    molecule=self,
                    atom=atom,
                    atom_type=atom.GetSymbol(),
                    idx=atom_idx,
                    num_valence_electrons=num_valence_electrons,
                    metal=atom_is_metal,
                    n_doubly_occ=n_doubly_occ
                )
            )

        return atoms


    def set_up_localized_configuration(self):
        """ Set up a localized configuration with localized orbital systems for the molecule."""
        return LocalizedConfiguration(self.orig_mol, self.atoms)

    def set_up_localized_configuration_nbo(self):
        """ Set up a localized configuration with localized orbital systems for the molecule."""
        return LocalizedConfigurationNBO(self.numbered_smiles, self.atoms, self.nbo_lines, self.threshold_ssi,
                                         self.organometallic, self.threshold_si, self.radicalic, self.orig_mol)

    def set_up_orbital_graph(self):
        """ Set up an orbital graph for the molecule."""
        return OrbitalGraph(self.localized_configuration, self.numbered_smiles, self.orig_mol, self.nbo, self.organometallic)

    def generate_reaction_paths(self, idx_list, max_length):
        """
        Generate reaction paths for the molecule.

        Args:
            idx_list (list): indices of the orbital systems that need to be included in paths.
            max_length (int): the maximal number of bonding systems that can be combined on a single fragment.

        Returns:
            tuple: A tuple containing two lists. The first list contains interfragment paths,
            and the second list contains modified reaction paths.
        """
        if len(self.numbered_smiles.split('.')) > 1:
            print('Determining intermolecular reactions...')
            intrafragment_paths = self.orbital_graph.get_intrafragment_paths(max_length=max_length)
            interfragment_paths = self.orbital_graph.get_interfragment_paths(intrafragment_paths)
            if idx_list is not None:
                interfragment_paths = self.filter_paths(interfragment_paths, idx_list)
            return interfragment_paths
        else:
            print('Determining intramolecular reactions...')
            intramolecular_paths = self.orbital_graph.get_intramolecular_paths(max_length=max_length)
            if idx_list is not None:
                intramolecular_paths = self.filter_paths(intramolecular_paths, idx_list)
            return intramolecular_paths

    def generate_products(self, original_paths, allow_zwitterions):
        """
        Generate products based on reaction paths.

        Args:
            original_paths (list): List of original reaction paths.
            allow_zwitterions (bool): A bool indicating whether zwitterionic SMILES are considered

        Returns:
            list: List of generated products.
        """
        products, products_with_paths, maps_between_smiles = self.orbital_graph.generate_products(original_paths, allow_zwitterions=allow_zwitterions)
        stereo_products = generate_stereoisomers(products_with_paths)

        return products, stereo_products, maps_between_smiles


    def filter_paths(self, paths, idx_list):
        """
        Filter paths based on the presence of specific elements.

        This method filters paths based on whether all the elements in the set
        of VOs extracted from the active orbital systems
        corresponding to the provided idx-list are present in each path.

        Args:
            paths (list of lists): A list of paths, where each path is represented
                as a list of elements.
            idx_list (list): A list of indices corresponding to active orbital
                systems.

        Returns:
            list of lists: A filtered list of paths containing only those paths
                that have all the Virtual Objects (VOs) extracted from the
                specified active orbital systems.
        """
        vo_list = []
        for orbital_system in self.orbital_graph.localized_configuration.active_orbital_systems_list:
            if orbital_system.idx in idx_list:
                vo_list += orbital_system.get_vos()

        filtered_paths = []
        vo_set = set(vo_list)
        for path in paths:
            if vo_set.issubset(set(path)):
                filtered_paths.append(path)
            else:
                continue

        return filtered_paths


def get_neighbors_dict(orig_mol):
    """
    Get a dictionary of atom neighbors for the given molecule.

    Args:
        orig_mol: The original molecule for which atom neighbors are to be retrieved.

    Returns:
        dict: A dictionary where the keys are atom map numbers and the values are lists 
        of atom map numbers representing the neighbors of each atom.
    """
    return {atom.GetAtomMapNum(): [neighbor.GetAtomMapNum()
        for neighbor in atom.GetNeighbors()] for atom in orig_mol.GetAtoms()}


def get_num_bonds_atom(atom):
    """ Get how many bonds an atom has. """
    num_bonds = 0
    for bond in atom.GetBonds():
        num_bonds += int(bond.GetBondTypeAsDouble())
    return num_bonds