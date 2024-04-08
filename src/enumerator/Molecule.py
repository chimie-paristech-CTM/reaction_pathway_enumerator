from typing import Dict, List
from rdkit import Chem
import numpy as np
import re
from itertools import permutations, product
from tqdm import tqdm
from enumerator.generate_smiles import fix_radical_counts_at_endpoints_path, increase_bond_order, decrease_bond_order #generate_smiles
from copy import deepcopy


ps = Chem.SmilesParserParams()
ps.removeHs = False

metal_symbols = ["Al", "Fe", "Cu", "Au", "Ag",  "Zn", "Ni",  "Sn",  "Pb",  "Pt",  "Hg",  "Ti", "Co", 
    "Cr",  "Mg",  "Mn",  "W",   "Bi",  "Sb",  "Cd",  "V",   "U",   "Pd",  "Rh",  "Ru" 
]

# TODO: metals added in principle, but this won't work well until you have a good description of the bonding in the graph
def atom_to_num_VOs(atom_symbol: str) -> int:
    """Returns the number of VOs the atom should be initialized with."""
    if atom_symbol == "H" or atom_symbol == "He":
        return 1 # s
    elif atom_symbol in metal_symbols:
        return 6 # s + d
    else:
        return 4 # s + p


class ValenceOrbital:
    """A class corresponding to individual valence orbitals."""

    def __init__(self, idx: int, atom_idx: int, atom_type: str):
        self.idx = idx
        self.atom_idx = atom_idx
        self.atom_type = atom_type
        self.num_electrons = 0
        self.paired = False

        # set an identifier attribute that is unique across the entire reacting system
        self.identifier = f'{self.atom_idx}_{self.idx}'

    def set_population(self, num_electrons):
        """Sets the number of electrons present in the valence orbital."""
        self.num_electrons = num_electrons

    def set_paired(self):
        """Sets self.paired-bool to indicate that the valence orbital is paired."""
        self.paired = True

    def set_unpaired(self):
        """Sets self.paired-bool to indicate that the valence orbital is unpaired."""
        self.paired = False

    def is_paired(self):
        """Returns bool indicating whether the valence orbital is paired or not."""
        return self.paired

    def __str__(self) -> str:
        return (
            f"self.atom_idx: {str(self.atom_idx)}; self.idx: {str(self.idx)};"
            f" self.num_electrons: {self.num_electrons}; self.paired: {self.paired}"
        )

    def __repr__(self) -> str:
        return self.__str__()


class Atom:
    """A class corresponding to individual atoms."""

    def __init__(
        self, molecule: "Molecule", atom_type: str, idx: int, num_valence_electrons: int, metal=False
    ):
        self.molecule = molecule
        self.atom_type = atom_type
        self.idx = idx
        self.valence_orbitals = []
        self.num_valence_orbitals = atom_to_num_VOs(self.atom_type)
        self.num_valence_electrons = num_valence_electrons
        self.metal = metal

        for vo_idx in range(self.num_valence_orbitals):
            self.valence_orbitals.append(
                ValenceOrbital(vo_idx, self.idx, self.atom_type)
            )
        
        if self.metal: # for metals, also add the empty p orbitals of the next shell
            for vo_idx in range(self.num_valence_orbitals, self.num_valence_orbitals + 3):
                self.valence_orbitals.append(
                    ValenceOrbital(vo_idx, self.idx, self.atom_type)
                )

        self.occupy_vos()

    def occupy_vos(self):
        """Occupies the valence orbitals associated with the atom, based on the number of valence electrons present."""
        n_doubly_occ = max(0, (self.num_valence_electrons - self.num_valence_orbitals))
        n_singly_occ = self.num_valence_electrons - n_doubly_occ * 2

        for vo in self.valence_orbitals:
            if vo.idx < n_singly_occ:
                vo.set_population(1)
            elif vo.idx < n_singly_occ + n_doubly_occ:
                vo.set_population(2)
    
    def __str__(self):
        return f'idx: {self.idx}, type: {self.atom_type}, vos: {self.valence_orbitals}'


class OrbitalSystem:
    """An abstract class corresponding to generic orbital systems."""

    def __init__(self, idx, vo_init=None):
        self.idx = idx
        self.vos = []

        if vo_init is not None:
            self.vos.append(vo_init)
    
    def add_vo(self, vo: "ValenceOrbital"):
        """Add a valence orbital to the list of valence orbitals."""
        self.vos.append(vo)

    def get_vos(self) -> list:
        """Return the list of valence orbitals."""
        return self.vos
    
    def get_num_electrons(self):
        """Return the total number of electrons in the valence orbitals."""
        return sum([vo.num_electrons for vo in self.vos])

    def __str__(self):
        pass

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.vos)


class LocalizedOrbitalSystem(OrbitalSystem):
    """A class corresponding to individual (localized) orbital systems."""

    def add_vo(self, vo: "ValenceOrbital"):
        """Add valence orbitals to the orbital system (and modify the pairing bools accordingly)."""
        self.vos.append(vo)
        if len(self.vos) == 1:
            self.vos[0].set_unpaired()
        else:
            for vo in self.vos:
                vo.set_paired()

    def get_atoms(self):
        """Retruns a list of atom indices present in the orbital system."""
        return [vo.atom_idx for vo in self.vos]

    def get_heavy_atoms(self):
        """Returns a list of heavy atom indices present in the orbital system."""
        return [vo.atom_idx for vo in self.vos if vo.atom_type != "H"]
        
    def is_2c3e_or_2c1e(self):
        """
        returns bool, indicating whether the localized orbital system is 2 center-1/3 electron.
        """
        if len(self.vos) == 2 and self.get_num_electrons() != 2:
            return True
        else:
            False

    def is_conventional(self):
        """
        returns bool, indicating whether the localized orbital system is not 3 center or 2c3e or 2c1e
        """
        if len(self.vos) == 1 or (len(self.vos) == 2 and self.get_num_electrons() == 2):
            return True
        else:
            return False

    def __str__(self):
        return f'localized orbital system {self.idx}: vos = {self.vos}'

    
class DelocalizedOrbitalSystem(OrbitalSystem):
    """A class corresponding to individual delocalized orbital systems."""
    
    def __str__(self):
        return f'delocalized orbital system {self.idx}: vos = {self.vos}'


class LocalizedConfiguration:

    def __init__(self, orig_mol, atoms):
        self.orbital_systems_list = self.set_up_localized_orbital_systems(orig_mol, atoms)
        self.active_orbital_systems_list = self.select_active_orbital_systems()
        self.vo_list = self.set_vo_list()
        self.vo_to_orbital_system_dict = self.get_vo_to_orbital_system_dict()

    # TODO: what about circular 3c bonds (e.g., interaction between ethylene and PdL2)?
    # TODO: should you include validity checks to ensure that the localized configuration makes sense (e.g., exotic boding situations resulting in incorrect vo pairing)
    def set_up_localized_orbital_systems(self, orig_mol, atoms):
        """Construct the initial orbital systems (either 1, 2 or 3 vos in a linear arrangment)."""
        orbital_systems = []

        # Create adjacency list representation for bonds. Initial_bonds is not symmetric.
        initial_bonds: Dict[int, List[int]] = dict()
        for bond in orig_mol.GetBonds():
            bond.SetIsAromatic(False)  # remove aromaticity properties
            atom_1 = bond.GetBeginAtom().GetAtomMapNum()
            atom_2 = bond.GetEndAtom().GetAtomMapNum()
            num_bonds = round(bond.GetBondTypeAsDouble())
            if atom_1 < atom_2:
                initial_bonds[atom_1] = initial_bonds.get(atom_1, []) + [atom_2] * num_bonds
            else:
                initial_bonds[atom_2] = initial_bonds.get(atom_2, []) + [atom_1] * num_bonds

        # construct all the orbital systems
        orbital_system_idx = 0
        for atom in atoms:
            if atom.idx in initial_bonds.keys():
                neighbors = initial_bonds[atom.idx].copy()
            else:
                neighbors = None
            for vo in atom.valence_orbitals:
                # generate lone pairs and empty valences
                if vo.num_electrons == 0 or vo.num_electrons == 2:
                    new_orbital_system = LocalizedOrbitalSystem(orbital_system_idx)
                    new_orbital_system.add_vo(vo)
                    orbital_systems.append(new_orbital_system)
                    orbital_system_idx += 1
                # generate orbital pairs
                elif vo.num_electrons == 1 and vo.paired == False:
                    new_orbital_system = LocalizedOrbitalSystem(orbital_system_idx)
                    new_orbital_system.add_vo(vo)
                    if neighbors is not None:
                        if len(neighbors) > 0:
                            neighbor_idx = neighbors.pop()
                            for partner_vo in atoms[neighbor_idx-1].valence_orbitals:
                                if (
                                    partner_vo.num_electrons == 1
                                    and partner_vo.paired == False
                                ):
                                    new_orbital_system.add_vo(partner_vo)
                                    partner_vo.set_paired()
                                    vo.set_paired()
                                    break
                    orbital_systems.append(new_orbital_system)
                    orbital_system_idx += 1

        return orbital_systems 

    # TODO: you are losing lone pairs here 
    def select_active_orbital_systems(self):
        """ Only keep 1 orbital system of a triple/double bond, and only keep 1 X-H bond for every atom X """
        active_orbital_systems = set()
        already_covered_systems = set()
        for orbital_system in self.orbital_systems_list:
            # these are the minimal characteristics to distinguish between orbital systems
            system_info = f'{orbital_system.get_num_electrons()}, {set(orbital_system.get_heavy_atoms())}, {len(orbital_system.get_atoms())}'
            if system_info not in already_covered_systems:
                already_covered_systems.add(system_info)
                active_orbital_systems.add(orbital_system)

        return active_orbital_systems

    def get_vo_to_orbital_system_dict(self):
        """    
        Get a dictionary mapping valence orbitals to their corresponding orbital systems.

        Returns:
            dict: A dictionary where keys are valence orbitals (VOs) and values are the orbital 
            systems to which the VOs belong.
        """
        vo_to_orbital_system_dict = {}
        for orbital_system in self.active_orbital_systems_list:
            for vo in orbital_system.vos:
                vo_to_orbital_system_dict[vo.identifier] = orbital_system
        
        return vo_to_orbital_system_dict

    def set_vo_list(self):
        """
        Obtain a list of all vos involved in active orbital systems across the localized configuration.
        """
        vos = []
        for orbital_system in self.active_orbital_systems_list:
            vos += orbital_system.get_vos()
        vos = list(set(vos))

        return vos

    def get_vos(self):
        """     
        Get the valence orbitals associated with the orbital systems.

        Returns:
        list: A list of valence orbitals (VOs) associated with the orbital systems.
        """
        return self.vo_list


class Reaction:

    def __init__(self, orig_mol, vo_list, existing_interactions, conventional_path=True):
        self.orig_mol = orig_mol
        self.orig_path = vo_list
        self.modified_path = deepcopy(vo_list)
        self.existing_interactions = existing_interactions

        self.ignore_first_vo = False
        self.ignore_final_vo = False

        if not conventional_path:
            self.connect_end_vos = False
            self.invert_vo_populations_to_complete_orig_path()
        else:
            self.connect_end_vos = True
        
        self.adjust_vo_populations_along_modified_path()

    def invert_vo_populations_to_complete_orig_path(self):
        """
        This method inverses vo populations if vos not at the endpoints, but still in the first or last orbital system, 
        contain a number of electrons differing from 1 (reflecting the inherent delocalization present in these systems). 
        In case of 2c3e/2c1e, the first/list vo is set inactive, since the actual reorganization of the path will need to 
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
        elif self.modified_path[start_idx].num_electrons != 1 and self.modified_path[end_idx].num_electrons == 1:
            # pass on excess/shortage of electrons from one side of the path to the other
            self.modified_path[end_idx].num_electrons = self.orig_path[start_idx].num_electrons
            self.modified_path[start_idx].num_electrons = 1 
        elif self.modified_path[start_idx].num_electrons == 1 and self.modified_path[end_idx].num_electrons != 1:
            # reverse from above
            self.modified_path[start_idx].num_electrons = self.orig_path[end_idx].num_electrons
            self.modified_path[end_idx].num_electrons = 1
    
    # TODO: for 3 center systems, you will still need to add a bond at the edges because you are breaking up the bonding system completely.
    # For now you can leave this however since this is inherently problematic with SMILES.
    def generate_smiles(self):
        """Generate an output SMILES string.

        Returns:
            str: the output SMILES
        """
        editable_mol = Chem.RWMol(self.orig_mol)  # editable version of the molecule
        start_idx, end_idx = self.get_start_end_idx()

        # modify atom properties
        for vo, modified_vo in zip(self.orig_path, self.modified_path):
            if vo.num_electrons != modified_vo.num_electrons:
                init_charge = editable_mol.GetAtomWithIdx(vo.atom_idx - 1).GetFormalCharge()
                new_charge = init_charge - (modified_vo.num_electrons - vo.num_electrons)
                editable_mol.GetAtomWithIdx(vo.atom_idx - 1).SetFormalCharge(new_charge)
    
        # modify bonding situation
        for i, vo in enumerate(self.orig_path[start_idx:end_idx]):
            if self.orig_path[i+1] in self.existing_interactions[vo.identifier]:
                editable_mol = decrease_bond_order(editable_mol, vo, self.orig_path[i+1])
            else:
                editable_mol = increase_bond_order(editable_mol, vo, self.orig_path[i+1])  
        
        # take care of the terminal vos of the path -> connect or leave radical site
        if self.connect_end_vos and self.orig_path[start_idx].is_paired() and self.orig_path[end_idx].is_paired():
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
            print(Chem.MolToSmiles(editable_mol))

        return Chem.MolToSmiles(editable_mol)

    def invert_vo_populations(self, vo1, vo2):
        """
        Switch the electronic populations of two VOs
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


class OrbitalGraph:
    """ 
    A class corresponding to an abstract graph, where nodes correspond to VOs, 
    and edges correspond to existing and potential (intra-/interfragment) interactions. 
    """
    def __init__(self, localized_configuration, numbered_smiles, orig_mol):
        self.localized_configuration = localized_configuration
        self.atom_to_fragment_dict = self.get_atom_to_fragment_dict(numbered_smiles)
        self.numbered_smiles = numbered_smiles
        self.orig_mol = orig_mol

        self.existing_interactions = {}
        self.potential_intrafragment_interactions = {}
        self.potential_interfragment_interactions = {}
        self.secondary_interactions = {}

        self.add_vos_to_graph()
        self.add_existing_interactions()
        self.add_potential_interactions()

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
            if vo not in self.potential_interfragment_interactions:
                self.potential_interfragment_interactions[vo.identifier] = set()
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
                # don't include orbitals on the same atom
                elif vo2.atom_idx == vo1.atom_idx:
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
                # if different fragments -> potential interfragment interactions
                else:
                    self.add_potential_interfragment_interaction(vo1, vo2)
    
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

    def add_potential_interfragment_interaction(self, source, destination):
        """
        Add a potential intrafragment interaction between valence orbitals.

        Args:
            source: The source valence orbital.
            destination: The destination valence orbital.
        """
        if source.identifier in self.potential_interfragment_interactions and destination.identifier in self.potential_interfragment_interactions \
            and destination not in self.potential_interfragment_interactions[source.identifier]:
            self.potential_interfragment_interactions[source.identifier].add(destination)

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

    def get_intrafragment_neighbors(self, vo):
        return self.potential_intrafragment_interactions.get(vo.identifier, [])
        
    def get_interfragment_neighbors(self, vo):
        return self.potential_interfragment_interactions.get(vo.identifier, [])

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
            elif len(partner_vos) == 1:
                if vo.num_electrons == 1 and partner_vos[0].num_electrons == 1:
                    new_path.append(vo)
                    new_path.append(partner_vos[0])
                elif vo.num_electrons == 1 and partner_vos[0].num_electrons != 1: # 2c1e/2c3e TODO: you should still break the bond between the partner and the vo
                    new_path.append(partner_vos[0]) # put the other vo first so that you can leave this "dangling"
                    new_path.append(vo) 
                if len(self.get_interacting_orbitals(partner_vos[0])) == 2: # 3c systems
                    remaining_vo = [vo3 for vo3 in self.get_interacting_orbitals(partner_vos[0]) if vo3 != vo]
                    new_path.append(remaining_vo[0])
            else:
                continue # in a 3c system, starting from either of the two extremes is enough to capture all possibilities
            all_intrafragment_paths[self.atom_to_fragment_dict[vo.atom_idx]].append(new_path)

        # iterate through the fragments
        for fragment_paths in all_intrafragment_paths:
            # initialize -- at first, we consider plausible extensions to each mono-orbital system path
            paths_to_extend = fragment_paths
            while len(paths_to_extend) > 0:
                new_paths_to_extend = []
                for path in paths_to_extend:
                    for neighbor in self.get_intrafragment_neighbors(path[-1]):
                        partners_of_neighbor = self.get_interacting_orbitals(neighbor)
                        if len(partners_of_neighbor) == 0 or len(partners_of_neighbor) == 2:
                            continue # you don't want to prematurely end paths
                        elif len(partners_of_neighbor) == 1:
                            if neighbor.num_electrons != 1 or partners_of_neighbor[0].num_electrons !=1:
                                continue 
                            # 2c1e/2c3e bonds should not be in the middle of the path either since this would also end the path
                            elif neighbor.num_electrons + partners_of_neighbor[0].num_electrons != 2:
                                continue
                            elif neighbor not in path and partners_of_neighbor[0].atom_idx not in [vo.atom_idx for vo in path]:
                                new_path = path.copy() # only a shallow copy is needed -> don't duplicate the VOs
                                new_path.append(neighbor)
                                new_path.append(partners_of_neighbor[0])
                                fragment_paths.append(new_path)
                                if self.get_number_of_delocalized_orbital_systems(path) < max_length:
                                    new_paths_to_extend.append(new_path)
                                else:
                                    continue
                            else:
                                continue # no crossings within path
                    
                paths_to_extend = new_paths_to_extend

        return all_intrafragment_paths
    
    # TODO: what if two lone pairs get selected as end-points???
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
                new_interfragment_path = []
                for fragment_path in combination:
                    new_interfragment_path += fragment_path
                # remove invalid paths -- you should only keep continuous paths, i.e., when the bridging vos had an interacting orbital to start with
                if any([len(self.get_interacting_orbitals(vo)) != 1 for vo in new_interfragment_path[1:-1]]):
                    continue
                all_interfragment_paths.append(new_interfragment_path)

        return all_interfragment_paths

    def generate_products(self, all_paths):
        products = []
        unique_products = set()
        # true_start_vo, true_end_vo = None, None
        for path in tqdm(all_paths):

            if (self.localized_configuration.vo_to_orbital_system_dict[path[0].identifier].is_conventional() and \
                self.localized_configuration.vo_to_orbital_system_dict[path[-1].identifier].is_conventional()):
                reaction = Reaction(self.orig_mol, path, self.existing_interactions, conventional_path=True)
            else:
                reaction = Reaction(self.orig_mol, path, self.existing_interactions, conventional_path=False) 

            smiles = reaction.generate_smiles()

            if smiles != None:
                smiles_without_numbering = clear_numbering(smiles)
                if smiles_without_numbering not in unique_products:
                    unique_products.add(smiles_without_numbering)
                    products.append(smiles)

        return products

    def __str__(self) -> str:
        return f'intra: {self.potential_intrafragment_interactions}; inter:{self.potential_interfragment_interactions}'


class ReactingSystem:
    """A class corresponding to reacting systems (can consist of multiple molecules)"""

    def __init__(self, smiles: str):
        self.orig_mol, self.numbered_smiles = self.parse_smiles(smiles)
        
        self.num_atoms = self.orig_mol.GetNumAtoms()
        self.atoms = self.set_up_atoms()

        self.localized_configuration = self.set_up_localized_configuration()
        self.orbital_graph = self.set_up_orbital_graph()

    def parse_smiles(self, smiles):
        """Get mol object with hydrogens, fully numbered and kekulized """
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)  # always add H's to make bonding correct
        Chem.Kekulize(mol) # change to kekulized smiles to remove aromatic bonds
        [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in mol.GetAtoms()]

        return mol, Chem.MolToSmiles(mol) 

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
                    metal=atom_is_metal
                )
            )

        return atoms
    
    def set_up_localized_configuration(self):
        """ Set up a localized configuration with localized orbital systems for the molecule."""
        return LocalizedConfiguration(self.orig_mol, self.atoms)

    def set_up_orbital_graph(self):
        """ Set up an orbital graph for the molecule."""
        return OrbitalGraph(self.localized_configuration, self.numbered_smiles, self.orig_mol)

    def generate_reaction_paths(self):
        """
        Generate reaction paths for the molecule.

        Returns:
            tuple: A tuple containing two lists. The first list contains interfragment paths,
            and the second list contains modified reaction paths.
        """
        intrafragment_paths = self.orbital_graph.get_intrafragment_paths()
        interfragment_paths = self.orbital_graph.get_interfragment_paths(intrafragment_paths)

        return interfragment_paths

    def generate_products(self, original_paths):
        """
        Generate products based on reaction paths.

        Args:
            original_paths (list): List of original reaction paths.

        Returns:
            list: List of generated products.
        """
        products = self.orbital_graph.generate_products(original_paths)
        return products


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


def clear_numbering(smiles):
    """
    Clear atom numbering in the SMILES representation.

    Args:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        str or None: The SMILES representation of the molecule with cleared atom numbering,
        or None if an error occurs during processing.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)
    except:
        return None


def generate_subsets_bit(nums):
    """
    Generate all possible subsets of a set using bitwise operations.

    Args:
        nums (list): A list of integers representing the set.

    Returns:
        list: A list of lists containing all possible subsets of the input set.

    Note:
        This function generates all possible subsets of a given set of integers using bitwise 
        operations. It iterates through all possible combinations of elements using a bitmask, 
        where each bit represents whether the corresponding element is included in the subset 
        or not. The resulting subsets are stored in a list of lists and returned.
    """
    subsets = []
    n = len(nums)

    for i in range(2 ** n):
        subset = []
        for j in range(n):
            if (i >> j) & 1:
                subset.append(nums[j])
        subsets.append(subset)

    return subsets
