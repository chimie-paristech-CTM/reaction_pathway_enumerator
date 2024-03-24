from typing import Dict, List
from rdkit import Chem
import numpy as np
import re
from itertools import permutations, product
from tqdm import tqdm
from enumerator.generate_smiles import generate_smiles
from copy import deepcopy


ps = Chem.SmilesParserParams()
ps.removeHs = False

electro_dict = {
    "H": 2.1,
    "B": 2.0,
    "C": 2.5,
    "N": 3.0,
    "O": 3.5,
    "F": 4.0,
    "Si": 1.8,
    "P": 2.1,
    "S": 2.5,
    "Cl": 3.0,
    "Br": 2.8,
}


def atom_to_num_VOs(atom_symbol: str) -> int:
    """Returns the number of VOs the atom should be initialized with."""
    if atom_symbol == "H" or atom_symbol == "He":
        return 1
    else:
        return 4


class ValenceOrbital:
    """A class corresponding to individual valence orbitals."""

    def __init__(self, idx: int, atom_idx: int, atom_type: str):
        self.idx = idx
        self.atom_idx = atom_idx
        self.atom_type = atom_type
        self.num_electrons = 0
        self.paired = False

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
        """ """
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
        self, molecule: "Molecule", atom_type: str, idx: int, num_valence_electrons: int
    ):
        self.molecule = molecule
        self.atom_type = atom_type
        self.idx = idx
        self.valence_orbitals = []
        self.num_valence_orbitals = atom_to_num_VOs(self.atom_type)
        self.num_valence_electrons = num_valence_electrons

        for vo_idx in range(self.num_valence_orbitals):
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


class LocalizedOrbitalSystem:
    """A class corresponding to individual (localized) orbital systems."""

    def __init__(self, idx: int):
        self.idx = idx
        self.vos = []
        self.num_electrons = 0
        self.polarity_set = False

    def add_vo(self, vo: "ValenceOrbital"):
        """Add valence orbitals to the orbital system (and modify the pairing bools accordingly)."""
        self.vos.append(vo)
        self.num_electrons += vo.num_electrons
        if len(self.vos) == 1:
            self.vos[0].set_unpaired()
        else:
            for vo in self.vos:
                vo.set_paired()

    def set_polarity(self):
        """
        If electronegativity spread bigger than, or equal to, 0.4, set the vo order of the orbital system, 
        with the negative pole first. Values are specific vos.
        """
        if len(self.vos) > 1:
            electronegativity_list = [electro_dict[vo.atom_type] for vo in self.vos]
            if max(electronegativity_list) - min(electronegativity_list) >= 0.399:
                self.vos = [
                    self.vos[np.argmax(electronegativity_list)],
                    self.vos[np.argmin(electronegativity_list)],
                ]
                self.polarity_set = True

    def return_vos_forward(self, reverse=False):
        """ Returns a list with copies of the VOs """
        if reverse:
            return deepcopy(self.vos).reverse()
        else:
            return deepcopy(self.vos)

    def is_lone_pair(self):
        """Returns True if orbital system corresponds to lone pair."""
        if len(self.vos) == 1 and self.num_electrons == 2:
            return True
        else:
            False
        
    def is_empty_valence(self):
        """Returns True if orbital system corresponds to empty valence."""
        if len(self.vos) == 1 and self.num_electrons == 0:
            return True
        else:
            return False

    def is_xh_bond(self):
        """Returns True if orbital system corresponds to an X-H bond."""
        return len(self.get_heavy_atoms()) != len(self.vos)

    def get_heavy_atoms(self):
        """Returns a list of heavy atom indices present in the orbital system."""
        return [vo.atom_idx for vo in self.vos if vo.atom_type != "H"]
    
    def get_atoms(self):
        return [vo.atom_idx for vo in self.vos]

    def __str__(self) -> str:
        return f"idx: {self.idx}; vos: {[str(vo) for vo in self.vos]}"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.vos)
    
class LocalizedConfiguration:
    def __init__(self, orig_mol, atoms):
        self.orbital_systems_list = self.set_up_localized_orbital_systems(orig_mol, atoms)
        self.active_orbital_systems_list = self.select_active_orbital_systems()
        self.vo_to_orbital_system_dict = self.get_vo_to_orbital_systems_dict()
    
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
                            for partner_vo in self.atoms[neighbor_idx-1].valence_orbitals:
                                if (
                                    partner_vo.num_electrons == 1
                                    and partner_vo.paired == False
                                ):
                                    new_orbital_system.add_vo(partner_vo)
                                    partner_vo.set_paired()
                                    break
                            vo.set_paired()
                    orbital_systems.append(new_orbital_system)
                    orbital_system_idx += 1

        return orbital_systems 
    
    def select_active_orbital_systems(self):
        """ Only keep 1 orbital system of a triple/double bond, and only keep 1 X-H bond for every atom X """
        active_orbital_systems = set()
        already_covered_systems = set()
        for orbital_system in self.orbital_systems:
            system_info = f'{orbital_system.num_electrons}, {set(orbital_system.get_heavy_atoms())}'
            if system_info not in already_covered_systems:
                already_covered_systems.add(system_info)
                active_orbital_systems.add(orbital_system)
    
        return active_orbital_systems

    def get_vo_to_orbital_system_dict(self):
        vo_to_orbital_system_dict = {}
        for orbital_system in self.active_orbital_systems_list:
            for vo in orbital_system.vos:
                vo_to_orbital_system_dict[vo] = orbital_system
        
        return vo_to_orbital_system_dict
    
    def get_vos(self):
        """ """
        return LocalizedConfiguration.vo_to_orbital_system_dict.keys()


class OrbitalGraph:
    """ A class corresponding to an abstract graph, where nodes correspond to VOs, and edges correspond to existing and potential (intra-/interfragment) interactions. """
    def __init__(self, localized_configuration, numbered_smiles, orig_mol):
        self.localized_configuration = localized_configuration
        self.atom_to_fragment_dict = self.get_atom_to_fragment_dict(numbered_smiles)
        self.orig_mol = orig_mol

        self.existing_interactions = {}
        self.potential_intrafragment_interactions = {}
        self.potential_interfragment_interactions = {}
        self.potential_interfragment_path_ending_interactions = {}

        self.add_vos_to_graph()

    def get_atom_to_fragment_dict(self, numbered_smiles):
        atom_to_fragment_dict = {}
        fragment_smiles_list = numbered_smiles.split('.')
        for i, fragment_smiles in enumerate(fragment_smiles_list):
            atom_numbers = re.findall(r':(\d+)\]', fragment_smiles)
            for atom_number in atom_numbers:
                atom_to_fragment_dict[int(atom_number)] = i
    
        return atom_to_fragment_dict

    def add_vos_to_graph(self):
        for vo in self.localized_configuration.get_vos():
            if vo not in self.potential_intrafragment_interactions:
                self.potential_intrafragment_interactions[vo] = []
            if vo not in self.potential_interfragment_interactions:
                self.potential_interfragment_interactions[vo] = []
            if vo not in self.potential_interfragment_path_ending_interactions:
                self.potential_interfragment_path_ending_interactions[vo] = []

    def add_existing_interactions(self):
        for orbital_system in self.localized_configuration.orbital_systems_list:
            if len(orbital_system.vos) > 1:
                for i, vo in enumerate(orbital_system.vos[:-1]):
                    self.existing_interactions[vo] = [orbital_system.vos[i+1]]
                    self.existing_interactions[orbital_system.vos[i+1]]
    
    def add_potential_interactions(self):
        neighbors_dict = get_neighbors_dict(self.orig_mol)
        for vo1 in self.localized_configuration.get_vos():
            owning_fragment = self.atom_to_fragment_dict[vo1.atom_idx]

            for vo2 in self.localized_configuration.get_vos():
                # don't include existing interactions
                if vo2 in self.existing_interactions[vo1] or self.existing_interactions[vo2]:
                    continue
                # if intrafragment interactions, we only consider atoms that are adjacent
                if owning_fragment == self.atom_to_fragment_dict[vo2.atom_idx]:                  
                    if  vo2.atom_idx == vo1.atom_idx: # no 2 VOs on same atom
                        continue
                    if vo2.atom_idx in neighbors_dict[vo1.atom_idx]: # if the idx in the candidate partner is in neighbor set => adjacent orbital systems
                            self.add_potential_intrafragment_interaction(vo1, vo2)
                    else:
                        continue
                # for interfragment interactions, we distinguish with path ending, and regular interfragment interactions
                else:
                    if len(self.existing_interactions[vo2]) != 0:
                        self.add_potential_interfragment_interaction(vo1, vo2)
                    else:
                        self.add_potential_interfragment_path_ending_interaction(vo1, vo2)
    
    def add_potential_intrafragment_interaction(self, source, destination):
        if source in self.potential_intrafragment_interactions and destination in self.potential_intrafragment_interactions \
            and destination not in self.potential_intrafragment_interactions[source]:
            self.potential_intrafragment_interactions[source].append(destination)

    def add_potential_interfragment_interaction(self, source, destination):
        if source in self.potential_interfragment_interactions and destination in self.potential_interfragment_interactions \
            and destination not in self.potential_interfragment_interactions[source]:
            self.potential_interfragment_interactions[source].append(destination)
    
    def add_potential_interfragment_path_ending_interaction(self, source, destination):
        if source in self.potential_interfragment_path_ending_interactions and destination in self.potential_interfragment_path_ending_interactions \
            and destination not in self.potential_interfragment_path_ending_interactions[source]:
            self.potential_interfragment_path_ending_interactions[source].append(destination)
    
    def get_interacting_orbitals(self, vo):
        return deepcopy(self.existing_interactions.get(vo, []))

    def get_intrafragment_neighbors(self, vo):
        return deepcopy(self.potential_intrafragment_interactions.get(vo, [])) 
        
    def get_interfragment_neighbors(self, vo):
        return deepcopy(self.potential_interfragment_interactions.get(vo, []))

    def get_interfragment_path_ending_neighbors(self, vo):
        return deepcopy(self.potential_interfragment_path_ending_interactions.get(vo, []))

    def get_intrafragment_paths(self, max_length=2):
        all_intrafragment_paths = [[] for _ in self.numbered_smiles.split('.')]
        # initialize
        for vo in self.localized_configuration.get_vos():
            all_intrafragment_paths[self.atom_to_fragment_dict[vo.atom_idx]].append([vo])
        
        for fragment_paths in all_intrafragment_paths:
            previous_length = 0
            for _ in range(max_length - 1):
                paths_to_extend = fragment_paths[previous_length:]
                previous_length = len(fragment_paths)
                for path in paths_to_extend:
                    for neighbor in self.get_intrafragment_neighbors(path[-1]):
                        partners_of_neighbor = self.get_interacting_orbitals(neighbor)
                        if len(partners_of_neighbor) == 0:
                            continue # you don't want to prematurely end paths
                        elif len(partners_of_neighbor) == 1:
                            if neighbor not in path and partners_of_neighbor not in path:
                                new_path = deepcopy(path)
                                new_path.append(neighbor)
                                new_path.append(partners_of_neighbor[0])
                                fragment_paths.append(new_path)
                            else:
                                continue
                        else:
                            print('not yet implemented!')

        # now the inverted paths should also be added if the path does not start with a singular vo without existing interactions
        # for fragment_paths in all_intrafragment_paths:
        #    for path in fragment_paths:
        #        if len(partners_of_neighbor) == 0

        return all_intrafragment_paths
    
    # TODO: I think I am doing this double!
    def get_interfragment_paths(self, all_intrafragment_paths):
        all_interfragment_paths = []
        # first permutate the fragment order
        potential_fragment_arrangements = list(permutations(list(range(len(all_intrafragment_paths))), len(all_intrafragment_paths)))
        for arrangement in potential_fragment_arrangements:
            intrafragment_path_tmp = [deepcopy(all_intrafragment_paths[i]) for i in arrangement]
            # invert the final fragment pathway if it starts with an unpaired vo
            if not self.existing_interactions[intrafragment_path_tmp[-1][0]].is_paired():
                intrafragment_path_tmp[-1] = intrafragment_path_tmp[-1][::-1]
            # now make all the possible combinations within the selected fragment order
            all_combinations_list = list(product(*intrafragment_path_tmp))
            for combination in all_combinations_list:
                new_interfragment_path = []
                for fragment in combination:
                    new_interfragment_path += fragment 
            
            # finalize the path -- TODO: path ending??????
            if len(self.existing_interactions[new_interfragment_path[0]]) == 1 and len(self.existing_interactions[new_interfragment_path[-1]]) == 1:
                new_interfragment_path.append(self.get_interacting_orbitals(new_interfragment_path[0]))
                all_interfragment_paths.append(new_interfragment_path)   
            elif len(self.existing_interactions[new_interfragment_path[0]]) == 1 and len(self.existing_interactions[new_interfragment_path[-1]]) == 0:
                all_interfragment_paths.append(new_interfragment_path[::-1]) 
            elif len(self.existing_interactions[new_interfragment_path[0]]) == 0 and len(self.existing_interactions[new_interfragment_path[-1]]) == 1:
                all_interfragment_paths.append(new_interfragment_path)
            else:
                continue # TODO: expand this

        return all_interfragment_paths

    def modify_reaction_paths(self, all_paths):
        products = []
        unique_products = set()
        for path in tqdm(all_paths):
            # ensure that the electron counts are fine
            modified_path = deepcopy(path)
            if not path[0].is_paired:
                if not path[-1].is_paired:
                    modified_path[0].num_electrons = 1
                    modified_path[-1].num_electrons = 1
                else:
                    modified_path[-1].num_electrons = path[0].num_electrons
                    modified_path[0].num_electrons = 1
            
            smiles = generate_smiles(
                self.orig_mol,
                    path,
                    modified_path,
                )
            if smiles != None:
                smiles_without_numbering = clear_numbering(smiles)
                if smiles_without_numbering not in unique_products:
                    unique_products.add(smiles_without_numbering)
                    products.append(smiles)
            
        products = list(set(products))
        return products  

    def __str__(self) -> str:
        return f'intra: {self.potential_intrafragment_interactions}; inter:{self.potential_interfragment_interactions}'


class ReactingSystem:
    """A class corresponding to reacting systems (can consist of multiple molecules)"""

    def __init__(self, smiles: str):
        self.orig_mol, self.numbered_smiles = self.parse_smiles(smiles)
        
        self.num_atoms = self.orig_mol.GetNumAtoms()
        self.atoms = self.set_up_atoms()
        self.atom_to_fragment_dict = self.get_atom_to_fragment_dict()

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
                )
            )

        return atoms
    
    def set_up_localized_configuration(self):
        """ """
        LocalizedConfiguration(self.orig_mol, self.atoms)

    def set_up_orbital_graph(self):
        return OrbitalGraph(self.localized_configuration, self.numbered_smiles, self.orig_mol)


def get_neighbors_dict(orig_mol):
        return {atom.GetAtomMapNum(): [neighbor.GetAtomMapNum()
            for neighbor in atom.GetNeighbors()] for atom in orig_mol.GetAtoms()}























class OrbitalSystemGraph:
    """ A class corresponding to an abstract graph, where nodes correspond to orbital systems, and edges correspond to potential (intra-/interfragment) interactions. """

    def __init__(self):

        self.potential_intrafragment_interaction_list = {}
        self.potential_interfragment_interaction_list = {}
        self.potential_interfragment_path_ending_interaction_list = {}
        self.xh_orbital_systems = set()


    def add_orbital_system(self, orbital_system):
        if orbital_system not in self.potential_intrafragment_interaction_list:
            self.potential_intrafragment_interaction_list[orbital_system] = []
        if orbital_system not in self.potential_interfragment_interaction_list:
            self.potential_interfragment_interaction_list[orbital_system] = []
        if orbital_system not in self.potential_interfragment_path_ending_interaction_list:
            self.potential_interfragment_path_ending_interaction_list[orbital_system] = []
        if orbital_system.is_xh_bond() and orbital_system not in self.xh_orbital_systems:
            self.xh_orbital_systems.add(orbital_system)
    
    def add_potential_intrafragment_interaction(self, source, destination):
        if source in self.potential_intrafragment_interaction_list and destination in self.potential_intrafragment_interaction_list \
            and destination not in self.potential_intrafragment_interaction_list[source]:
            self.potential_intrafragment_interaction_list[source].append(destination)

    def add_potential_interfragment_interaction(self, source, destination):
        if source in self.potential_interfragment_interaction_list and destination in self.potential_interfragment_interaction_list \
            and destination not in self.potential_interfragment_interaction_list[source]:
            self.potential_interfragment_interaction_list[source].append(destination)
    
    def add_potential_interfragment_path_ending_interaction(self, source, destination):
        if source in self.potential_interfragment_path_ending_interaction_list and destination in self.potential_interfragment_path_ending_interaction_list \
            and destination not in self.potential_interfragment_path_ending_interaction_list[source]:
            self.potential_interfragment_path_ending_interaction_list[source].append(destination)
    
    def get_intrafragment_neighbors(self, orbital_system, with_xh=True):
        if with_xh == True:
            return self.potential_intrafragment_interaction_list.get(orbital_system, []) # TODO: refine this!!!!
        else:
            neighbors = self.potential_intrafragment_interaction_list.get(orbital_system, [])
            return  [neighbor for neighbor in neighbors if neighbor not in self.xh_orbital_systems] 
        
    def get_interfragment_neighbors(self, orbital_system, with_xh=True):
        if with_xh == True:
            return self.potential_interfragment_interaction_list.get(orbital_system, []) # TODO: refine this!!!!
        else:
            neighbors = self.potential_interfragment_interaction_list.get(orbital_system, [])
            return  [neighbor for neighbor in neighbors if neighbor not in self.xh_orbital_systems] 

    def get_interfragment_path_ending_neighbors(self, orbital_system, with_xh=True):
        if with_xh == True:
            return self.potential_interfragment_path_ending_interaction_list.get(orbital_system, []) # TODO: refine this!!!!
        else:
            neighbors = self.potential_interfragment_path_ending_interaction_list.get(orbital_system, [])
            return  [neighbor for neighbor in neighbors if neighbor not in self.xh_orbital_systems] 

    def __str__(self) -> str:
        return f'intra: {self.potential_intrafragment_interaction_list}; inter:{self.potential_interfragment_interaction_list}'


# TODO: delocalization???
class Molecule:
    """A class corresponding to molecular systems (can consist of multiple molecules)"""

    def __init__(self, smiles: str):
        self.orig_mol, self.numbered_smiles = self.parse_smiles(smiles)
        
        self.num_atoms = self.orig_mol.GetNumAtoms()
        self.atoms = self.get_atoms()
        self.atom_to_fragment_dict = self.determine_atom_to_fragment_dict()

        self.orbital_systems = self.get_orbital_systems() # TODO: set polarity always???
        self.active_orbital_systems = self.select_active_orbital_systems()

        self.orbital_system_graph = OrbitalSystemGraph()
        self.add_orbital_systems_to_graph()
        self.add_potential_interactions_to_graph()

    def parse_smiles(self, smiles):
        """Get mol object with hydrogens, fully numbered and kekulized """
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)  # always add H's to make bonding correct
        Chem.Kekulize(mol) # change to kekulized smiles to remove aromatic bonds
        [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in mol.GetAtoms()]

        return mol, Chem.MolToSmiles(mol) 

    def get_atoms(self):
        """Process rdkit_atoms, add them to the editable version of the molecule, and create Atom objects."""
        atoms = []

        rd_periodic_table = Chem.GetPeriodicTable()
        for atom in self.orig_mol.GetAtoms():
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
                )
            )

        return atoms
    
    def get_orbital_systems(self):
        """Construct the initial orbital systems."""
        orbital_systems = []

        # Create adjacency list representation for bonds. Initial_bonds is not symmetric.
        initial_bonds: Dict[int, List[int]] = dict()
        for bond in self.orig_mol.GetBonds():
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
        for atom in self.atoms:
            if atom.idx in initial_bonds.keys():
                neighbors = initial_bonds[atom.idx].copy()
            else:
                neighbors = None
            for vo in atom.valence_orbitals:
                # generate lone pairs and empty valences
                if vo.num_electrons == 0 or vo.num_electrons == 2:
                    new_orbital_system = OrbitalSystem(orbital_system_idx)
                    new_orbital_system.add_vo(vo)
                    orbital_systems.append(new_orbital_system)
                    orbital_system_idx += 1
                # generate orbital pairs
                elif vo.num_electrons == 1 and vo.paired == False:
                    new_orbital_system = OrbitalSystem(orbital_system_idx)
                    new_orbital_system.add_vo(vo)
                    if neighbors is not None:
                        if len(neighbors) > 0:
                            neighbor_idx = neighbors.pop()
                            for partner_vo in self.atoms[neighbor_idx-1].valence_orbitals:
                                if (
                                    partner_vo.num_electrons == 1
                                    and partner_vo.paired == False
                                ):
                                    new_orbital_system.add_vo(partner_vo)
                                    partner_vo.set_paired()
                                    break
                            vo.set_paired()
                    orbital_systems.append(new_orbital_system)
                    orbital_system_idx += 1

        return orbital_systems 

    def select_active_orbital_systems(self):
        """ Only keep 1 orbital system of a triple/double bond, and only keep 1 X-H bond for every atom X """
        active_orbital_systems = set()
        already_covered_systems = set()
        for orbital_system in self.orbital_systems:
            system_info = f'{orbital_system.num_electrons}, {set(orbital_system.get_heavy_atoms())}'
            if system_info not in already_covered_systems:
                already_covered_systems.add(system_info)
                active_orbital_systems.add(orbital_system)
    
        return active_orbital_systems

    def add_potential_interactions_to_graph(self):
        """ Add potential inter- and intrafragment interactions to the orbital system graph. """
        # get the neighbors for every atom
        neighbors_dict = self.get_neighbors_dict()

        # now construct all the interactions
        for orbital_system in self.active_orbital_systems:
            owning_fragment = self.atom_to_fragment_dict[orbital_system.vos[0].atom_idx]
            atom_idx1 = orbital_system.get_atoms()

            for candidate_interaction_partner in self.active_orbital_systems:
                # if intrafragment interactions, we only consider atoms that are adjacent
                if owning_fragment == self.atom_to_fragment_dict[candidate_interaction_partner.vos[0].atom_idx]:
                    atom_idx2 = candidate_interaction_partner.get_atoms()                   
                    if bool(set(set(atom_idx1) & set(atom_idx2))): # no overlap
                        continue
                    for idx in atom_idx1:
                        if bool(set(neighbors_dict[idx]) & set(atom_idx2)): # if any of the idx in the candidate partner is in neighbor set => adjacent orbital systems
                            self.orbital_system_graph.add_potential_intrafragment_interaction(orbital_system, candidate_interaction_partner)
                        else:
                            continue
                # for interfragment interactions, we distinguish with path ending, and regular interfragment interactions
                else:
                    if len(atom_idx2) > 1:
                        self.orbital_system_graph.add_potential_interfragment_interaction(orbital_system, candidate_interaction_partner)
                    else:
                        self.orbital_system_graph.add_potential_interfragment_path_ending_interaction(orbital_system, candidate_interaction_partner)

    def construct_reaction_paths(self, max_length):
        # start by generating all the intrafragment paths
        all_intrafragment_paths = [[] for _ in self.numbered_smiles.split('.')]
        # first, get all the paths consisting of a single orbital system
        for orbital_system in self.active_orbital_systems:
            all_intrafragment_paths[self.atom_to_fragment_dict[orbital_system.get_atoms()[0]]].append([orbital_system])

        # iterate through the paths, if no "natural" end-point, i.e., a single VO or a neighbor of the first orbital system, is reached, 
        # then continue appending orbital systems
        for fragment_paths in all_intrafragment_paths:
            previous_length = 0
            for _ in range(max_length-1):
                paths_to_extend = fragment_paths[previous_length:]
                previous_length = len(fragment_paths)
                for path in paths_to_extend:
                    if len(path[-1].vos) > 1: # and not path[0] in self.orbital_system_graph.get_neighbors(path[-1]): # TODO: this would stop everytime you add a second neighbor
                        for neighbor in self.orbital_system_graph.get_intrafragment_neighbors(path[-1]):
                            if neighbor not in path:
                                new_path = deepcopy(path)
                                new_path.append(neighbor)
                                fragment_paths.append(new_path)

        #print(self.numbered_smiles)
        #for i, fragment_paths in enumerate(all_intrafragment_paths):
        #    print(i)
        #    for j, path in enumerate(fragment_paths):
        #        print(j, path)

        #print(all_intrafragment_paths)
        #fraise KeyError

        #TODO: double check that everything works as intended
        # now we combine intrafragment paths
        all_interfragment_paths = []
        # first permutate the fragment order
        potential_fragment_arrangements = list(permutations(list(range(len(all_intrafragment_paths))), len(all_intrafragment_paths)))
        for arrangement in potential_fragment_arrangements:
            intrafragment_paths_tmp = [all_intrafragment_paths[i].copy() for i in arrangement]
            # invert the final fragment pathways
            for path in intrafragment_paths_tmp[-1]:
                path = path[::-1]

            # now make all the possible combinations within the selected fragment order
            all_combinations_list = list(product(*intrafragment_paths_tmp))
            for combination in all_combinations_list:
                new_interfragment_path = []
                for fragment in combination:
                    new_interfragment_path += fragment
                if not any(len(bonding_system) < 2 for bonding_system in new_interfragment_path[1:]):
                    all_interfragment_paths.append(new_interfragment_path)
                else:
                    continue
                if len(new_interfragment_path[0].vos) == 1 and len(new_interfragment_path[-1].vos) != 1:
                    for orbital_system in self.orbital_system_graph.get_interfragment_path_ending_neighbors(new_interfragment_path[-1]):
                        if str(orbital_system) not in str(new_interfragment_path): # you made copies, so you need to take string to compare the systems
                            new_interfragment_path_alt = new_interfragment_path + [orbital_system]
                            all_interfragment_paths.append(new_interfragment_path_alt)

        return all_interfragment_paths

    def modify_reaction_paths(self, all_paths):
        products = []
        unique_products = set()
        modified_paths = []
        for path in tqdm(all_paths):
            modified_path = []
            vos = []
            for orbital_system in path:
                vos += deepcopy(orbital_system.vos)
            if len(path[0].vos) == 1 and len(path[-1].vos) == 1:
                for i in range(0, len(vos), 2):
                    modified_path += [construct_new_paired_orbital_system(vos[i], vos[i+1])]
            elif len(path[0].vos) == 1 and len(path[-1].vos) == 2:
                if path[0].vos[0].num_electrons == path[-1].vos[0].num_electrons:
                    continue
                else:
                    for i in range(0, len(vos) - 1, 2):
                        modified_path += [construct_new_paired_orbital_system(vos[i], vos[i+1])]
                    if path[0].vos[0].num_electrons == 2:
                        modified_path += [construct_new_single_vo_system(vos[-1], vos[-1].num_electrons + 1)]
                    elif path[0].vos[0].num_electrons == 0:
                        modified_path += [construct_new_single_vo_system(vos[-1], vos[-1].num_electrons - 1)]
                    else:
                        continue
                    modified_paths.append(modified_path)
            elif len(path[0].vos) == 2 and len(path[-1].vos) == 2:
                print(vos[-1].num_electrons, vos[0].num_electrons)
                for i in range(1, len(vos)-2, 2):
                    modified_path += [construct_new_paired_orbital_system(vos[i], vos[i+1])]
                # concerted reaction
                modified_paths.append(modified_path + [construct_new_paired_orbital_system(vos[0], vos[-1])])
                # ionic path
                modified_paths.append(modified_path + [construct_new_single_vo_system(vos[-1], 0)] \
                                      + [construct_new_single_vo_system(vos[0], 2)])
                print(self.numbered_smiles)
                print(modified_path + [construct_new_single_vo_system(vos[-1], 0)] \
                                      + [construct_new_single_vo_system(vos[0], 2)])
            else:
                print('This path type has not yet been implemented!')

            for modified_path in modified_paths:
                smiles = generate_smiles(
                        self.orig_mol,
                        path,
                        modified_path,
                    )
                if smiles != None:
                    smiles_without_numbering = clear_numbering(smiles)
                    if smiles_without_numbering not in unique_products:
                        print(path)
                        print()
                        print(modified_path)
                        print()
                        print(smiles)
                        unique_products.add(smiles_without_numbering)
                        products.append(smiles)
        
        products = list(set(products))

        print(self.numbered_smiles)
        print(products)
        print(len(products))
        return products

    # TODO: Clean this up!
    def modify_specific_path(self):
        path = [self.orbital_systems[15], self.orbital_systems[0], 
                self.orbital_systems[6], self.orbital_systems[11]]
        print(path)
        print(self.numbered_smiles)
        modified_path = []
        vos = []
        for orbital_system in path:
            vos += deepcopy(orbital_system.vos)
        if len(path[0].vos) == 1 and len(path[-1].vos) == 1:
            for i in range(0, len(vos), 2):
                modified_path += [construct_new_paired_orbital_system(vos[i], vos[i+1])]
        elif len(path[0].vos) == 1 and len(path[-1].vos) == 2:
            if path[0].vos[0].num_electrons == path[-1].vos[0].num_electrons:
                print('lol')
            else:
                for i in range(0, len(vos) - 1, 2):
                    modified_path += [construct_new_paired_orbital_system(vos[i], vos[i+1])]
                if path[0].vos[0].num_electrons == 2:
                    modified_path += [construct_new_single_vo_system(vos[-1], vos[-1].num_electrons + 1)]
                elif path[0].vos[0].num_electrons == 0:
                    modified_path += [construct_new_single_vo_system(vos[-1], vos[-1].num_electrons - 1)]
                else:
                    print('lol2')
        elif len(path[0].vos) == 2 and len(path[-1].vos) == 2:
            for i in range(1, len(vos)-2, 2):
                modified_path += [construct_new_paired_orbital_system(vos[i], vos[i+1])]
            modified_path += [construct_new_paired_orbital_system(vos[0], vos[-1])]
        else:
            print('This path type has not yet been implemented!')

        print(modified_path)
        smiles = generate_smiles(
                    self.orig_mol,
                    path,
                    modified_path,
                )
        print(smiles) 

    def determine_atom_to_fragment_dict(self):
        atom_to_fragment_dict = {}
        fragment_smiles_list = self.numbered_smiles.split('.')
        for i, fragment_smiles in enumerate(fragment_smiles_list):
            atom_numbers = re.findall(r':(\d+)\]', fragment_smiles)
            for atom_number in atom_numbers:
                atom_to_fragment_dict[int(atom_number)] = i
    
        return atom_to_fragment_dict
    
    def get_neighbors_dict(self):
        return {atom.GetAtomMapNum(): [neighbor.GetAtomMapNum()
            for neighbor in atom.GetNeighbors()] for atom in self.orig_mol.GetAtoms()}
    
    def add_orbital_systems_to_graph(self):
        for orbital_system in self.orbital_systems:
            self.orbital_system_graph.add_orbital_system(orbital_system)


def remove_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item not in unique_list and item[::-1] not in unique_list:
            unique_list.append(item)
    return unique_list

# TODO: update!!!!
def set_polarization_bonding_systems(bonding_systems):
    """
    Sets an ordering of the bonding systems.

    Args:
        bonding_systems (list): list of bonding systems.
    """
    for bonding_system in bonding_systems:
        if len(bonding_system) > 1:
            bonding_system.set_polarity()


def construct_new_paired_orbital_system(vo1, vo2, idx=-1):
    """Auxiliary function to construct a new paired orbital system from 2 existing vos.

    Args:
        vo1 (ValenceOrbital): first valence orbital object.
        vo2 (ValenceOrbital): second valence orbital object.
        idx (int): the index of the bonding system to be formed.

    Returns:
        OrbitalSystem: the new bonding system.
    """
    new_orbital_system = OrbitalSystem(idx)
    vo1.set_population(1)
    vo2.set_population(1)
    new_orbital_system.add_vo(vo1)
    new_orbital_system.add_vo(vo2)

    return new_orbital_system

def construct_new_single_vo_system(vo, pop, idx=-1):
    new_orbital_system = OrbitalSystem(idx)
    vo.set_population(pop)
    new_orbital_system.add_vo(deepcopy(vo))

    return new_orbital_system

def clear_numbering(smiles):
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)















