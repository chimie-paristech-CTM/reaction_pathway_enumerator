from typing import Dict, List
from rdkit import Chem
import numpy as np
import re


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


class OrbitalSystem:
    """A class corresponding to individual orbital systems."""

    def __init__(self, idx: int):
        self.idx = idx
        self.vos = []
        self.num_electrons = 0
        self.polarity_set = False

    def add_vo(self, vo: "ValenceOrbital"):
        """Add valence orbitals to the bonding system (and modify the pairing bools accordingly)."""
        self.vos.append(vo)
        self.num_electrons += vo.num_electrons
        if len(self.vos) == 1:
            self.vos[0].set_unpaired()
        else:
            for vo in self.vos:
                vo.set_paired()

    def set_polarity(self):
        """
        If electronegativity spread bigger than, or equal to, 0.4, set the vo order of the bonding system, 
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

    def reverse_vo_order(self):
        """Reverses the order of the vos."""
        if len(self.vos) == 2:
            self.vos.reverse()

    def is_lone_pair(self):
        """Returns True if bonding system corresponds to lone pair."""
        if len(self.vos) == 1 and self.num_electrons == 2:
            return True
        else:
            False
        
    def is_empty_valence(self):
        """Returns True if bonding system corresponds to empty valence."""
        if len(self.vos) == 1 and self.num_electrons == 0:
            return True
        else:
            return False

    def is_xh_bond(self):
        """Returns True if bonding system corresponds to an X-H bond."""
        return len(self.get_heavy_atoms()) != len(self.vos)

    def get_heavy_atoms(self):
        """Returns a list of heavy atom indices present in the bonding system."""
        return [vo.atom_idx for vo in self.vos if vo.atom_type != "H"]
    
    def get_atoms(self):
        return [vo.atom_idx for vo in self.vos]

    def __str__(self) -> str:
        return f"idx: {self.idx}; vos: {[str(vo) for vo in self.vos]}"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.vos)


class OrbitalSystemGraph:
    """ A class corresponding to an abstract graph, where nodes correspond to orbital systems, and edges correspond to potential interactions. """

    def __init__(self):
        self.potential_interaction_list = {}
        self.xh_orbital_systems = set()

    def add_orbital_system(self, orbital_system):
        if orbital_system not in self.potential_interaction_list:
            self.potential_interaction_list[orbital_system] = []
            if orbital_system.is_xh_bond():
                self.xh_orbital_systems.add(orbital_system)
            elif orbital_system.is_lone_pair():
                self.lone_pairs.add(orbital_system)
            elif orbital_system.is_empty_valence():
                self.empty_valences.add(orbital_system)
    
    def add_potential_interaction(self, source, destination):
        if source in self.potential_interaction_list and destination in self.potential_interaction_list and destination not in self.potential_interaction_list[source]:
            self.potential_interaction_list[source].append(destination)
    
    def get_neighbors(self, orbital_system, with_xh=True):
        if with_xh == True:
            return self.potential_interaction_list.get(orbital_system, []) # TODO: refine this!!!!
        else:
            neighbors = self.potential_interaction_list.get(orbital_system, [])
            return  [neighbor for neighbor in neighbors if neighbor not in self.xh_orbital_systems] 

    def __str__(self) -> str:
        return f'{self.potential_interaction_list}'


# TODO: delocalization???
class Molecule:
    """A class corresponding to molecular systems (can consist of multiple molecules)"""

    def __init__(self, smiles: str):
        self.orig_mol, self.numbered_smiles = self.parse_smiles(smiles)
        
        self.num_atoms = self.orig_mol.GetNumAtoms()
        self.atoms = self.get_atoms()

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
        active_orbital_systems = set()
        already_covered_systems = set()
        for orbital_system in self.orbital_systems:
            system_info = f'{orbital_system.num_electrons}, {set(orbital_system.get_atoms())}'
            if system_info not in already_covered_systems:
                already_covered_systems.add(system_info)
                active_orbital_systems.add(orbital_system)
    
        return active_orbital_systems

    def add_potential_interactions_to_graph(self):
        # first determine which atoms belong to which fragment of the molecular system
        atom_to_fragment_dict = self.determine_atom_to_fragment_dict()
        # also get the neighbors for every atom
        neighbors_dict = self.get_neighbors_dict()

        # now construct all the interactions
        for orbital_system in self.active_orbital_systems:
            owning_fragment = atom_to_fragment_dict[orbital_system.vos[0].atom_idx]
            atom_idx1 = orbital_system.get_atoms()

            for candidate_interaction_partner in self.active_orbital_systems:
                # if different fragment, add interaction
                if owning_fragment != atom_to_fragment_dict[candidate_interaction_partner.vos[0].atom_idx]:
                    self.orbital_system_graph.add_potential_interaction(orbital_system, candidate_interaction_partner)
                else:
                    # on same fragment, the atoms need to be adjacent
                    atom_idx2 = candidate_interaction_partner.get_atoms()
                    for idx in atom_idx1:
                        if bool(set(neighbors_dict[idx]) & set(atom_idx2)): # if any of the idx in the candidate partner is in neighbor set => adjacent orbital systems
                            self.orbital_system_graph.add_potential_interaction(orbital_system, candidate_interaction_partner)
                        else:
                            continue

    def construct_orbital_system_paths(self, max_length:int):
        all_paths = []
        # first, get all the paths consisting of a single orbital system
        for orbital_system in self.active_orbital_systems:
            all_paths.append([orbital_system])

        # iterate through the paths, if no "natural" end-point, i.e., a single VO or a neighbor of the first orbital system, is reached, 
        # then continue appending orbital systems
        previous_length = 0
        for _ in range(max_length-2):
            paths_to_extend = all_paths[previous_length:]
            previous_length = len(all_paths)
            for path in paths_to_extend:
                if len(path[-1].vos) == 2: # and not path[0] in self.orbital_system_graph.get_neighbors(path[-1]): # TODO: this would stop everytime you add a second neighbor
                    for neighbor in self.orbital_system_graph.get_neighbors(path[-1]):
                        if neighbor not in path:
                            new_path = path.copy()
                            new_path.append(neighbor)
                            all_paths.append(new_path)

        # finish the path construction by adding endpoints if they are not already there
        for path in all_paths[previous_length:]:
            if len(path[-1].vos) == 2 and not path[0] in self.orbital_system_graph.get_neighbors(path[-1]):
                if path[0].is_lone_pair():
                    for empty_valence in self.orbital_system_graph.empty_valences:
                        new_path = path.copy()
                        new_path.append(empty_valence)
                        all_paths.append(new_path)
                elif path[0].is_empty_valence():
                    for lone_pair in self.orbital_system_graph.lone_pairs:
                        new_path = path.copy()
                        new_path.append(lone_pair)
                        all_paths.append(new_path)
                else:
                    for neighbor in self.orbital_system_graph.get_neighbors(path[-1]):
                        if neighbor not in path:
                            new_path = path.copy()
                            new_path.append(neighbor)
                            all_paths.append(new_path)
        

        print(len(all_paths))
        return all_paths

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






















