from typing import Optional, Dict, List
from rdkit import Chem
import copy
import numpy as np


electro_dict = {'H': 2.1, 'B': 2.0, 'C': 2.5, 'N': 3.0, 'O': 3.5, 'F': 4.0, 
                'Si': 1.8, 'P': 2.1, 'S': 2.5, 'Cl': 3.0, 'Br': 2.8}


def atom_to_num_VOs(atom_symbol: str) -> int:
    """Returns the number of VOs the atom should be initialized with."""
    if atom_symbol == 'H' or atom_symbol == 'He':
        return 1
    else:
        return 4


class ValenceOrbital:
    """ A class corresponding to individual valence orbitals. """
    def __init__(self, idx: int, atom_idx: int, atom_type: str):
        self.idx = idx
        self.atom_idx = atom_idx
        self.atom_type = atom_type
        self.num_electrons = 0
        self.paired = False

    def set_population(self, num_electrons):
        """ Sets the number of electrons present in the valence orbital. """
        self.num_electrons = num_electrons

    def set_paired(self):
        """ Sets self.paired-bool to indicate that the valence orbital is paired. """
        self.paired = True

    def set_unpaired(self):
        """ Sets self.paired-bool to indicate that the valence orbital is unpaired. """
        self.paired = False

    def __str__(self) -> str:
        return f"self.atom_idx: {str(self.atom_idx)}; self.idx: {str(self.idx)};" \
            f" self.num_electrons: {self.num_electrons}; self.paired: {self.paired}"

    def __repr__(self) -> str:
        return self.__str__()


class Atom:
    """ A class corresponding to individual atoms. """
    def __init__(self, molecule: 'Molecule', atom_type: str, idx: int, num_valence_electrons: int):
        self.molecule = molecule
        self.atom_type = atom_type
        self.idx = idx
        self.valence_orbitals = []
        self.num_valence_orbitals = atom_to_num_VOs(self.atom_type)
        self.num_valence_electrons = num_valence_electrons
    
        for vo_idx in range(self.num_valence_orbitals):
            self.valence_orbitals.append(ValenceOrbital(vo_idx, self.idx, self.atom_type))
        
        self.occupy_vos()

    def occupy_vos(self):
        """ Occupies the valence orbitals associated with the atom, based on the number of valence electrons present. """
        n_doubly_occ = self.num_valence_electrons % (self.num_valence_orbitals)
        n_singly_occ = self.num_valence_electrons - n_doubly_occ * 2

        for vo in self.valence_orbitals:
            if vo.idx < n_singly_occ:
                vo.set_population(1)
            elif vo.idx < n_singly_occ + n_doubly_occ:
                vo.set_population(2)


class BondingSystem:
    """ A class corresponding to individual bonding systems. """
    def __init__(self, idx: int):
        self.idx = idx
        self.vos = []
        self.num_electrons = 0
        self.polarity_set = False

    def add_vo(self, vo: 'ValenceOrbital'):
        """ Add valence orbitals to the bonding system (and modify the pairing bools accordingly). """
        self.vos.append(vo)
        self.num_electrons += vo.num_electrons
        if len(self.vos) == 1:
            self.vos[0].set_unpaired()
        else:
            for vo in self.vos:
                vo.set_paired()

    def set_polarity(self):
        """ 
        If electronegativity spread bigger than 0.4, set the vo order of the bonding system, with the negative pole first. 
        Values are specific vos.
        """
        if len(self.vos) > 1:
            electronegativity_list = [electro_dict[vo.atom_type] for vo in self.vos]
            if max(electronegativity_list) - min(electronegativity_list) >= 0.395:
                self.vos = [self.vos[np.argmax(electronegativity_list)], self.vos[np.argmin(electronegativity_list)]]
                self.polarity_set = True

    def reverse_vo_order(self):
        """ Reverses the order of the vos. """
        if len(self.vos) == 2:
            self.vos.reverse()

    def is_lone_pair(self):
        """ Returns True if bonding system corresponds to lone pair. """
        if len(self.vos) == 1 and self.num_electrons == 2:
            return True
        
    def is_xh_bond(self):
        """ Returns True if bonding system corresponds to an X-H bond. """
        return len(self.get_heavy_atoms()) != len(self.vos)
    
    def get_heavy_atoms(self):
        """ Returns a list of heavy atom indices present in the bonding system. """
        return [vo.atom_idx for vo in self.vos if vo.atom_type != 'H']

    def __str__(self) -> str:
        return f"idx: {self.idx}; vos: {[str(vo) for vo in self.vos]}"

    def __repr__(self) -> str:
        return self.__str__()
    
    def __len__(self) -> int:
        return len(self.vos)


class Molecule:
    """ A class corresponding to molecular systems (can consist of multiple molecules) """
    def __init__(self, smi: str):
        self.smi = smi
        self.orig_molecule = Chem.MolFromSmiles(smi)
        self.orig_molecule = Chem.AddHs(self.orig_molecule)  # always add H's to make bonding correct
        Chem.Kekulize(self.orig_molecule)  # change to kekulized smiles to remove aromatic bonds
        self.num_atoms = self.orig_molecule.GetNumAtoms()

        self.atoms = self.get_atoms()
        self.bonding_systems = self.get_bonding_systems()

    def get_atoms(self):
        """ Process rdkit_atoms, add them to the editable version of the molecule, and create Atom objects. """
        atoms = []

        rd_periodic_table = Chem.GetPeriodicTable()
        for idx, atom in enumerate(self.orig_molecule.GetAtoms()):
            atom.SetIsAromatic(False)  # remove aromaticity properties
            num_valence_electrons = rd_periodic_table.GetNOuterElecs(atom.GetSymbol()) - atom.GetFormalCharge()
            atoms.append(Atom(molecule=self, atom_type=atom.GetSymbol(), idx=idx, num_valence_electrons=num_valence_electrons))

        return atoms

    def get_bonding_systems(self):
        """ Construct the initial bonding systems. """
        bonding_systems = []

        # Create adjacency list representation for bonds. Initial_bonds is not symmetric.
        initial_bonds: Dict[int, List[int]] = dict()
        for bond in self.orig_molecule.GetBonds():
            bond.SetIsAromatic(False)  # remove aromaticity properties
            atom_1 = bond.GetBeginAtomIdx()
            atom_2 = bond.GetEndAtomIdx()
            num_bonds = round(bond.GetBondTypeAsDouble())
            initial_bonds[atom_1] = initial_bonds.get(atom_1, []) + [atom_2] * num_bonds

        # construct all the bonding systems
        bonding_system_idx = 0
        for atom in self.atoms:
            if atom.idx in initial_bonds.keys():
                neighbors = initial_bonds[atom.idx].copy()
            else:
                neighbors = None
            for vo in atom.valence_orbitals:
                if vo.num_electrons == 0 or vo.num_electrons == 2:
                    new_bonding_system = BondingSystem(bonding_system_idx)
                    new_bonding_system.add_vo(vo)
                    bonding_systems.append(new_bonding_system)
                    bonding_system_idx += 1
                elif vo.num_electrons == 1 and vo.paired == False:
                    new_bonding_system = BondingSystem(bonding_system_idx)
                    new_bonding_system.add_vo(vo)
                    if neighbors is not None:
                        if len(neighbors) > 0: 
                            neighbor_idx = neighbors.pop()
                            for partner_vo in self.atoms[neighbor_idx].valence_orbitals:
                                if partner_vo.num_electrons == 1 and partner_vo.paired == False:
                                    new_bonding_system.add_vo(partner_vo)
                                    partner_vo.set_paired()
                                    break
                            vo.set_paired()
                    bonding_systems.append(new_bonding_system)
                    bonding_system_idx += 1

        return bonding_systems
