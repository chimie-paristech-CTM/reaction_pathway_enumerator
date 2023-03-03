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
        n_doubly_occ = self.num_valence_electrons // (self.num_valence_orbitals + 1)
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
        self.polarity = None

    def add_vo(self, vo: 'ValenceOrbital'):
        """ Add valence orbitals to the bonding system (and modify the pairing bools accordingly). """
        self.vos.append(vo)
        if len(self.vos) == 1:
            self.vos[0].set_unpaired()
        else:
            for vo in self.vos:
                vo.set_paired()

    def set_polarity(self):
        """ 
        Sets self.polarity to a dictionary with keys 'pos_pole' and 'neg_pole' 
        if electronegativity spread bigger than 0.4. Values are specific vos.
        """

        if len(self.vos) > 1:
            electronegativity_list = [electro_dict[vo.atom_type] for vo in self.vos]
            if max(electronegativity_list) - min(electronegativity_list) >= 0.4:
                self.polarity = {'pos_pole': self.vo[np.argmax(electronegativity_list)], 
                                'neg_pole': self.vo[np.argmin(electronegativity_list)]}

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

        self.atoms = []
        self.bonding_systems = []

        # Process rdkit_atoms, add them to the editable version of the molecule, and create Atom objects
        rd_periodic_table = Chem.GetPeriodicTable()
        for idx, atom in enumerate(self.orig_molecule.GetAtoms()):
            atom.SetIsAromatic(False)  # remove aromaticity properties
            num_valence_electrons = rd_periodic_table.GetNOuterElecs(atom.GetSymbol()) - atom.GetFormalCharge()
            self.atoms.append(Atom(molecule=self, atom_type=atom.GetSymbol(), idx=idx, num_valence_electrons=num_valence_electrons))

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
                    self.bonding_systems.append(new_bonding_system)
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
                    self.bonding_systems.append(new_bonding_system)
                    bonding_system_idx += 1


def permutate_bonding_systems(molecule: 'Molecule', idx_list: list):
    """ Permutate a subset of the bonding systems.

    Args:
        molecule (Molecule): a molecule object.
        idx_list (list): the list of bonding systems that need to be permutated.

    Returns:
        old_bonding_systems (list): (copy of) the old bonding systems.
        new_bonding_systems (list): new bonding systems formed through permutation.
    """
    bonding_system_init = molecule.bonding_systems[idx_list[0]]
    reaction_path = [vo for vo in bonding_system_init.vos]
    for idx in idx_list[1:-1]:
        if len(molecule.bonding_systems[idx].vos) == 1: # you don't want a lone pair or empty orbital halfway a sequence
            return None, None
        next_bonding_system = molecule.bonding_systems[idx]
        reaction_path += [vo for vo in next_bonding_system.vos]
    bonding_system_end = molecule.bonding_systems[idx_list[-1]]
    reaction_path += [vo for vo in bonding_system_end.vos]

    atom_list = [vo.atom_idx for vo in reaction_path]
    if len(atom_list) != len(set(atom_list)): # you don't want two vos on the same atom to be part of a single reactive event
        return None, None



    # save a copy of the bonding systems being modified
    old_bonding_systems = [copy.deepcopy(molecule.bonding_systems[idx]) for idx in idx_list]

    num_electrons = sum([vo.num_electrons for vo in reaction_path])
    new_bonding_systems = []
    if len(reaction_path) % 2 == 0:
        for i in range(1, len(reaction_path) - 1, 2):
            new_bonding_system = construct_new_bonding_system(reaction_path[i], reaction_path[i+1])
            new_bonding_systems.append(new_bonding_system)
        new_bonding_system = construct_new_bonding_system(reaction_path[0], reaction_path[-1])
        new_bonding_systems.append(new_bonding_system)
    elif len(reaction_path) % 2 == 1:
        if len(bonding_system_init) == 1:
            for i in range(len(reaction_path) - 1, 2):
                new_bonding_system = construct_new_bonding_system(reaction_path[i], reaction_path[i+1]) 
                new_bonding_systems.append(new_bonding_system)
            new_bonding_system = BondingSystem(-1)
            reaction_path[-1].set_population(num_electrons - (len(reaction_path) - 1))
            new_bonding_system.add_vo(reaction_path[-1])
            new_bonding_systems.append(new_bonding_system)
        if len(bonding_system_end) == 1:
            for i in range(1, len(reaction_path),2):
                new_bonding_system = construct_new_bonding_system(reaction_path[i], reaction_path[i+1])
                new_bonding_systems.append(new_bonding_system)
            new_bonding_system = BondingSystem(-1)
            reaction_path[0].set_population(num_electrons - (len(reaction_path) - 1))
            new_bonding_system.add_vo(reaction_path[0])
            new_bonding_systems.append(new_bonding_system)

    return old_bonding_systems, new_bonding_systems


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
    

def generate_smiles(orig_mol, old_bonding_systems, new_bonding_systems):
    """ Generate an output SMILES string.

    Args:
        orig_mol (rdkit.Mol): the rdkit mol-object corresponding to the input system
        old_bonding_systems (list): (copy of) the old bonding systems.
        new_bonding_systems (list): new bonding systems.

    Returns:
        str: the output SMILES
    """
    editable_molecule = Chem.RWMol(orig_mol) # editable version of the molecule

    for bonding_system in old_bonding_systems:
        if len(bonding_system) == 1:
            init_charge = editable_molecule.GetAtomWithIdx(bonding_system.vos[0].atom_idx).GetFormalCharge()
            if bonding_system.vos[0].num_electrons == 2:
               editable_molecule.GetAtomWithIdx(bonding_system.vos[0].atom_idx).SetFormalCharge(init_charge + 1)
            elif bonding_system.vos[0].num_electrons == 0:
                editable_molecule.GetAtomWithIdx(bonding_system.vos[0].atom_idx).SetFormalCharge(init_charge - 1)
            elif bonding_system.vos[0].num_electrons == 1:
                num_radicals = editable_molecule.GetAtomWithIdx(bonding_system.vos[0].atom_idx).GetNumRadicalElectrons()
                editable_molecule.GetAtomWithIdx(bonding_system.vos[0].atom_idx).SetNumRadicalElectrons(num_radicals - 1) 
        else:
            current_bond = editable_molecule.GetBondBetweenAtoms(bonding_system.vos[0].atom_idx, bonding_system.vos[1].atom_idx)
            if current_bond.GetBondType() is Chem.rdchem.BondType.SINGLE:
                editable_molecule.RemoveBond(bonding_system.vos[0].atom_idx, bonding_system.vos[1].atom_idx)
            elif current_bond.GetBondType() is Chem.rdchem.BondType.DOUBLE:
                editable_molecule.RemoveBond(bonding_system.vos[0].atom_idx, bonding_system.vos[1].atom_idx)
                editable_molecule.AddBond(bonding_system.vos[0].atom_idx, bonding_system.vos[1].atom_idx, Chem.rdchem.BondType.SINGLE)
            elif current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
                editable_molecule.RemoveBond(bonding_system.vos[0].atom_idx, bonding_system.vos[1].atom_idx)
                editable_molecule.AddBond(bonding_system.vos[0].atom_idx, bonding_system.vos[1].atom_idx, Chem.rdchem.BondType.DOUBLE)
    
    for bonding_system in new_bonding_systems:
        if len(bonding_system) == 1:
            init_charge = editable_molecule.GetAtomWithIdx(bonding_system.vos[0].atom_idx).GetFormalCharge()
            if bonding_system.vos[0].num_electrons == 2:
               editable_molecule.GetAtomWithIdx(bonding_system.vos[0].atom_idx).SetFormalCharge(init_charge - 1)
            elif bonding_system.vos[0].num_electrons == 0:
                editable_molecule.GetAtomWithIdx(bonding_system.vos[0].atom_idx).SetFormalCharge(init_charge + 1) 
            elif bonding_system.vos[0].num_electrons == 1:
                num_radicals = editable_molecule.GetAtomWithIdx(bonding_system.vos[0].atom_idx).GetNumRadicalElectrons()
                editable_molecule.GetAtomWithIdx(bonding_system.vos[0].atom_idx).SetNumRadicalElectrons(num_radicals + 1) 
        else:
            current_bond = editable_molecule.GetBondBetweenAtoms(bonding_system.vos[0].atom_idx, bonding_system.vos[1].atom_idx)
            if current_bond is None:
                editable_molecule.AddBond(bonding_system.vos[0].atom_idx, bonding_system.vos[1].atom_idx, Chem.rdchem.BondType.SINGLE) 
            elif current_bond.GetBondType() is Chem.rdchem.BondType.SINGLE:
                editable_molecule.RemoveBond(bonding_system.vos[0].atom_idx, bonding_system.vos[1].atom_idx)
                editable_molecule.AddBond(bonding_system.vos[0].atom_idx, bonding_system.vos[1].atom_idx, Chem.rdchem.BondType.DOUBLE) 
            elif current_bond.GetBondType() is Chem.rdchem.BondType.DOUBLE:
                editable_molecule.RemoveBond(bonding_system.vos[0].atom_idx, bonding_system.vos[1].atom_idx)
                editable_molecule.AddBond(bonding_system.vos[0].atom_idx, bonding_system.vos[1].atom_idx, Chem.rdchem.BondType.TRIPLE)
            elif current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
                pass # This should likely give an error

    return Chem.MolToSmiles(editable_molecule)


if __name__ == '__main__':
    mol = Molecule('C.C#[N+]')
    old_bonding_systems, new_bonding_systems = permutate_bonding_systems(mol, [1,8])
    smiles = generate_smiles(mol.orig_molecule, old_bonding_systems, new_bonding_systems)
    print(mol.smi, smiles)
