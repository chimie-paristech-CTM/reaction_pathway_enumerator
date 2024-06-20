import re
from typing import Dict, List
from enumerator.orbital_systems import LocalizedOrbitalSystem
from enumerator.utils import ordering_smiles
from enumerator.utils_nbo import extract_2nd_interaction_dict
from rdkit import Chem

metal_symbols = ["Al", "Fe", "Cu", "Au", "Ag",  "Zn", "Ni",  "Sn",  "Pb",  "Pt",  "Hg",  "Ti", "Co",
    "Cr",  "Mg",  "Mn",  "W",   "Bi",  "Sb",  "Cd",  "V",   "U",   "Pd",  "Rh",  "Ru"]

extra_valence_symbols = ["P", "S", "Cl", "As", "Se",  "Br", "Sb",  "Te",  "I"]


# TODO: metals added in principle, but this won't work well until you have a good description of the bonding in the graph

# TODO: What about 3rd row elements (and upper) when you have availability of d orbitals (P 5 bonds and S 6 bonds)
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
        self.lone_pair_idx = None

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

    def set_lone_pair_idx(self, lp_idx):
        """Sets the lone pair index from an NBO calculation."""
        self.lone_pair_idx = lp_idx

    def __str__(self) -> str:
        return (
            f"self.atom_idx: {str(self.atom_idx)}; self.idx: {str(self.idx)};"
            f" self.num_electrons: {self.num_electrons}; self.paired: {self.paired}"
        )

    def __repr__(self) -> str:
        return self.__str__()


class AtomNBO:
    """A class corresponding to individual atoms."""

    def __init__(
            self, molecule: "Molecule", atom, atom_type: str, idx: int, num_valence_electrons: int,
            metal=False
    ):
        self.molecule = molecule
        self.atom = atom
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

        if self.metal:  # for metals, also add the empty p orbitals of the next shell
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


class LocalizedConfigurationNBO:
    def __init__(self, numbered_smiles, atoms, nbo_lines):
        self.orbital_systems_list, self.raw_smiles = self.set_up_localized_orbital_systems(numbered_smiles, atoms, nbo_lines)
        self.secondary_interactions = extract_2nd_interaction_dict(numbered_smiles, nbo_lines)
        self.active_orbital_systems_list = self.select_active_orbital_systems()
        self.vo_list = self.set_vo_list()
        self.vo_to_orbital_system_dict = self.get_vo_to_orbital_system_dict()
        self.bonding_antibonding_system_list = self.map_bonding_antibonding(numbered_smiles, nbo_lines)


    # TODO: what about circular 3c bonds (e.g., interaction between ethylene and PdL2)?
    # TODO: should you include validity checks to ensure that the localized configuration makes sense (e.g., exotic boding situations resulting in incorrect vo pairing)?
    def set_up_localized_orbital_systems(self, numbered_smiles, atoms, nbo_lines):
        """Construct the initial orbital systems (either 1, 2 or 3 vos in a linear arrangment)."""
        orbital_systems = []
        initial_bonds: Dict[int, List[int]] = dict()
        lone_pairs: Dict[int, List[int]] = dict()

        smiles_list = numbered_smiles.split('.')

        for idx, smiles in enumerate(smiles_list):

            ordered_smiles = ordering_smiles(smiles)

            line_0 = " ------------------ Lewis ------------------------------------------------------\n"
            line_1 = " ---------------- non-Lewis ----------------------------------------------------\n"
            idx_0 = nbo_lines[idx].index(line_0)
            idx_1 = nbo_lines[idx].index(line_1)

            for line in nbo_lines[idx][idx_0 + 1: idx_1]:

                if 'BD' in line:
                    atom_1 = int(line[25:28])
                    atom_2 = int(line[31:34])
                    atom_1_in_numbered_smiles = int(ordered_smiles[atom_1 - 1].split(':')[-1])
                    atom_2_in_numbered_smiles = int(ordered_smiles[atom_2 - 1].split(':')[-1])
                    if atom_1 < atom_2:
                        initial_bonds[atom_1_in_numbered_smiles] = initial_bonds.get(atom_1_in_numbered_smiles, []) + [
                            atom_2_in_numbered_smiles]
                    else:
                        initial_bonds[atom_2_in_numbered_smiles] = initial_bonds.get(atom_2_in_numbered_smiles, []) + [
                            atom_1_in_numbered_smiles]

                if 'LP' in line:
                    atom = int(line[25:28])
                    lp_idx = int(line[0:4])
                    atom_in_numbered_smiles = int(ordered_smiles[atom - 1].split(':')[-1])
                    lone_pairs[atom_in_numbered_smiles] = lone_pairs.get(atom_in_numbered_smiles, []) + [lp_idx]

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
                    if vo.num_electrons == 2:
                        lp_idxs = lone_pairs[atom.idx]
                        lp_idx = lp_idxs.pop()
                        vo.set_lone_pair_idx(lp_idx)
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
                            for partner_vo in atoms[neighbor_idx - 1].valence_orbitals:
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

        rd_periodic_table = Chem.GetPeriodicTable()
        new_mol = Chem.Mol()
        ed_new_mol = Chem.RWMol(new_mol)
        for atom in atoms:
            z = atom.atom.GetAtomicNum()
            symbol = atom.atom.GetSymbol()
            new_atom_idx = ed_new_mol.AddAtom(Chem.Atom(z))
            new_atom = ed_new_mol.GetAtomWithIdx(new_atom_idx)
            new_atom.SetAtomMapNum(atom.idx)
            formal_charge = (
                    rd_periodic_table.GetNOuterElecs(symbol)
                    - atom.num_valence_electrons
            )
            new_atom.SetFormalCharge(formal_charge)

        rdkit_bond_dict = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}

        for idx_1 in initial_bonds.keys():
            for idx_2 in initial_bonds[idx_1]:
                bond_type = rdkit_bond_dict[initial_bonds[idx_1].count(idx_2)]
                if ed_new_mol.GetBondBetweenAtoms(idx_1 - 1, idx_2 - 1) == None:
                    ed_new_mol.AddBond(idx_1 - 1, idx_2 - 1, bond_type)


        return orbital_systems, Chem.MolToSmiles(ed_new_mol)

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

    def map_bonding_antibonding(self, numbered_smiles, nbo_lines):

        smiles_list = numbered_smiles.split('.')
        map_bonds = {}

        for idx, smiles in enumerate(smiles_list):

            line_0 = " ------------------ Lewis ------------------------------------------------------\n"
            line_1 = " ---------------- non-Lewis ----------------------------------------------------\n"
            line_2 = " NHO DIRECTIONALITY AND BOND BENDING (deviation from line of nuclear centers at\n"
            idx_0 = nbo_lines[idx].index(line_0)
            idx_1 = nbo_lines[idx].index(line_1)
            idx_2 = nbo_lines[idx].index(line_2)

            bond_antibond_pairs = []

            for line in nbo_lines[idx][idx_0: idx_1]:

                if 'BD' in line:
                    bond_idx = int(line[0:4])
                    atom_1_bd = int(line[25:28])
                    atom_2_bd = int(line[31:34])

                    for line in nbo_lines[idx][idx_1: idx_2]:
                        if 'BD*' in line:
                            atom_1_antibd = int(line[25:28])
                            atom_2_antibd = int(line[31:34])
                            if (atom_1_antibd == atom_1_bd) and (atom_2_antibd == atom_2_bd):
                                antibond_idx = int(line[0:4])
                                bond_antibond_pairs.append((bond_idx, antibond_idx))
                                break

            map_bonds[idx] = bond_antibond_pairs

        return map_bonds
