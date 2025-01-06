import re
from typing import Dict, List
from enumerator.orbital_systems import LocalizedOrbitalSystem
from enumerator.utils import ordering_smiles
from enumerator.utils_nbo import extract_secondary_interactions_raw, check_lp_within_secondary_interaction
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
        self.lp_idx = None
        self.lv_idx = None

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
        self.lp_idx = lp_idx

    def set_lone_vacancy_idx(self, lv_idx):
        """Sets the lone vacancy index from an NBO calculation."""
        self.lv_idx = lv_idx

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
            metal=False, n_doubly_occ=None
    ):
        self.molecule = molecule
        self.atom = atom
        self.atom_type = atom_type
        self.idx = idx
        self.valence_orbitals = []
        self.num_valence_orbitals = atom_to_num_VOs(self.atom_type)
        self.num_valence_electrons = num_valence_electrons
        self.metal = metal
        self.n_doubly_occ = n_doubly_occ

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
        if self.n_doubly_occ:
            n_doubly_occ = self.n_doubly_occ
        else:
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
    def __init__(self, numbered_smiles, atoms, nbo_lines, threshold_strong_sec_interaction, organometallic, threshold_sec_interaction):
        self.threshold_ssi = threshold_strong_sec_interaction
        self.mapping_orbital_system_bonds = {}
        self.orbital_systems_list, self.raw_smiles, self.orbital_system_idx = self.set_up_localized_orbital_systems(numbered_smiles, atoms, nbo_lines, organometallic)
        self.secondary_interactions_raw = extract_secondary_interactions_raw(numbered_smiles, nbo_lines, organometallic, threshold_sec_interaction)
        self.active_orbital_systems_list = self.select_active_orbital_systems()
        self.strong_sec_int_orbital_systems_list = set()
        self.vo_list = self.set_vo_list()
        self.vo_to_orbital_system_dict = self.get_vo_to_orbital_system_dict()
        self.secondary_interaction_vos_systems = self.get_secondary_interacting_vos()

    # TODO: what about circular 3c bonds (e.g., interaction between ethylene and PdL2)?
    # TODO: should you include validity checks to ensure that the localized configuration makes sense (e.g., exotic boding situations resulting in incorrect vo pairing)?
    def set_up_localized_orbital_systems(self, numbered_smiles, atoms, nbo_lines, organometallic):
        """Construct the initial orbital systems (either 1, 2 or 3 vos in a linear arrangment)."""
        orbital_systems = []
        initial_bonds: Dict[int, List[int]] = dict()
        lone_pairs: Dict[int, List[int]] = dict()
        lone_vacancy: Dict[int, List[int]] = dict()
        so_orbs: Dict[int, List[int]] = dict()

        smiles_list = numbered_smiles.split('.')

        for idx, smiles in enumerate(smiles_list):
            ordered_smiles = ordering_smiles(smiles, organometallic)

            line_0 = " ------------------ Lewis ------------------------------------------------------\n"
            idx_0 = nbo_lines[idx].index(line_0)

            for line in nbo_lines[idx][idx_0 + 1:]:

                if 'BD*' in line:
                    break

                if 'RY' in line:  #case for single atom
                    break

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

                # lone pair
                if 'LP' in line:
                    atom = int(line[25:28])
                    lp_idx = int(line[20:22])
                    atom_in_numbered_smiles = int(ordered_smiles[atom - 1].split(':')[-1])
                    lone_pairs[atom_in_numbered_smiles] = lone_pairs.get(atom_in_numbered_smiles, []) + [lp_idx]

                # lone vacancy
                if 'LV' in line:
                    atom = int(line[25:28])
                    lv_idx = int(line[20:22])
                    print(atom)
                    atom_in_numbered_smiles = int(ordered_smiles[atom - 1].split(':')[-1])
                    lone_vacancy[atom_in_numbered_smiles] = lone_vacancy.get(atom_in_numbered_smiles, []) + [lv_idx]

                # singly occupied
                if 'SO' in line:
                    atom = int(line[25:28])
                    so_idx = int(line[20:22])
                    atom_in_numbered_smiles = int(ordered_smiles[atom - 1].split(':')[-1])
                    so_orbs[atom_in_numbered_smiles] = so_orbs.get(atom_in_numbered_smiles, []) + [so_idx]

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
                        atom_lp_idx = f"{atom.idx}_{lp_idx}"
                        self.mapping_orbital_system_bonds[
                            atom_lp_idx] = self.mapping_orbital_system_bonds.get(atom_lp_idx, []) + [
                            new_orbital_system]
                    if vo.num_electrons == 0 and atom.idx in lone_vacancy.keys():
                        if lone_vacancy[atom.idx]:
                            lv_idxs = lone_vacancy[atom.idx]
                            lv_idx = lv_idxs.pop()
                            vo.set_lone_vacancy_idx(lv_idxs)
                            atom_lv_idx = f"{atom.idx}#{lv_idx}"
                            self.mapping_orbital_system_bonds[
                                atom_lv_idx] = self.mapping_orbital_system_bonds.get(atom_lv_idx, []) + [
                                new_orbital_system]
                if vo.num_electrons == 1:
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
                            for partner_vo in atoms[neighbor_idx - 1].valence_orbitals:
                                if (
                                        partner_vo.num_electrons == 1
                                        and partner_vo.paired == False
                                ):
                                    new_orbital_system.add_vo(partner_vo)
                                    partner_vo.set_paired()
                                    vo.set_paired()
                                    bond_between_atoms = f"{atom.idx}-{neighbor_idx}"
                                    self.mapping_orbital_system_bonds[
                                        bond_between_atoms] = self.mapping_orbital_system_bonds.get(bond_between_atoms, []) + [new_orbital_system]
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

        return orbital_systems, Chem.MolToSmiles(ed_new_mol), orbital_system_idx

    # TODO: you are losing lone pairs here
    def select_active_orbital_systems(self):
        """ Only keep 1 orbital system of a triple/double bond, and only keep 1 X-H bond for every atom X """
        active_orbital_systems = set()
        already_covered_systems = set()
        for orbital_system in self.orbital_systems_list:
            # these are the minimal characteristics to distinguish between orbital systems
            system_info = f'{orbital_system.get_num_electrons()}, {set(orbital_system.get_heavy_atoms())}, {len(orbital_system.get_atoms())}, {orbital_system.get_lp_idx()}'
            if orbital_system.is_lp():
                lp_idx = orbital_system.vos[0].lp_idx
                if not check_lp_within_secondary_interaction(self.secondary_interactions_raw, lp_idx):
                    already_covered_systems.add(system_info)

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

    def get_secondary_interacting_vos(self):

        secondary_interactions = self.secondary_interactions_raw
        secondary_interaction_vos = []

        if secondary_interactions:
            for interaction in secondary_interactions:
                vos = []
                donor = interaction[0]
                acceptor = interaction[1]
                energy = interaction[2]
                donor_orbital_systems = self.mapping_orbital_system_bonds[donor]
                acceptor_orbital_systems = self.mapping_orbital_system_bonds[acceptor]
                metal_in_donor = False

                for orbital_system in donor_orbital_systems:
                    if orbital_system in self.active_orbital_systems_list:
                        for vo in orbital_system.vos:
                            vos.append(vo)
                            if vo.atom_type in metal_symbols:
                                metal_in_donor = True

                for orbital_system in acceptor_orbital_systems:
                    if orbital_system in self.active_orbital_systems_list:
                        for idx, vo in enumerate(orbital_system.vos):
                            vos.append(vo)

                            # the secondary interaction in metals contain several atoms ... sometimes you need only the metal center and one more ... case 2A cobalt
                            if metal_in_donor and idx == 0:
                                secondary_interaction_vos.append(vos[::-1].copy())
                            if vo.atom_type in metal_symbols:
                                secondary_interaction_vos.append(vos[-2:][::-1].copy())

                if energy > self.threshold_ssi:
                    new_orbital_system = LocalizedOrbitalSystem(self.orbital_system_idx)
                    for vo in vos:
                        new_orbital_system.add_vo(vo)
                    self.orbital_system_idx += 1
                    self.strong_sec_int_orbital_systems_list.add(new_orbital_system)

                secondary_interaction_vos.append(vos)
                extended_secondary_interactions_vos = self.get_extended_secondary_interactions_vos(secondary_interactions)

                if extended_secondary_interactions_vos:
                    for extended_vos in extended_secondary_interactions_vos:
                        secondary_interaction_vos.append(extended_vos)

        return secondary_interaction_vos

    def get_extended_secondary_interactions_vos(self, secondary_interactions):

        all_extended_systems = []

        for idx, first_pair in enumerate(secondary_interactions):
            donor_first = first_pair[0]
            acceptor_first = first_pair[1]
            extended_system = [donor_first, acceptor_first]
            for extended_pair in secondary_interactions[idx + 1:]:
                donor_extended = extended_pair[0]
                acceptor_extended = extended_pair[1]
                if acceptor_first == donor_extended and donor_first != acceptor_extended and acceptor_extended not in extended_system:
                    extended_system.append(acceptor_extended)
                    acceptor_first = acceptor_extended
                    donor_first = donor_extended

            # cyclic systems, if you start in the second pair, you will finish with the first pair case of fulvene
            if int(extended_system[-1].split('-')[0]) < int(extended_system[0].split('-')[0]):
                del extended_system[-1]

            if len(extended_system) > 2:
                all_extended_systems.append(extended_system)

        extended_secondary_interaction_vos = []
        if all_extended_systems:
            for interaction in all_extended_systems:
                vos = []
                for pair in interaction:
                    orbital_systems = self.mapping_orbital_system_bonds[pair]

                    for orbital_system in orbital_systems:
                        if orbital_system in self.active_orbital_systems_list:
                            for vo in orbital_system.vos:
                                vos.append(vo)

                extended_secondary_interaction_vos.append(vos)

        return extended_secondary_interaction_vos

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

