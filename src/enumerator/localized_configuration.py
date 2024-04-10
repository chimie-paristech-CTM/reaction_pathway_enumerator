from typing import Dict, List
from enumerator.orbital_systems import LocalizedOrbitalSystem

metal_symbols = ["Al", "Fe", "Cu", "Au", "Ag",  "Zn", "Ni",  "Sn",  "Pb",  "Pt",  "Hg",  "Ti", "Co", 
    "Cr",  "Mg",  "Mn",  "W",   "Bi",  "Sb",  "Cd",  "V",   "U",   "Pd",  "Rh",  "Ru"]


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


class LocalizedConfiguration:

    def __init__(self, orig_mol, atoms):
        self.orbital_systems_list = self.set_up_localized_orbital_systems(orig_mol, atoms)
        self.active_orbital_systems_list = self.select_active_orbital_systems()
        self.vo_list = self.set_vo_list()
        self.vo_to_orbital_system_dict = self.get_vo_to_orbital_system_dict()

    # TODO: what about circular 3c bonds (e.g., interaction between ethylene and PdL2)?
    # TODO: should you include validity checks to ensure that the localized configuration makes sense (e.g., exotic boding situations resulting in incorrect vo pairing)?
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
    