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
        """Returns a list of atom indices present in the orbital system."""
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

    def is_lp(self):
        """
        returns bool, indicating wheter the localized orbital system is a lone pair or not
        """
        if len(self.vos) == 1:
            if self.vos[0].num_electrons == 2:
                return True
            else:
                return False
        else:
            return False

    def get_lp_idx(self):
        """
        returns lone pair index, if not return 0
        """
        if len(self.vos) == 1:
            if self.vos[0].num_electrons == 2:
                return self.vos[0].lp_idx
            else:
                return 0
        else:
            return 0

    def __str__(self):
        return f'localized orbital system {self.idx}: vos = {self.vos}'

    
class DelocalizedOrbitalSystem(OrbitalSystem):
    """A class corresponding to individual delocalized orbital systems."""
    
    def __str__(self):
        return f'delocalized orbital system {self.idx}: vos = {self.vos}'