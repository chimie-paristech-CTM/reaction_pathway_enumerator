from rdkit import Chem


def generate_smiles(orig_mol, old_bonding_systems, new_bonding_systems):
    """Generate an output SMILES string.

    Args:
        orig_mol (rdkit.Mol): the rdkit mol-object corresponding to the input system
        old_bonding_systems (list): (copy of) the old bonding systems.
        new_bonding_systems (list): new bonding systems.

    Returns:
        str: the output SMILES
    """
    editable_molecule = Chem.RWMol(orig_mol)  # editable version of the molecule

    # adjust charges and spin on atoms based on information from the original state of the vos
    for bonding_system in old_bonding_systems:
        if len(bonding_system) == 1:
            init_charge = editable_molecule.GetAtomWithIdx(
                bonding_system.vos[0].atom_idx-1
            ).GetFormalCharge()
            if bonding_system.vos[0].num_electrons == 2:
                editable_molecule.GetAtomWithIdx(
                    bonding_system.vos[0].atom_idx-1
                ).SetFormalCharge(init_charge + 1)
            elif bonding_system.vos[0].num_electrons == 0:
                editable_molecule.GetAtomWithIdx(
                    bonding_system.vos[0].atom_idx-1
                ).SetFormalCharge(init_charge - 1)
            elif bonding_system.vos[0].num_electrons == 1:
                num_radicals = editable_molecule.GetAtomWithIdx(
                    bonding_system.vos[0].atom_idx-1
                ).GetNumRadicalElectrons()
                editable_molecule.GetAtomWithIdx(
                    bonding_system.vos[0].atom_idx-1
                ).SetNumRadicalElectrons(num_radicals - 1)
        else:
            current_bond = editable_molecule.GetBondBetweenAtoms(
                bonding_system.vos[0].atom_idx-1, bonding_system.vos[1].atom_idx-1
            )
            if current_bond.GetBondType() is Chem.rdchem.BondType.SINGLE:
                editable_molecule.RemoveBond(
                    bonding_system.vos[0].atom_idx-1, bonding_system.vos[1].atom_idx-1
                )
            elif current_bond.GetBondType() is Chem.rdchem.BondType.DOUBLE:
                editable_molecule.RemoveBond(
                    bonding_system.vos[0].atom_idx-1, bonding_system.vos[1].atom_idx-1
                )
                editable_molecule.AddBond(
                    bonding_system.vos[0].atom_idx-1,
                    bonding_system.vos[1].atom_idx-1,
                    Chem.rdchem.BondType.SINGLE,
                )
            elif current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
                editable_molecule.RemoveBond(
                    bonding_system.vos[0].atom_idx-1, bonding_system.vos[1].atom_idx-1
                )
                editable_molecule.AddBond(
                    bonding_system.vos[0].atom_idx-1,
                    bonding_system.vos[1].atom_idx-1,
                    Chem.rdchem.BondType.DOUBLE,
                )
    # adjust charges and spin on atoms based on information from the updated state of the vos
    for bonding_system in new_bonding_systems:
        if len(bonding_system) == 1:
            init_charge = editable_molecule.GetAtomWithIdx(
                bonding_system.vos[0].atom_idx-1
            ).GetFormalCharge()
            if bonding_system.vos[0].num_electrons == 2:
                editable_molecule.GetAtomWithIdx(
                    bonding_system.vos[0].atom_idx-1
                ).SetFormalCharge(init_charge - 1)
            elif bonding_system.vos[0].num_electrons == 0:
                editable_molecule.GetAtomWithIdx(
                    bonding_system.vos[0].atom_idx-1
                ).SetFormalCharge(init_charge + 1)
            elif bonding_system.vos[0].num_electrons == 1:
                num_radicals = editable_molecule.GetAtomWithIdx(
                    bonding_system.vos[0].atom_idx-1
                ).GetNumRadicalElectrons()
                editable_molecule.GetAtomWithIdx(
                    bonding_system.vos[0].atom_idx-1
                ).SetNumRadicalElectrons(num_radicals + 1)
        else:
            current_bond = editable_molecule.GetBondBetweenAtoms(
                bonding_system.vos[0].atom_idx-1, bonding_system.vos[1].atom_idx-1
            )
            if current_bond is None:
                editable_molecule.AddBond(
                    bonding_system.vos[0].atom_idx-1,
                    bonding_system.vos[1].atom_idx-1,
                    Chem.rdchem.BondType.SINGLE,
                )
            elif current_bond.GetBondType() is Chem.rdchem.BondType.SINGLE:
                editable_molecule.RemoveBond(
                    bonding_system.vos[0].atom_idx-1, bonding_system.vos[1].atom_idx-1
                )
                editable_molecule.AddBond(
                    bonding_system.vos[0].atom_idx-1,
                    bonding_system.vos[1].atom_idx-1,
                    Chem.rdchem.BondType.DOUBLE,
                )
            elif current_bond.GetBondType() is Chem.rdchem.BondType.DOUBLE:
                editable_molecule.RemoveBond(
                    bonding_system.vos[0].atom_idx-1, bonding_system.vos[1].atom_idx-1
                )
                editable_molecule.AddBond(
                    bonding_system.vos[0].atom_idx-1,
                    bonding_system.vos[1].atom_idx-1,
                    Chem.rdchem.BondType.TRIPLE,
                )
            elif current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
                pass  # This should likely give an error

    # check if 2 neighbors carry radicals/charges, if so, don't return anything:
    for atom in editable_molecule.GetAtoms():
        if atom.GetNumRadicalElectrons() != 0:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetNumRadicalElectrons() != 0:
                    return None
        if atom.GetFormalCharge() != 0:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetFormalCharge() != 0:
                    return None

    # if 1 atom carries both a lone pair and an empty orbital, sanitization will add Hs -> you don't want that!      
    if len(editable_molecule.GetAtoms()) != len(Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(editable_molecule))).GetAtoms()):
        return None

    return Chem.MolToSmiles(editable_molecule)
