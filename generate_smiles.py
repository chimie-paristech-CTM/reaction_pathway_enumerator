from rdkit import Chem
import copy
import numpy as np


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
