from rdkit import Chem
import re

def decrease_bond_order(editable_mol, vo1, vo2):
    current_bond = editable_mol.GetBondBetweenAtoms(
                vo1.atom_idx - 1, vo2.atom_idx - 1
            )
    if current_bond.GetBondType() is Chem.rdchem.BondType.SINGLE:
        editable_mol.RemoveBond(
            vo1.atom_idx - 1, vo2.atom_idx - 1
        )
    elif current_bond.GetBondType() is Chem.rdchem.BondType.DOUBLE:
        editable_mol.RemoveBond(
            vo1.atom_idx - 1, vo2.atom_idx - 1
        )
        editable_mol.AddBond(
            vo1.atom_idx - 1, vo2.atom_idx - 1,
            Chem.rdchem.BondType.SINGLE,
        )
    elif current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
        editable_mol.RemoveBond(
            vo1.atom_idx-1, vo2.atom_idx - 1
        )
        editable_mol.AddBond(
            vo1.atom_idx - 1, vo2.atom_idx - 1,
            Chem.rdchem.BondType.DOUBLE
        )
    
    return editable_mol


def increase_bond_order(editable_mol, vo1, vo2):
    current_bond = editable_mol.GetBondBetweenAtoms(
                vo1.atom_idx - 1, vo2.atom_idx - 1
            )
    if current_bond is None:
        editable_mol.AddBond(
            vo1.atom_idx - 1, vo2.atom_idx - 1,
            Chem.rdchem.BondType.SINGLE,
        ) 
    elif current_bond.GetBondType() is Chem.rdchem.BondType.SINGLE:
        editable_mol.RemoveBond(
            vo1.atom_idx - 1, vo2.atom_idx - 1
        )
        editable_mol.AddBond(
            vo1.atom_idx - 1, vo2.atom_idx - 1,
            Chem.rdchem.BondType.DOUBLE
        ) 
    elif current_bond.GetBondType() is Chem.rdchem.BondType.DOUBLE:
        editable_mol.RemoveBond(
            vo1.atom_idx - 1, vo2.atom_idx - 1
        )
        editable_mol.AddBond(
            vo1.atom_idx - 1, vo2.atom_idx - 1,
            Chem.rdchem.BondType.TRIPLE,
        )
    elif current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
        editable_mol.RemoveBond(
            vo1.atom_idx-1, vo2.atom_idx - 1
        )
        editable_mol.AddBond(
            vo1.atom_idx - 1, vo2.atom_idx - 1,
            Chem.rdchem.BondType.QUADRUPLE
        ) 

    return editable_mol


def fix_radical_counts_at_endpoints_path(editable_mol, vo_start, vo_end):
    current_num_unpaired_elec_start = editable_mol.GetAtomWithIdx(vo_start.atom_idx - 1).GetNumRadicalElectrons()
    editable_mol.GetAtomWithIdx(vo_start.atom_idx - 1).SetNumRadicalElectrons(current_num_unpaired_elec_start - 1) 
    current_num_unpaired_elec_end = editable_mol.GetAtomWithIdx(vo_end.atom_idx - 1).GetNumRadicalElectrons()
    editable_mol.GetAtomWithIdx(vo_end.atom_idx - 1).SetNumRadicalElectrons(current_num_unpaired_elec_end - 1)

    return editable_mol


def get_neighbors_dict(orig_mol):
    """
    Get a dictionary of atom neighbors for the given molecule.

    Args:
        orig_mol: The original molecule for which atom neighbors are to be retrieved.

    Returns:
        dict: A dictionary where the keys are atom map numbers and the values are lists 
        of atom map numbers representing the neighbors of each atom.
    """
    return {atom.GetAtomMapNum(): [neighbor.GetAtomMapNum()
        for neighbor in atom.GetNeighbors()] for atom in orig_mol.GetAtoms()}


def clear_numbering(smiles):
    """
    Clear atom numbering in the SMILES representation.

    Args:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        str or None: The SMILES representation of the molecule with cleared atom numbering,
        or None if an error occurs during processing.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)
    except:
        return None


def ordering_smiles(numbered_smiles):
    """
    Ordering the numbered SMILES representation

    Args:
        numbered_smiles (str): The numbered SMILES representation of the molecule.

    Returns:
        list: The numbered SMILES representation of the molecule ordered by atom numbering.
    """


    pattern = r'\[(.*?)\]'
    smiles_elements = re.findall(pattern, numbered_smiles)
    ordered_smiles = sorted(smiles_elements, key=(lambda x: int(x.split(':')[-1])))

    return ordered_smiles
