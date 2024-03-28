from rdkit import Chem


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


# TODO: this is not yet 3c bond proof -> you can have those as endpoints
def generate_smiles(orig_mol, path, modified_path, existing_interactions):
    """Generate an output SMILES string.

    Args:
        orig_mol (rdkit.Mol): the rdkit mol-object corresponding to the input system

    Returns:
        str: the output SMILES
    """
    editable_mol = Chem.RWMol(orig_mol)  # editable version of the molecule

    # modify atom properties
    for vo, modified_vo in zip(path, modified_path):
        #print(path) 
        #print(modified_path)
        if vo.num_electrons != modified_vo.num_electrons:
            init_charge = editable_mol.GetAtomWithIdx(vo.atom_idx - 1).GetFormalCharge()
            new_charge = init_charge - (modified_vo.num_electrons - vo.num_electrons)
            editable_mol.GetAtomWithIdx(vo.atom_idx - 1).SetFormalCharge(new_charge)
    
    # modify bonding situation
    for i, vo in enumerate(path[:-1]):
        if path[i+1] in existing_interactions[vo]:
            editable_mol = decrease_bond_order(editable_mol, vo, path[i+1])
        else:
            editable_mol = increase_bond_order(editable_mol, vo, path[i+1])   
    if path[0].is_paired() and path[-1].is_paired():
        editable_mol = increase_bond_order(editable_mol, path[0], path[-1]) # finish covalent path
    elif (not path[0].is_paired() and path[-1].is_paired()) and path[0].num_electrons == 1 and modified_path[-1].num_electrons == 1: # fix radical sites
        editable_mol = fix_radical_counts_at_endpoints_path(editable_mol, path[0], path[-1])
    elif (not path[-1].is_paired() and path[0].is_paired()) and path[-1].num_electrons == 1 and modified_path[0].num_electrons == 1:
       editable_mol = fix_radical_counts_at_endpoints_path(editable_mol, path[-1], path[0]) 

    # if 1 atom carries both a lone pair and an empty orbital, sanitization will add Hs -> you don't want that!
    try:      
        if len(editable_mol.GetAtoms()) != len(Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(editable_mol))).GetAtoms()):
            return None
    except Exception as e:
        print(e)
        print('lol')
        print(path, modified_path)
        print(Chem.MolToSmiles(editable_mol))
        #raise KeyError

    return Chem.MolToSmiles(editable_mol)
