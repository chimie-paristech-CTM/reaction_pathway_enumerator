from rdkit import Chem


def generate_smiles(orig_mol, path, modified_path, existing_interactions):
    """Generate an output SMILES string.

    Args:
        orig_mol (rdkit.Mol): the rdkit mol-object corresponding to the input system

    Returns:
        str: the output SMILES
    """
    editable_molecule = Chem.RWMol(orig_mol)  # editable version of the molecule

    # modify atom properties
    for vo, modified_vo in zip(path, modified_path):
        if vo.num_electrons != modified_vo.num_electrons:
            init_charge = editable_molecule.GetAtomWithIdx(vo.atom_idx - 1).GetFormalCharge()
            new_charge = init_charge - (modified_vo.num_electrons - vo.num_electrons)
            editable_molecule.GetAtomWithIdx(vo.atom_idx - 1).SetFormalCharge(new_charge)
        # TODO: do this also for radicals!!!
    
    # break original bonds
    # TODO: this is not OK!!! -> 1 or 2 vos selected first???
    for vo in path[1::2]:
        interacting_vo = existing_interactions[vo]
        current_bond = editable_molecule.GetBondBetweenAtoms(
                vo.atom_idx - 1, interacting_vo.atom_idx - 1
            )
        if current_bond.GetBondType() is Chem.rdchem.BondType.SINGLE:
                editable_molecule.RemoveBond(
                    vo.atom_idx - 1, interacting_vo.atom_id - 1
                )
        elif current_bond.GetBondType() is Chem.rdchem.BondType.DOUBLE:
                editable_molecule.RemoveBond(
                    vo.atom_idx - 1, interacting_vo.atom_idx - 1
                )
                editable_molecule.AddBond(
                    vo.atom_idx - 1, interacting_vo.atom_idx - 1,
                    Chem.rdchem.BondType.SINGLE,
                )
        elif current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
                editable_molecule.RemoveBond(
                    vo.atom_idx-1, interacting_vo.atom_idx - 1
                )
                editable_molecule.AddBond(
                    vo.atom_idx - 1, interacting_vo.atom_idx - 1,
                    Chem.rdchem.BondType.DOUBLE
                )
                

    # form new bonds
    for vo in path[1]:
         pass


    # if 1 atom carries both a lone pair and an empty orbital, sanitization will add Hs -> you don't want that!
    try:      
        if len(editable_molecule.GetAtoms()) != len(Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(editable_molecule))).GetAtoms()):
            return None
    except:
        print(Chem.MolToSmiles(editable_molecule))
        raise KeyError

    return Chem.MolToSmiles(editable_molecule)
