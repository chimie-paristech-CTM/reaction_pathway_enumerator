from rdkit.Chem import AllChem  # type: ignore
from rdkit.Chem import Descriptors
from rdkit import Chem  # type: ignore
import os
import shutil
from typing import List, Tuple, Optional
import logging
import subprocess
import re
import numpy as np

import contextlib
from pathlib import Path

ps = Chem.SmilesParserParams()
ps.removeHs = False


@contextlib.contextmanager
def make_tmp_directory():
    """Makes a temporary directory to do things in, and then reverts back on exit."""
    prev_cwd = Path.cwd()
    if not os.path.exists("tmp_{}".format(os.getpid())):
        os.mkdir("tmp_{}".format(os.getpid()))
    os.chdir("tmp_{}".format(os.getpid()))
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        # shutil.rmtree('tmp_{}/'.format(os.getpid()))


def mol_to_coords(
    mol: "Chem.Mol", smi: str, optimizer: str = "rdkit", solvent: str = None, nproc: int = 4
) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    """
    Returns the atoms and coordinates of a molecule as arrays.
    Supported Geometry Optimizers: 'rdkit' (ETKDG method), 'xtb' (GFN2-xTB method)
    """
    if optimizer not in {"rdkit", "xtb"}:
        raise ValueError("Invalid optimizer. Supported optimizers: rdkit, xtb")

    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)  # for reproducibility
    lines = Chem.MolToMolBlock(mol).split("\n")
    atoms = []
    atom_coords = []

    logging.info(f"Converting {smi} to coordinates")
    # string parsing RDKit mol block
    for idx, line in enumerate(lines):
        if idx < 3:
            continue
        elif idx == 3:
            num_atoms = mol.GetNumAtoms()
        elif idx <= 3 + num_atoms:
            x, y, z, atom = [c for c in line.split(" ") if len(c) > 0][:4]
            x, y, z = map(float, [x, y, z])
            atom_coords.append((x, y, z))
            atoms.append(atom)

    # geometry optimization beyond RDKit with XTB if specified
    if optimizer == "xtb":
        if len(atoms) == 1:
            # no need for optimization on single atom
            return atoms, atom_coords

        charge = Chem.rdmolops.GetFormalCharge(mol)
        num_unpaired_electrons = Descriptors.NumRadicalElectrons(mol)
        xyzfile = output_3d_coords(atoms, atom_coords, output_format="xyz")

        with make_tmp_directory():
            with open("tmp.xyz", "w") as f:
                f.write(xyzfile)

            if solvent != None:
                command = f"xtb tmp.xyz --opt normal --gfn 2 --chrg {charge} --uhf {num_unpaired_electrons} --alpb {solvent} -P {nproc}"
            else:
                command = f"xtb tmp.xyz --opt normal --gfn 2 --chrg {charge} --uhf {num_unpaired_electrons} -P {nproc}"
            subprocess.check_call(
                command.split(),
                stdout=open("xtblog.txt", "w"),
                stderr=open(os.devnull, "w"),
            )

            with open("xtbopt.xyz", "r") as f:
                lines = f.readlines()
                atoms = []
                atom_coords = []
                for line in lines[2:]:
                    atom, x, y, z = line.split()
                    x, y, z = map(float, [x, y, z])
                    atom_coords.append((x, y, z))
                    atoms.append(atom)

            for output_file in [
                "tmp.xyz",
                "xtbopt.xyz",
                "xtbopt.log",
                "xtbtopo.mol",
                "xtbrestart",
                "wbo",
                "tmp.ges",
                "charges",
                "xtblog.txt",
            ]:
                if os.path.exists(output_file):
                    os.remove(output_file)

    return atoms, atom_coords


def get_molecule_energy(
    molecule: str, optimizer: str = "rdkit", solvent: str = None, nproc: int = 4
) -> Optional[float]:
    """
    Returns the (potential) energy of a single molecule (given as a smiles string) in hartree.
    Uses xTB (GFN2-xTB) calculations for energy calculation.
    Geometry optimization is done either by RDKit (ETKDG method) or xTB (GFN2-xTB method).
    Energy calculations are attempted for n_attempts times. If they all fail, then
    this function returns None.
    """
    mol = Chem.MolFromSmiles(molecule)
    canonical_smi = Chem.MolToSmiles(mol, canonical=True)
    if canonical_smi != molecule:
        return get_molecule_energy(canonical_smi, optimizer=optimizer, solvent=solvent, nproc=nproc)
    mol = Chem.AddHs(mol)

    if molecule == "[H+]":
        return 0.0

    if len(mol.GetAtoms()) == 1:
        atoms, atom_coords = [mol.GetAtoms()[0].GetSymbol()], [(0.0, 0.0, 0.0)]
    else:
        try:
            atoms, atom_coords = mol_to_coords(mol, molecule, optimizer=optimizer, solvent=solvent)
        except Exception as e:
            logging.warning("{} in xTB geometry opt for {}".format(e, molecule))
            return None

    bond_matrix_initial = obtain_bond_matrix(mol, None)
    charge = Chem.rdmolops.GetFormalCharge(mol)
    num_unpaired_electrons = Descriptors.NumRadicalElectrons(mol)
    xyzfile = output_3d_coords(atoms, atom_coords, output_format="xyz")

    with make_tmp_directory():
        with open("tmp.xyz", "w") as f:
            f.write(xyzfile)

        energy = None
        if solvent != None:
            command = f"xtb tmp.xyz --gfn 2 --chrg {charge} --uhf {num_unpaired_electrons} --alpb {solvent} -P {nproc}"
        elif len(mol.GetAtoms()) == 1:
            command = f"xtb tmp.xyz --gfn 2 --chrg {charge} --uhf {num_unpaired_electrons} -P {nproc}"
        else:
            command = f"xtb tmp.xyz --opt normal --gfn 2 --chrg {charge} --uhf {num_unpaired_electrons} -P {nproc}"

        subprocess.run(
            command,
            shell=True,
            stdout=open("xtblog.txt", "w"),
            stderr=open(os.devnull, "w"),
        )

        if os.path.isfile("NOT_CONVERGED"):
            return None

        with open("xtblog.txt", "rb") as f:
            lines = f.readlines()
            for line in reversed(lines):
                try:
                    current_line = line.decode("utf-8")
                except:
                    continue
                if "TOTAL ENERGY" in current_line or ":: total energy" in current_line:
                    energy = float(current_line.split()[-3])
                    break

        if energy is None:
            return None
        bond_matrix_final = obtain_bond_matrix(mol, 'xtbopt.xyz')

        if (bond_matrix_initial != bond_matrix_final).all():
            return None

        logging.info("Energy of {} is {} hartree".format(molecule, energy))
    return energy


def output_3d_coords(
    atoms: List[str],
    atom_coords: List[Tuple[float, float, float]],
    output_format: str = "xyz",
) -> str:
    """
    Returns the coordinates of a molecule in the specified output format.
    Supported Output Formats: 'xyz', 'turbo'
    """
    if output_format not in {"xyz", "turbo"}:
        raise ValueError("Invalid output format. Supported formats: xyz, turbo")

    logging.info("Outputting coordinates in {} format".format(output_format))
    if output_format == "xyz":
        output = str(len(atoms)) + "\n\n"
        for atom, (x, y, z) in zip(atoms, atom_coords):
            output += " ".join([atom, str(x), str(y), str(z)]) + "\n"

    elif output_format == "turbo":
        output = "$coord"
        for atom, (x, y, z) in zip(atoms, atom_coords):
            output += "\n" + " ".join([str(x), str(y), str(z), atom.lower()])
        output += "\n$end\n"

    return output


def get_system_energy(
    smi: str, optimizer: str = "rdkit", solvent: str = None, nproc: int = 4
) -> Optional[float]:
    """
    Returns the (potential) energy of the system (given as a smiles string) in eV.
    Uses xTB (GFN2-xTB) calculations for energy calculation.
    Geometry optimization is done either by RDKit (ETKDG method) or xTB (GFN2-xTB method).
    Energy calculations are attempted for n_attempts times. If they all fail, then
    this function returns None.
    """
    molecules = smi.split(".")
    total_energy = 0.0

    for molecule in molecules:
        molecule_energy = get_molecule_energy(
            molecule, optimizer=optimizer, solvent=solvent, nproc=nproc
        )
        if molecule_energy is None:
            return None
        else:
            total_energy += molecule_energy

    shutil.rmtree("tmp_{}".format(os.getpid()))
    return total_energy


def obtain_bond_matrix(mol, xyzfile):
    """_summary_

    Args:
        geom (_type_): _description_
    """

    pt = Chem.GetPeriodicTable()

    num_atoms = mol.GetNumAtoms()
    bond_matrix = np.zeros((num_atoms, num_atoms), dtype=int)

    if num_atoms == 1:
        return np.ones((num_atoms), dtype=int)

    if xyzfile is None and mol:

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_matrix[i][j] = bond_matrix[j][i] = 1

    else:
        with open(xyzfile, 'r') as f:
            lines = f.readlines()[2:]

        labels = []
        geom = []

        for line in lines:
            label, x, y, z = line.split()
            labels.append(label)
            geom.append((float(x), float(y), float(z)))

        for i in range(num_atoms):
            for j in range(i + 1, len(geom)):
                cov_bond = pt.GetRcovalent(labels[i]) + pt.GetRcovalent(labels[j])
                distance = np.linalg.norm(np.array(geom[i]) - np.array(geom[j]))
                if distance < cov_bond + 0.25:
                    bond_matrix[i][j] = bond_matrix[j][i] = 1

    return bond_matrix
