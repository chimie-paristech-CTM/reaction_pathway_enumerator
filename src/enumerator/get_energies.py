from rdkit.Chem import AllChem  # type: ignore
from rdkit.Chem import Descriptors
from rdkit import Chem  # type: ignore
import os
import shutil
from typing import List, Tuple, Optional
import logging
import subprocess

import contextlib
from pathlib import Path

HARTREE_TO_EV = 27.2114


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
    mol: "Chem.Mol", smi: str, optimizer: str = "rdkit"
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

            command = "xtb tmp.xyz --opt normal --gfn 2 --chrg {} --uhf {}".format(
                charge, num_unpaired_electrons
            )
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
    molecule: str, optimizer: str = "rdkit", n_attempts=10
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
        return get_molecule_energy(canonical_smi, optimizer=optimizer)
    mol = Chem.AddHs(mol)

    if canonical_smi == "[H+]":
        return 0.0

    if len(mol.GetAtoms()) == 1:
        atoms, atom_coords = [mol.GetAtoms()[0].GetSymbol()], [(0.0, 0.0, 0.0)]
    else:
        try:
            atoms, atom_coords = mol_to_coords(mol, molecule, optimizer=optimizer)
        except Exception as e:
            logging.warning("{} in xTB geometry opt for {}".format(e, molecule))
            return None

    charge = Chem.rdmolops.GetFormalCharge(mol)
    num_unpaired_electrons = Descriptors.NumRadicalElectrons(mol)
    xyzfile = output_3d_coords(atoms, atom_coords, output_format="xyz")

    with make_tmp_directory():
        with open("tmp.xyz", "w") as f:
            f.write(xyzfile)

        energy = None
        command = f"xtb tmp.xyz --gfn 2 --chrg {charge} --uhf {num_unpaired_electrons}"

        subprocess.run(
            command,
            shell=True,
            stdout=open("xtblog.txt", "w"),
            stderr=open(os.devnull, "w"),
        )
        with open("xtblog.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "TOTAL ENERGY" in line or "total energy" in line:
                    energy = float(line.split()[-3])

        if energy is None:
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
    smi: str, optimizer: str = "rdkit", n_attempts=10
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
            molecule, optimizer=optimizer, n_attempts=n_attempts
        )
        if molecule_energy is None:
            return None
        else:
            total_energy += molecule_energy

    shutil.rmtree("tmp_{}".format(os.getpid()))
    return total_energy * HARTREE_TO_EV
