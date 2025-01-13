import autode as ade
from autode import Molecule
from autode.wrappers.G16 import g16
import logging
import os
import subprocess
import sys
import math
from enumerator.utils import ordering_smiles


def read_from_chk(smiles, dir_nbo):
    """Extract the NBO from a directory"""
    cwd = os.getcwd()
    try:
        os.chdir(dir_nbo)
    except Exception as e:
        print(f'Directory {dir_nbo} not existed.')
        #sys.exit()

    smiles_list = smiles.split('.')
    dict_nbo_lines = {}
    for idx, smi in enumerate(smiles_list):
        nbo_lines = extract_nbo_lines(f"r{idx}_NBO.log")
        dict_nbo_lines[idx] = nbo_lines
    os.chdir(cwd)
    return dict_nbo_lines


def get_nbo(smiles, mult, nproc):
    """Execute a NBO calculation with G16"""
    smiles_list = smiles.split('.')
    dict_nbo_lines = {}
    for idx, smi in enumerate(smiles_list):
        nbo_lines = exec_nbo_calculation(idx, smi, mult, g16_path='/opt/gaussian/g16/C01/g16', n_cores=nproc)
        dict_nbo_lines[idx] = nbo_lines
    return dict_nbo_lines


def exec_nbo_calculation(idx, smiles, mult, g16_path, n_cores=16, basis_set='def2svp', functional='pbe1pbe'):

    cwd = os.getcwd()
    working_directory = os.path.join(cwd, 'calc')
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    os.chdir(working_directory)
    if mult != -1:
        molecule = Molecule(smiles=smiles, name=f"r{idx}", mult=mult)
    else:
        molecule = Molecule(smiles=smiles, name=f"r{idx}")
    g16.keywords.set_functional(functional)
    g16.keywords.set_opt_basis_set(basis_set)
    ade.Config.n_cores = n_cores
    ade.Config.max_core = 1000
    ade.Config.num_conformers = 600
    ade.Config.rmsd_threshold = 0.15
    molecule.find_lowest_energy_conformer(hmethod=g16)
    molecule.optimise(method=g16)
    generate_input_gaussian(molecule, n_cores, basis_set, functional)
    run_g16(g16_path, molecule.name)
    try:
        if normal_termination(f"{molecule.name}_NBO.log"):
            nbo_lines = extract_nbo_lines(f"{molecule.name}_NBO.log")
            os.chdir(cwd)
            return nbo_lines
        else:
            raise CalculationError(f"{molecule.name}_NBO.log")

    except CalculationError:
        sys.exit()


def generate_input_gaussian(molecule, n_cores, basis_set='def2svp', functional='m062x'):
    """Generate the required input file for G16"""

    name = molecule.name

    logging.info(f"Generating input file for {name}")
    with open(f"{name}_NBO.com", 'w') as file:
        file.write(f"%nprocshared={n_cores} \n")
        file.write(f"%mem={n_cores}000MB \n")
        file.write(f"# {functional} {basis_set} pop=NBO7 \n\n")
        file.write(f"NBO input \n\n")
        file.write(f" {molecule.charge} {molecule.mult} \n")
        for atom in molecule.atoms:
            file.write(f"{atom.atomic_symbol}   {atom.coord[0]}   {atom.coord[1]}   {atom.coord[2]} \n")
        file.write("\n")
        file.write(r"$nbo $end")
        file.write("\n\n")


def run_g16(g16_path, name):
    """Launch a G16 calculation"""

    g16_command = os.path.join(g16_path, 'g16')
    name += '_NBO.com'
    command_line = f"{g16_command} {name}"
    with open('log_file.out', 'w') as out:
        subprocess.run(f"{command_line}", shell=True, stdout=out, stderr=out)


def normal_termination(name):
    """Check for normal termination in a Gaussian output"""

    with open(name, 'r') as file:
        lines = file.readlines()[::-1]

    for line in lines:
        if 'Normal termination' in line:
            return True

    return False


def extract_nbo_lines(name):
    """Extract NBO lines"""

    with open(name, 'r') as file:
        lines = file.readlines()

    line_0 = "Perform NBO analysis...executing"
    line_1 = "NBO analysis completed in"

    append = False
    nbo_lines = []

    for line in lines:

        if line_1 in line:
            break

        if line_0 in line:
            append = True

        if append:
            nbo_lines.append(line)

    return nbo_lines


def extract_electrons_based_bond_matrix(nbo_lines, smiles_list, organometallic):

    electrons_per_atom = dict()
    lp_per_atom = dict()

    line_0 = " ------------------ Lewis ------------------------------------------------------\n"
    line_1 = " ---------------- non-Lewis ----------------------------------------------------\n"

    for idx_smi, smiles in enumerate(smiles_list):
        ordered_smiles = ordering_smiles(smiles, organometallic)
        idxs_0 = [i for i, x in enumerate(nbo_lines[idx_smi]) if x == line_0]
        idxs_1 = [i for i, x in enumerate(nbo_lines[idx_smi]) if x == line_1]

        # when S=2, NBO split electrons into alpha/beta ... therefore, the amount of electrons in BD and LP should be divided by 2
        num_idxs = len(idxs_0)
        for idx_0, idx_1 in zip(idxs_0, idxs_1):
            for line in nbo_lines[idx_smi][idx_0 + 1: idx_1]:

                if 'BD' in line:
                    atom_1 = int(line[25:28])
                    atom_2 = int(line[31:34])
                    atom_1_in_numbered_smiles = int(ordered_smiles[atom_1 - 1].split(':')[-1])
                    atom_2_in_numbered_smiles = int(ordered_smiles[atom_2 - 1].split(':')[-1])

                    electrons_per_atom[atom_1_in_numbered_smiles] = electrons_per_atom.get(atom_1_in_numbered_smiles, 0) + 1/num_idxs
                    electrons_per_atom[atom_2_in_numbered_smiles] = electrons_per_atom.get(atom_2_in_numbered_smiles, 0) + 1/num_idxs

                if 'LP' in line:
                    atom = int(line[25:28])
                    atom_in_numbered_smiles = int(ordered_smiles[atom - 1].split(':')[-1])
                    electrons_per_atom[atom_in_numbered_smiles] = electrons_per_atom.get(atom_in_numbered_smiles, 0) + 2/num_idxs
                    lp_per_atom[atom_in_numbered_smiles] = lp_per_atom.get(atom_in_numbered_smiles, 0) + 1/num_idxs

        # for conjugated systems, is it possible that the LP is not located in the same atom of the LV ... and you will have atoms with a fractional number of electrons
        if num_idxs == 2:
            for atom in lp_per_atom:
                if electrons_per_atom[atom].is_integer():
                    continue
                else:
                    other_atoms = [idx for idx in electrons_per_atom if electrons_per_atom[idx].is_integer() == False and idx != atom]
                    electrons_per_atom[atom] = float(math.floor(electrons_per_atom[atom]))
                    for other_atom in other_atoms:
                        electrons_per_atom[other_atom] = float(math.ceil(electrons_per_atom[other_atom]))

    return electrons_per_atom, lp_per_atom


def extract_secondary_interactions_raw(numbered_smiles, nbo_lines, organometallic, threshold=11.5):

    smiles_list = numbered_smiles.split('.')
    interactions = []

    for idx, smiles in enumerate(smiles_list):
        ordered_smiles = ordering_smiles(smiles, organometallic)

        line_0 = " SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS\n"
        line_1 = " NATURAL BOND ORBITALS (Summary):\n"
        idx_0 = nbo_lines[idx].index(line_0)
        idx_1 = nbo_lines[idx].index(line_1)

        for line in nbo_lines[idx][idx_0 + 7: idx_1 - 2]:
            if line.startswith(' within unit') or line.startswith(' from unit') or line.isspace():
                continue

            if 'None above threshold' in line:
                break

            lp_idx = None
            lv_idx = None

            energy = float(line.split()[-3])

            if energy > threshold:

                if line[35:37] == 'RY':
                    continue

                if line[7:9] == "LP":
                    donor_atom_idxs = (int(line[17:19]),)
                    lp_idx = (int(line[11:13]))
                elif line[7:9] == "BD":
                    donor_atom_idxs = (int(line[17:19]), int(line[23:25]))

                if line[35:38] == 'BD*':
                    acceptor_atom_idxs = (int(line[45:47]), int(line[51:53]))

                if line[35:37] == 'LV':
                    acceptor_atom_idxs = (int(line[45:47]),)
                    lv_idx = (int(line[39:41]))

                donor_idx_numbered_smiles = [ordered_smiles[atom_idx - 1].split(':')[-1] for atom_idx in donor_atom_idxs]
                if lp_idx:
                    donor_bond = f"{donor_idx_numbered_smiles[0]}_{lp_idx}"
                else:
                    donor_bond = f"{donor_idx_numbered_smiles[0]}-{donor_idx_numbered_smiles[1]}"

                acceptor_idx_numbered_smiles = [ordered_smiles[atom_idx - 1].split(':')[-1] for atom_idx in acceptor_atom_idxs]
                if lv_idx:
                    acceptor_bond = f"{acceptor_idx_numbered_smiles[0]}#{lv_idx}"
                else:
                    acceptor_bond = f"{acceptor_idx_numbered_smiles[0]}-{acceptor_idx_numbered_smiles[1]}"

                if any((acceptor_bond, donor_bond) == (item[0], item[1]) for item in interactions):
                    continue
                else:
                    interactions.append((donor_bond, acceptor_bond, energy))

    return interactions


def check_lp_within_secondary_interaction(interactions, lp_idx):

    if interactions:
        for interaction in interactions:
            donor_key = interaction[0]
            if '_' in donor_key:
                lp_donor_idx = int(donor_key.split('_')[-1])
                if lp_donor_idx == lp_idx:
                    return True
    return False

class CalculationError(Exception):
    """Custom exception for calculation errors."""
    def __init__(self, name):
        message = f'G16 calculation not finished for {name} ...'
        super().__init__(message)

