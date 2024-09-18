import autode as ade
from autode import Molecule
from autode.wrappers.G16 import g16
import logging
import os
import subprocess
import sys
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


def get_nbo(smiles):
    """Execute a NBO calculation with G16"""
    smiles_list = smiles.split('.')
    dict_nbo_lines = {}
    for idx, smi in enumerate(smiles_list):
        nbo_lines = exec_nbo_calculation(idx, smi, g16_path='/opt/gaussian/g16/C01/g16')
        dict_nbo_lines[idx] = nbo_lines
    return dict_nbo_lines


def exec_nbo_calculation(idx, smiles, g16_path, n_cores=16, basis_set='def2svp', functional='pbe1pbe'):

    cwd = os.getcwd()
    working_directory = os.path.join(cwd, 'calc')
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    os.chdir(working_directory)
    molecule = Molecule(smiles=smiles, name=f"r{idx}")
    g16.keywords.set_functional(functional)
    g16.keywords.set_opt_basis_set(basis_set)
    ade.Config.n_cores = n_cores
    ade.Config.max_core = 1000
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


def extract_electrons_based_bond_matrix(nbo_lines, smiles_list):

    electrons_per_atom = dict()

    line_0 = " ------------------ Lewis ------------------------------------------------------\n"
    line_1 = " ---------------- non-Lewis ----------------------------------------------------\n"

    for idx_smi, smiles in enumerate(smiles_list):
        ordered_smiles = ordering_smiles(smiles)
        idx_0 = nbo_lines[idx_smi].index(line_0)
        idx_1 = nbo_lines[idx_smi].index(line_1)

        for line in nbo_lines[idx_smi][idx_0 + 1: idx_1]:

            if 'BD' in line:
                atom_1 = int(line[25:28])
                atom_2 = int(line[31:34])
                atom_1_in_numbered_smiles = int(ordered_smiles[atom_1 - 1].split(':')[-1])
                atom_2_in_numbered_smiles = int(ordered_smiles[atom_2 - 1].split(':')[-1])

                electrons_per_atom[atom_1_in_numbered_smiles] = electrons_per_atom.get(atom_1_in_numbered_smiles, 0) + 1
                electrons_per_atom[atom_2_in_numbered_smiles] = electrons_per_atom.get(atom_2_in_numbered_smiles, 0) + 1

            if 'LP' in line:
                atom = int(line[25:28])
                atom_in_numbered_smiles = int(ordered_smiles[atom - 1].split(':')[-1])
                electrons_per_atom[atom_in_numbered_smiles] = electrons_per_atom.get(atom_in_numbered_smiles, 0) + 2

    return electrons_per_atom


def extract_secondary_interactions(numbered_smiles, nbo_lines, threshold=11.5):

    smiles_list = numbered_smiles.split('.')
    interactions = []

    for idx, smiles in enumerate(smiles_list):
        ordered_smiles = ordering_smiles(smiles)

        line_0 = " SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS\n"
        line_1 = " NATURAL BOND ORBITALS (Summary):\n"
        idx_0 = nbo_lines[idx].index(line_0)
        idx_1 = nbo_lines[idx].index(line_1)

        for line in nbo_lines[idx][idx_0 + 7: idx_1 - 2]:
            if line.startswith(' within unit') or line.startswith(' from unit') or line.isspace():
                continue

            lp_idx = None

            if float(line.split()[-3]) > threshold:

                if line[35:37] == 'RY':
                    continue

                if line[7:9] == "LP":
                    donor_atom_idxs = (int(line[17:19]),)
                    lp_idx = (int(line[11:13]))
                elif line[7:9] == "BD":
                    donor_atom_idxs = (int(line[17:19]), int(line[23:25]))

                if line[35:38] == 'BD*':
                    acceptor_atom_idxs = (int(line[45:47]), int(line[51:53]))

                donor_idx_numbered_smiles = [ordered_smiles[atom_idx - 1].split(':')[-1] for atom_idx in donor_atom_idxs]
                if lp_idx:
                    donor_bond = f"{donor_idx_numbered_smiles[0]}_{lp_idx}"
                else:
                    donor_bond = f"{donor_idx_numbered_smiles[0]}-{donor_idx_numbered_smiles[1]}"
                acceptor_idx_numbered_smiles = [ordered_smiles[atom_idx - 1].split(':')[-1] for atom_idx in acceptor_atom_idxs]
                acceptor_bond = f"{acceptor_idx_numbered_smiles[0]}-{acceptor_idx_numbered_smiles[1]}"

                interactions.append((donor_bond, acceptor_bond))

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

