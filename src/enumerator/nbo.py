import autode as ade
from autode import Molecule
from autode.wrappers.G16 import g16
import logging
import os
import subprocess


def exec_nbo_calculation(idx, smiles, g16_path, n_cores=8, basis_set='def2svp', functional='pbe1pbe'):

    molecule = Molecule(smiles=smiles, name=f"r{idx}")
    g16.keywords.set_functional(functional)
    g16.keywords.set_opt_basis_set(basis_set)
    ade.Config.n_cores = n_cores
    ade.Config.max_core = 1000
    molecule.find_lowest_energy_conformer(hmethod=g16)
    molecule.optimise(method=g16)
    generate_input_gaussian(molecule, n_cores, basis_set, functional)
    run_g16(g16_path, molecule.name)


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


def check_normal_termination(name):
    """Check for normal termination in a Gaussian output"""
    name += '_NBO.log'

    with open(name, 'r') as file:
        lines = file.readlines()[::-1]

    for line in lines:
        if 'Normal termination' in line:
            return True

    return False
