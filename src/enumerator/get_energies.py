from rdkit import Chem

from typing import List, Tuple, Optional, Dict, Optional
import numpy as np
import os
import logging
from rdkit.Chem import Descriptors, AllChem
from rdkit import Chem  # type: ignore
from pathlib import Path
import torch
from torch import Tensor

import subprocess
import shutil


def smi_to_coords(smi: str, optimizer: str = 'rdkit') -> Tuple[List[str], List[Tuple[float, float, float]]]:
    """
    Returns the atoms and coordinates of a molecule (given as a smiles string) as arrays.
    Supported Geometry Optimizers: 'rdkit' (ETKDG method), 'xtb' (GFN2-xTB method)
    """
    if optimizer not in {'rdkit', 'xtb'}:
        raise ValueError('Invalid optimizer. Supported optimizers: rdkit, xtb')

    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)  # for reproducibility
    lines = Chem.MolToMolBlock(mol).split('\n')
    atoms = []
    atom_coords = []

    logging.info('Converting SMILES {} to coordinates'.format(smi))
    # string parsing RDKit mol block
    for idx, line in enumerate(lines):
        if idx < 3:
            continue
        elif idx == 3:
            num_atoms = mol.GetNumAtoms()
        elif idx <= 3 + num_atoms:
            x, y, z, atom = [c for c in line.split(' ') if len(c) > 0][:4]
            x, y, z = map(float, [x, y, z])
            atom_coords.append((x, y, z))
            atoms.append(atom)

    # geometry optimization beyond RDKit with XTB if specified
    if optimizer == 'xtb':
        if len(atoms) == 1:
            # no need for optimization on single atom
            return atoms, atom_coords

        logging.info('Optimizing geometry with xTB')
        charge = Chem.rdmolops.GetFormalCharge(mol)
        num_unpaired_electrons = Descriptors.NumRadicalElectrons(mol)
        xyzfile = output_3d_coords(atoms, atom_coords, output_format='xyz')

        with make_tmp_directory():
            with open("tmp.xyz", "w") as f:
                f.write(xyzfile)

            command = "xtb tmp.xyz --opt normal --gfn 2 --chrg {} --uhf {}".format(charge, num_unpaired_electrons)
            subprocess.check_call(command.split(), stdout=open('xtblog.txt', 'w'), stderr=open(os.devnull, 'w'))

            with open("xtbopt.xyz", "r") as f:
                lines = f.readlines()
                atoms = []
                atom_coords = []
                for line in lines[2:]:
                    atom, x, y, z = line.split()
                    x, y, z = map(float, [x, y, z])
                    atom_coords.append((x, y, z))
                    atoms.append(atom)

            for output_file in ['tmp.xyz', 'xtbopt.xyz', 'xtbopt.log', 'xtbtopo.mol', 'xtbrestart', 'wbo', 'tmp.ges', 'charges', 'xtblog.txt']:
                if os.path.exists(output_file):
                    os.remove(output_file)

    return atoms, atom_coords


def make_tmp_directory():
    """Makes a temporary directory to do things in, and then reverts back on exit."""
    prev_cwd = Path.cwd()
    if not os.path.exists('tmp_{}'.format(os.getpid())):
        os.mkdir('tmp_{}'.format(os.getpid()))
    os.chdir('tmp_{}'.format(os.getpid()))
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree('tmp_{}/'.format(os.getpid()))


def output_3d_coords(atoms: List[str], atom_coords: List[Tuple[float, float, float]], output_format: str = 'xyz') -> str:
    """
    Returns the coordinates of a molecule in the specified output format.
    Supported Output Formats: 'xyz', 'turbo'
    """
    if output_format not in {'xyz', 'turbo'}:
        raise ValueError('Invalid output format. Supported formats: xyz, turbo')

    logging.info('Outputting coordinates in {} format'.format(output_format))
    if output_format == 'xyz':
        output = str(len(atoms)) + '\n\n'
        for atom, (x, y, z) in zip(atoms, atom_coords):
            output += ' '.join([atom, str(x), str(y), str(z)]) + '\n'

    elif output_format == 'turbo':
        output = '$coord'
        for atom, (x, y, z) in zip(atoms, atom_coords):
            output += '\n' + ' '.join([str(x), str(y), str(z), atom.lower()])
        output += '\n$end\n'

    return output


class AimnetCalculator():
    """
    Uses the AIMNet-NSE model by Zubatyuk et al to calculate molecular energies with Machine Learning.
    Reference: https://chemrxiv.org/engage/chemrxiv/article-details/60c75793702a9b15d318cb0c
    Parts of this code is from the model's associated Github Repo, and the models used were downloaded from the associated Zenodo link.
    """

    def __init__(self, models_base_path: Optional[str] = None):
        if models_base_path is None:
            models_base_path = os.path.join(os.path.dirname(__file__), 'aimnet_models')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rd_periodic_table = Chem.GetPeriodicTable()
        self.models = [torch.jit.load(os.path.join(models_base_path, 'aimnet-nse-cv{}.jpt'.format(i))).to(self.device) for i in range(5)]

    def predict(self, data: Dict[str, Tensor], smi: str) -> float:
        """
        Predicts the energy of a molecule in tensorial representation
        Returns DFT calculations for single atom (with SMILES specified in smi)
        """
        #from mech_pred.utils import get_orca_energy
        if data['numbers'].size(dim=1) == 1:
            # if the molecule is a single atom, use Orca/DFT to calculate the energy since AIMNet doesn't work
            if smi == '[H+]':
                return 0.0  # proton has 0 energy
            # return get_orca_energy(smi, optimizer='rdkit')
        # change data to have alpha and beta total charge
        data['charge'] = 0.5 * torch.stack([data['charge'] - data['mult'] + 1, data['charge'] + data['mult'] - 1], dim=-1)
        return float(np.mean([model(data)['energy'][-1].cpu().numpy() for model in self.models]).tolist())

    def mol2data(self, smi: str, charge: float, mult: float, optimizer: str = 'xtb') -> Dict[str, Tensor]:
        """
        Changes a smiles string to the coordinate tensorial representation needed by the model.
        If optimizer == 'rdkit', then the default rdkit geometry optimization force field is used.
        If optimizer == 'xtb', then the BFGS method from xtb is used.
        """
        atoms, coords = smi_to_coords(smi, optimizer=optimizer)
        coord = np.array(coords)
        numbers = np.array([self.rd_periodic_table.GetAtomicNumber(a) for a in atoms])
        coord = torch.tensor(coord, dtype=torch.float).unsqueeze(0).repeat(1, 1, 1).to(self.device)
        numbers = torch.tensor(numbers, dtype=torch.long).unsqueeze(0).repeat(1, 1).to(self.device)
        charge = torch.tensor([charge]).to(self.device)  # cation, neutral, anion
        mult = torch.tensor([mult]).to(self.device)
        return dict(coord=coord, numbers=numbers, charge=charge, mult=mult)

    def get_energy(self, smi: str, optimizer: str = 'xtb') -> float:
        """
        Returns the energy predicted by AIMNet-NSE on the given molecular system, in eV.
        If optimizer == 'rdkit', then the default rdkit geometry optimization force field is used.
        If optimizer == 'xtb', then the BFGS method from xtb is used.
        """

        molecules = smi.split('.')
        system_energy = 0.0
        for molecule in molecules:
            molecule_charge = Chem.rdmolops.GetFormalCharge(Chem.MolFromSmiles(molecule))
            num_unpaired_electrons = Descriptors.NumRadicalElectrons(Chem.MolFromSmiles(molecule))
            logging.info('Calculating AIMNet energy for {}'.format(molecule))
            spin_multiplicity = num_unpaired_electrons + 1
            data = self.mol2data(molecule, charge=molecule_charge, mult=spin_multiplicity, optimizer=optimizer)
            with torch.jit.optimized_execution(False), torch.no_grad():
                pred = self.predict(data, smi=molecule)
            system_energy += pred
        return system_energy
    