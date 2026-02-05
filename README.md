[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/chimie-paristech-CTM/reaction_possibility_enumerator/master/LICENSE)

# Reaction Enumeration Based on NBO-Informed Molecular Graphs

This repository contains the implementation of a **reaction possibility enumerator** based on **valence orbital permutations and rearrangements**, using NBO-informed molecular graphs.

## Setting Up the Environment

Create the conda environment using the provided configuration file:

```bash
conda env create --file environment.yml
```

Activate the environment and install the enumerator package from the repository root:

```bash
conda activate enumerator
pip install .
```

For the enumeration protocol based on NBO analysis, Gaussian must be available. On HPC systems, this is typically handled by loading the appropriate module, for example:

```bash
module load gaussian/G16_C01
```

## Execution

The enumerator can be run via:

```bash
python run_enumerator.py [-h] [--smiles SMILES] [--idx-list IDX_LIST [IDX_LIST ...]]
                         [--solvent SOLVENT] [--max-length MAX_LENGTH]
                         [--allow-zwitterions] [--print-configuration]
                         [--print-all-paths] [--nbo] [--nbo-dir NBO_DIR]
                         [--threshold-sec-interaction THRESHOLD_SEC_INTERACTION]
                         [--threshold-strong-sec-interaction THRESHOLD_STRONG_SEC_INTERACTION]
                         [--ts-tools] [--nproc NPROC]
                         [--num-unpaired-elec NUM_UNPAIRED_ELEC]
```

### Command-line options
* ``--smiles`` : SMILES representation of the reactants.
* ``--idx-list``: Constrain the enumeration protocol to a specific set of valence orbitals.
* ``--solvent``: Solvent used for xTB and NBO calculations.
* ``--print-all-paths``: Print all possible paths connecting the reactants to each product.
* ``--nbo``: Enable NBO calculations.
* ``--nbo-dir``: Extract NBO information from a previously completed calculation.
* ``--max-length``: Maximum number of valence orbitals included in an intrafragment path.
* ``--allow-zwitterions``: Boolean variable for wheter to print also zwitterions structuresAlso include zwitterionic structures in the output.
* ``--print-configuration``: Print the constructed valence orbital graph.
* ``--threshold-sec-interaction``: Energy threshold for filtering secondary interactions from NBO analysis
(default: 12.0 kcal mol⁻¹).
* ``--threshold-strong-sec-interaction``: Energy threshold for filtering strong secondary interactions from NBO analysis
(default: 85.0 kcal mol⁻¹).
* ``--ts-tools``: Generate a TXT file with reactions formatted for subsequent use in [TS-Tools](https://github.com/chimie-paristech-CTM/TS-tools)
* ``--nproc``: Number of processors used for xTB and NBO calculations.
* ``--num-unpaired-elec``: Number of unpaired electrons (for diradical species; RDKit defaults to multiplicity 1).

A typical run including NBO calculations:

```python run_enumerator.py --smiles "C=C.C=CC=C" --nbo --nproc 4 --ts-tools```

## Reaction network generation

In combination with the [TS-Tools](https://github.com/chimie-paristech-CTM/TS-tools) package, reaction networks can be generated using the `run_network.py` script. 

## References

If (parts of) this workflow are used as part of a publication please cite the associated paper:
```
@article{vo_enumerator,
  author = {Alfonso-Ramos, Javier E. and Stuyver, Thijs},
  title = {},
  journal = {Chemrxiv},
  year = {2026},
}
```