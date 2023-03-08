from enumerator.enumerate import enumerate_reaction_possibilities, get_energies

import logging
import time

#mol = Molecule('CCCC(=O)OC(=O)N.C#N')
#mol = Molecule('C.C#[N+2]')
# mol = Molecule('F')

if __name__=="__main__": 
    start_time = time.time()
    products = enumerate_reaction_possibilities()
    get_energies(products)
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
