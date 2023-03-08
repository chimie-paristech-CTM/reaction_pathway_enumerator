from enumerator.enumerate import enumerate

import logging
import time

#mol = Molecule('CCCC(=O)OC(=O)N.C#N')
#mol = Molecule('C.C#[N+2]')
# mol = Molecule('F')

if __name__=="__main__": 
    start_time = time.time()
    enumerate()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
