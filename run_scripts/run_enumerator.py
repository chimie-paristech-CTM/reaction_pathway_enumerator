from enumerator.enumerate import get_thermodynamically_feasible_products
from rdkit import RDLogger

import logging
import time

if __name__ == "__main__":
    RDLogger.DisableLog('rdApp.*') 
    start_time = time.time()
    get_thermodynamically_feasible_products()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
