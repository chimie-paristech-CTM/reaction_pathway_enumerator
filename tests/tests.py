from enumerator.enumerate import enumerate_potential_products
from enumerator.reacting_system import ReactingSystem
from rdkit import Chem
import unittest
from rdkit import rdBase
from rdkit import RDLogger

# Suppress RDKit warnings
rdBase.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.*')

def get_products_without_map_numbers(product_list):
    final_product_list = []
    for product in product_list:
        mol = Chem.MolFromSmiles(product)
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
        final_product_list.append(Chem.MolToSmiles(mol))

    return final_product_list

class TestMyModule(unittest.TestCase):
    def test_formose_reaction(self):
        formose_products = enumerate_potential_products('C=O.C=CO.O', None, allow_zwitterions=False)
        formose_products = get_products_without_map_numbers(formose_products)

        self.assertIn('O.O=CCCO', formose_products)
        self.assertIn('C=CO.OCO', formose_products)
        self.assertIn('C=COCO.O', formose_products)
        self.assertIn('CC=O.OCO', formose_products)

    def test_eas_reaction(self):
        eas_products = enumerate_potential_products('Cc1ccccc1.Cl.ClCl', None, allow_zwitterions=False)
        eas_products = get_products_without_map_numbers(eas_products)

        self.assertIn('CC1=CC(Cl)C(Cl)C=C1.Cl', eas_products)
        self.assertIn('Cc1ccc(Cl)cc1.Cl.Cl', eas_products)
        self.assertIn('Cc1cccc(Cl)c1.Cl.Cl', eas_products)
        self.assertIn('Cc1ccccc1Cl.Cl.Cl', eas_products)

    def test_deprotonation_reaction1(self):
        deprot_products = enumerate_potential_products('O.O=C1C=C[NH+]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)C(=O)N1', None, allow_zwitterions=False)
        deprot_products = get_products_without_map_numbers(deprot_products)

        self.assertIn('O=c1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1.[OH3+]', deprot_products)


    # TODO: Molecules with P will only work once you have NBO implemented -> for now use P+O-
    def test_deprotonation_reaction2(self):
        deprot_products = enumerate_potential_products('O=P([O-])(O)O.O=C1C=C[NH+]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)C(=O)N1', None, allow_zwitterions=False)
        deprot_products = get_products_without_map_numbers(deprot_products)
        
        self.assertIn('O=P(O)(O)O.O=c1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1', deprot_products)



if __name__ == "__main__":
    unittest.main()
