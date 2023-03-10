from enumerator.Molecule import Molecule, ValenceOrbital, BondingSystem
from enumerator.generate_products import (
    determine_reaction_type,
    set_polarization_bonding_systems,
    construct_new_bonding_system,
)
from enumerator.generate_products import (
    enumerate_reaction_possibilities,
    split_single_bonding_system,
)
import unittest


class TestMyModule(unittest.TestCase):
    def test_determine_reaction_type(self):
        # Test the case of two valence orbitals.
        bs1 = BondingSystem(1)
        vo1 = ValenceOrbital(1, 1, "C")
        vo2 = ValenceOrbital(2, 2, "C")
        bs1.add_vo(vo1)
        bs1.add_vo(vo2)
        self.assertEqual(determine_reaction_type(bs1), "concerted")

        # Test the case of one valence orbital with 2 electrons.
        bs2 = BondingSystem(2)
        vo3 = ValenceOrbital(3, 2, "C")
        bs2.add_vo(vo3)
        vo3.set_population(0)
        self.assertEqual(determine_reaction_type(bs2), "electrophilic")

        # Test the case of one valence orbital with 1 electron.
        bs3 = BondingSystem(3)
        vo4 = ValenceOrbital(4, 1, "C")
        bs3.add_vo(vo4)
        vo4.set_population(1)
        self.assertEqual(determine_reaction_type(bs3), "radical")

        # Test the case of one valence orbital with 0 electrons.
        bs4 = BondingSystem(4)
        vo5 = ValenceOrbital(5, 0, "C")
        bs4.add_vo(vo5)
        vo5.set_population(2)
        self.assertEqual(determine_reaction_type(bs4), "nucleophilic")

    def test_set_polarization_bonding_systems(self):
        # Test with a list of 2 bonding systems, one polar and one non-polar.
        bs1 = BondingSystem(1)
        vo1 = ValenceOrbital(1, 0, "H")
        vo2 = ValenceOrbital(2, 1, "C")
        bs1.add_vo(vo1)
        bs1.add_vo(vo2)

        bs2 = BondingSystem(2)
        vo3 = ValenceOrbital(3, 2, "C")
        bs2.add_vo(vo3)

        bs3 = BondingSystem(3)
        vo4 = ValenceOrbital(1, 0, "C")
        vo5 = ValenceOrbital(1, 0, "C")
        bs3.add_vo(vo4)
        bs3.add_vo(vo5)

        bonding_systems = [bs1, bs2, bs3]
        set_polarization_bonding_systems(bonding_systems)
        self.assertTrue(bonding_systems[0].polarity_set)
        self.assertTrue(bs1.vos[0].atom_type == "C" and bs1.vos[1].atom_type == "H")
        self.assertFalse(bonding_systems[1].polarity_set)
        self.assertFalse(bonding_systems[2].polarity_set)

    def test_construct_new_bonding_system(self):
        # Test with two valence orbitals.
        vo1 = ValenceOrbital(1, 0, "C")
        vo2 = ValenceOrbital(2, 1, "C")
        bs1 = construct_new_bonding_system(vo1, vo2, idx=1)
        self.assertIsInstance(bs1, BondingSystem)
        self.assertEqual(len(bs1), 2)
        self.assertEqual(bs1.vos[0], vo1)
        self.assertEqual(bs1.vos[1], vo2)
        self.assertEqual(bs1.idx, 1)

    def test_enumerate_reaction_possibilites(self):
        max_length = 3
        products = enumerate_reaction_possibilities(Molecule("C=C.C=C"), max_length)
        self.assertIsInstance(products, list)

    def test_split_single_bonding_system(self):
        vo1 = ValenceOrbital(1, 0, "C")
        vo2 = ValenceOrbital(2, 1, "C")
        old_bonding_system = construct_new_bonding_system(vo1, vo2)
        num_electrons = 2
        population_first_vo = 1
        new_bonding_systems = split_single_bonding_system(
            old_bonding_system, num_electrons, population_first_vo
        )
        self.assertIsInstance(new_bonding_systems, list)
        self.assertEqual(len(new_bonding_systems), 2)
        self.assertIsInstance(new_bonding_systems[0], BondingSystem)
        self.assertIsInstance(new_bonding_systems[1], BondingSystem)
        self.assertEqual(new_bonding_systems[0].idx, -1)
        self.assertEqual(new_bonding_systems[1].idx, -1)
        self.assertEqual(len(new_bonding_systems[0].vos), 1)
        self.assertEqual(len(new_bonding_systems[1].vos), 1)
        self.assertEqual(
            new_bonding_systems[0].vos[0].num_electrons, population_first_vo
        )
        self.assertEqual(
            new_bonding_systems[1].vos[0].num_electrons,
            num_electrons - population_first_vo,
        )

    def test_enumerate_reaction_possibilities1(self):
        molecule = Molecule("CC")
        expected_products = [
            "[H]C([H])=C([H])[H].[H][H]",
            "[H].[H][C]([H])C([H])([H])[H]",
            "[H+].[H-].[H]C([H])=C([H])[H]",
            "[H+].[H][C-]([H])C([H])([H])[H]",
            "[H]C([H])([H])C([H])([H])[H]",
            "[H].[H].[H]C([H])=C([H])[H]",
        ]
        products = enumerate_reaction_possibilities(molecule, 4)
        for expected_product in expected_products:
            self.assertIn(expected_product, products)

    def test_enumerate_reaction_possibilities2(self):
        molecule = Molecule("C.C#N")
        expected_products = [
            "[H]C([H])=[N].[H][C]([H])[H]",
            "[H].[H]C(=[N])C([H])([H])[H]",
            "[H+].[H]C(=[N-])C([H])([H])[H]",
            "[H]C([H])=NC([H])([H])[H]",
            "[H]C([H])([H])C#N.[H][H]",
            "[C]#N.[H][C]([H])[H].[H][H]",
        ]
        products = enumerate_reaction_possibilities(molecule, 4)
        for expected_product in expected_products:
            self.assertIn(expected_product, products)

    def test_enumerate_reaction_possibilities3(self):
        molecule = Molecule("[CH3].N")
        expected_products = [
            "[H]C([H])([H])[H].[H][N][H]",
            "[H][C]([H])N([H])[H].[H][H]",
            "[H].[H]N([H])C([H])([H])[H]",
            "[H][C-][H].[H][N+]([H])([H])[H]",
        ]
        products = enumerate_reaction_possibilities(molecule, 4)
        for expected_product in expected_products:
            self.assertIn(expected_product, products)

    def test_enumerate_reaction_possibilities4(self):
        molecule = Molecule("[CH3+].N")
        expected_products = [
            "[H]C([H])([H])[N+]([H])([H])[H]",
            "[H+].[H]N([H])C([H])([H])[H]",
        ]
        products = enumerate_reaction_possibilities(molecule, 4)
        for expected_product in expected_products:
            self.assertIn(expected_product, products)

    def test_enumerate_reaction_possibilities4(self):
        molecule = Molecule("[CH3-].N")
        expected_products = ["[H]C([H])([H])[H].[H][N-][H]"]
        products = enumerate_reaction_possibilities(molecule, 4)
        for expected_product in expected_products:
            self.assertIn(expected_product, products)


if __name__ == "__main__":
    unittest.main()
