"""Regression tests for grounded and rod-to-rod spring elements."""

import unittest

from .model import GROUND, HydraulicCylinder, LinearSpring, SpringElement
from .static_network import PressureLine, StaticHydraulicSystem


class SpringElementTests(unittest.TestCase):
    def test_coupled_rods_share_load_through_the_spring(self):
        a, b = HydraulicCylinder(20), HydraulicCylinder(20)
        anchor = SpringElement(LinearSpring(100), a.rod, GROUND)
        coupling = SpringElement(LinearSpring(200), a.rod, b.rod)
        system = StaticHydraulicSystem((), cylinders=(a, b), springs=(anchor, coupling))

        result = system.solve({a.rod: -100, b.rod: -200})

        self.assertAlmostEqual(result.rod_extension_mm[a.rod], -3.)
        self.assertAlmostEqual(result.rod_extension_mm[b.rod], -4.)

    def test_grounded_spring_preserves_three_cylinder_static_solution(self):
        a, b, c = (HydraulicCylinder(20, rod_diameter_mm=diameter) for diameter in (8, 16, 12))
        supply = PressureLine("P1", (a.cap_end, b.cap_end, c.cap_end))
        return_line = PressureLine("P2", (a.rod_end, b.rod_end, c.rod_end))
        spring = SpringElement(LinearSpring(100), c.rod, GROUND)
        system = StaticHydraulicSystem((supply, return_line), springs=(spring,))

        result = system.solve({a.rod: -100, b.rod: -180})

        self.assertAlmostEqual(result.rod_extension_mm[a.rod], -7 / 9)
        self.assertAlmostEqual(result.rod_extension_mm[b.rod], -5 / 9)
        self.assertAlmostEqual(result.rod_extension_mm[c.rod], 4 / 3)


if __name__ == "__main__":
    unittest.main()
