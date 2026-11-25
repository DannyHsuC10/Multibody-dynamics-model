"""Regression tests for grounded and rod-to-rod spring elements."""

from dataclasses import replace
import unittest

from .design import CylinderDesign, import_project_circuit, pressure_groups, solve_design
from .model import GROUND, HydraulicCylinder, LinearSpring, SpringElement
from .static_network import PressureLine, StaticHydraulicSystem


class VehicleCircuitImportTests(unittest.TestCase):
    def test_named_single_port_preserves_pressure_and_compliance(self):
        data = {
            "rocker_dampers": [{
                "name": "Cylinder", "kind": "hydraulic",
                "a": {"mount": "FR"}, "b": {"mount": "member:csv:b:Crossbar"},
                "hydraulic": {"bore_mm": 20., "rod_mm": 10., "port_a": "Supply", "port_b": ""},
            }],
            "hydraulic_lines": {
                "Supply": {"pressure_mpa": 1.5, "compliance_cm3_mpa": 5.},
                "P1": {"pressure_mpa": .5, "compliance_cm3_mpa": 2.},
            },
        }
        cylinders, hoses, anchors, lines, legacy = import_project_circuit(data)
        self.assertEqual(legacy, 0)
        self.assertEqual(cylinders[1].project_endpoints, ("FR 搖臂", "Crossbar · B 端"))
        self.assertIsNone(CylinderDesign("Manual").project_endpoints)
        self.assertEqual(pressure_groups(hoses, anchors, reserved_names=lines), {(1, "cap"): "Supply"})
        cylinders[1] = replace(cylinders[1], prescribed_compression_mm=2.)
        pressures, compression, reactions = solve_design(
            cylinders, hoses, {}, {}, anchors=anchors, project_lines=lines)
        area = HydraulicCylinder(20.).area_mm2
        expected_pressure = 1.5 + 2. * area / 5000.
        self.assertAlmostEqual(pressures["Supply"], expected_pressure)
        self.assertNotIn("P1", pressures)
        self.assertAlmostEqual(compression[1], 2.)
        self.assertAlmostEqual(reactions[1], area * expected_pressure)


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


class DisplacementControlTests(unittest.TestCase):
    def test_identical_cylinders_can_have_unique_reactions_with_unknown_pressures(self):
        a, b, c = (HydraulicCylinder(20, rod_diameter_mm=10) for _ in range(3))
        p1 = PressureLine("P1", (a.cap_end, b.cap_end, c.cap_end))
        p2 = PressureLine("P2", (a.rod_end, b.rod_end, c.rod_end))
        spring = SpringElement(LinearSpring(100), c.rod, GROUND)
        system = StaticHydraulicSystem((p1, p2), springs=(spring,))

        with self.assertRaisesRegex(ValueError, "cannot satisfy"):
            system.solve({a.rod: -100, b.rod: -180})

        result = system.solve({}, prescribed_extension_mm={a.rod: -1, b.rod: -2})

        self.assertAlmostEqual(result.rod_extension_mm[c.rod], 3.)
        self.assertAlmostEqual(result.required_rod_forces_n[a.rod], -300.)
        self.assertAlmostEqual(result.required_rod_forces_n[b.rod], -300.)
        self.assertEqual(result.pressures_mpa, {"P1": None, "P2": None})

    def test_mixed_force_and_displacement_reproduce_original_equilibrium(self):
        a, b, c = (HydraulicCylinder(20, rod_diameter_mm=diameter) for diameter in (8, 16, 12))
        p1 = PressureLine("P1", (a.cap_end, b.cap_end, c.cap_end))
        p2 = PressureLine("P2", (a.rod_end, b.rod_end, c.rod_end))
        spring = SpringElement(LinearSpring(100), c.rod, GROUND)
        system = StaticHydraulicSystem((p1, p2), springs=(spring,))

        result = system.solve({b.rod: -180}, prescribed_extension_mm={a.rod: -7 / 9})

        self.assertAlmostEqual(result.rod_extension_mm[b.rod], -5 / 9)
        self.assertAlmostEqual(result.rod_extension_mm[c.rod], 4 / 3)
        self.assertAlmostEqual(result.required_rod_forces_n[a.rod], -100.)
        self.assertIsNotNone(result.pressures_mpa["P1"])

    def test_prescribed_displacements_must_conserve_oil(self):
        a, b = (HydraulicCylinder(20, rod_diameter_mm=10) for _ in range(2))
        line = PressureLine("P1", (a.cap_end, b.cap_end))
        system = StaticHydraulicSystem((line,))
        with self.assertRaisesRegex(ValueError, "oil-volume conservation"):
            system.solve({}, prescribed_extension_mm={a.rod: -1, b.rod: -1})

        result = system.solve({}, prescribed_extension_mm={a.rod: -1, b.rod: 1})
        self.assertEqual(result.required_rod_forces_n, {a.rod: None, b.rod: None})


if __name__ == "__main__":
    unittest.main()
