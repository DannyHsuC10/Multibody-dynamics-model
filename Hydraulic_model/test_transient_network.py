"""Regression tests for transient forces and pressure equalization."""

import unittest
from math import exp

import numpy as np

from .model import GROUND, HydraulicCylinder, LinearSpring, SpringElement
from .static_network import PressureLine, StaticHydraulicSystem
from .transient_network import TransientSettings, simulate_transient


class TransientNetworkTests(unittest.TestCase):
    def test_imported_compliance_and_precharge_act_on_prescribed_rods(self):
        cylinder = HydraulicCylinder(20.)
        line = PressureLine("Supply", (cylinder.cap_end,), compliance_cm3_mpa=5., initial_pressure_mpa=1.5)
        system = StaticHydraulicSystem((line,))
        settings = TransientSettings(duration_s=.15, ramp_s=.05, line_delay_enabled=True)

        trace = simulate_transient(system, {}, {cylinder.rod: -2.}, settings)

        pressure_change = 2. * cylinder.area_mm2 / 5000.
        self.assertAlmostEqual(trace.pressure_change_mpa["Supply"][-1], pressure_change, places=6)
        reaction = trace.required_rod_forces_n[cylinder.rod]
        self.assertAlmostEqual(reaction[0], -1.5 * cylinder.area_mm2)
        self.assertAlmostEqual(reaction[-1], -(1.5 + pressure_change) * cylinder.area_mm2, places=4)

    def test_damper_uses_relative_velocity_and_equal_opposite_forces(self):
        a, b = HydraulicCylinder(20.), HydraulicCylinder(20.)
        element = SpringElement(LinearSpring(100.), a.rod, b.rod, damping_n_s_mm=2.)
        system = StaticHydraulicSystem((), cylinders=(a, b), springs=(element,))
        settings = TransientSettings(duration_s=.2, ramp_s=.1, damping_n_s_mm=0.)
        result = simulate_transient(system, {}, {a.rod: 1., b.rod: 0.}, settings)
        middle = np.argmin(np.abs(result.time_s - .05))

        self.assertAlmostEqual(result.required_rod_forces_n[a.rod][middle], 50. + 2. * 18.75)
        self.assertAlmostEqual(result.required_rod_forces_n[b.rod][middle], -50. - 2. * 18.75)
        self.assertAlmostEqual(result.required_rod_forces_n[a.rod][-1], 100.)

    def test_prescribed_rod_drives_a_free_rod_through_damping_alone(self):
        a, b = HydraulicCylinder(20.), HydraulicCylinder(20.)
        element = SpringElement(None, a.rod, b.rod, damping_n_s_mm=.1)
        system = StaticHydraulicSystem((), cylinders=(a, b), springs=(element,))
        settings = TransientSettings(duration_s=.3, ramp_s=.1, damping_n_s_mm=0.)

        result = simulate_transient(system, {}, {a.rod: 1.}, settings)

        self.assertAlmostEqual(result.rod_extension_mm[b.rod][-1], 1., places=5)

    def test_ramped_force_settles_to_the_spring_deflection(self):
        cylinder = HydraulicCylinder(20.)
        spring = SpringElement(LinearSpring(100.), cylinder.rod, GROUND)
        system = StaticHydraulicSystem((), cylinders=(cylinder,), springs=(spring,))
        settings = TransientSettings(duration_s=.3, ramp_s=.05, damping_n_s_mm=.5)

        trace = simulate_transient(system, {cylinder.rod: -100.}, {}, settings)

        extension = trace.rod_extension_mm[cylinder.rod]
        self.assertEqual(extension[0], 0.)
        self.assertAlmostEqual(extension[-1], -1., places=5)

    def test_delay_equalizes_ports_with_the_specified_time_constant(self):
        a, b = HydraulicCylinder(20.), HydraulicCylinder(20.)
        line = PressureLine("P1", (a.cap_end, b.cap_end))
        system = StaticHydraulicSystem((line,))
        prescribed = {a.rod: -1., b.rod: 0.}
        ideal = simulate_transient(system, {}, prescribed, TransientSettings(duration_s=.25, ramp_s=.05))
        settings = TransientSettings(duration_s=.25, ramp_s=.05, line_delay_enabled=True, line_time_constant_s=.05)
        delayed = simulate_transient(system, {}, prescribed, settings)

        a_pressure = delayed.port_pressure_change_mpa["P1", a.rod, "cap"]
        b_pressure = delayed.port_pressure_change_mpa["P1", b.rod, "cap"]
        end_ramp = np.searchsorted(delayed.time_s, .05)
        at_tau = np.searchsorted(delayed.time_s, .1)
        gap = a_pressure - b_pressure
        self.assertGreater(gap[end_ramp], 0.)
        elapsed = delayed.time_s[at_tau] - delayed.time_s[end_ramp]
        self.assertAlmostEqual(gap[at_tau] / gap[end_ramp], exp(-elapsed / .05), places=5)
        self.assertAlmostEqual(delayed.pressure_change_mpa["P1"][-1], ideal.pressure_change_mpa["P1"][-1], places=5)
        self.assertAlmostEqual(delayed.required_rod_forces_n[b.rod][-1], -b.area_mm2 * b_pressure[-1])


if __name__ == "__main__":
    unittest.main()
