import unittest

import numpy as np

from MultibodyGeometryLab.deformable import build_dynamics_model, sample_element_springs
from MultibodyGeometryLab.geometry import build_vehicle
from MultibodyGeometryLab.hydraulic_circuits import HydraulicNetwork
from MultibodyGeometryLab.project import default_project
from MultibodyGeometryLab.solver import solve_vehicle


class HydraulicCircuitTests(unittest.TestCase):
    def setUp(self):
        self.lines = {
            "supply": {"pressure_mpa": 1., "compliance_cm3_mpa": 5.},
            "return": {"pressure_mpa": .6, "compliance_cm3_mpa": 4.},
        }
        self.cylinder = {"bore_mm": 25., "rod_mm": 12., "port_a": "supply", "port_b": "return"}

    def test_two_ports_and_three_way_branch_share_pressure(self):
        network = HydraulicNetwork.build(self.lines, [self.cylinder] * 3)
        pressure, force, stiffness = network.response(np.zeros(3))
        np.testing.assert_allclose(pressure, [1.e6, .6e6])
        cap = np.pi * .025**2 / 4
        annulus = cap - np.pi * .012**2 / 4
        np.testing.assert_allclose(force, 1.e6 * cap - .6e6 * annulus)
        moved_pressure, moved_force, _ = network.response([.001, 0., 0.])
        self.assertLess(moved_pressure[0], pressure[0])
        self.assertGreater(moved_pressure[1], pressure[1])
        np.testing.assert_allclose(moved_force, force - stiffness[:, 0] * .001)
        self.assertLess(moved_force[2], force[2])

    def test_moving_one_arm_changes_force_at_another_cylinder(self):
        data = default_project()
        data["hydraulic_lines"] = self.lines

        def cylinder(name, a, b):
            return {"name": name, "a": {"mount": a[0], "xyz": a[1]}, "b": {"mount": b[0], "xyz": b[1]},
                    "kind": "hydraulic", "hydraulic": self.cylinder}

        data["rocker_dampers"] = [
            cylinder("cross", ("FL:UCA", [.01, .49, .30]), ("FR:UPRIGHT", [.01, -.49, .19])),
            cylinder("right", ("body", [.03, -.30, .56]), ("FR:LCA", [.03, -.46, .13])),
            cylinder("rear", ("body", [-1.5, .30, .55]), ("RL:LCA", [-1.5, .46, .13])),
        ]
        model = build_dynamics_model(build_vehicle(data), data["geometry"]["vehicle"], data["dynamics"])
        rest = sample_element_springs(model, solve_vehicle(model.geometry, data["motion"]))
        data["motion"]["FL"]["position_mm"] = 10.
        moved = sample_element_springs(model, solve_vehicle(model.geometry, data["motion"]))
        self.assertAlmostEqual(moved["RL"]["rear"].length, rest["RL"]["rear"].length)
        self.assertGreater(abs(moved["RL"]["rear"].force - rest["RL"]["rear"].force), 1.)
