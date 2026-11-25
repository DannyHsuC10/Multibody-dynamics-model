import unittest

import numpy as np

from Hydraulic_model.design import CylinderDesign, DamperDesign, import_project_circuit, solve_design
from Hydraulic_model.vehicle_coupling import vehicle_design
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

    def test_workbench_circuit_returns_reactions_for_vehicle_strokes(self):
        data = default_project()
        data["hydraulic_lines"] = self.lines
        data["rocker_dampers"] = [
            {"name": name, "a": {"mount": "body", "xyz": fixed},
             "b": {"mount": f"{corner}:LCA", "xyz": moving},
             "kind": "hydraulic", "hydraulic": self.cylinder}
            for name, corner, fixed, moving in (
                ("front", "FL", [.03, .30, .55], [.03, .46, .13]),
                ("rear", "RL", [-1.5, .30, .55], [-1.5, .46, .13]),
            )
        ]
        cylinders, hoses, anchors, lines, _ = import_project_circuit(data)
        cylinders[3] = CylinderDesign("Accumulator", bore_mm=16., rod_mm=8., compression_force_n=5.)
        hoses.add(((1, "cap"), (3, "cap")))
        springs = {1: DamperDesign("Accumulator spring", stiffness_n_mm=20.),
                   2: DamperDesign("Cylinder damper", stiffness_n_mm=0., damping_n_s_mm=2.)}
        links = {(1, "a"): None, (1, "b"): 3, (2, "a"): 1, (2, "b"): 2}
        data["hydraulic_design"] = vehicle_design(cylinders, hoses, anchors, lines, springs, links)
        model = build_dynamics_model(build_vehicle(data), data["geometry"]["vehicle"], data["dynamics"])
        data["motion"]["FL"].update(position_mm=12., velocity_mm_s=30.)
        result = solve_vehicle(model.geometry, data["motion"])
        springs_at_pose = sample_element_springs(model, result)
        readings = [springs_at_pose[corner][name] for corner, name in (("FL", "front"), ("RL", "rear"))]
        prescribed = {key: CylinderDesign(cylinders[key].name, cylinders[key].bore_mm, cylinders[key].rod_mm,
                                          prescribed_compression_mm=-reading.deflection * 1000.)
                      for key, reading in enumerate(readings, 1)}
        _, _, reactions = solve_design({**cylinders, **prescribed}, hoses, springs, links,
                                       anchors=anchors, project_lines=lines)
        damping = 2000. * (readings[0].velocity - readings[1].velocity)
        self.assertNotAlmostEqual(readings[0].deflection, 0.)
        self.assertAlmostEqual(readings[1].deflection, 0.)
        self.assertAlmostEqual(readings[0].force, reactions[1] - damping)
        self.assertAlmostEqual(readings[1].force, reactions[2] + damping)
