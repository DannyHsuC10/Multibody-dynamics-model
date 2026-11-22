import csv
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from MultibodyGeometryLab.deformable import build_dynamics_model, sample_element_springs
from MultibodyGeometryLab.elements import DAMPER_POINT, Member, default_elements
from MultibodyGeometryLab.geometry import POINTS, build_vehicle, static_geometry
from MultibodyGeometryLab.hardpoint_import import import_hardpoint_csv
from MultibodyGeometryLab.project import default_project, load_project, save_project
from MultibodyGeometryLab.rigid_body import simulate_rigid_road
from MultibodyGeometryLab.solver import solve_vehicle


class HardpointImportTests(unittest.TestCase):
    def test_csv_rocker_geometry_and_coefficients_drive_coupled_dynamics(self):
        base = default_project()
        arms, _, centers = static_geometry(base["geometry"])
        supplemental = base["geometry"]["kinematics_hardpoints"]["corners"]
        members = [
            {"name": "Roll", "a": {"xyz": [0.04, -.20, .36], "mount": "FR"},
             "b": {"xyz": [0.04, .20, .29], "mount": "FL"}},
            {"name": "Heave", "a": {"xyz": [0.04, -.23, .36], "mount": "FR"},
             "b": {"xyz": [0.04, .23, .36], "mount": "FL"}},
        ]
        with TemporaryDirectory() as directory:
            source, project = (Path(directory) / name for name in ("geometry.csv", "project.json"))
            metadata = json.dumps({"version": 1, "deformable_members": members})
            with source.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(("name", "x_m", "y_m", "z_m", "jacobian_project"))
                for name, corners in arms.items():
                    for key, point in {**corners, **supplemental[name], "WC": centers[name]}.items():
                        writer.writerow((f"{name}_{key}", *point, metadata))
                        metadata = ""
            data = import_hardpoint_csv(base, source)
            data["rocker_dampers"][0].update(spring_n_m=45000., damper_ns_m=0.)
            save_project(project, data)
            data = load_project(project)

        geometry = build_vehicle(data)
        np.testing.assert_allclose(geometry.corners[0].reference[POINTS.index("RACK")], supplemental["FL"]["RACK"])
        model = build_dynamics_model(geometry, data["geometry"]["vehicle"], data["dynamics"])
        self.assertEqual(model.unbalanced_corners, ("RL", "RR"))
        self.assertTrue(all(member.kind == "rigid" for corner in model.geometry.corners for member in corner.members))
        self.assertTrue(all(not members for members in model.elastic_members.values()))
        neutral = solve_vehicle(model.geometry, data["motion"])
        rest = sample_element_springs(model, neutral)["FL"]
        np.testing.assert_allclose(rest["Roll"].endpoints, [members[0][key]["xyz"] for key in ("a", "b")], atol=1e-12)
        data["motion"]["FL"].update(position_mm=8., velocity_mm_s=10.)
        moved = sample_element_springs(model, solve_vehicle(model.geometry, data["motion"]))["FL"]["Roll"]
        self.assertGreater(abs(moved.deflection), 1e-5)
        self.assertAlmostEqual(moved.force - rest["Roll"].force, -45000. * moved.deflection)
        data["dynamics"]["elements"] = default_elements(csv_mechanism=True)
        for corner in ("RL", "RR"):
            data["dynamics"]["elements"][corner].append(asdict(Member("支撐", DAMPER_POINT, "UBJ", "spring_damper")))
        model = build_dynamics_model(geometry, data["geometry"]["vehicle"], data["dynamics"])
        self.assertEqual(model.unbalanced_corners, ())
        data["dynamics"]["simulation"].update(bump_mm=0., duration_seconds=.5)
        frames = simulate_rigid_road(model, data["dynamics"], data["geometry"]["vehicle"]["mass"])
        np.testing.assert_allclose(frames[-1].body.displacement, 0., atol=1e-10)

        data["rocker_dampers"][0].update(kind="hydraulic", hydraulic_force_n=125.)
        tie_rod = next(member for member in data["dynamics"]["elements"]["FL"] if member["b"] == "RACK")
        tie_rod.update(kind="hydraulic", hydraulic_force_n=-260.)
        data["dynamics"]["elements"]["FL"].append(asdict(Member("導桿", "RACK", "WC", "rigid")))
        model = build_dynamics_model(build_vehicle(data), data["geometry"]["vehicle"], data["dynamics"])
        readings = sample_element_springs(model, solve_vehicle(model.geometry, data["motion"]))["FL"]
        self.assertAlmostEqual(readings["Roll"].force, 125.)
        self.assertAlmostEqual(readings[tie_rod["name"]].force, -260.)
        moved = solve_vehicle(model.geometry, data["motion"]).corners["FL"]
        self.assertNotIn(("TOE", "RACK"), moved.rigid_links)
        self.assertIn(("RACK", "WC"), moved.rigid_links)

    def test_direct_csv_pushrod_mounts_on_each_supported_rigid_body(self):
        base = default_project()
        arms, _, centers = static_geometry(base["geometry"])
        supplemental = base["geometry"]["kinematics_hardpoints"]["corners"]
        pin, pickup = [.010, .300, .600], [.030, .490, .128]
        for outboard in ("LCA", "UCA", "UPRIGHT"):
            with self.subTest(outboard=outboard), TemporaryDirectory() as directory:
                source = Path(directory) / "direct.csv"
                metadata = {"version": 1, "pushrod_mounts": {"FL": outboard}, "deformable_members": []}
                with source.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("name", "x_m", "y_m", "z_m", "jacobian_project"))
                    writer.writerow(("FL_PUSH_IN", *pin, json.dumps(metadata)))
                    writer.writerow(("FL_PUSH_OUT", *pickup, ""))
                    for name, points in arms.items():
                        for key, point in {**points, "WC": centers[name], "RACK": supplemental[name]["RACK"],
                                           "TOE": supplemental[name]["TOE"]}.items():
                            writer.writerow((f"{name}_{key}", *point, ""))
                data = import_hardpoint_csv(base, source)
                rod = data["pushrods"]["FL"]
                self.assertEqual((rod["inboard"], rod["outboard"]), ("body", outboard))
                model = build_dynamics_model(build_vehicle(data), data["geometry"]["vehicle"], data["dynamics"])

                def sample(travel, speed=0.):
                    data["motion"]["FL"].update(position_mm=travel, velocity_mm_s=speed)
                    return sample_element_springs(model, solve_vehicle(model.geometry, data["motion"]))["FL"]["FL_PUSH_ROD"]

                neutral = sample(0.)
                np.testing.assert_allclose(neutral.endpoints, [pin, pickup], atol=1e-12)
                moved = sample(10., 50.)
                self.assertGreater(abs(moved.deflection), .001)
                np.testing.assert_allclose(moved.endpoints[0], pin, atol=1e-12)
                difference = (sample(10.005).length - sample(9.995).length) / .0002
                self.assertAlmostEqual(moved.velocity, difference, places=6)
