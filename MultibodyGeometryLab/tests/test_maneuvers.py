"""Regressions for prescribed driver inputs and spatial road profiles."""

import unittest

import numpy as np

from MultibodyGeometryLab.dynamics_settings import default_drive, default_vehicle_test, validate_vehicle_test
from MultibodyGeometryLab.geometry import build_vehicle
from MultibodyGeometryLab.maneuvers import ISORoad, Maneuver
from MultibodyGeometryLab.project import default_project
from MultibodyGeometryLab.solver import CornerSolver


class ManeuverTests(unittest.TestCase):
    def test_cruise_releases_to_the_requested_throttle_step(self):
        settings = default_vehicle_test("cruise_step")
        drive = default_drive()
        test = Maneuver(settings, drive, (), 0., -1.6)
        torque, integral_rate, target = test.propulsion(settings["start_seconds"] - .1, 7.9, 0.)
        self.assertGreater(torque, 0.)
        self.assertGreater(integral_rate, 0.)
        self.assertEqual(target, 8.)
        self.assertEqual(test.propulsion(settings["start_seconds"], 7.9, .1), (12., 0., None))

    def test_iso_profiles_scale_with_class_and_use_a_shared_same_side_track(self):
        corners = ("FL", "FR", "RL", "RR")
        station = np.array([3., 3., 3., 3.])
        speed = np.full(4, 8.)
        roads = [ISORoad(road_class, 7, corners, 0.) for road_class in ("A", "B", "C")]
        height, rate = roads[0].sample(station, speed)
        self.assertEqual(height[0], height[2])
        self.assertEqual(height[1], height[3])
        self.assertNotEqual(height[0], height[1])
        for multiplier, road in enumerate(roads[1:], 1):
            np.testing.assert_allclose(road.sample(station, speed), 2**multiplier * np.array([height, rate]))
        h = 1e-5
        before, _ = roads[0].sample(station - h * speed, speed)
        after, _ = roads[0].sample(station + h * speed, speed)
        np.testing.assert_allclose(rate, (after - before) / (2. * h), atol=1e-7)
        with self.assertRaises(ValueError):
            validate_vehicle_test({**default_vehicle_test("iso_road"), "road_seed": 1.2})

    def test_moving_rack_preserves_full_kinematic_derivatives(self):
        solver = CornerSolver(build_vehicle(default_project()).corners[0])
        travel = .012, -.08, .15
        rack = .004, .015, -.05
        path = solver.wheel_travel(travel[0], rack=rack[0])
        motion = solver.wheel_motion(path, travel[1], rack[1], rack[2])
        result = solver.wheel_result(path, travel[1], travel[2], motion=motion)
        expected = solver.solve(travel, rack)
        np.testing.assert_allclose(result.position, expected.position, atol=1e-8)
        np.testing.assert_allclose(result.velocity, expected.velocity, atol=1e-8)
        np.testing.assert_allclose(result.acceleration, expected.acceleration, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
