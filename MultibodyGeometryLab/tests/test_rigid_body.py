"""Mass, force, and moment balance regressions for the rigid chassis."""

import unittest

import numpy as np

from MultibodyGeometryLab.deformable import build_dynamics_model
from MultibodyGeometryLab.geometry import FIXED, POINTS, build_vehicle
from MultibodyGeometryLab.project import default_project
from MultibodyGeometryLab.rigid_body import simulate_rigid_drive, simulate_rigid_lateral, simulate_rigid_road


INERTIA = {"ixx": 55., "iyy": 95., "izz": 125.}
HUB = POINTS.index("WC")


def road_settings(settings, bump_mm, inertia):
    return {
        **settings,
        "chassis_inertia": inertia.copy(),
        "simulation": {"target": "FL", "bump_mm": bump_mm,
                       "bump_seconds": .24, "duration_seconds": .50},
    }


class RigidBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project = default_project()
        cls.vehicle = project["geometry"]["vehicle"]
        cls.settings = road_settings(project["dynamics"], 20., INERTIA)
        cls.geometry = build_vehicle(project)
        cls.model = build_dynamics_model(cls.geometry, cls.vehicle, cls.settings)
        cls.frames = simulate_rigid_road(cls.model, cls.settings, cls.vehicle["mass"])

    def test_no_input_preserves_equilibrium_with_asymmetric_wheel_masses(self):
        project = default_project()
        settings = road_settings(project["dynamics"], 0., INERTIA)
        settings["corners"]["FL"]["mass_kg"] = 15.
        settings["corners"]["RR"]["mass_kg"] = 8.
        model = build_dynamics_model(self.geometry, self.vehicle, settings)
        mass = self.vehicle["mass"]
        frames = simulate_rigid_road(model, settings, mass)
        sprung_mass = mass - sum(spec.mass for spec in model.corners.values())
        wheel_moments = sum(model.corners[corner.name].mass * np.asarray(corner.reference[HUB])
                            for corner in self.geometry.corners)
        self.assertAlmostEqual(frames[-1].time, .50)
        for frame in frames:
            # The geometry specifies total-vehicle CG, so the sprung pivot
            # and wheel masses together must reproduce that CG at rest.
            np.testing.assert_allclose(sprung_mass * frame.pivot + wheel_moments,
                                       mass * np.asarray(self.geometry.cg), atol=1e-7, rtol=0)
            np.testing.assert_allclose(frame.translation, 0., atol=1e-8, rtol=0)
            np.testing.assert_allclose(frame.rotation, np.eye(3), atol=1e-8, rtol=0)
            np.testing.assert_allclose(frame.body.acceleration, 0., atol=1e-7, rtol=0)
            np.testing.assert_allclose(frame.body.angular_acceleration, 0., atol=1e-7, rtol=0)

    def test_single_wheel_bump_rotates_chassis_and_changes_with_roll_inertia(self):
        baseline_angles = np.array([frame.body.angles_deg for frame in self.frames])
        self.assertGreater(np.ptp(baseline_angles[:, 0]), .08)
        self.assertGreater(np.ptp(baseline_angles[:, 1]), .08)

        heavier_roll = road_settings(self.settings, 20., {**INERTIA, "ixx": 105.})
        frames = simulate_rigid_road(self.model, heavier_roll, self.vehicle["mass"])
        early = next(i for i, frame in enumerate(self.frames) if frame.time >= .04)
        self.assertAlmostEqual(frames[early].time, self.frames[early].time)
        initial_roll_accel = abs(self.frames[early].body.angular_acceleration[0])
        increased_roll_mass_accel = abs(frames[early].body.angular_acceleration[0])
        self.assertGreater(initial_roll_accel, .1)
        self.assertLess(increased_roll_mass_accel, .85 * initial_roll_accel)

    def test_sprung_chassis_force_and_moment_balance(self):
        mass = self.vehicle["mass"] - sum(spec.mass for spec in self.model.corners.values())
        inertia = np.diag([INERTIA[axis] for axis in ("ixx", "iyy", "izz")])
        samples = [frame for frame in self.frames if .08 <= frame.time <= .4]
        self.assertGreater(len(samples), 4)
        for frame in samples:
            with self.subTest(time=frame.time):
                np.testing.assert_allclose(frame.rotation.T @ frame.rotation, np.eye(3), atol=1e-10, rtol=0)
                np.testing.assert_allclose(frame.translation, frame.body.displacement, atol=1e-10, rtol=0)
                net_force = np.array([0., 0., -mass * self.model.gravity])
                net_torque = np.zeros(3)
                for corner in self.model.corners:
                    chassis_loads = frame.mount_loads[corner]
                    self.assertEqual(set(chassis_loads.by_mount), {*FIXED, "DAMPER"})
                    for name, force in chassis_loads.by_mount.items():
                        if name == "DAMPER":
                            point = frame.dampers[corner].fixed_point
                        else:
                            point = frame.result.corners[corner].position[POINTS.index(name)]
                        world_lever = frame.rotation @ (point - frame.pivot)
                        net_force += force
                        net_torque += np.cross(world_lever, force)

                np.testing.assert_allclose(net_force, mass * frame.body.acceleration, atol=1e-5, rtol=0)
                world_inertia = frame.rotation @ inertia @ frame.rotation.T
                angular_momentum = world_inertia @ frame.body.angular_velocity
                momentum_rate = world_inertia @ frame.body.angular_acceleration
                momentum_rate += np.cross(frame.body.angular_velocity, angular_momentum)
                np.testing.assert_allclose(net_torque, momentum_rate, atol=1e-5, rtol=0)

    def test_lateral_cg_force_ramps_and_transfers_load_between_sides(self):
        total_mass = self.vehicle["mass"]
        sprung_mass = total_mass - sum(spec.mass for spec in self.model.corners.values())
        inertia = np.diag([INERTIA[axis] for axis in ("ixx", "iyy", "izz")])
        for force in (600., -600.):
            with self.subTest(force=force):
                settings = {**self.settings, "lateral": {
                    "force_y_n": force, "ramp_seconds": .2, "duration_seconds": .5,
                }}
                frames = simulate_rigid_lateral(self.model, settings, total_mass)
                self.assertEqual(frames[0].body.force_y, 0.)
                at_ramp_end = next(frame for frame in frames if frame.time == .2)
                self.assertEqual(at_ramp_end.body.force_y, force)
                end = frames[-1]
                self.assertEqual(end.body.force_y, force)
                self.assertLess(force * end.body.angles_deg[0], 0.)
                left = end.dampers["FL"].tire_force + end.dampers["RL"].tire_force
                right = end.dampers["FR"].tire_force + end.dampers["RR"].tire_force
                self.assertGreater(force * (left - right), 0.)
                mount_force_y = sum(load.total[1] for load in end.mount_loads.values())
                self.assertAlmostEqual(force + mount_force_y, sprung_mass * end.body.acceleration[1], places=5)
                for frame in frames:
                    self.assertAlmostEqual(frame.body.angles_deg[2], 0., places=10)
                    forward = frame.rotation[:, 0]
                    omega = frame.body.angular_velocity
                    alpha = frame.body.angular_acceleration
                    self.assertAlmostEqual(np.cross(omega, forward)[1], 0., places=10)
                    self.assertAlmostEqual(
                        (np.cross(alpha, forward) + np.cross(omega, np.cross(omega, forward)))[1], 0., places=10)
                    torque_z = 0.
                    for name, load in frame.mount_loads.items():
                        for mount, reaction in load.by_mount.items():
                            point = (frame.dampers[name].fixed_point if mount == "DAMPER"
                                     else frame.result.corners[name].position[POINTS.index(mount)])
                            torque_z += np.cross(frame.rotation @ (point - frame.pivot), reaction)[2]
                    world_inertia = frame.rotation @ inertia @ frame.rotation.T
                    momentum_rate = world_inertia @ alpha + np.cross(omega, world_inertia @ omega)
                    self.assertAlmostEqual(torque_z, momentum_rate[2], places=5)

    def test_four_motor_launch_turns_and_limits_torque_at_each_wheel(self):
        settings = {**self.settings, "drive": {
            **self.settings["drive"], "steer_deg": 3., "initial_speed_m_s": 0.,
            "torque_nm": 100., "duration_seconds": .5,
        }}
        frames = simulate_rigid_drive(self.model, settings, self.vehicle["mass"])
        start, end = frames[0], frames[-1]
        self.assertAlmostEqual(np.mean([start.result.corners[name].toe_deg for name in ("FL", "FR")]), 3., places=5)
        self.assertGreater(end.body.planar_speed, .2)
        self.assertGreater(end.body.displacement[1], 0.)
        self.assertGreater(end.body.angles_deg[2], 0.)
        drive = end.drive
        self.assertEqual(settings["drive"]["gear_ratio"], 11.2)
        self.assertTrue(np.all((drive.motor_torque_by_wheel_nm > 0.) & (drive.motor_torque_by_wheel_nm < 22.31)))
        self.assertGreater(np.ptp(drive.motor_torque_by_wheel_nm), 1e-3)
        np.testing.assert_allclose(drive.motor_rpm_by_wheel, np.abs(drive.wheel_speed_rad_s) * 11.2 * 60 / (2 * np.pi))
        np.testing.assert_allclose(drive.wheel_torque_nm, drive.motor_torque_by_wheel_nm * 11.2 * settings["drive"]["gear_efficiency"])
        self.assertTrue(np.all(drive.fx > 0.))


if __name__ == "__main__":
    unittest.main()
