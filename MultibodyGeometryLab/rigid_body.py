"""Six chassis coordinates coupled to the four constrained suspension paths."""

from dataclasses import dataclass, replace
from math import atan2, hypot, isfinite, sqrt

import numpy as np

from .animation import AnimationFrame
from .deformable import DynamicsModel, sample_dampers, sample_element_springs, suspension_terms
from .rocker import coupled_terms
from .dynamics import MAX_INTEGRATION_STEP, MAX_STEPS, ramp_force, rk4_step, road_profile, sample_times
from .driving import THROTTLE_RAMP_SECONDS, DriveModel, DriveSample, steering_rack
from .dynamics_settings import (SIMULATION_LIMITS, bounded, default_aero, require_fields,
                                validate_aero, validate_chassis_inertia, validate_drive, validate_lateral,
                                validate_vehicle_test)
from .heave import HeaveSample
from .mount_loads import compute_mount_loads
from .maneuvers import COAST_STOP_SPEED, Maneuver
from .solver import CornerSolver, Unreachable, VehicleResult, WHEEL, WheelTravel


DISPLACEMENT = slice(0, 3)
ORIENTATION = slice(3, 7)
VELOCITY = slice(7, 10)
OMEGA = slice(10, 13)
WHEELS = slice(13, 21)
SPIN = slice(21, 25)
SPEED_INTEGRAL = 25
STEER_ANGLE = 26
STEER_RATE = 27


@dataclass(frozen=True)
class RigidBodySample:
    """Sprung-CG motion in ground axes, with roll, pitch and yaw in degrees."""

    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    angles_deg: np.ndarray
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray
    force_y: float = 0.
    aero_drag: float = 0.
    aero_downforce: float = 0.

    @property
    def planar_speed(self):
        return hypot(*self.velocity[:2])

    @property
    def turning_radius(self):
        vx, vy = self.velocity[:2]
        ax, ay = self.acceleration[:2]
        cross = abs(vx * ay - vy * ax)
        speed = self.planar_speed
        return speed**3 / cross if speed > .1 and cross > speed**3 / 500. else None


@dataclass(frozen=True)
class _Evaluation:
    paths: tuple[WheelTravel, ...]
    rotation: np.ndarray
    acceleration: np.ndarray
    hub_accelerations: np.ndarray
    tire_forces: np.ndarray
    road_heights: np.ndarray
    force_y: float
    tire_forces_xy: np.ndarray
    step_limit: float
    spin_acceleration: np.ndarray | None = None
    drive_sample: DriveSample | None = None
    aero_drag: float = 0.
    aero_downforce: float = 0.
    speed_integral_rate: float = 0.
    steered_motion: tuple | None = None
    steer_acceleration: float = 0.


def _rotation_matrix(quaternion):
    # Unit quaternion (w, x, y, z) maps chassis coordinates to ground coordinates.
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _angles_deg(rotation):
    """Euler angles of Rz(yaw) Ry(pitch) Rx(roll)."""
    projected_x = hypot(rotation[0, 0], rotation[1, 0])
    pitch = atan2(-rotation[2, 0], projected_x)
    if projected_x <= 1e-12:
        return np.degrees([atan2(-rotation[1, 2], rotation[1, 1]), pitch, 0.])
    roll = atan2(rotation[2, 1], rotation[2, 2])
    yaw = atan2(rotation[1, 0], rotation[0, 0])
    return np.degrees([roll, pitch, yaw])


def _rate_bound(response, coefficients):
    weighted = np.sqrt(coefficients)
    return np.linalg.norm(weighted[:, None] * response * weighted, ord=np.inf)


class _RigidChassisEquation:
    def __init__(self, model, total_mass, inertia, target, amplitude, width, lateral_force=lambda _: 0.,
                 *, hold_heading=False, drive_settings=None, aero_settings=None, test_settings=None):
        self.model = model
        self.solvers = tuple(CornerSolver(item) for item in model.geometry.corners)
        self.names = tuple(solver.geometry.name for solver in self.solvers)
        self.specs = tuple(model.corners[name] for name in self.names)
        self.locked_columns = 6 + np.flatnonzero([solver.locked for solver in self.solvers])
        self.masses = np.array([spec.mass for spec in self.specs])
        self.body_mass = total_mass - self.masses.sum()
        if not isfinite(self.body_mass) or self.body_mass <= 0.:
            raise ValueError("車輛總質量須大於四輪輪端質量總和，才能求解剛體車身。")
        self.inertia = inertia
        self.drive = None if drive_settings is None else DriveModel(drive_settings, model.geometry.corners)
        front = tuple(solver for solver in self.solvers if solver.geometry.axle == "front")
        rack = 0.
        if test_settings is not None:
            rack = steering_rack(front, 0., calibrate_zero=True)
        elif self.drive is not None:
            rack = steering_rack(front, drive_settings["steer_deg"])
        self.racks = tuple(rack if solver.geometry.axle == "front" else 0. for solver in self.solvers)
        self.paths = tuple(solver.wheel_travel(0., rack=position)
                           for solver, position in zip(self.solvers, self.racks))
        hubs = np.array([solver.reference[WHEEL] for solver in self.solvers])
        self.maneuver = self.steer_map = None
        front_hub = np.mean([hub[0] for hub, solver in zip(hubs, self.solvers) if solver.geometry.axle == "front"])
        rear_hub = np.mean([hub[0] for hub, solver in zip(hubs, self.solvers) if solver.geometry.axle == "rear"])
        if test_settings is not None:
            self.maneuver = Maneuver(test_settings, drive_settings, self.names, front_hub, rear_hub)
            angle = self.maneuver.angle
            if angle:
                bidirectional = self.maneuver.preset in ("steer_sine", "circle")
                start, end = (-abs(angle), abs(angle)) if bidirectional else sorted((0., angle))
                angles = np.linspace(start, end, 9)
                racks = [steering_rack(front, value, calibrate_zero=True) for value in angles]
                mapping = np.polynomial.Polynomial.fit(angles, racks, 5)
                self.steer_map = mapping - mapping(0.) + rack, mapping.deriv(), mapping.deriv(2)
        # The measured CG and static tire loads refer to the complete vehicle.
        # Removing the four wheel masses gives the CG of the sprung body.
        self.pivot = (total_mass * np.asarray(model.geometry.cg) - self.masses @ hubs) / self.body_mass
        self.pivot.setflags(write=False)
        self.aero_offset = None
        if aero_settings is not None:
            self.drag_area = aero_settings["drag_area_m2"]
            self.downforce_area = aero_settings["downforce_area_m2"]
            if self.drag_area or self.downforce_area:
                balance = aero_settings["front_balance_percent"] / 100.
                center_x = rear_hub + balance * (front_hub - rear_hub)
                self.aero_offset = np.array([center_x, 0., aero_settings["pressure_center_height_m"]]) - self.pivot
                self.air_density = aero_settings["air_density_kg_m3"]
        self.amplitudes = np.array([amplitude if target in (name, "all") else 0. for name in self.names])
        self.width = width
        self.lateral_force = lateral_force
        self.hold_heading = hold_heading
        self.wheel_columns = 6 + np.arange(len(self.names))
        self.force_rows = np.eye(6 + len(self.names))[6:]
        self.tire_stiffnesses = np.full(len(self.names), model.tire_stiffness)
        self.tire_dampings = np.full(len(self.names), model.tire_damping)

    def evaluate(self, state, time):
        wheel_states = state[WHEELS].reshape(-1, 2)
        speeds = wheel_states[:, 1]
        racks = self.racks
        steer = rack_rate = rack_acceleration = steer_acceleration = 0.
        if self.steer_map is not None:
            if self.maneuver.preset == "circle":
                steer, steer_rate = state[STEER_ANGLE], state[STEER_RATE]
                steer_acceleration = self.maneuver.circle_rate_change(
                    time, state[DISPLACEMENT], state[VELOCITY], steer, steer_rate)
            else:
                steer, steer_rate, steer_acceleration = self.maneuver.steering(time)
            rack_map, slope, curvature = self.steer_map
            racks = tuple(float(rack_map(steer)) if solver.geometry.axle == "front" else 0. for solver in self.solvers)
            rack_rate = float(slope(steer) * steer_rate)
            rack_acceleration = float(slope(steer) * steer_acceleration + curvature(steer) * steer_rate**2)
        self.paths = tuple(solver.wheel_travel(travel, previous, rack=rack)
                           for solver, (travel, _), previous, rack
                           in zip(self.solvers, wheel_states, self.paths, racks))
        rotation = _rotation_matrix(state[ORIENTATION])
        omega = state[OMEGA]
        offsets = (np.array([path.position[WHEEL] for path in self.paths]) - self.pivot) @ rotation.T
        tangents = np.array([path.tangent[WHEEL] for path in self.paths]) @ rotation.T
        curvatures = np.array([path.curvature[WHEEL] for path in self.paths]) @ rotation.T
        relative_velocity = tangents * speeds[:, None]
        relative_bias = curvatures * speeds[:, None]**2
        steered_motion = None
        if rack_rate or rack_acceleration:
            steered_motion = tuple(
                solver.wheel_motion(path, speed, rack_rate, rack_acceleration)
                if solver.geometry.axle == "front" else solver.wheel_motion(path, speed)
                for solver, path, speed in zip(self.solvers, self.paths, speeds))
            relative_velocity = np.array([velocity[WHEEL] for velocity, _ in steered_motion]) @ rotation.T
            relative_bias = np.array([bias[WHEEL] for _, bias in steered_motion]) @ rotation.T
        rotational_velocity = np.cross(omega, offsets)
        hub_velocity = state[VELOCITY] + rotational_velocity + relative_velocity
        bias = np.cross(omega, rotational_velocity + 2 * relative_velocity) + relative_bias
        hub_heights = self.pivot[2] + state[DISPLACEMENT][2] + offsets[:, 2]
        road_height, road_velocity = road_profile(time, 1., self.width)
        road_heights = self.amplitudes * road_height
        road_velocities = self.amplitudes * road_velocity
        if self.maneuver is not None and self.maneuver.road is not None:
            hub_x = self.pivot[0] + state[DISPLACEMENT][0] + offsets[:, 0]
            road_heights, road_velocities = self.maneuver.road.sample(hub_x, hub_velocity[:, 0])
        tire_forces = np.array([self.model.tire_force(z, v, h, rate)
                                for z, v, h, rate in zip(hub_heights, hub_velocity[:, 2], road_heights, road_velocities)])

        jacobian = self.acceleration_jacobians(offsets, tangents)
        ground_inertia = (rotation * self.inertia) @ rotation.T
        mass_matrix = np.einsum("ijk,ijl,i->kl", jacobian, jacobian, self.masses)
        mass_matrix[:3, :3] += self.body_mass * np.eye(3)
        mass_matrix[3:6, 3:6] += ground_inertia
        # A locked suspension path has a zero tangent. A positive dummy
        # diagonal leaves its zero relative motion intact and makes the
        # acceleration solve invertible with one slot per wheel.
        mass_matrix[self.locked_columns, self.locked_columns] = 1.

        wheel_forces = -self.masses[:, None] * bias
        wheel_forces[:, 2] += tire_forces - self.masses * self.model.gravity
        force_y = self.lateral_force(time)
        normal_load = tire_forces.sum()
        horizontal = np.zeros_like(wheel_forces)
        spin_acceleration = drive_sample = None
        command = None
        integral_rate = rolling_resistance = 0.
        drive_step = MAX_INTEGRATION_STEP
        if self.drive is None:
            if normal_load > 0.:
                horizontal[:, 1] = -force_y * (tire_forces / normal_load)
        else:
            axes = np.array([solver.wheel_axis_at(path.position) for solver, path in zip(self.solvers, self.paths)])
            axes = axes @ rotation.T
            if self.maneuver is not None:
                command, integral_rate, speed_target = self.maneuver.propulsion(
                    time, hypot(*state[VELOCITY][:2]), state[SPEED_INTEGRAL])
                rolling_resistance = self.maneuver.rolling_resistance
            horizontal, spin_acceleration, drive_sample, drive_step = self.drive.contact(
                time, state[SPIN], axes, hub_velocity, tire_forces, command=command,
                rolling_resistance=rolling_resistance)
            if self.maneuver is not None:
                drive_sample = replace(drive_sample, test_torque_request_nm=command, test_steer_deg=steer,
                                       test_target_speed_m_s=speed_target)
        wheel_forces += horizontal
        rhs = np.einsum("ijk,ij->k", jacobian, wheel_forces)
        rhs[1] += force_y
        # Move the horizontal contact force from the hub to ground level.
        contact_height = hub_heights - road_heights
        rhs[3] += contact_height @ horizontal[:, 1]
        rhs[4] -= contact_height @ horizontal[:, 0]
        if self.drive is not None:
            rhs[3:6] -= self.drive.inertia * (spin_acceleration @ axes)
            if drive_sample.rolling_torque_nm is not None:
                rhs[3:6] -= drive_sample.rolling_torque_nm @ axes
        aero_drag = aero_downforce = 0.
        if self.aero_offset is not None:
            vx, vy = state[VELOCITY][:2]
            speed = hypot(vx, vy)
            pressure = .5 * self.air_density * speed**2
            aero_drag = pressure * self.drag_area
            aero_downforce = pressure * self.downforce_area
            # Drag opposes horizontal travel in still air. Downforce points toward the ground.
            drag_factor = .5 * self.air_density * speed * self.drag_area
            aero_force = np.array([-drag_factor * vx, -drag_factor * vy, -aero_downforce])
            rhs[:3] += aero_force
            rhs[3:6] += np.cross(rotation @ self.aero_offset, aero_force)
        rhs[2] -= self.body_mass * self.model.gravity
        rhs[3:6] -= np.cross(omega, ground_inertia @ omega)
        velocities = None if steered_motion is None or not rack_rate else tuple(v for v, _ in steered_motion)
        spring_rates, damping_rates = self.apply_suspension(rhs, speeds, velocities)

        # The mass-normalized stiffness and damping eigenvalues are bounded
        # by the absolute row sums of their matrices in force coordinates.
        contact_and_spring = np.vstack((jacobian[:, 2, :], self.force_rows))
        loads = np.column_stack((rhs, contact_and_spring.T))
        if self.hold_heading:
            yaw_distribution = self.yaw_distribution(offsets[:, 0], tire_forces, normal_load)
            yaw_load = jacobian[:, 1, :].T @ yaw_distribution
            yaw_load[3] += contact_height @ yaw_distribution
            loads = np.column_stack((loads, yaw_load))
        try:
            solutions = np.linalg.solve(mass_matrix, loads)
        except np.linalg.LinAlgError as error:
            raise Unreachable("車身與懸吊的質量矩陣無法求解。") from error
        if self.hold_heading:
            # The chassis forward axis remains in the ground X–Z plane.
            # Its Y acceleration is (alpha × forward + omega × (omega × forward)).y.
            forward = rotation[:, 0]
            heading_jacobian = np.cross(forward, [0., 1., 0.])
            heading_bias = np.cross(omega, np.cross(omega, forward))[1]
            yaw_response = solutions[:, -1]
            effectiveness = heading_jacobian @ yaw_response[3:6]
            if abs(effectiveness) <= 1e-12:
                raise Unreachable("輪胎側向反力無法維持車頭方向。")
            solutions = solutions[:, :-1]
            heading_rate_change = heading_jacobian @ solutions[3:6]
            heading_rate_change[0] += heading_bias
            correction = -heading_rate_change / effectiveness
            solutions = solutions + np.outer(yaw_response, correction)
            horizontal[:, 1] += yaw_distribution * correction[0]
        acceleration = solutions[:, 0]
        if not np.isfinite(acceleration).all():
            raise Unreachable("車身或輪心加速度超出數值範圍。")
        hub_accelerations = bias + jacobian @ acceleration
        response = contact_and_spring @ solutions[:, 1:]
        stiffness_rate = _rate_bound(response, np.concatenate((self.tire_stiffnesses, spring_rates)))
        damping_rate = _rate_bound(response, np.concatenate((self.tire_dampings, damping_rates)))
        travel_rate = np.max(np.abs(speeds) + np.sqrt(.004 * np.abs(acceleration[6:])))
        rotation_rate = np.linalg.norm(omega) + sqrt(.05 * np.linalg.norm(acceleration[3:6]))
        step_limit = min(drive_step, .35 / sqrt(stiffness_rate), .7 / max(damping_rate, 1e-12),
                         .002 / max(travel_rate, 1e-12), .025 / max(rotation_rate, 1e-12))
        return _Evaluation(self.paths, rotation, acceleration, hub_accelerations, tire_forces, road_heights,
                           force_y, horizontal[:, :2], step_limit, spin_acceleration, drive_sample,
                           aero_drag, aero_downforce, integral_rate, steered_motion, steer_acceleration)

    @staticmethod
    def yaw_distribution(positions_x, tire_forces, normal_load):
        """Distribute a unit ground yaw moment with zero net added lateral force."""
        if normal_load <= 0.:
            raise Unreachable("輪胎離地，無法分配維持車頭方向的側向反力。")
        shares = tire_forces / normal_load
        lever = positions_x - shares @ positions_x
        spread = shares @ (lever * lever)
        if spread <= 1e-10:
            raise Unreachable("接地輪胎的縱向力臂不足，無法平衡側向力矩。")
        return shares * lever / spread

    def enforce_heading(self, state):
        """Remove numerical drift in heading and its first derivative."""
        quaternion = state[ORIENTATION]
        forward = _rotation_matrix(quaternion)[:, 0]
        half_yaw = .5 * atan2(forward[1], forward[0])
        w, x, y, z = quaternion.copy()
        c, s = np.cos(half_yaw), np.sin(half_yaw)
        quaternion[:] = [c*w + s*z, c*x + s*y, c*y - s*x, c*z - s*w]
        horizontal = hypot(forward[0], forward[1])
        if horizontal <= 1e-8:
            raise Unreachable("車頭方向接近垂直，無法維持偏航角。")
        state[OMEGA][2] = forward[2] / horizontal * state[OMEGA][0]

    def acceleration_jacobians(self, offsets, tangents):
        """Map chassis linear/angular acceleration and wheel travel acceleration to each hub."""
        count = len(self.solvers)
        matrix = np.zeros((count, 3, 6 + count))
        matrix[:, :, :3] = np.eye(3)
        # alpha × r = -skew(r) alpha.
        x, y, z = offsets.T
        matrix[:, 0, 4], matrix[:, 0, 5] = z, -y
        matrix[:, 1, 3], matrix[:, 1, 5] = -z, x
        matrix[:, 2, 3], matrix[:, 2, 4] = y, -x
        matrix[np.arange(count), :, self.wheel_columns] = tangents
        return matrix

    def apply_suspension(self, rhs, speeds, velocities=None):
        coupled = coupled_terms(self.model.geometry, self.paths, speeds, velocities=velocities)
        stiffness_rates = []
        damping_rates = []
        for i, (name, path, speed) in enumerate(zip(self.names, self.paths, speeds)):
            velocity = None if velocities is None else velocities[i]
            force, stiffness, damping = suspension_terms(self.model, name, path, speed, velocity=velocity)
            if name in coupled:
                force += coupled[name][0]
                stiffness += coupled[name][1]
                damping += coupled[name][2]
            # The spring/damper is internal to the vehicle, so it does virtual
            # work only along relative suspension travel, never whole-body motion.
            rhs[6+i] += force
            stiffness_rates.append(stiffness)
            damping_rates.append(damping)
        return np.asarray(stiffness_rates), np.asarray(damping_rates)

    def rates(self, state, time):
        return _rates(state, self.evaluate(state, time))

    def frame(self, state, time, evaluation):
        acceleration = evaluation.acceleration
        speeds = state[WHEELS].reshape(-1, 2)[:, 1]
        motions = evaluation.steered_motion or (None,) * len(self.names)
        corners = {name: solver.wheel_result(path, speed, relative_acceleration, motion=motion)
                   for name, solver, path, speed, relative_acceleration, motion in zip(
                       self.names, self.solvers, evaluation.paths, speeds, acceleration[6:], motions)}
        result = VehicleResult(corners)
        dampers = sample_dampers(self.model, result)
        dampers = {name: replace(dampers[name], tire_force=float(force), road_height=float(height))
                   for name, force, height in zip(self.names, evaluation.tire_forces, evaluation.road_heights)}
        rotation = evaluation.rotation
        mount_loads = compute_mount_loads(self.model, result, dampers, 0., rotation=rotation,
                                          wheel_accelerations=dict(zip(self.names, evaluation.hub_accelerations)),
                                          tire_forces_xy=dict(zip(self.names, evaluation.tire_forces_xy)))
        body = RigidBodySample(
            state[DISPLACEMENT].copy(), state[VELOCITY].copy(), acceleration[:3].copy(),
            _angles_deg(rotation), state[OMEGA].copy(), acceleration[3:6].copy(), evaluation.force_y,
            evaluation.aero_drag, evaluation.aero_downforce,
        )
        for array in (rotation, body.displacement, body.velocity, body.acceleration, body.angles_deg,
                      body.angular_velocity, body.angular_acceleration):
            array.setflags(write=False)
        heave = HeaveSample(float(body.displacement[2]), float(body.velocity[2]), float(body.acceleration[2]), 0.)
        return AnimationFrame(time, result, rotation, self.pivot, body.displacement, dampers=dampers,
                              heave=heave, mount_loads=mount_loads, body=body,
                              element_springs=sample_element_springs(self.model, result),
                              drive=evaluation.drive_sample)


def _rates(state, evaluation):
    derivative = np.empty_like(state)
    derivative[DISPLACEMENT] = state[VELOCITY]
    derivative[VELOCITY] = evaluation.acceleration[:3]
    derivative[OMEGA] = evaluation.acceleration[3:6]
    real, vector = state[3], state[4:7]
    omega = state[OMEGA]
    # Ground angular velocity premultiplies the chassis-to-ground quaternion.
    derivative[3] = -.5 * (omega @ vector)
    derivative[4:7] = .5 * (real * omega + np.cross(omega, vector))
    derivative[WHEELS] = np.column_stack((state[WHEELS].reshape(-1, 2)[:, 1],
                                         evaluation.acceleration[6:])).ravel()
    if evaluation.spin_acceleration is not None:
        derivative[SPIN] = evaluation.spin_acceleration
    if len(state) > SPEED_INTEGRAL:
        derivative[SPEED_INTEGRAL] = evaluation.speed_integral_rate
    if len(state) > STEER_RATE:
        derivative[STEER_ANGLE] = state[STEER_RATE]
        derivative[STEER_RATE] = evaluation.steer_acceleration
    return derivative


def simulate_rigid_road(model: DynamicsModel, settings: dict, total_mass: float) -> tuple[AnimationFrame, ...]:
    """Integrate a free rigid chassis and the four wheel travels over a road pulse."""
    simulation = settings.get("simulation") if isinstance(settings, dict) else None
    require_fields(simulation, ("target", *SIMULATION_LIMITS), "模擬設定")
    target = simulation["target"]
    if target not in (*model.corners, "all"):
        raise ValueError("路面擾動輪位必須是 FL、FR、RL、RR 或四輪。")
    limits = {key: bounded(simulation[key], *limit) for key, limit in SIMULATION_LIMITS.items()}
    duration, width = limits["duration_seconds"], limits["bump_seconds"]
    if width > duration:
        raise ValueError("擾動持續時間不可超過模擬時間。")
    equation = _RigidChassisEquation(model, total_mass, _inertia(settings), target, limits["bump_mm"] / 1000, width)
    return _integrate(equation, duration, width, (width / 2, width))


def simulate_rigid_lateral(model: DynamicsModel, settings: dict, total_mass: float) -> tuple[AnimationFrame, ...]:
    """Apply a held lateral force at the sprung CG while maintaining heading."""
    lateral = validate_lateral(settings.get("lateral") if isinstance(settings, dict) else None)
    force, ramp, duration = (lateral[key] for key in ("force_y_n", "ramp_seconds", "duration_seconds"))
    equation = _RigidChassisEquation(model, total_mass, _inertia(settings), "all", 0., ramp,
                                     lambda time: ramp_force(time, force, ramp), hold_heading=True)
    return _integrate(equation, duration, ramp, (ramp,))


def _inertia(settings):
    inertia = validate_chassis_inertia(settings.get("chassis_inertia"))
    return np.array([inertia[key] for key in ("ixx", "iyy", "izz")])


def simulate_rigid_drive(model: DynamicsModel, settings: dict, total_mass: float) -> tuple[AnimationFrame, ...]:
    drive = validate_drive(settings.get("drive"))
    aero = validate_aero(settings.get("aero", default_aero()))
    equation = _RigidChassisEquation(model, total_mass, _inertia(settings), "all", 0., .25,
                                     drive_settings=drive, aero_settings=aero)
    return _integrate(equation, drive["duration_seconds"], THROTTLE_RAMP_SECONDS, (THROTTLE_RAMP_SECONDS,))


def simulate_vehicle_test(model: DynamicsModel, settings: dict, total_mass: float) -> tuple[AnimationFrame, ...]:
    test = validate_vehicle_test(settings.get("vehicle_test"))
    drive = validate_drive(settings.get("drive"))
    aero = validate_aero(settings.get("aero", default_aero()))
    equation = _RigidChassisEquation(model, total_mass, _inertia(settings), "all", 0., .25,
                                     drive_settings=drive, aero_settings=aero, test_settings=test)
    checkpoints = tuple(time for time in equation.maneuver.checkpoints() if 0. < time < test["duration_seconds"])
    return _integrate(equation, test["duration_seconds"], 0., checkpoints)


def _integrate(equation, duration, input_window, checkpoints):
    test = equation.maneuver
    if test is not None:
        state = np.zeros(28 if test.preset == "circle" else 26)
    else:
        state = np.zeros(21 if equation.drive is None else 25)
    state[3] = 1.
    if equation.drive is not None:
        speed = equation.drive.initial_speed if test is None else test.initial_speed
        state[7] = speed
        state[SPIN] = speed / equation.drive.radius
    frames = []
    current = 0.
    steps = 0
    step_budget = MAX_STEPS if test is None else max(MAX_STEPS, int(duration * 2000))
    coast = test is not None and test.preset == "coast"
    throttle_step = test.start if test is not None and test.preset in ("throttle_step", "cruise_step") else None
    try:
        evaluation = equation.evaluate(state, current)
        for output in sample_times(duration, *checkpoints):
            while output - current > 1e-12:
                input_step = input_window / 32 if current < input_window else MAX_INTEGRATION_STEP
                step_limit = min(input_step, evaluation.step_limit)
                if steps >= step_budget or step_limit < 1e-8:
                    raise Unreachable("超過積分步數限制，請縮短模擬時間或調整剛性、阻尼與轉動慣量。")
                step = min(output - current, step_limit)
                rates = equation.rates
                if output == throttle_step and step == output - current:
                    # Integrate up to the discontinuity from the left, then sample the new command.
                    before_step = np.nextafter(output, -np.inf)
                    rates = lambda stage, time: equation.rates(stage, min(time, before_step))
                state = rk4_step(rates, state, current, step, _rates(state, evaluation))
                state[ORIENTATION] /= np.linalg.norm(state[ORIENTATION])
                if equation.hold_heading:
                    equation.enforce_heading(state)
                current += step
                steps += 1
                evaluation = equation.evaluate(state, current)
            current = output
            frames.append(equation.frame(state, output, evaluation))
            if coast and hypot(*state[VELOCITY][:2]) <= COAST_STOP_SPEED:
                break
    except Unreachable as error:
        raise Unreachable(f"剛體車身於 {current:.3f} s 無法繼續。{error}") from error
    return tuple(frames)
