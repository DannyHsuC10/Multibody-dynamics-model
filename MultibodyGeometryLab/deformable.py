"""Variable-length direct-acting spring/dampers and vertical tire contacts."""

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .dynamics_settings import MOUNT_REFERENCES, validate_dynamics
from .elements import DAMPER_POINT, DEFORMABLE_KINDS, PUSH_ROD, Member, default_elements
from .geometry import MOVING, POINTS, VehicleGeometry
from .rod_kinematics import RodMotion, axis_motion
from .rocker import length_gradient, result_jets, rocker_states
from .solver import CornerSolver, Unreachable, VehicleResult, WHEEL


ELEMENT_POINT_INDEX = {name: index for index, name in enumerate((*POINTS, DAMPER_POINT))}
EXTERNAL_INDEX = len(POINTS)
ZERO_3D = np.zeros(3)
ZERO_3D.setflags(write=False)


@dataclass(frozen=True)
class DamperSpec:
    """One coilover and a lumped unsprung mass at its corner's wheel center."""

    mass: float
    stiffness: float
    damping: float
    fixed_point: np.ndarray
    attachment_index: int
    static_length: float
    free_length: float
    static_load: float
    static_axis: np.ndarray
    hydraulic_force: float = 0.

    def kinematics(self, position, velocity, acceleration=None):
        """Return length, its first derivative, and optionally its second."""
        point = self.attachment_index
        return _length_motion(position[point] - self.fixed_point, velocity[point],
                              None if acceleration is None else acceleration[point])

    def spring_force(self, length):
        """Positive force pushes the two attachments apart."""
        return self.hydraulic_force + self.stiffness * (self.free_length - length)

    def damping_force(self, speed):
        return -self.damping * speed


@dataclass(frozen=True)
class ElasticSpec:
    """One spring and damper acting between two named mechanism nodes."""

    name: str
    start: int
    end: int
    stiffness: float
    damping: float
    static_length: float
    free_length: float
    body_mount: np.ndarray
    hydraulic_force: float = 0.

    def relative(self, array, fixed):
        a = fixed if self.start == EXTERNAL_INDEX else array[self.start]
        b = fixed if self.end == EXTERNAL_INDEX else array[self.end]
        return b - a

    def kinematics(self, position, velocity, acceleration=None):
        return _length_motion(self.relative(position, self.body_mount), self.relative(velocity, ZERO_3D),
                              None if acceleration is None else self.relative(acceleration, ZERO_3D))

    def spring_force(self, length):
        return self.hydraulic_force + self.stiffness * (self.free_length - length)

    def damping_force(self, speed):
        return -self.damping * speed

    def endpoint_force(self, position, velocity):
        """Chassis-coordinate force applied to the element's end node."""
        relative = self.relative(position, self.body_mount)
        length, speed, _ = _length_motion(relative, self.relative(velocity, ZERO_3D), None)
        return relative * ((self.spring_force(length) + self.damping_force(speed)) / length)


@dataclass(frozen=True)
class ElasticResult:
    """Length, change from static, speed in SI, and positive compressive force."""

    length: float
    deflection: float
    velocity: float
    force: float
    endpoints: tuple | None = None
    acceleration: float | None = None


@dataclass(frozen=True)
class ConstraintLayout:
    """Ordered physical Jacobian rows and matching endpoint indices."""

    start: np.ndarray
    end: np.ndarray
    lengths: np.ndarray
    rank: int


@dataclass(frozen=True)
class DynamicsModel:
    geometry: VehicleGeometry
    corners: dict[str, DamperSpec]
    gravity: float
    tire_stiffness: float
    tire_damping: float
    free_radius: float
    elastic_members: dict[str, tuple[ElasticSpec, ...]] | None = None
    constraint_layouts: dict[str, ConstraintLayout] | None = None
    unbalanced_corners: tuple[str, ...] = ()

    def tire_force(self, wheel_z, wheel_vz, road_height, road_velocity):
        """Upward-only normal force from ground-frame wheel motion."""
        compression = self.free_radius + road_height - wheel_z
        if compression <= 0.0:
            return 0.0
        return max(0.0, self.tire_stiffness * compression + self.tire_damping * (road_velocity - wheel_vz))


@dataclass(frozen=True)
class DamperResult:
    fixed_point: np.ndarray
    moving_point: np.ndarray
    length: float
    deflection: float
    velocity: float
    acceleration: float
    spring_force: float
    damping_force: float
    force: float
    tire_force: float | None
    road_height: float | None
    orientation: RodMotion

    @property
    def inclination_deg(self) -> float:
        """Unsigned inclination from the chassis vertical, in degrees."""
        return 90.0 - abs(self.orientation.elevation_deg)

    @property
    def force_on_chassis(self) -> np.ndarray:
        """Current spring/damper force at the chassis mount, in chassis XYZ and N."""
        force = (self.force / self.length) * (self.fixed_point - self.moving_point)
        force.setflags(write=False)
        return force


def mount_origin_mm(reference: str, points: np.ndarray, attachment: str) -> np.ndarray:
    """Return the origin of the mount input, in static chassis XYZ and mm."""
    if reference not in MOUNT_REFERENCES:
        raise ValueError("未知的避震器車身接點基準。")
    if reference == "body_coordinates":
        return np.zeros(3)
    index = POINTS.index(attachment) if reference == "attachment" else WHEEL
    return np.asarray(points[index], dtype=float) * 1000.0


def with_axle_pushrod_mounts(geometry: VehicleGeometry, dynamics: dict) -> dict:
    """Migrate legacy mounts using the existing left mount's Y and Z on each axle."""
    settings = validate_dynamics(dynamics)
    if "pushrod_mounts" in settings:
        return settings
    references = {corner.name: np.asarray(corner.reference) for corner in geometry.corners}
    mounts = {}
    for axle, corner in (("front", "FL"), ("rear", "RL")):
        legacy = settings["corners"][corner]
        origin = mount_origin_mm(legacy["mount_reference"], references[corner], legacy["attachment"])
        _, y, z = origin + np.asarray(legacy["mount_mm"], dtype=float)
        mounts[axle] = {"y_mm": abs(float(y)), "z_mm": float(z)}
    return validate_dynamics({**settings, "pushrod_mounts": mounts})


def _length_motion(relative, velocity, acceleration):
    length = float(np.linalg.norm(relative))
    if not np.isfinite(length) or length <= 1e-8:
        raise Unreachable("避震器兩端接點重合，無法求出作用方向。")
    axis = relative / length
    speed = float(axis @ velocity)
    if acceleration is None:
        return length, speed, None
    transverse = velocity - speed * axis
    rate_derivative = float(axis @ acceleration + (transverse @ transverse) / length)
    return length, speed, rate_derivative


def _body_mount(corner, settings, axle_mounts):
    reference = np.asarray(corner.reference)
    attachment = "UBJ" if axle_mounts is not None else settings["attachment"]
    if attachment not in MOVING:
        raise ValueError(f"{corner.name} · 避震器輪端接點必須為 {', '.join(MOVING)} 之一。")
    if axle_mounts is not None:
        mount = axle_mounts[corner.axle]
        side = 1.0 if corner.name.endswith("L") else -1.0
        fixed = np.array([reference[POINTS.index(attachment), 0],
                          side * mount["y_mm"] / 1000.0, mount["z_mm"] / 1000.0])
    else:
        origin_mm = mount_origin_mm(settings.get("mount_reference", "wheel_center"), reference, attachment)
        fixed = (origin_mm + np.asarray(settings["mount_mm"], dtype=float)) / 1000.0
    fixed.setflags(write=False)
    return POINTS.index(attachment), fixed


def _elastic_members(corner, path, fixed, settings):
    specs, ratios = [], []
    for member in corner.members:
        if member.kind not in DEFORMABLE_KINDS:
            continue
        k = settings["spring_n_m"] if member.stiffness_n_m is None else member.stiffness_n_m
        c = settings["damper_ns_m"] if member.damping_ns_m is None else member.damping_ns_m
        hydraulic = member.hydraulic_force_n if member.kind == "hydraulic" else 0.
        if member.kind == "hydraulic":
            k = c = 0.
        spec = ElasticSpec(member.name, ELEMENT_POINT_INDEX[member.a], ELEMENT_POINT_INDEX[member.b],
                           k, c, 0., 0., fixed, hydraulic)
        try:
            length, ratio, _ = spec.kinematics(path.position, path.tangent)
        except Unreachable as error:
            raise ValueError(f"{corner.name} · {member.name} · {error}") from error
        specs.append(replace(spec, static_length=length, free_length=length))
        ratios.append(ratio)
    return tuple(specs), ratios


def _fit_preloads(geometry, paths, loads, elastic_members, local_ratios):
    """Balance the wheel loads with the combined stiffness of local and cross-wheel members."""
    names = tuple(paths)
    states = {corner.name: corner.rocker.state(path.position, path.tangent, path.curvature)
              for corner in geometry.corners if corner.rocker is not None
              for path in (paths[corner.name],)}
    jets = {name: (path.position, path.tangent, path.curvature) for name, path in paths.items()}
    rockers, columns, stiffness, hydraulic = [], [], [], []
    for damper in geometry.rocker_dampers:
        sample = damper.sample(states, jets)
        rockers.append(replace(damper, static_length=sample.length))
        columns.append(length_gradient(damper, sample, names))
        stiffness.append(damper.stiffness)
        hydraulic.append(damper.hydraulic_force)
    local_indices = {}
    for name, ratios in local_ratios.items():
        indices = local_indices[name] = []
        for spec, ratio in zip(elastic_members[name], ratios):
            indices.append(len(columns))
            columns.append([ratio if wheel == name else 0. for wheel in names])
            stiffness.append(spec.stiffness)
            hydraulic.append(spec.hydraulic_force)

    load = np.array([loads[name] for name in names])
    ratios = np.asarray(columns, dtype=float).reshape(len(columns), len(names)).T
    scale = np.sqrt(stiffness)
    remaining = load - ratios @ hydraulic
    preload = scale * np.linalg.lstsq(ratios * scale, remaining, rcond=1e-8)[0] if columns else np.empty(0)
    residual = remaining - ratios @ preload
    unbalanced = tuple(name for name, value in zip(names, residual) if abs(value) > 1e-5)
    for name, indices in local_indices.items():
        fitted = []
        for spec, index in zip(elastic_members[name], indices):
            if spec.stiffness:
                free_length = spec.static_length + preload[index] / spec.stiffness
                if not np.isfinite(free_length) or free_length <= 0.:
                    raise ValueError(f"{name} · {spec.name} · 靜態輪荷對應的彈簧自由長度無效。")
                spec = replace(spec, free_length=free_length)
            fitted.append(spec)
        elastic_members[name] = tuple(fitted)
    return tuple(replace(spec, preload=float(force)) for spec, force in zip(rockers, preload)), unbalanced


def build_dynamics_model(geometry: VehicleGeometry, vehicle_settings, dynamics_settings) -> DynamicsModel:
    """Build the rigid graph, axial force elements and vertical tire contacts."""
    gravity = float(vehicle_settings["gravity"])
    tire_stiffness = float(vehicle_settings["tire_stiffness"])
    free_radius = float(vehicle_settings["free_radius"])
    axle_mounts = dynamics_settings.get("pushrod_mounts")
    elements = dynamics_settings.get("elements")
    if elements is None and (geometry.csv_mechanism or geometry.rocker_dampers or any(
        corner.rocker is not None or corner.pushrod_configured for corner in geometry.corners
    )):
        elements = default_elements(csv_mechanism=geometry.csv_mechanism)
    corners = {}
    updated = []
    elastic_members = None if elements is None else {}
    layouts = None if elements is None else {}
    paths, loads, local_ratios = {}, {}, {}
    for corner in geometry.corners:
        name = corner.name
        settings = dynamics_settings["corners"][name]
        mass = float(settings["mass_kg"])
        stiffness = float(settings["spring_n_m"])
        damping = float(settings["damper_ns_m"])
        point, fixed = _body_mount(corner, settings, axle_mounts)
        if elements is not None:
            members = tuple(Member(**item) if isinstance(item, dict) else item for item in elements[name])
            if corner.rocker is not None or corner.pushrod_configured:
                members = tuple(member for member in members if member.name != PUSH_ROD)
            corner = replace(corner, members=members, body_mount=tuple(fixed))
            # The primary legacy readout is reserved for DAMPER–UBJ.
            point = POINTS.index("UBJ")
            if not any(DAMPER_POINT in (member.a, member.b) for member in members):
                fixed = np.asarray(corner.reference[point]) + [0., 0., .25]
                fixed.setflags(write=False)
                corner = replace(corner, body_mount=tuple(fixed))
        updated.append(corner)
        solver = CornerSolver(corner)
        path = solver.wheel_travel(0.)
        static_axis = path.position[point] - fixed
        static_axis.setflags(write=False)
        try:
            length, ratio, _ = _length_motion(static_axis, path.tangent[point], None)
        except Unreachable as error:
            raise ValueError(f"{name} · {error}") from error
        static_load = tire_stiffness * max(0.0, free_radius - path.position[WHEEL, 2])
        load_to_balance = mass * gravity - static_load
        if elements is None:
            if abs(ratio) <= 1e-6:
                raise ValueError(f"{name} · 靜態時避震器相對輪心的行程比為零，請調整接點。")
            free_length = length + (load_to_balance / ratio) / stiffness
            if not np.isfinite(free_length) or free_length <= 0:
                raise ValueError(f"{name} · 無法由靜態輪荷得到正的彈簧自由長度，請調整接點或剛性。")
        else:
            paths[name] = path
            loads[name] = 0. if solver.locked else load_to_balance
            elastic_members[name], local_ratios[name] = _elastic_members(corner, path, fixed, settings)
            stiffness, damping, free_length = 0., 0., length
            endpoints = np.asarray([[ELEMENT_POINT_INDEX[a], ELEMENT_POINT_INDEX[b]]
                                    for a, b in solver.rigid_links], dtype=int).reshape(-1, 2)
            endpoints.setflags(write=False)
            lengths = np.array(solver.rigid_lengths, dtype=float, copy=True)
            lengths.setflags(write=False)
            layouts[name] = ConstraintLayout(endpoints[:, 0], endpoints[:, 1], lengths, solver.rigid_rank)
        corners[name] = DamperSpec(mass, stiffness, damping, fixed, point, length, free_length, static_load, static_axis)
    unbalanced = ()
    if elements is not None:
        geometry = replace(geometry, corners=tuple(updated))
        rockers, unbalanced = _fit_preloads(geometry, paths, loads, elastic_members, local_ratios)
        geometry = replace(geometry, rocker_dampers=rockers)
        for name, specs in elastic_members.items():
            primary = next((spec for spec in specs if spec.name == PUSH_ROD), None)
            if primary is not None:
                corners[name] = replace(corners[name], stiffness=primary.stiffness, damping=primary.damping,
                                        free_length=primary.free_length, hydraulic_force=primary.hydraulic_force)
    return DynamicsModel(geometry, corners, gravity, tire_stiffness,
                         float(dynamics_settings["tire_damping_ns_m"]), free_radius, elastic_members, layouts,
                         unbalanced)


def suspension_terms(model, name, path, speed):
    """Return generalized spring force plus tangent stiffness and damping bounds."""
    specs = (model.corners[name],) if model.elastic_members is None else model.elastic_members[name]
    generalized_force = stiffness = damping = 0.
    for spec in specs:
        try:
            length, ratio, curvature = spec.kinematics(path.position, path.tangent, path.curvature)
        except Unreachable as error:
            raise Unreachable(f"{name} · {error}") from error
        force = spec.spring_force(length) + spec.damping_force(ratio * speed)
        generalized_force += force * ratio
        stiffness += (spec.stiffness * ratio**2 + abs(force * curvature)
                      + 2 * spec.damping * abs(ratio * curvature * speed))
        damping += spec.damping * ratio**2
    return generalized_force, stiffness, damping


def elastic_node_forces(model, name, pose, damper):
    """Elastic forces on the original nodes plus the Push Rod chassis mount."""
    out = np.zeros((len(ELEMENT_POINT_INDEX), 3))
    if model.elastic_members is None:
        out[EXTERNAL_INDEX] = damper.force_on_chassis
        out[model.corners[name].attachment_index] -= out[EXTERNAL_INDEX]
    else:
        for spec in model.elastic_members[name]:
            force = spec.endpoint_force(pose.position, pose.velocity)
            out[spec.start] -= force
            out[spec.end] += force
    return out


def sample_element_springs(model, result):
    """Sample each edited spring/damper on the current solved node state."""
    samples = {}
    for name, specs in (model.elastic_members or {}).items():
        pose = result.corners[name]
        corner_samples = {}
        for spec in specs:
            try:
                length, speed, acceleration = spec.kinematics(pose.position, pose.velocity, pose.acceleration)
            except Unreachable as error:
                raise Unreachable(f"{name} · {spec.name} · {error}") from error
            corner_samples[spec.name] = ElasticResult(
                length, length - spec.static_length, speed, spec.spring_force(length) + spec.damping_force(speed),
                acceleration=acceleration,
            )
        samples[name] = corner_samples
    states = rocker_states(model.geometry, result.corners)
    jets = result_jets(result.corners)
    for damper in model.geometry.rocker_dampers:
        sample = damper.sample(states, jets)
        reading = ElasticResult(sample.length, sample.length - damper.static_length,
                                sample.speed, sample.force, sample.positions, sample.acceleration)
        for endpoint in damper.endpoints:
            if endpoint.mount != "body":
                samples.setdefault(endpoint.mount, {})[damper.name] = reading
    return samples


def sample_dampers(model: DynamicsModel, result: VehicleResult,
                   road: Mapping[str, tuple[float, float]] | None = None, *,
                   body_position: float = 0., body_velocity: float = 0.) -> dict[str, DamperResult]:
    """Sample deformable elements from a solved state without a fixed-length constraint."""
    samples = {}
    for name, spec in model.corners.items():
        state = result.corners[name]
        try:
            length, speed, acceleration = spec.kinematics(state.position, state.velocity, state.acceleration)
        except Unreachable as error:
            raise Unreachable(f"{name} · {error}") from error
        spring_force = spec.spring_force(length)
        damping_force = spec.damping_force(speed)
        road_height = tire_force = None
        if road is not None:
            road_height, road_velocity = road.get(name, (0.0, 0.0))
            tire_force = model.tire_force(state.position[WHEEL, 2] + body_position,
                                          state.velocity[WHEEL, 2] + body_velocity, road_height, road_velocity)
        moving = state.position[spec.attachment_index].copy()
        moving.setflags(write=False)
        orientation = axis_motion(spec.static_axis, moving - spec.fixed_point,
                                  state.velocity[spec.attachment_index], state.acceleration[spec.attachment_index])
        samples[name] = DamperResult(spec.fixed_point, moving, length, length - spec.static_length,
                                     speed, acceleration, spring_force, damping_force, spring_force + damping_force,
                                     tire_force, road_height, orientation)
    return samples
