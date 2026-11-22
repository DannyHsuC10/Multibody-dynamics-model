"""Massless pushrods and rockers driving spring/dampers across suspension corners."""

from dataclasses import dataclass
from math import acos, atan2, cos, sin

import numpy as np

from ._geometry_support.hardpoints import PUSHROD_INBOARD, PUSHROD_OUTBOARD
from .geometry import POINTS
from .solver import Unreachable


def unit(vector, label):
    length = np.linalg.norm(vector)
    if length <= 1e-8:
        raise ValueError(f"{label}：旋轉軸或力臂長度過小。")
    return vector / length


@dataclass(frozen=True)
class WheelPickup:
    name: str
    joint: int
    pickup: np.ndarray
    joint_reference: np.ndarray
    pickup_map: np.ndarray
    upright: tuple | None

    @classmethod
    def from_pickup(cls, name, points, pickup_key, pickup_xyz):
        pickup = np.asarray(pickup_xyz, dtype=float)
        upright = None
        if pickup_key == "PUSH_OUT_UPRIGHT":
            nodes = tuple(POINTS.index(key) for key in ("UBJ", "LBJ", "WC", "TOE"))
            coordinates = np.array([points[POINTS[node]] for node in nodes])
            weights = np.linalg.solve(np.vstack((coordinates.T, np.ones(4))), np.r_[pickup, 1.])
            upright = (nodes, weights)
            joint_name = "UBJ"
            pickup_map = np.eye(3)
        else:
            arm, joint_name = ("UCA", "UBJ") if pickup_key == "PUSH_OUT_UCA" else ("LCA", "LBJ")
            base = np.asarray(points[arm + "_F"])
            axis = unit(np.asarray(points[arm + "_R"]) - base, name)
            projection = np.eye(3) - np.outer(axis, axis)
            joint_radial = projection @ (points[joint_name] - base)
            radius_sq = joint_radial @ joint_radial
            if radius_sq < 1e-16:
                raise ValueError(f"{name}：A 臂球接點位於鉸鏈軸上。")
            pickup_radial = projection @ (pickup - base)
            scale = joint_radial @ pickup_radial / radius_sq
            twist = np.cross(axis, joint_radial) @ pickup_radial / radius_sq
            pickup_map = scale * projection + twist * np.cross(axis, np.eye(3)).T
        return cls(name, POINTS.index(joint_name), pickup, np.asarray(points[joint_name]), pickup_map, upright)

    def pickup_motion(self, position, velocity, acceleration):
        if self.upright is not None:
            nodes, weights = self.upright
            return tuple(weights @ array[list(nodes)] for array in (position, velocity, acceleration))
        return (self.pickup + self.pickup_map @ (position[self.joint] - self.joint_reference),
                self.pickup_map @ velocity[self.joint], self.pickup_map @ acceleration[self.joint])

    def apply_pickup_force(self, forces, force):
        if self.upright is not None:
            nodes, weights = self.upright
            forces[list(nodes)] += weights[:, None] * force
        else:
            forces[self.joint] += self.pickup_map.T @ force


@dataclass(frozen=True)
class Rocker(WheelPickup):
    pivot: np.ndarray
    axis: np.ndarray
    pin: np.ndarray
    radial: np.ndarray
    transverse: np.ndarray
    rod_length_sq: float
    branch: float

    @classmethod
    def from_points(cls, name, points, extra):
        outboard = PUSHROD_OUTBOARD & extra.keys()
        if not PUSHROD_INBOARD <= extra.keys() or len(outboard) != 1:
            raise ValueError(f"{name}：搖臂需要完整的推桿與搖臂硬點。")
        key = next(iter(outboard))
        pickup = WheelPickup.from_pickup(name, points, key, extra[key])
        pivot = np.asarray(extra["ROCKER_A"])
        axis = unit(np.asarray(extra["ROCKER_B"]) - pivot, name)
        pin = np.asarray(extra["PUSH_IN"])
        radial = pin - pivot - axis * (axis @ (pin - pivot))
        transverse = np.cross(axis, radial)
        rod = pin - pickup.pickup
        branch_slope = rod @ transverse
        if abs(branch_slope) <= 1e-9:
            raise ValueError(f"{name}：靜態推桿與搖臂位於死點。")
        return cls(pickup.name, pickup.joint, pickup.pickup, pickup.joint_reference, pickup.pickup_map, pickup.upright,
                   pivot, axis, pin, radial, transverse, float(rod @ rod), float(np.sign(branch_slope)))

    def state(self, position, velocity, acceleration):
        pickup, speed, accel = self.pickup_motion(position, velocity, acceleration)
        offset = self.pin - self.radial - pickup
        a, b = offset @ self.radial, offset @ self.transverse
        magnitude = np.hypot(a, b)
        target = .5 * (self.rod_length_sq - offset @ offset - self.radial @ self.radial)
        if magnitude <= 1e-12 or abs(target) >= magnitude:
            raise Unreachable(f"{self.name} · 推桿超出搖臂可達行程或已到死點。")
        theta = atan2(b, a) - self.branch * acos(target / magnitude)
        cosine, sine = cos(theta), sin(theta)
        radial = cosine * self.radial + sine * self.transverse
        tangent = cosine * self.transverse - sine * self.radial
        pin = self.pin - self.radial + radial
        rod = pin - pickup
        denominator = rod @ tangent
        if abs(denominator) <= 1e-9:
            raise Unreachable(f"{self.name} · 推桿與搖臂接近死點。")
        gradient = rod / denominator
        rate = gradient @ speed
        rod_rate = tangent * rate - speed
        second = gradient @ (accel + radial * rate**2) - rod_rate @ rod_rate / denominator
        return RockerState(self, cosine, sine, rate, second, gradient, pin, pickup)

@dataclass(frozen=True)
class RockerState:
    rocker: Rocker
    cosine: float
    sine: float
    rate: float
    second: float
    gradient: np.ndarray
    pin: np.ndarray
    pickup: np.ndarray

    def endpoint(self, xyz):
        axis = self.rocker.axis
        offset = xyz - self.rocker.pivot
        radial = offset - axis * (axis @ offset)
        transverse = np.cross(axis, radial)
        rotated = self.cosine * radial + self.sine * transverse
        tangent = self.cosine * transverse - self.sine * radial
        return xyz - radial + rotated, tangent, rotated


@dataclass(frozen=True)
class RockerEndpoint:
    mount: str
    xyz: np.ndarray
    pickup: WheelPickup | None = None


@dataclass(frozen=True)
class RockerDamper:
    name: str
    endpoints: tuple[RockerEndpoint, RockerEndpoint]
    stiffness: float
    damping: float
    static_length: float = 0.
    preload: float = 0.
    hydraulic_force: float = 0.
    is_hydraulic: bool = False

    @classmethod
    def from_record(cls, record, *, endpoints=None):
        if endpoints is None:
            endpoints = tuple(RockerEndpoint(record[key]["mount"], np.asarray(record[key]["xyz"], dtype=float))
                              for key in ("a", "b"))
        if record["kind"] == "hydraulic":
            return cls(record["name"], endpoints, 0., 0., hydraulic_force=record["hydraulic_force_n"], is_hydraulic=True)
        return cls(record["name"], endpoints, record["spring_n_m"], record["damper_ns_m"])

    def axial_force(self, length, speed):
        return self.hydraulic_force + self.preload - self.stiffness * (length - self.static_length) - self.damping * speed

    def sample(self, states, poses):
        positions, tangents, velocities, accelerations = [], [], [], []
        zero = np.zeros(3)
        for endpoint in self.endpoints:
            if endpoint.mount == "body":
                position, tangent, velocity, acceleration = endpoint.xyz, zero, zero, zero
            elif endpoint.pickup is not None:
                pose = poses[endpoint.mount]
                position, velocity, acceleration = endpoint.pickup.pickup_motion(*pose)
                tangent = zero
            else:
                state = states[endpoint.mount]
                position, tangent, radial = state.endpoint(endpoint.xyz)
                velocity = tangent * state.rate
                acceleration = tangent * state.second - radial * state.rate**2
            positions.append(position)
            tangents.append(tangent)
            velocities.append(velocity)
            accelerations.append(acceleration)
        relative = positions[1] - positions[0]
        length = np.linalg.norm(relative)
        if length <= 1e-8:
            raise Unreachable(f"{self.name} · 避震器兩端重合。")
        axis = relative / length
        velocity = velocities[1] - velocities[0]
        speed = axis @ velocity
        curvature = axis @ (accelerations[1] - accelerations[0]) + (velocity @ velocity - speed**2) / length
        return RockerSample(length, speed, curvature, self.axial_force(length, speed),
                            axis, tuple(positions), tuple(tangents), tuple(velocities), tuple(accelerations))


@dataclass(frozen=True)
class RockerSample:
    length: float
    speed: float
    acceleration: float
    force: float
    axis: np.ndarray
    positions: tuple
    tangents: tuple
    endpoint_velocities: tuple
    endpoint_accelerations: tuple


def rocker_states(geometry, corners):
    return {corner.name: corner.rocker.state(pose.position, pose.velocity, pose.acceleration)
            for corner in geometry.corners if corner.rocker is not None
            for pose in (corners[corner.name],)}


def result_jets(corners):
    return {name: (pose.position, pose.velocity, pose.acceleration) for name, pose in corners.items()}


def length_gradient(damper, sample, names):
    gradient = dict.fromkeys(names, 0.)
    for endpoint, velocity, sign in zip(damper.endpoints, sample.endpoint_velocities, (-1., 1.)):
        if endpoint.mount != "body":
            gradient[endpoint.mount] += sign * (sample.axis @ velocity)
    return np.array(list(gradient.values()))


def coupled_terms(geometry, paths, speeds):
    """Return axial forces and absolute stiffness/damping row bounds by wheel."""
    if not geometry.rocker_dampers:
        return {}
    names = tuple(corner.name for corner in geometry.corners)
    jets = {name: (path.position, path.tangent, path.curvature) for name, path in zip(names, paths)}
    states = {corner.name: corner.rocker.state(*jets[corner.name])
              for corner in geometry.corners if corner.rocker is not None}
    wheel_rates = np.asarray(speeds)
    terms = np.zeros((len(names), 3))
    for damper in geometry.rocker_dampers:
        sample = damper.sample(states, jets)
        ratio = length_gradient(damper, sample, names)
        speed = ratio @ wheel_rates
        force = damper.axial_force(sample.length, speed)
        terms[:, 0] += force * ratio
        tangent = np.zeros((3, len(names)))
        second = np.zeros(len(names))
        for endpoint, velocity, acceleration, sign in zip(
            damper.endpoints, sample.endpoint_velocities, sample.endpoint_accelerations, (-1., 1.)
        ):
            if endpoint.mount != "body":
                index = names.index(endpoint.mount)
                tangent[:, index] += sign * velocity
                second[index] += sign * (sample.axis @ acceleration)
        normal = tangent - np.outer(sample.axis, ratio)
        hessian = normal.T @ normal / sample.length + np.diag(second)
        elastic = damper.stiffness * np.outer(ratio, ratio) - force * hessian
        viscous = damper.damping * np.outer(ratio, hessian @ wheel_rates)
        terms[:, 1] += np.sum(np.abs(elastic + viscous), axis=1)
        terms[:, 2] += damper.damping * np.abs(ratio) * np.sum(np.abs(ratio))
    return dict(zip(names, terms))


def rocker_node_forces(geometry, result):
    """Equivalent wheel node forces and the remaining chassis reactions."""
    states = rocker_states(geometry, result.corners)
    jets = result_jets(result.corners)
    nodes, chassis = {}, {}
    for damper in geometry.rocker_dampers:
        sample = damper.sample(states, jets)
        owner = next(endpoint.mount for endpoint in damper.endpoints if endpoint.mount != "body")
        for endpoint, tangent, sign in zip(damper.endpoints, sample.tangents, (-1., 1.)):
            force = sign * sample.force * sample.axis
            name = owner if endpoint.mount == "body" else endpoint.mount
            chassis.setdefault(name, np.zeros(3))
            chassis[name] += force
            if endpoint.mount != "body":
                forces = nodes.setdefault(name, np.zeros((len(POINTS), 3)))
                if endpoint.pickup is not None:
                    endpoint.pickup.apply_pickup_force(forces, force)
                else:
                    state = states[name]
                    state.rocker.apply_pickup_force(forces, (force @ tangent) * state.gradient)
    for name, forces in nodes.items():
        chassis[name] -= forces.sum(axis=0)
    return nodes, chassis
