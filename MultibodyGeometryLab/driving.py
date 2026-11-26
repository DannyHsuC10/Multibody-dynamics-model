"""Pacejka contact and four independently speed-limited in-wheel motors."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dynamics import ramp_force
from .solver import Unreachable


THROTTLE_RAMP_SECONDS = .25


@dataclass(frozen=True)
class DriveSample:
    motor_torque_by_wheel_nm: np.ndarray
    motor_rpm_by_wheel: np.ndarray
    wheel_torque_nm: np.ndarray
    wheel_speed_rad_s: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    slip_ratio: np.ndarray
    slip_angle_deg: np.ndarray
    test_torque_request_nm: float | None = None
    test_steer_deg: float | None = None
    test_target_speed_m_s: float | None = None
    rolling_torque_nm: np.ndarray | None = None

    @property
    def motor_torque_nm(self):
        return float(np.mean(self.motor_torque_by_wheel_nm))

    @property
    def motor_rpm(self):
        return float(np.mean(self.motor_rpm_by_wheel))


def steering_rack(solvers, target_deg, *, calibrate_zero=False):
    """Calibrate one fixed rack displacement to the mean front toe at static ride height."""
    if target_deg == 0. and not calibrate_zero:
        return 0.

    def mean_toe(rack):
        angles = []
        for solver in solvers:
            axis = solver.wheel_axis_at(solver.wheel_travel(0., rack=rack).position)
            angles.append(np.degrees(np.arctan2(-axis[0], axis[1])))
        return np.mean(angles)

    rack = 0.
    try:
        for _ in range(18):
            angle = mean_toe(rack)
            if abs(angle - target_deg) < 1e-5:
                return float(rack)
            slope = (mean_toe(rack + 1e-5) - angle) / 1e-5
            if abs(slope) < 1.:
                break
            rack += np.clip((target_deg - angle) / slope, -.005, .005)
            if abs(rack) > .06:
                break
    except Unreachable as error:
        raise Unreachable("前輪轉角超出目前齒條與硬點可達範圍。") from error
    raise Unreachable("無法達到指定前輪轉角。請減少轉角或檢查轉向硬點。")


class DriveModel:
    def __init__(self, settings, corners):
        try:
            from Motor import motor
            from Pacejka_tire_model.model import api
        except ImportError as error:
            raise ValueError("馬達行駛需要同一專案中的 Motor 與 Pacejka_tire_model。") from error

        self.api = api
        self.params = api.load_tir("Hoosier_43075_R20_W8_candidate.tir")
        curve_path = Path(motor.__file__).with_name("motor_speed_vs_torque.csv")
        self.rpm, torque = np.loadtxt(curve_path, delimiter=",", skiprows=1, unpack=True)
        self.torque_limit = np.maximum(torque, 0.)
        self.mirror = np.array([1. if corner.name.endswith("L") else -1. for corner in corners])
        self.radius = np.array([corner.wheel_radius for corner in corners])
        self.command_limit = settings["torque_nm"]
        self.target_slip_ratio = settings["target_slip_ratio"]
        self.slip_kp = settings["slip_kp"]
        self.train_rate = settings["gear_ratio"]
        self.gear_eta = settings["gear_efficiency"]
        self.inertia = settings["wheel_inertia"]
        self.initial_speed = settings["initial_speed_m_s"]

        probe = api.CalculationInput(FZ=self.params["FNOMIN"], SA=0., SL=.001, tir_params=self.params)
        zero = probe.with_inputs(SL=0.)
        self.slip_stiffness = abs(api.calculate_fx(probe).item() - api.calculate_fx(zero).item()) * 1000

    def get_motor_torque_data(self, rpm):
        return np.interp(abs(rpm), self.rpm, self.torque_limit, right=0.)

    def contact(self, time, wheel_speed, axes, hub_velocity, normal_load, *, command=None, rolling_resistance=0.):
        forward = np.column_stack((axes[:, 1], -axes[:, 0], np.zeros(len(axes))))
        forward /= np.linalg.norm(forward, axis=1)[:, None]
        lateral = np.column_stack((-forward[:, 1], forward[:, 0], forward[:, 2]))
        vx = np.einsum("ij,ij->i", forward, hub_velocity)
        vy = np.einsum("ij,ij->i", lateral, hub_velocity)
        rolling = self.radius * wheel_speed
        # Regularize slip at standstill and fade steady-state offsets to zero at rest.
        slip_speed = np.maximum.reduce((abs(vx), abs(rolling), np.full_like(vx, 1.)))
        kappa = (rolling - vx) / slip_speed
        alpha = np.arctan2(vy, np.maximum(abs(vx), 1.))
        camber = -np.arcsin(np.clip(axes[:, 2], -1., 1.))
        p = self.params
        condition = self.api.CalculationInput(
            FZ=np.clip(normal_load, p["FZMIN"], p["FZMAX"]),
            SA=np.degrees(np.clip(self.mirror * alpha, p["ALPMIN"], p["ALPMAX"])),
            SL=np.clip(kappa, p["KPUMIN"], p["KPUMAX"]),
            IA=np.degrees(np.clip(self.mirror * camber, p["CAMMIN"], p["CAMMAX"])),
            V=np.hypot(vx, vy), tir_params=p,
        )
        contact_scale = np.minimum(normal_load / p["FZMIN"], 1.)
        contact_scale *= np.minimum(np.maximum.reduce((abs(vx), abs(vy), abs(rolling))), 1.)
        fx = self.api.calculate_fx(condition) * contact_scale
        fy = self.api.calculate_fy(condition) * self.mirror * contact_scale
        rpm = np.abs(wheel_speed) * self.train_rate * 60 / (2 * np.pi)
        torque_limit = self.get_motor_torque_data(rpm)
        throttle_ramp = ramp_force(time, 1., THROTTLE_RAMP_SECONDS) if command is None else 1.
        requested_limit = self.command_limit if command is None else max(command, 0.)
        torque_request = self.slip_kp * (self.target_slip_ratio - kappa)
        torque = np.minimum(throttle_ramp * np.clip(torque_request, 0., requested_limit), torque_limit)
        if command is not None and command < 0.:
            torque = -np.minimum(-command, torque_limit)
        wheel_torque = torque * self.train_rate * self.gear_eta
        wheel_net_torque = wheel_torque - self.radius * fx
        loss = None
        if rolling_resistance:
            loss = self.radius * normal_load * rolling_resistance * np.tanh(rolling / .2)
            wheel_net_torque -= loss
        wheel_acceleration = wheel_net_torque / self.inertia
        horizontal = fx[:, None] * forward + fy[:, None] * lateral
        spin_rate = self.slip_stiffness * (normal_load / p["FNOMIN"]) * self.radius**2 / (self.inertia * slip_speed)
        # The proportional term adds stiffness where the torque command is unsaturated.
        feedback_active = ((torque_request >= 0.) & (torque_request <= requested_limit)
                           & (torque < torque_limit) & (requested_limit > 0.))
        spin_rate += (feedback_active * throttle_ramp * self.slip_kp * self.train_rate
                      * self.gear_eta * self.radius / (self.inertia * slip_speed))
        step_limit = min(.002, 1. / max(float(np.max(spin_rate)), 1.))
        sample = DriveSample(torque, rpm, wheel_torque, wheel_speed.copy(),
                             fx, fy, kappa, np.degrees(alpha), rolling_torque_nm=loss)
        return horizontal, wheel_acceleration, sample, step_limit
