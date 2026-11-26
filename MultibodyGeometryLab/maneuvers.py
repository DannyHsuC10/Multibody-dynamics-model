"""Driver input schedules and repeatable ISO 8608 road profiles."""

from math import atan2, cos, degrees, hypot, pi, sin

import numpy as np

from .driving import THROTTLE_RAMP_SECONDS


STEER_STEP_SECONDS = .15
COAST_STOP_SPEED = .1
CRUISE_TESTS = {"steer_step", "steer_ramp", "steer_sine", "circle", "iso_road"}
ISO_ROUGHNESS = {"A": 16e-6, "B": 64e-6, "C": 256e-6}


def smooth_input(time, duration):
    """Quintic ramp and its first two time derivatives."""
    if time <= 0.:
        return 0., 0., 0.
    if time >= duration:
        return 1., 0., 0.
    u = time / duration
    return (u**3 * (10. - u * (15. - 6. * u)),
            30. * u**2 * (1. - u)**2 / duration,
            60. * u * (1. - u) * (1. - 2. * u) / duration**2)


class ISORoad:
    def __init__(self, road_class, seed, corners, front_x):
        self.front_x = front_x
        self.tracks = np.array([0 if name.endswith("L") else 1 for name in corners])
        # One-sided displacement PSD at 0.1 cycles/m, with waviness exponent 2.
        edges = np.geomspace(.02, 3., 65)
        frequency = np.sqrt(edges[:-1] * edges[1:])
        psd = ISO_ROUGHNESS[road_class] * (frequency / .1)**-2
        self.amplitude = np.sqrt(2. * psd * np.diff(edges))
        self.wave_number = 2. * pi * frequency
        self.phase = np.random.default_rng(seed).uniform(0., 2. * pi, (2, len(frequency)))[self.tracks]

    def sample(self, hub_x, hub_vx):
        station = hub_x - self.front_x
        phase = station[:, None] * self.wave_number + self.phase
        height = np.cos(phase) @ self.amplitude
        slope = -np.sin(phase) @ (self.amplitude * self.wave_number)
        # Blend over the first metre of each track from the level starting surface.
        entry = np.clip(station, 0., 1.)
        weight = entry**3 * (10. - entry * (15. - 6. * entry))
        weight_slope = 30. * entry**2 * (1. - entry)**2
        return weight * height, (weight * slope + weight_slope * height) * hub_vx


class Maneuver:
    def __init__(self, settings, drive, names, front_x, rear_x):
        self.settings = settings
        self.preset = settings["preset"]
        self.initial_speed = settings.get("speed_m_s", 0.)
        self.start = settings.get("start_seconds", 0.)
        self.duration = settings["duration_seconds"]
        self.torque_limit = drive["torque_nm"]
        self.rolling_resistance = settings.get("rolling_resistance", 0.)
        self.angle = settings.get("steer_deg", 0.)
        self.road = None
        if self.preset == "circle":
            self.wheelbase = front_x - rear_x
            if self.wheelbase <= 0.:
                raise ValueError("定圓測試需要有效的前後軸距。")
            nominal_angle = degrees(atan2(self.wheelbase, settings["radius_m"]))
            self.angle = min(25., nominal_angle + max(4., .5 * nominal_angle))
        elif self.preset == "iso_road":
            self.road = ISORoad(settings["road_class"], settings["road_seed"], names, front_x)

    def steering(self, time):
        elapsed = time - self.start
        if self.preset == "steer_sine":
            omega = 2. * pi * self.settings["frequency_hz"]
            fade, rate, acceleration = smooth_input(elapsed, min(.35, pi / (2. * omega)))
            sine, cosine = sin(omega * elapsed), cos(omega * elapsed)
            return (self.angle * fade * sine,
                    self.angle * (rate * sine + fade * omega * cosine),
                    self.angle * ((acceleration - fade * omega**2) * sine + 2. * rate * omega * cosine))
        duration = self.settings.get("rise_seconds", STEER_STEP_SECONDS)
        return tuple(self.angle * value for value in smooth_input(elapsed, duration))

    def circle_rate_change(self, time, displacement, velocity, angle, rate):
        """Pure pursuit of a left-hand circle, with a damped steering actuator."""
        x, y = displacement[:2]
        vx, vy = velocity[:2]
        speed = hypot(vx, vy)
        radius = self.settings["radius_m"]
        ahead = min(max(.6 * speed, 3.) / radius, .6)
        phase = atan2(y - radius, x) + ahead
        dx, dy = radius * cos(phase) - x, radius * (1. + sin(phase)) - y
        lateral = (vx * dy - vy * dx) / speed if speed > .1 else dy
        target = degrees(atan2(2. * self.wheelbase * lateral, dx**2 + dy**2))
        target = float(np.clip(target, -self.angle, self.angle)) * smooth_input(time, self.settings["rise_seconds"])[0]
        return 64. * (target - angle) - 16. * rate

    def propulsion(self, time, speed, speed_integral):
        cruise = self.preset in CRUISE_TESTS or (self.preset == "cruise_step" and time < self.start)
        if cruise:
            error = self.initial_speed - speed
            request = 35. * error + 20. * speed_integral
            torque = float(np.clip(request, -self.torque_limit, self.torque_limit))
            integral_rate = error if abs(request) < self.torque_limit or error * request < 0. else 0.
            return torque, integral_rate, self.initial_speed
        if self.preset in ("acceleration", "throttle_step", "cruise_step") and time >= self.start:
            throttle = self.settings["throttle_percent"] / 100.
            if self.preset == "acceleration":
                throttle *= smooth_input(time, THROTTLE_RAMP_SECONDS)[0]
            return throttle * self.torque_limit, 0., None
        return 0., 0., None

    def checkpoints(self):
        elapsed = 0.
        if self.preset in ("steer_ramp", "circle"):
            elapsed = self.settings["rise_seconds"]
        elif self.preset == "steer_step":
            elapsed = STEER_STEP_SECONDS
        elif self.preset == "acceleration":
            elapsed = THROTTLE_RAMP_SECONDS
        return self.start, self.start + elapsed
