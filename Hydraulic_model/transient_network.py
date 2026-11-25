"""Lumped transient model with elastic oil volume and inertial piston rods."""

from dataclasses import dataclass
from math import ceil, isfinite, sqrt

import numpy as np


@dataclass(frozen=True)
class TransientSettings:
    duration_s: float = 1.
    ramp_s: float = .1
    mass_kg: float = 1.
    damping_n_s_mm: float = .2
    bulk_modulus_mpa: float = 1500.
    line_volume_ml: float = 100.
    line_delay_enabled: bool = False
    line_time_constant_s: float = .02

    def __post_init__(self):
        positive = (self.duration_s, self.ramp_s, self.mass_kg, self.bulk_modulus_mpa, self.line_volume_ml)
        if not all(isfinite(value) and value > 0. for value in positive):
            raise ValueError("時間、質量、油液體積模數與參考油量須為有限正值。")
        if self.ramp_s > self.duration_s:
            raise ValueError("輸入上升時間不可大於模擬時間。")
        if not isfinite(self.damping_n_s_mm) or self.damping_n_s_mm < 0.:
            raise ValueError("活塞桿阻尼須為大於或等於零的有限數值。")
        if self.line_delay_enabled and not (
            isfinite(self.line_time_constant_s) and self.line_time_constant_s > 0.
        ):
            raise ValueError("油路延遲的時間常數須為有限正值。")


@dataclass(frozen=True)
class TransientTrace:
    time_s: np.ndarray
    rod_extension_mm: dict
    pressure_change_mpa: dict
    required_rod_forces_n: dict
    port_pressure_change_mpa: dict


def _ramp_fraction(time, ramp_s):
    phase = min(time / ramp_s, 1.)
    return phase**3 * (10. - phase * (15. - 6. * phase))


def _integration_rate(stiffness, mass, damping):
    if not stiffness.size:
        return 0.
    rate = sqrt(np.linalg.norm(stiffness, ord=np.inf) / mass) + np.linalg.norm(damping, ord=np.inf) / mass
    if not isfinite(rate):
        raise ValueError("質量、剛性或阻尼超出可積分的數值範圍。")
    return rate


def _integrate_state(times, ramp_s, operator, forcing, rate, velocity_forcing=None, initial_force=None):
    size = len(forcing)
    history = np.zeros((len(times), size))
    if not size:
        return history

    maximum_step = .35 / rate if rate else float("inf")
    interval = np.diff(times)
    step_limits = np.where(times[1:] <= ramp_s, min(maximum_step, ramp_s / 24), maximum_step)
    subdivisions = np.maximum(1., np.ceil(interval / step_limits))
    if not np.isfinite(subdivisions).all() or subdivisions.sum() > 100_000:
        raise ValueError("瞬態積分步數過多。可縮短模擬時間，或檢查質量、阻尼及油路參數。")

    if (not np.isfinite(operator).all() or not np.isfinite(forcing).all()
            or (initial_force is not None and not np.isfinite(initial_force).all())
            or (velocity_forcing is not None and not np.isfinite(velocity_forcing).all())):
        raise ValueError("瞬態方程超出可計算的數值範圍。")
    state = np.zeros(size)

    def rates(time, value):
        derivative = operator @ value + _ramp_fraction(time, ramp_s) * forcing
        if initial_force is not None:
            derivative += initial_force
        if velocity_forcing is not None and time < ramp_s:
            phase = time / ramp_s
            derivative += (30. * phase**2 * (1. - phase)**2 / ramp_s) * velocity_forcing
        return derivative

    for index, (start, duration, count) in enumerate(zip(times, interval, subdivisions.astype(int)), 1):
        step = duration / count
        for substep in range(count):
            time = start + step * substep
            first = rates(time, state)
            second = rates(time + step / 2, state + step * first / 2)
            third = rates(time + step / 2, state + step * second / 2)
            fourth = rates(time + step, state + step * third)
            state += step * (first + 2 * second + 2 * third + fourth) / 6
        if not np.isfinite(state).all():
            raise ValueError("瞬態結果超出可計算的數值範圍。")
        history[index] = state
    return history


def _port_network(lines, rods):
    """Give each port a chamber and each line a mean pressure for resistive exchange."""
    ports = tuple((line.name, port) for line in lines for port in line.ports)
    port_areas = np.zeros((len(ports), len(rods)))
    for index, (_, port) in enumerate(ports):
        port_areas[index, rods[port.cylinder.rod]] = port.pressure_force_direction * port.area_mm2

    averaging = np.zeros((len(lines), len(ports)))
    exchange = np.zeros((len(ports), len(ports)))
    chamber_counts = np.empty(len(ports))
    start = 0
    for index, line in enumerate(lines):
        size = len(line.ports)
        section = slice(start, start + size)
        averaging[index, section] = 1. / size
        exchange[section, section] = np.full((size, size), 1. / size) - np.eye(size)
        chamber_counts[section] = size
        start += size
    return ports, port_areas, averaging, exchange, chamber_counts


def simulate_transient(system, forces_n, prescribed_extension_mm, settings: TransientSettings) -> TransientTrace:
    """Ramp forces and prescribed motion from a reference equilibrium at rest.

    With ideal lines, pressure is -B/V * Gx, where Gx is the oil volume
    change. With delay, the line volume is split evenly between its ports.
    Chamber pressure changes with piston flow and relaxes toward the line
    mean with time constant tau. Exchange sums to zero in each line.
    """
    rods, applied, prescribed = system.boundary_conditions(forces_n, prescribed_extension_mm)
    stiffness, damping, area = system.assemble_matrices(rods)
    mass = settings.mass_kg / 1000.  # N per mm/s²
    volume = settings.line_volume_ml * 1000.  # mm³
    damping += settings.damping_n_s_mm * np.eye(len(rods))
    if not np.isfinite(damping).all():
        raise ValueError("阻尼超出可計算的數值範圍。")
    if mass <= 0. or not isfinite(volume):
        raise ValueError("質量或參考油量超出可計算的數值範圍。")
    inverse_compliance = np.array([settings.bulk_modulus_mpa / volume if line.compliance_cm3_mpa is None
                                   else .001 / line.compliance_cm3_mpa for line in system.lines])
    initial_pressures = np.array([line.initial_pressure_mpa for line in system.lines])
    pressure_coefficient = -inverse_compliance[:, None] * area
    effective_stiffness = stiffness - area.T @ pressure_coefficient
    if not np.isfinite(effective_stiffness).all():
        raise ValueError("油液或彈簧剛性超出可計算的數值範圍。")

    free = [index for rod, index in rods.items() if rod not in prescribed]
    fixed = [rods[rod] for rod in prescribed]
    target = np.array(list(prescribed.values()))
    size = len(free)
    delay = settings.line_delay_enabled and bool(system.lines)
    rod_stiffness = stiffness if delay else effective_stiffness
    free_stiffness = rod_stiffness[np.ix_(free, free)]
    drive = applied[free] - rod_stiffness[np.ix_(free, fixed)] @ target
    free_damping = damping[np.ix_(free, free)]
    rate = _integration_rate(effective_stiffness[np.ix_(free, free)], mass, free_damping)
    zero, identity = np.zeros((size, size)), np.eye(size)
    damping_matrix = -free_damping / mass
    initial_drive = area[:, free].T @ initial_pressures / mass
    initial_force = np.r_[np.zeros(size), initial_drive]
    velocity_drive = -(damping[np.ix_(free, fixed)] @ target) / mass
    velocity_forcing = np.r_[np.zeros(size), velocity_drive]
    if delay:
        inverse_tau = 1. / settings.line_time_constant_s
        ports, port_areas, averaging, exchange, counts = _port_network(system.lines, rods)
        port_count = len(ports)
        # Each port has compliance V / (number of ports * B). Pressure
        # differences drive equal-and-opposite flows through the ideal manifold.
        inverse_by_line = {line.name: value for line, value in zip(system.lines, inverse_compliance)}
        port_inverse_compliance = np.array([inverse_by_line[line] for line, _ in ports]) * counts
        port_rate = -port_inverse_compliance[:, None] * port_areas
        locked_stiffness = stiffness - port_areas.T @ port_rate
        rate = _integration_rate(locked_stiffness[np.ix_(free, free)], mass, free_damping) + inverse_tau
        if not isfinite(rate):
            raise ValueError("油路延遲時間常數超出可積分的數值範圍。")
        operator = np.block([
            [zero, identity, np.zeros((size, port_count))],
            [-free_stiffness / mass, damping_matrix, port_areas[:, free].T / mass],
            [np.zeros((port_count, size)), port_rate[:, free], inverse_tau * exchange],
        ])
        forcing = np.r_[np.zeros(size), drive / mass, np.zeros(port_count)]
        velocity_forcing = np.r_[velocity_forcing, port_rate[:, fixed] @ target]
        initial_force = np.r_[initial_force, np.zeros(port_count)]
    else:
        operator = np.block([[zero, identity], [-free_stiffness / mass, damping_matrix]])
        forcing = np.r_[np.zeros(size), drive / mass]
    intervals = max(1, ceil(min(12_000, max(120., 2. * rate) * settings.duration_s)))
    # Sample the ramp separately to retain its shape even on short inputs.
    ramp_times = np.linspace(0., settings.ramp_s, 25)
    times = np.unique(np.r_[np.linspace(0., settings.duration_s, intervals + 1), ramp_times])
    phase = np.minimum(times, settings.ramp_s) / settings.ramp_s
    fraction = phase**3 * (10. - 15. * phase + 6. * phase**2)
    extension = np.zeros((len(times), len(rods)))
    extension[:, fixed] = fraction[:, None] * target
    history = _integrate_state(times, settings.ramp_s, operator, forcing, rate, velocity_forcing, initial_force)
    extension[:, free] = history[:, :size]
    port_pressure_traces = {}
    if delay:
        port_pressure = history[:, 2 * size:]
        pressure = port_pressure @ averaging.T
        hydraulic_force = port_pressure @ port_areas[:, fixed]
        port_pressure_traces = {(line, port.cylinder.rod, port.side): series
                                for (line, port), series in zip(ports, port_pressure.T)}
    else:
        pressure = extension @ pressure_coefficient.T
        hydraulic_force = pressure @ area[:, fixed]
    required = extension @ stiffness[:, fixed] - hydraulic_force - initial_pressures @ area[:, fixed]
    if fixed:
        ramp_velocity = 30. * phase**2 * (1. - phase)**2 / settings.ramp_s
        ramp_acceleration = 60. * phase * (1. - phase) * (1. - 2. * phase) / settings.ramp_s / settings.ramp_s
        required += history[:, size:2 * size] @ damping[np.ix_(free, fixed)]
        required += ramp_velocity[:, None] * (damping[np.ix_(fixed, fixed)] @ target)
        required += mass * ramp_acceleration[:, None] * target
    if not (np.isfinite(extension).all() and np.isfinite(pressure).all() and np.isfinite(required).all()):
        raise ValueError("瞬態結果超出可計算的數值範圍。請檢查輸入與油路參數。")

    return TransientTrace(
        times,
        {rod: extension[:, index] for rod, index in rods.items()},
        {line.name: pressure[:, index] for index, line in enumerate(system.lines)},
        dict(zip(prescribed, required.T)),
        port_pressure_traces,
    )
