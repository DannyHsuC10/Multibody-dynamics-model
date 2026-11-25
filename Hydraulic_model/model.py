"""Quasi-static, incompressible hydraulics with ideal supply and return lines."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isclose, isfinite, pi
from types import MappingProxyType
from typing import Literal, Protocol


class RodLoad(Protocol):
    """A linear axial load referenced to zero rod extension."""

    @property
    def stiffness_n_mm(self) -> float:
        ...


@dataclass(frozen=True, eq=False)
class PistonRod:
    """Mechanical connection point. Positive axial force extends the rod."""


@dataclass(frozen=True)
class LinearSpring:
    """An unpreloaded spring opposing the motion of an output cylinder."""

    stiffness_n_mm: float

    def __post_init__(self):
        if not isfinite(self.stiffness_n_mm) or self.stiffness_n_mm <= 0.:
            raise ValueError("Spring stiffness must be finite and positive.")

    def stroke_for_force(self, force_n: float) -> float:
        return force_n / self.stiffness_n_mm


@dataclass(frozen=True)
class ChassisMount:
    """A fixed connection point in the vehicle chassis frame."""


CHASSIS = ChassisMount()
# Keep older Python models working with the same fixed reference.
MechanicalGround = ChassisMount
GROUND = CHASSIS


@dataclass(frozen=True)
class SpringElement:
    """A two-terminal linear spring and viscous damper in parallel."""

    spring: LinearSpring | None
    end_a: PistonRod | ChassisMount
    end_b: PistonRod | ChassisMount
    damping_n_s_mm: float = 0.

    def __post_init__(self):
        if not all(isinstance(end, (PistonRod, ChassisMount)) for end in (self.end_a, self.end_b)):
            raise ValueError("Spring endpoints must be piston rods or a chassis mount.")
        if self.end_a == self.end_b:
            raise ValueError("A spring must connect two distinct mechanical points.")
        if not isfinite(self.damping_n_s_mm) or self.damping_n_s_mm < 0.:
            raise ValueError("Damper coefficient must be finite and nonnegative.")
        if self.spring is None and self.damping_n_s_mm == 0.:
            raise ValueError("The element needs spring stiffness or damping.")


@dataclass(frozen=True)
class HydraulicCylinder:
    """A cylinder with cap-end and rod-end ports and an optional spring load."""

    bore_mm: float
    load: RodLoad | None = None
    rod_diameter_mm: float | None = None
    rod: PistonRod = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (not isfinite(self.bore_mm) or self.bore_mm <= 0.
                or not isfinite(self.area_mm2) or self.area_mm2 <= 0.):
            raise ValueError("Cylinder bore must be finite and positive and have a representable area.")
        if self.rod_diameter_mm is not None and not 0. < self.rod_diameter_mm < self.bore_mm:
            raise ValueError("Rod diameter must be positive and smaller than the cylinder bore.")
        object.__setattr__(self, "rod", PistonRod())

    @property
    def area_mm2(self) -> float:
        radius = self.bore_mm / 2.
        return pi * (radius * radius)

    def volume_for_stroke(self, stroke_mm: float) -> float:
        return self.area_mm2 * stroke_mm

    def force_at_pressure(self, pressure_mpa: float) -> float:
        # 1 MPa = 1 N/mm².
        return self.area_mm2 * pressure_mpa

    @property
    def cap_end(self) -> "HydraulicPort":
        return HydraulicPort(self, "cap")

    @property
    def rod_end(self) -> "HydraulicPort":
        return HydraulicPort(self, "rod")


def rod_stiffness(cylinder: HydraulicCylinder, attachment: RodLoad | None) -> float:
    """Combine the cylinder's own load with a linear rod attachment."""
    stiffness = 0.
    for load in (cylinder.load, attachment):
        if load is not None:
            value = load.stiffness_n_mm
            if not isfinite(value) or value < 0.:
                raise ValueError("Rod load stiffness must be finite and nonnegative.")
            stiffness += value
    if not isfinite(stiffness):
        raise ValueError("Combined rod stiffness is outside the numerical range.")
    return stiffness


@dataclass(frozen=True)
class HydraulicPort:
    """A connectable cylinder chamber with its own effective piston area."""

    cylinder: HydraulicCylinder
    side: Literal["cap", "rod"]

    def __post_init__(self):
        if self.side not in ("cap", "rod"):
            raise ValueError("Cylinder port side must be 'cap' or 'rod'.")
        if self.side == "rod" and self.cylinder.rod_diameter_mm is None:
            raise ValueError("Set rod_diameter_mm to connect the rod-end port.")
        if self.area_mm2 <= 0.:
            raise ValueError("Cylinder port must have a representable positive area.")

    @property
    def area_mm2(self) -> float:
        if self.side == "cap":
            return self.cylinder.area_mm2
        bore = self.cylinder.bore_mm
        rod = self.cylinder.rod_diameter_mm
        # Factor the difference of squares to retain precision for a thin annulus.
        return self.cylinder.area_mm2 * ((bore - rod) / bore) * (1. + rod / bore)

    def volume_for_stroke(self, stroke_mm: float) -> float:
        return self.area_mm2 * stroke_mm

    def force_at_pressure(self, pressure_mpa: float) -> float:
        return self.area_mm2 * pressure_mpa

    @property
    def pressure_force_direction(self) -> int:
        return 1 if self.side == "cap" else -1


@dataclass(frozen=True)
class HydraulicLine:
    """Connect one master port to parallel slave ports.

    Passing a cylinder directly is shorthand for connecting its cap-end port.
    """

    master: HydraulicPort | HydraulicCylinder
    slaves: tuple[HydraulicPort | HydraulicCylinder, ...]

    def __post_init__(self):
        object.__setattr__(self, "slaves", tuple(self.slaves))
        master = self.master_port.cylinder
        slaves = tuple(port.cylinder for port in self.slave_ports)
        if not slaves:
            raise ValueError("The line requires at least one slave cylinder.")
        if len({id(cylinder) for cylinder in (master, *slaves)}) != 1 + len(slaves):
            raise ValueError("Each connected cylinder must be a separate instance.")

    @property
    def master_port(self) -> HydraulicPort:
        return self.master if isinstance(self.master, HydraulicPort) else self.master.cap_end

    @property
    def slave_ports(self) -> tuple[HydraulicPort, ...]:
        return tuple(port if isinstance(port, HydraulicPort) else port.cap_end for port in self.slaves)


@dataclass(frozen=True)
class CylinderResponse:
    """Stroke in the hydraulic actuation direction and pressure force magnitude.

    The master moves to shrink its supply chamber. Each slave moves to expand
    its supply chamber: extension for the cap end, retraction for the rod end.
    The signed extension and hydraulic force are positive toward rod extension.
    """

    stroke_mm: float
    force_n: float
    extension_mm: float
    hydraulic_force_n: float


@dataclass(frozen=True)
class HydraulicResponse:
    """Supply-line gauge pressure and the corresponding cylinder responses."""

    pressure_mpa: float
    master: CylinderResponse
    slaves: tuple[CylinderResponse, ...]
    master_input_force_n: float


@dataclass(frozen=True)
class HydraulicSystem:
    """Displace oil out of the master supply chamber into the slave chambers.

    Unconnected chambers are vented. An optional closed return line connects
    the remaining chambers at the model's 0 MPa gauge reference. Its oil
    volume must balance independently of the supply line.
    """

    line: HydraulicLine
    return_line: HydraulicLine | None = None
    rod_loads: Mapping[PistonRod, RodLoad] = field(default_factory=dict)

    def __post_init__(self):
        loads = dict(self.rod_loads)
        self._check_rods(loads)
        object.__setattr__(self, "rod_loads", MappingProxyType(loads))
        if self.return_line is None:
            return
        supply_master = self.line.master_port
        return_master = self.return_line.master_port
        supply_slaves = {id(port.cylinder): port.side for port in self.line.slave_ports}
        return_slaves = {id(port.cylinder): port.side for port in self.return_line.slave_ports}
        if (supply_master.cylinder is not return_master.cylinder or supply_master.side == return_master.side
                or supply_slaves.keys() != return_slaves.keys()
                or any(side == return_slaves[cylinder] for cylinder, side in supply_slaves.items())):
            raise ValueError("The return line must connect the opposite port of each supply-line cylinder.")

    def solve(self, master_stroke_mm: float, *,
              rod_forces_n: Mapping[PistonRod, float] | None = None) -> HydraulicResponse:
        """Prescribe oil-displacing travel and optionally apply forces to any rod."""
        if not isfinite(master_stroke_mm) or master_stroke_mm < 0.:
            raise ValueError("Master stroke must be finite and nonnegative.")
        return self._solve(master_stroke_mm, None, rod_forces_n)

    def solve_force(self, master_force_n: float, *,
                    rod_forces_n: Mapping[PistonRod, float] | None = None) -> HydraulicResponse:
        """Prescribe one master rod force, positive toward rod extension."""
        if not isfinite(master_force_n):
            raise ValueError("Master rod force must be finite.")
        return self._solve(None, master_force_n, rod_forces_n)

    def _check_rods(self, connections):
        ports = (self.line.master_port, *self.line.slave_ports)
        if connections.keys() - {port.cylinder.rod for port in ports}:
            raise ValueError("Mechanical connections must use rods from this hydraulic system.")

    def _solve(self, master_stroke, master_input, rod_forces_n):
        forces = {} if rod_forces_n is None else dict(rod_forces_n)
        self._check_rods(forces)
        if not all(isfinite(force) for force in forces.values()):
            raise ValueError("External rod forces must be finite.")
        master = self.line.master_port
        ports = self.line.slave_ports
        master_area = master.area_mm2
        master_direction = -master.pressure_force_direction
        master_stiffness = rod_stiffness(master.cylinder, self.rod_loads.get(master.cylinder.rod))
        master_bias = master_direction * forces.get(master.cylinder.rod, 0.)
        stiffnesses = [rod_stiffness(port.cylinder, self.rod_loads.get(port.cylinder.rod)) for port in ports]
        applied = [port.pressure_force_direction * forces.get(port.cylinder.rod, 0.) for port in ports]

        # Slave equilibrium gives k * x = A * p + F. One zero-stiffness
        # slave can set the pressure, then take up the remaining oil volume.
        free_slaves = [index for index, stiffness in enumerate(stiffnesses) if stiffness == 0.]
        if len(free_slaves) > 1:
            raise ValueError("Multiple unloaded slave rods leave oil displacement indeterminate.")
        volume_per_mpa = 0.
        volume_at_zero_pressure = 0.
        for port, stiffness, force in zip(ports, stiffnesses, applied):
            if stiffness == 0.:
                continue
            area = port.area_mm2
            volume_per_mpa += area * (area / stiffness)
            volume_at_zero_pressure += area * (force / stiffness)

        if free_slaves:
            free_index = free_slaves[0]
            pressure_at_zero_stroke = -applied[free_index] / ports[free_index].area_mm2
            pressure_per_mm = 0.
        else:
            if not isfinite(volume_per_mpa) or volume_per_mpa <= 0.:
                raise ValueError("The hydraulic configuration is outside the numerical range.")
            pressure_at_zero_stroke = -volume_at_zero_pressure / volume_per_mpa
            pressure_per_mm = master_area / volume_per_mpa

        if master_stroke is None:
            equivalent_stiffness = master_stiffness + master_area * pressure_per_mm
            if not isfinite(equivalent_stiffness):
                raise ValueError("Equivalent master stiffness is outside the numerical range.")
            if equivalent_stiffness <= 0.:
                raise ValueError("Prescribed force requires a restoring stiffness to determine master travel.")
            applied_master = master_direction * master_input + master_bias
            master_stroke = (applied_master - master_area * pressure_at_zero_stroke) / equivalent_stiffness
            pressure = (pressure_per_mm * applied_master
                        + master_stiffness * pressure_at_zero_stroke) / equivalent_stiffness
        else:
            pressure = pressure_at_zero_stroke + pressure_per_mm * master_stroke
        if pressure < 0.:
            raise ValueError("The load requires negative supply pressure at the zero-return reference.")

        master_force = master.force_at_pressure(pressure)
        required_input = master_direction * (master_force + master_stiffness * master_stroke - master_bias)
        master_response = CylinderResponse(master_stroke, master_force, master_direction * master_stroke,
                                           master.pressure_force_direction * master_force)
        slaves = []
        outputs = [pressure, master_stroke, master_force, required_input]
        for port, stiffness, force_applied in zip(ports, stiffnesses, applied):
            force = port.force_at_pressure(pressure)
            if stiffness > 0.:
                stroke = (force + force_applied) / stiffness
            else:
                remaining = master.volume_for_stroke(master_stroke) - (
                    volume_at_zero_pressure + volume_per_mpa * pressure
                )
                stroke = remaining / port.area_mm2
            direction = port.pressure_force_direction
            slaves.append(CylinderResponse(stroke, force, direction * stroke, direction * force))
            outputs.extend((stroke, force))
        if not all(isfinite(value) for value in outputs):
            raise ValueError("The hydraulic response is outside the numerical range.")
        if self.return_line is not None:
            strokes = {id(port.cylinder): response.stroke_mm for port, response in zip(ports, slaves)}
            return_volumes = [port.volume_for_stroke(strokes[id(port.cylinder)])
                              for port in self.return_line.slave_ports]
            returning = sum(return_volumes)
            required = self.return_line.master_port.volume_for_stroke(master_stroke)
            tolerance = 1e-12 * max((abs(volume) for volume in return_volumes), default=0.)
            if not isfinite(returning) or not isfinite(required) or not isclose(
                returning, required, rel_tol=1e-9, abs_tol=tolerance
            ):
                raise ValueError("Closed return-line oil volume does not balance at zero return pressure.")
        return HydraulicResponse(pressure, master_response, tuple(slaves), required_input)
