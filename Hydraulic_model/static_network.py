"""Force-controlled static equilibrium of closed, incompressible pressure lines."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

import numpy as np

if __package__:
    from .model import HydraulicCylinder, HydraulicPort, PistonRod, RodLoad, SpringElement, rod_stiffness
else:
    from model import HydraulicCylinder, HydraulicPort, PistonRod, RodLoad, SpringElement, rod_stiffness


@dataclass(frozen=True)
class PressureLine:
    """Ports sharing one gauge pressure and a fixed total oil volume."""

    name: str
    ports: tuple[HydraulicPort, ...]

    def __post_init__(self):
        object.__setattr__(self, "ports", tuple(self.ports))
        if not self.name or not self.ports:
            raise ValueError("A pressure line needs a name and at least one port.")


@dataclass(frozen=True)
class StaticHydraulicResponse:
    pressures_mpa: Mapping[str, float]
    rod_extension_mm: Mapping[PistonRod, float]


@dataclass(frozen=True)
class StaticHydraulicSystem:
    """Solve rod force balances and the oil-volume balance in every line.

    The initial geometry defines zero rod extension and zero volume change.
    Unconnected chambers are vented at zero gauge pressure. A positive rod
    force or displacement acts in the direction of rod extension.
    """

    lines: tuple[PressureLine, ...]
    rod_loads: Mapping[PistonRod, RodLoad] = field(default_factory=dict)
    cylinders: tuple[HydraulicCylinder, ...] = ()
    springs: tuple[SpringElement, ...] = ()
    _cylinders: tuple[HydraulicCylinder, ...] = field(init=False, repr=False)

    def __post_init__(self):
        lines = tuple(self.lines)
        if len({line.name for line in lines}) != len(lines):
            raise ValueError("Pressure line names must be unique.")
        declared_cylinders = tuple(self.cylinders)
        cylinders = {cylinder.rod: cylinder for cylinder in declared_cylinders}
        springs = tuple(self.springs)
        ports = set()
        for line in lines:
            for port in line.ports:
                rod = port.cylinder.rod
                key = (rod, port.side)
                if key in ports:
                    raise ValueError("Each hydraulic port can belong to only one line.")
                ports.add(key)
                cylinders[rod] = port.cylinder
        loads = dict(self.rod_loads)
        if loads.keys() - cylinders.keys():
            raise ValueError("Mechanical loads must connect to rods in this hydraulic system.")
        if not cylinders:
            raise ValueError("The system needs at least one cylinder.")
        for spring in springs:
            if any(isinstance(end, PistonRod) and end not in cylinders for end in (spring.end_a, spring.end_b)):
                raise ValueError("Spring endpoints must connect to rods in this system or to ground.")
        object.__setattr__(self, "lines", lines)
        object.__setattr__(self, "cylinders", declared_cylinders)
        object.__setattr__(self, "springs", springs)
        object.__setattr__(self, "_cylinders", tuple(cylinders.values()))
        object.__setattr__(self, "rod_loads", MappingProxyType(loads))

    def solve(self, rod_forces_n: Mapping[PistonRod, float]) -> StaticHydraulicResponse:
        rods = {cylinder.rod: index for index, cylinder in enumerate(self._cylinders)}
        if rod_forces_n.keys() - rods.keys() or not all(isfinite(force) for force in rod_forces_n.values()):
            raise ValueError("Apply finite forces to rods in this hydraulic system.")

        stiffness = np.diag([rod_stiffness(cylinder, self.rod_loads.get(cylinder.rod))
                             for cylinder in self._cylinders])
        for element in self.springs:
            direction = np.zeros(len(rods))
            for endpoint, sign in ((element.end_a, -1.), (element.end_b, 1.)):
                if isinstance(endpoint, PistonRod):
                    direction[rods[endpoint]] = sign
            # Both rods point along the same axis. Spring extension is x_B - x_A,
            # and its reaction acts with opposite signs on the two endpoints.
            stiffness += element.spring.stiffness_n_mm * np.outer(direction, direction)
        if not np.isfinite(stiffness).all():
            raise ValueError("The assembled spring stiffness is outside the numerical range.")
        applied = np.array([rod_forces_n.get(rod, 0.) for rod in rods], dtype=float)
        chamber_areas = np.zeros((len(self.lines), len(rods)))
        for index, line in enumerate(self.lines):
            for port in line.ports:
                chamber_areas[index, rods[port.cylinder.rod]] += port.pressure_force_direction * port.area_mm2

        # G is the derivative of chamber volume with respect to rod extension.
        # The equilibrium equations are Kx - G.T p = F and Gx = 0.
        area_scale = max((port.area_mm2 for line in self.lines for port in line.ports), default=1.)
        stiffness_scale = np.max(np.diag(stiffness)) or 1.
        geometry = chamber_areas / area_scale
        matrix = np.block([[stiffness / stiffness_scale, -geometry.T],
                           [geometry, np.zeros((len(self.lines), len(self.lines)))]])
        rhs = np.r_[applied, np.zeros(len(self.lines))]
        solution, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
        residual = np.linalg.norm(matrix @ solution - rhs, ord=np.inf)
        if residual > 1e-9 * max(1., np.linalg.norm(rhs, ord=np.inf)):
            raise ValueError("The applied forces cannot satisfy rod equilibrium and oil-volume conservation.")
        if rank < matrix.shape[0]:
            raise ValueError("Static displacement or pressure is not unique. Check cylinder areas and rod loads.")

        extension = solution[:len(rods)] / stiffness_scale
        pressure = solution[len(rods):] / area_scale
        if not np.isfinite(solution).all() or not np.isfinite(extension).all() or not np.isfinite(pressure).all():
            raise ValueError("The static response is outside the numerical range.")
        return StaticHydraulicResponse(
            MappingProxyType({line.name: float(value) for line, value in zip(self.lines, pressure)}),
            MappingProxyType(dict(zip(rods, map(float, extension)))),
        )
