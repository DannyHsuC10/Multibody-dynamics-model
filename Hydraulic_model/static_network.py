"""Force- and displacement-controlled equilibrium of pressure lines."""

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
    """Ports sharing one gauge pressure, with optional volume compliance."""

    name: str
    ports: tuple[HydraulicPort, ...]
    compliance_cm3_mpa: float | None = None
    initial_pressure_mpa: float = 0.

    def __post_init__(self):
        object.__setattr__(self, "ports", tuple(self.ports))
        if not self.name or not self.ports:
            raise ValueError("A pressure line needs a name and at least one port.")
        if self.compliance_cm3_mpa is not None and not (
            isfinite(self.compliance_cm3_mpa) and self.compliance_cm3_mpa > 0.
        ):
            raise ValueError("Line compliance must be positive and finite.")
        if not isfinite(self.initial_pressure_mpa) or self.initial_pressure_mpa < 0.:
            raise ValueError("Initial line pressure must be nonnegative and finite.")


@dataclass(frozen=True)
class StaticHydraulicResponse:
    pressures_mpa: Mapping[str, float | None]
    rod_extension_mm: Mapping[PistonRod, float]
    required_rod_forces_n: Mapping[PistonRod, float | None]


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
                raise ValueError("Spring endpoints must connect to rods in this system or to the chassis mount.")
        object.__setattr__(self, "lines", lines)
        object.__setattr__(self, "cylinders", declared_cylinders)
        object.__setattr__(self, "springs", springs)
        object.__setattr__(self, "_cylinders", tuple(cylinders.values()))
        object.__setattr__(self, "rod_loads", MappingProxyType(loads))

    def boundary_conditions(self, rod_forces_n, prescribed_extension_mm):
        rods = {cylinder.rod: index for index, cylinder in enumerate(self._cylinders)}
        prescribed = {} if prescribed_extension_mm is None else dict(prescribed_extension_mm)
        if rod_forces_n.keys() - rods.keys() or not all(isfinite(force) for force in rod_forces_n.values()):
            raise ValueError("Apply finite forces to rods in this hydraulic system.")
        if prescribed.keys() - rods.keys() or not all(isfinite(value) for value in prescribed.values()):
            raise ValueError("Prescribe finite displacements on rods in this hydraulic system.")
        if prescribed.keys() & rod_forces_n.keys():
            raise ValueError("Specify either force or displacement on each rod.")
        applied = np.array([rod_forces_n.get(rod, 0.) for rod in rods], dtype=float)
        return rods, applied, prescribed

    def assemble_matrices(self, rods):
        """Assemble mechanical stiffness, damping and chamber volume derivatives."""
        stiffness = np.diag([rod_stiffness(cylinder, self.rod_loads.get(cylinder.rod))
                             for cylinder in self._cylinders])
        damping = np.zeros_like(stiffness)
        for element in self.springs:
            direction = np.zeros(len(rods))
            # The chassis mount has zero displacement and velocity in this frame.
            for endpoint, sign in ((element.end_a, -1.), (element.end_b, 1.)):
                if isinstance(endpoint, PistonRod):
                    direction[rods[endpoint]] = sign
            # Both rods point along the same axis. Reactions act in opposite
            # directions at the ends and depend on relative motion.
            coupling = np.outer(direction, direction)
            if element.spring is not None:
                stiffness += element.spring.stiffness_n_mm * coupling
            damping += element.damping_n_s_mm * coupling
        if not np.isfinite(stiffness).all():
            raise ValueError("The assembled spring stiffness is outside the numerical range.")
        chamber_areas = np.zeros((len(self.lines), len(rods)))
        for index, line in enumerate(self.lines):
            for port in line.ports:
                chamber_areas[index, rods[port.cylinder.rod]] += port.pressure_force_direction * port.area_mm2
        return stiffness, damping, chamber_areas

    def solve(self, rod_forces_n: Mapping[PistonRod, float], *,
              prescribed_extension_mm: Mapping[PistonRod, float] | None = None) -> StaticHydraulicResponse:
        """Return rod displacements and the required forces on prescribed rods.

        Pressures or required forces without a unique value are returned as
        None in displacement mode when the rod displacements are determined.
        """
        rods, applied, prescribed = self.boundary_conditions(rod_forces_n, prescribed_extension_mm)
        stiffness, _, chamber_areas = self.assemble_matrices(rods)

        # Kx - G.T p = F and Gx + C(p - p0) = 0. C is zero for ideal lines.
        # Move prescribed displacements to the RHS.
        area_scale = max((port.area_mm2 for line in self.lines for port in line.ports), default=1.)
        stiffness_scale = np.max(np.diag(stiffness)) or 1.
        stiffness /= stiffness_scale
        geometry = chamber_areas / area_scale
        fixed = [rods[rod] for rod in prescribed]
        free = [index for rod, index in rods.items() if rod not in prescribed]
        scaled_extension = np.zeros(len(rods))
        scaled_extension[fixed] = np.array(list(prescribed.values()), dtype=float) * stiffness_scale
        free_geometry = geometry[:, free]
        compliance = np.asarray([(line.compliance_cm3_mpa or 0.) * 1000. for line in self.lines])
        scaled_compliance = (compliance / area_scale) * (stiffness_scale / area_scale)
        initial = np.asarray([line.initial_pressure_mpa for line in self.lines])
        matrix = np.block([[stiffness[np.ix_(free, free)], -free_geometry.T],
                           [free_geometry, np.diag(scaled_compliance)]])
        rhs = np.r_[applied[free] - stiffness[free] @ scaled_extension,
                    scaled_compliance * (initial * area_scale) - geometry @ scaled_extension]
        if not np.isfinite(rhs).all():
            raise ValueError("The static response is outside the numerical range.")
        solution, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
        residual = np.abs(matrix @ solution - rhs)
        incompatible = np.max(residual, initial=0.) > 1e-9 * max(1., np.max(np.abs(rhs), initial=0.))
        if prescribed:
            displacement_scale = max(stiffness_scale, np.max(np.abs(scaled_extension), initial=0.),
                                     np.max(np.abs(solution[:len(free)]), initial=0.))
            incompatible |= np.max(residual[len(free):], initial=0.) > 1e-9 * displacement_scale
        if incompatible:
            if prescribed:
                raise ValueError("The prescribed displacements and forces cannot satisfy static equilibrium "
                                 "and oil-volume conservation.")
            raise ValueError("The applied forces cannot satisfy rod equilibrium and oil-volume conservation.")

        pressure_unique = np.ones(len(self.lines), dtype=bool)
        reaction_unique = np.ones(len(fixed), dtype=bool)
        if rank < matrix.shape[0]:
            if not prescribed:
                raise ValueError("Static displacement or pressure is not unique. Check cylinder areas and rod loads.")
            # Only report quantities invariant to every remaining null-space direction.
            null_space = np.linalg.svd(matrix, full_matrices=False)[2][rank:]
            free_null, pressure_null = np.split(null_space, [len(free)], axis=1)
            if np.any(np.linalg.norm(free_null, axis=0) > 1e-9):
                raise ValueError("Free rod displacement is not unique. Prescribe another displacement or add a load.")
            pressure_unique = np.linalg.norm(pressure_null, axis=0) <= 1e-9
            reaction_rows = np.hstack((stiffness[np.ix_(fixed, free)], -geometry[:, fixed].T))
            reaction_unique = (np.linalg.norm(reaction_rows @ null_space.T, axis=1)
                               <= 1e-9 * np.linalg.norm(reaction_rows, axis=1))

        scaled_extension[free] = solution[:len(free)]
        extension = scaled_extension / stiffness_scale
        scaled_pressure = solution[len(free):]
        pressure = scaled_pressure / area_scale
        required = (stiffness @ scaled_extension - geometry.T @ scaled_pressure)[fixed]
        if not (np.isfinite(extension).all() and np.isfinite(pressure).all() and np.isfinite(required).all()):
            raise ValueError("The static response is outside the numerical range.")
        return StaticHydraulicResponse(
            MappingProxyType({line.name: float(value) if unique else None
                              for line, value, unique in zip(self.lines, pressure, pressure_unique)}),
            MappingProxyType(dict(zip(rods, map(float, extension)))),
            MappingProxyType({rod: float(value) if unique else None
                              for rod, value, unique in zip(prescribed, required, reaction_unique)}),
        )
