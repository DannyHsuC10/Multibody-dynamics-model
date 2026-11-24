"""Editable cylinder and hose topology for the hydraulic workbench."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

if __package__:
    from .model import GROUND, HydraulicCylinder, LinearSpring, SpringElement
    from .static_network import PressureLine, StaticHydraulicSystem
else:
    from model import GROUND, HydraulicCylinder, LinearSpring, SpringElement
    from static_network import PressureLine, StaticHydraulicSystem


Port = tuple[int, Literal["cap", "rod"]]
Hose = tuple[Port, Port]
SpringEnd = tuple[int, Literal["a", "b"]]
SpringTarget = int | None  # None represents mechanical ground.


@dataclass(frozen=True)
class CylinderDesign:
    name: str
    bore_mm: float = 20.
    rod_mm: float = 10.
    compression_force_n: float = 0.

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("請輸入液壓缸名稱。")
        if not isfinite(self.bore_mm) or not 0. < self.rod_mm < self.bore_mm:
            raise ValueError("缸徑須為有限正值，且桿徑須大於 0 並小於缸徑。")
        if not isfinite(self.compression_force_n):
            raise ValueError("活塞桿外力須為有限數值。")


@dataclass(frozen=True)
class SpringDesign:
    name: str
    stiffness_n_mm: float = 100.

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("請輸入彈簧名稱。")
        if not isfinite(self.stiffness_n_mm) or self.stiffness_n_mm <= 0.:
            raise ValueError("彈簧剛性須為大於 0 的有限數值。")


def pressure_groups(hoses: set[Hose]) -> dict[Port, str]:
    """Assign the same line name to every port in a connected component."""
    neighbors: dict[Port, set[Port]] = {}
    for first, second in hoses:
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)
    groups = {}
    line_count = 0
    for port in sorted(neighbors):
        if port in groups:
            continue
        line_count += 1
        name = f"P{line_count}"
        groups[port] = name
        stack = [port]
        while stack:
            for neighbor in neighbors[stack.pop()]:
                if neighbor not in groups:
                    groups[neighbor] = name
                    stack.append(neighbor)
    return groups


def solve_design(cylinders: Mapping[int, CylinderDesign], hoses: set[Hose],
                 springs: Mapping[int, SpringDesign], spring_links: Mapping[SpringEnd, SpringTarget]):
    if not cylinders:
        raise ValueError("請先新增液壓缸。")
    incomplete = [item.name for key, item in springs.items()
                  if (key, "a") not in spring_links or (key, "b") not in spring_links]
    if incomplete:
        raise ValueError("請接妥彈簧的 A、B 兩端：" + "、".join(incomplete))
    groups = pressure_groups(hoses)
    connected = {key for key, _ in groups}
    connected.update(target for target in spring_links.values() if target is not None)
    isolated = [item.name for key, item in cylinders.items() if key not in connected]
    if isolated:
        raise ValueError("請先接管或連接活塞桿：" + "、".join(isolated))

    model = {key: HydraulicCylinder(item.bore_mm, rod_diameter_mm=item.rod_mm)
             for key, item in cylinders.items()}
    ports_by_line = {}
    for (key, side), line_name in groups.items():
        cylinder = model[key]
        port = cylinder.cap_end if side == "cap" else cylinder.rod_end
        ports_by_line.setdefault(line_name, []).append(port)
    lines = tuple(PressureLine(name, tuple(ports)) for name, ports in ports_by_line.items())
    mechanical_springs = []
    for key, item in springs.items():
        first, second = (spring_links[(key, end)] for end in ("a", "b"))
        if first == second:
            raise ValueError(f"{item.name} 的兩端須接到不同的機械接點。")
        end_a = GROUND if first is None else model[first].rod
        end_b = GROUND if second is None else model[second].rod
        mechanical_springs.append(SpringElement(LinearSpring(item.stiffness_n_mm), end_a, end_b))
    forces = {model[key].rod: -item.compression_force_n for key, item in cylinders.items()}
    system = StaticHydraulicSystem(lines, cylinders=tuple(model.values()), springs=tuple(mechanical_springs))
    result = system.solve(forces)
    compression = {key: -result.rod_extension_mm[cylinder.rod] for key, cylinder in model.items()}
    return result.pressures_mpa, compression
