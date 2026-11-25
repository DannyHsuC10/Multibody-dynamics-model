"""Editable cylinder and hose topology for the hydraulic workbench."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

if __package__:
    from .model import CHASSIS, HydraulicCylinder, LinearSpring, SpringElement
    from .static_network import PressureLine, StaticHydraulicSystem
else:
    from model import CHASSIS, HydraulicCylinder, LinearSpring, SpringElement
    from static_network import PressureLine, StaticHydraulicSystem


Port = tuple[int, Literal["cap", "rod"]]
Hose = tuple[Port, Port]
SpringEnd = tuple[int, Literal["a", "b"]]
SpringTarget = int | None  # None represents the fixed chassis mount.


@dataclass(frozen=True)
class CylinderDesign:
    name: str
    bore_mm: float = 20.
    rod_mm: float = 10.
    compression_force_n: float = 0.
    prescribed_compression_mm: float | None = None
    project_source: tuple[str, str] | None = field(default=None, compare=False)
    project_endpoints: tuple[str, str] | None = field(default=None, compare=False)

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("請輸入液壓缸名稱。")
        if not isfinite(self.bore_mm) or not 0. < self.rod_mm < self.bore_mm:
            raise ValueError("缸徑須為有限正值，且桿徑須大於 0 並小於缸徑。")
        if not isfinite(self.compression_force_n):
            raise ValueError("活塞桿外力須為有限數值。")
        if self.prescribed_compression_mm is not None and not isfinite(self.prescribed_compression_mm):
            raise ValueError("指定位移須為有限數值。")


@dataclass(frozen=True)
class DamperDesign:
    name: str
    stiffness_n_mm: float = 100.
    damping_n_s_mm: float = 0.

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("請輸入避震器名稱。")
        if not isfinite(self.stiffness_n_mm) or self.stiffness_n_mm < 0.:
            raise ValueError("彈簧剛性須為大於或等於 0 的有限數值。")
        if not isfinite(self.damping_n_s_mm) or self.damping_n_s_mm < 0.:
            raise ValueError("阻尼係數須為大於或等於 0 的有限數值。")
        if self.stiffness_n_mm == self.damping_n_s_mm == 0.:
            raise ValueError("彈簧剛性或阻尼係數至少有一項須大於 0。")


@dataclass(frozen=True)
class ProjectLine:
    pressure_mpa: float
    compliance_cm3_mpa: float


def project_cylinder_records(data):
    """Yield the source key, display name and record for each vehicle cylinder."""
    for record in data.get("rocker_dampers", ()):
        yield ("csv", record["name"]), record["name"], record
    pushrods = data.get("pushrods", {})
    for corner, record in pushrods.items():
        if record["inboard"] == "body":
            yield ("direct", corner), f"{corner} · 直連推桿", record
    for corner, records in data.get("dynamics", {}).get("elements", {}).items():
        for record in records:
            if record["name"] != "Push Rod" or (corner not in pushrods and corner not in data.get("csv_rockers", ())):
                yield (corner, record["name"]), f"{corner} · {record['name']}", record


def project_endpoint_labels(source, record):
    scope, name = source
    parts = {"UCA": "上 A 臂", "LCA": "下 A 臂", "UPRIGHT": "upright"}
    if scope == "direct":
        return "車體", f"{name} {parts[record['outboard']]}"
    if scope != "csv":
        return tuple(f"{scope} · {record[end]}" for end in ("a", "b"))

    def label(endpoint):
        if not isinstance(endpoint, dict) or not endpoint.get("mount"):
            return "未設定"
        mount = endpoint["mount"]
        if mount in ("body", "CHASSIS"):
            return "車體"
        if mount.startswith("member:"):
            _, group, end, member = mount.split(":", 3)
            if group == "direct":
                member = f"{member.removesuffix('_PUSH_ROD')} · 直連推桿"
            elif group != "csv":
                member = f"{group} · {member}"
            return f"{member} · {end.upper()} 端"
        corner, _, part = mount.partition(":")
        return f"{corner} {parts.get(part, part or '搖臂')}"

    return tuple(label(record.get(end)) for end in ("a", "b"))


def import_project_circuit(data):
    """Read the three possible cylinder sources in a validated vehicle project."""
    lines = {name: ProjectLine(**settings) for name, settings in data.get("hydraulic_lines", {}).items()}
    ports = {name: [] for name in lines}
    cylinders = {}
    legacy_count = 0
    names = set()
    for source, label, record in project_cylinder_records(data):
        if record["kind"] != "hydraulic":
            continue
        config = record.get("hydraulic")
        if config is None:
            legacy_count += 1
            continue
        key = len(cylinders) + 1
        name = label
        suffix = 2
        while name in names:
            name = f"{label} ({suffix})"
            suffix += 1
        names.add(name)
        cylinders[key] = CylinderDesign(name, config["bore_mm"], config["rod_mm"],
                                        project_source=source, project_endpoints=project_endpoint_labels(source, record))
        for side, field in (("cap", "port_a"), ("rod", "port_b")):
            if config[field]:
                ports[config[field]].append((key, side))
    anchors = {ends[0]: name for name, ends in ports.items() if ends}
    hoses = {tuple(sorted((first, second))) for ends in ports.values() for first, second in zip(ends, ends[1:])}
    return cylinders, hoses, anchors, lines, legacy_count


def pressure_groups(hoses: set[Hose], anchors: Mapping[Port, str] | None = None, *, reserved_names=()) -> dict[Port, str]:
    """Assign the same line name to every port in a connected component."""
    anchors = {} if anchors is None else anchors
    neighbors: dict[Port, set[Port]] = {port: set() for port in anchors}
    for first, second in hoses:
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)
    groups = {}
    reserved = set(anchors.values()) | set(reserved_names)
    line_count = 0
    for port in sorted(neighbors):
        if port in groups:
            continue
        component = {port}
        stack = [port]
        while stack:
            for neighbor in neighbors[stack.pop()]:
                if neighbor not in component:
                    component.add(neighbor)
                    stack.append(neighbor)
        names = {anchors[member] for member in component if member in anchors}
        if len(names) > 1:
            raise ValueError("同一油路不可合併兩組已匯入的管路設定。")
        if names:
            name = names.pop()
        else:
            line_count += 1
            name = f"P{line_count}"
            while name in reserved:
                line_count += 1
                name = f"P{line_count}"
        groups.update(dict.fromkeys(sorted(component), name))
    return groups


def assemble_design(cylinders: Mapping[int, CylinderDesign], hoses: set[Hose],
                    springs: Mapping[int, DamperDesign], spring_links: Mapping[SpringEnd, SpringTarget],
                    *, require_connected: bool = True, anchors=None, project_lines=None):
    if not cylinders:
        raise ValueError("請先新增液壓缸。")
    incomplete = [item.name for key, item in springs.items()
                  if (key, "a") not in spring_links or (key, "b") not in spring_links]
    if incomplete:
        raise ValueError("請接妥避震器的 A、B 兩端：" + "、".join(incomplete))
    groups = pressure_groups(hoses, anchors, reserved_names=project_lines or ())
    connected = {key for key, _ in groups}
    connected.update(target for target in spring_links.values() if target is not None)
    isolated = [item.name for key, item in cylinders.items()
                if key not in connected and item.prescribed_compression_mm is None]
    if require_connected and isolated:
        raise ValueError("請先接管或連接活塞桿：" + "、".join(isolated))

    model = {key: HydraulicCylinder(item.bore_mm, rod_diameter_mm=item.rod_mm)
             for key, item in cylinders.items()}
    ports_by_line = {}
    for (key, side), line_name in groups.items():
        cylinder = model[key]
        port = cylinder.cap_end if side == "cap" else cylinder.rod_end
        ports_by_line.setdefault(line_name, []).append(port)
    settings = project_lines or {}
    lines = tuple(
        PressureLine(name, tuple(ports), settings[name].compliance_cm3_mpa, settings[name].pressure_mpa)
        if name in settings else PressureLine(name, tuple(ports))
        for name, ports in ports_by_line.items()
    )
    mechanical_springs = []
    for key, item in springs.items():
        first, second = (spring_links[(key, end)] for end in ("a", "b"))
        if first == second:
            raise ValueError(f"{item.name} 的兩端須接到不同的機械接點。")
        end_a = CHASSIS if first is None else model[first].rod
        end_b = CHASSIS if second is None else model[second].rod
        spring = LinearSpring(item.stiffness_n_mm) if item.stiffness_n_mm else None
        mechanical_springs.append(SpringElement(spring, end_a, end_b, damping_n_s_mm=item.damping_n_s_mm))
    forces = {model[key].rod: -item.compression_force_n for key, item in cylinders.items()
              if item.prescribed_compression_mm is None}
    prescribed = {model[key].rod: -item.prescribed_compression_mm for key, item in cylinders.items()
                  if item.prescribed_compression_mm is not None}
    system = StaticHydraulicSystem(lines, cylinders=tuple(model.values()), springs=tuple(mechanical_springs))
    return system, model, forces, prescribed


def solve_design(cylinders: Mapping[int, CylinderDesign], hoses: set[Hose],
                 springs: Mapping[int, DamperDesign], spring_links: Mapping[SpringEnd, SpringTarget],
                 *, anchors=None, project_lines=None):
    system, model, forces, prescribed = assemble_design(
        cylinders, hoses, springs, spring_links, anchors=anchors, project_lines=project_lines)
    result = system.solve(forces, prescribed_extension_mm=prescribed)
    compression = {key: -result.rod_extension_mm[cylinder.rod] for key, cylinder in model.items()}
    required_forces = {}
    for key, cylinder in model.items():
        if cylinder.rod in result.required_rod_forces_n:
            force = result.required_rod_forces_n[cylinder.rod]
            required_forces[key] = -force if force is not None else None
    return result.pressures_mpa, compression, required_forces
