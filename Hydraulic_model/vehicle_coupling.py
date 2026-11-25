"""Use the workbench's static circuit as a force element in the vehicle solver."""

from dataclasses import asdict, dataclass

import numpy as np

from .design import CylinderDesign, DamperDesign, ProjectLine, assemble_design


DEFAULT_COMPLIANCE = 5.  # cm³/MPa for hoses first drawn in the workbench


def vehicle_design(cylinders, hoses, anchors, lines, springs, links):
    cylinder_data = {}
    for key, cylinder in cylinders.items():
        record = asdict(cylinder)
        record.pop("project_endpoints")
        if cylinder.project_source is not None:
            record["project_source"] = list(cylinder.project_source)
        cylinder_data[str(key)] = record
    return {
        "cylinders": cylinder_data,
        "hoses": [[list(first), list(second)] for first, second in sorted(hoses)],
        "anchors": [[key, side, line] for (key, side), line in sorted(anchors.items())],
        "lines": {name: asdict(line) for name, line in lines.items()},
        "springs": {str(key): asdict(spring) for key, spring in springs.items()},
        "links": [[key, end, target] for (key, end), target in sorted(links.items())],
    }


def read_vehicle_design(data):
    try:
        cylinders = {}
        for key, record in data["cylinders"].items():
            source = record["project_source"]
            if source is not None:
                if (not isinstance(source, (list, tuple)) or len(source) != 2
                        or not all(isinstance(item, str) and item for item in source)):
                    raise ValueError("液壓缸的車輛來源無效。")
                source = tuple(source)
            cylinders[int(key)] = CylinderDesign(**{**record, "project_source": source})
        hoses = {(tuple(first), tuple(second)) for first, second in data["hoses"]}
        anchors = {(key, side): line for key, side, line in data["anchors"]}
        lines = {name: ProjectLine(**record) for name, record in data["lines"].items()}
        springs = {int(key): DamperDesign(**record) for key, record in data["springs"].items()}
        links = {(key, end): target for key, end, target in data["links"]}
        ports = set(anchors) | {port for hose in hoses for port in hose}
        if any(key not in cylinders or side not in ("cap", "rod") for key, side in ports):
            raise ValueError("液壓設計包含無效的油口。")
        if any(name not in lines for name in anchors.values()):
            raise ValueError("液壓設計包含未定義的專案油路。")
        if any(key not in springs or end not in ("a", "b")
               or (target is not None and target not in cylinders) for (key, end), target in links.items()):
            raise ValueError("液壓設計包含無效的避震器接點。")
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("無法讀取多體專案的液壓設計。") from error
    return cylinders, hoses, anchors, lines, springs, links


@dataclass(frozen=True)
class VehicleCircuit:
    neutral_force: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray

    def axial_response(self, extension_m, speed_m_s):
        force = self.neutral_force - self.stiffness @ extension_m - self.damping @ speed_m_s
        return force, self.stiffness, self.damping


def assemble_vehicle_circuit(data, sources):
    cylinders, hoses, anchors, lines, springs, links = read_vehicle_design(data)
    source_indices = {source: index for index, source in enumerate(sources) if source is not None}
    bound = {key: source_indices.get(item.project_source) for key, item in cylinders.items()
             if item.project_source is not None}
    if not bound or None in bound.values() or set(bound.values()) != set(source_indices.values()) or (
        len(bound) != len(source_indices)
    ):
        raise ValueError("液壓設計中的車載缸與多體模型不符。請重新匯入車輛油路並確認各缸的來源。")

    system, model, forces, prescribed = assemble_design(
        cylinders, hoses, springs, links, require_connected=False, anchors=anchors, project_lines=lines,
        default_compliance_cm3_mpa=DEFAULT_COMPLIANCE)
    rods = {model[key].rod: index for index, key in enumerate(model)}
    vehicle_rods = tuple(model[key].rod for key in bound)
    for rod in vehicle_rods:
        forces.pop(rod, None)
        prescribed[rod] = 0.
    _, damping, _ = system.assemble_matrices(rods)
    driven = [rods[rod] for rod in vehicle_rods]
    free = [index for rod, index in rods.items() if rod not in prescribed]
    if np.any(damping[free]):
        raise ValueError("自由活塞桿上的避震器阻尼需要瞬態求解。連動時請指定該桿位移或將其阻尼設為零。")

    def required_force():
        try:
            result = system.solve(forces, prescribed_extension_mm=prescribed)
        except ValueError as error:
            raise ValueError(f"無法求解液壓連動。{error}") from error
        reaction = [result.required_rod_forces_n[rod] for rod in vehicle_rods]
        if any(force is None for force in reaction):
            raise ValueError("液壓連動的車載缸反力沒有唯一解。")
        return np.asarray(reaction)

    # The workbench equilibrium is affine for fixed topology and compliance.
    # Evaluate the offset and unit slopes once, then reuse them at every solver step.
    at_rest = required_force()
    columns = []
    for rod in vehicle_rods:
        prescribed[rod] = 1.
        columns.append(required_force() - at_rest)
        prescribed[rod] = 0.
    indices = tuple(bound.values())
    shape = (len(sources), len(sources))
    neutral = np.zeros(len(sources))
    stiffness = np.zeros(shape)
    viscous = np.zeros(shape)
    # Workbench reactions are external forces needed to impose rod extension.
    # Apply their opposite to the vehicle and convert N/mm and N·s/mm to SI.
    neutral[list(indices)] = -at_rest
    stiffness[np.ix_(indices, indices)] = 1000. * np.column_stack(columns)
    viscous[np.ix_(indices, indices)] = 1000. * damping[np.ix_(driven, driven)]
    if not (np.isfinite(stiffness).all() and np.isfinite(viscous).all()):
        raise ValueError("液壓連動的剛性或阻尼超出數值範圍。")
    return VehicleCircuit(neutral, stiffness, viscous)
