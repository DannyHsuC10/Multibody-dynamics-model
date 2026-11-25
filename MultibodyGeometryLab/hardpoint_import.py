"""Import suspension hardpoints and export the shared static geometry."""

import copy
import csv
import json
from pathlib import Path

import numpy as np

from ._geometry_support import hardpoints
from .dynamics_settings import require_fields
from .elements import parse_member_endpoint
from .hydraulic_circuits import CORNERS_PATTERN
from .pushrod_settings import COEFFICIENTS, OUTBOARD_KEYS, pushrod_from_hardpoints, validate_force

EXCHANGE_FORMAT = "SuspensionHardpoints"
ALIASES = {"UFI": "UCA_F", "URI": "UCA_R", "LFI": "LCA_F", "LRI": "LCA_R",
           "WHEEL_CENTER": "WC", "TIE_ROD_IN": "RACK", "TIE_ROD_OUT": "TOE",
           "PUSH_OUT_LCA": "PUSH_OUT", "PUSHROD_IN": "PUSH_IN", "PUSHROD_OUT": "PUSH_OUT"}
CSV_POINTS = set(hardpoints.POINTS) | hardpoints.TIE_ROD_POINTS | {"WC", "FORWARD"} | (
    hardpoints.PUSHROD_INBOARD | hardpoints.PUSHROD_OUTBOARD)


def validate_rocker_dampers(records):
    if not isinstance(records, list):
        raise ValueError("搖臂避震器必須為清單。")
    names = set()
    result = copy.deepcopy(records)
    for record in result:
        if isinstance(record, dict):
            validate_force(record)
        fields = ("name", "a", "b", *COEFFICIENTS)
        if isinstance(record, dict) and "hydraulic" in record:
            fields = (*fields, "hydraulic")
        require_fields(record, fields, "變形元件")
        name = record["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("搖臂避震器需要唯一的名稱。")
        names.add(name)
        for key in ("a", "b"):
            endpoint = record[key]
            require_fields(endpoint, ("xyz", "mount"), f"{name}/{key}")
            if endpoint["mount"] == "CHASSIS":
                endpoint["mount"] = "body"
            linked = parse_member_endpoint(endpoint["mount"])
            if linked is not None and linked[:2] == ("csv", name):
                raise ValueError(f"{name}/{key}：不可連接同一元件。")
            if linked is None and endpoint["mount"] not in (*hardpoints.CORNERS, "body") and not (
                isinstance(endpoint["mount"], str) and CORNERS_PATTERN.fullmatch(endpoint["mount"])
            ):
                raise ValueError(f"{name}/{key}：無法辨識搖臂或車身接點。")
            hardpoints._coordinate(endpoint["xyz"], f"{name}/{key}")
        if {record["a"]["mount"], record["b"]["mount"]} == {"body"}:
            raise ValueError(f"{name}：至少一端須接在搖臂上。")
    return result


def import_hardpoint_csv(data, path):
    """Overlay the geometry in a CSV onto a multibody project."""
    path = Path(path)
    points = {corner: {} for corner in hardpoints.CORNERS}
    metadata = None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"name", "x_m", "y_m", "z_m"} <= set(reader.fieldnames or ()):
            raise ValueError("硬點 CSV 需要 name、x_m、y_m、z_m 欄位。")
        for row in reader:
            label = (row["name"] or "").strip().upper()
            corner, _, name = label.partition("_")
            name = ALIASES.get(name, name)
            if corner in points and name in CSV_POINTS:
                if name in points[corner]:
                    raise ValueError(f"CSV 第 {reader.line_num} 行：硬點 {label} 重複。")
                try:
                    xyz = [float(row[key]) for key in ("x_m", "y_m", "z_m")]
                    hardpoints._coordinate(xyz, label)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"CSV 第 {reader.line_num} 行：{label} 需要有效的 XYZ 公尺座標。") from error
                points[corner][name] = xyz
            if row.get("jacobian_project"):
                try:
                    entry = json.loads(row["jacobian_project"])
                except ValueError as error:
                    raise ValueError(f"CSV 第 {reader.line_num} 行：jacobian_project JSON 無效。") from error
                if not isinstance(entry, dict) or entry.get("version") != 1 or metadata is not None:
                    raise ValueError("硬點 CSV 需要一筆版本 1 的 jacobian_project 設定。")
                metadata = entry

    return _apply_hardpoints(data, points, metadata or {}, path.name)


def import_hardpoint_json(data, exchange, source):
    if (not isinstance(exchange, dict) or exchange.get("format") != EXCHANGE_FORMAT
            or type(exchange.get("version")) is not int or exchange["version"] != 1
            or not isinstance(exchange.get("points"), dict)):
        raise ValueError("需要 SuspensionHardpoints v1 硬點 JSON。")
    points = {corner: {} for corner in hardpoints.CORNERS}
    for label, xyz in exchange["points"].items():
        corner, _, name = label.strip().upper().partition("_")
        name = ALIASES.get(name, name)
        if corner in points and name in CSV_POINTS:
            if name in points[corner]:
                raise ValueError(f"硬點 {corner}_{name} 重複。")
            points[corner][name] = hardpoints._coordinate(xyz, label).tolist()
    data = copy.deepcopy(data)
    tires = exchange.get("tires", {})
    if isinstance(tires, dict):
        for axle, corner in (("front", "FL"), ("rear", "RL")):
            tire = tires.get(corner)
            if isinstance(tire, dict) and "camber_deg" in tire:
                data["geometry"][axle]["static_camber_deg"] = tire["camber_deg"]
        front = tires.get("FL")
        if isinstance(front, dict):
            for key, setting in (("radius_mm", "free_radius"), ("width_mm", "tire_width")):
                if type(front.get(key)) in (int, float):
                    data["geometry"]["vehicle"][setting] = front[key] / 1000
    return _apply_hardpoints(data, points, exchange, source)


def export_hardpoint_exchange(data):
    from .geometry import static_geometry

    geometry = data["geometry"]
    arms, car, centers = static_geometry(geometry)
    if hardpoints.enabled(geometry):
        centers = data.get("wheel_centers", centers)
    supplemental = geometry.get(hardpoints.KINEMATICS_KEY, {}).get("corners", {})
    points, tires = {}, {}
    for corner in hardpoints.CORNERS:
        axle = "front" if corner.startswith("F") else "rear"
        local = {**supplemental.get(corner, {}), **arms[corner], "WC": centers[corner]}
        mirror = [1, 1 if corner.endswith("L") else -1, 1]
        for key, endpoint in (("TOE", "outer"), ("RACK", "inner")):
            local[key] = np.asarray(centers[corner]) + np.multiply(mirror, data["steering"][axle][endpoint])
        rod = data.get("pushrods", {}).get(corner)
        if rod:
            for key in hardpoints.PUSHROD_INBOARD | hardpoints.PUSHROD_OUTBOARD:
                local.pop(key, None)
            local["PUSH_IN"] = rod["PUSH_IN"]
            local[OUTBOARD_KEYS[rod["outboard"]]] = rod["PUSH_OUT"]
            if rod["inboard"] == "rocker":
                local.update({key: rod[key] for key in ("ROCKER_A", "ROCKER_B")})
        points.update({f"{corner}_{key}": np.asarray(xyz).tolist() for key, xyz in local.items()})
        tires[corner] = {"width_mm": float(car.tire_width) * 1000, "radius_mm": float(car.free_radius) * 1000,
                         "toe_deg": None, "camber_deg": geometry[axle].get("static_camber_deg", 0.)}
    points["CG"] = [-geometry["vehicle"]["cg_to_front"], 0., geometry["vehicle"]["cg_height"]]
    members = []
    for record in data.get("rocker_dampers", []):
        mounts = [record[end]["mount"] for end in ("a", "b")]
        if all(mount in (*hardpoints.CORNERS, "body") for mount in mounts):
            member = {"name": record["name"]}
            for end, mount in zip(("a", "b"), mounts):
                member[end] = {"xyz": record[end]["xyz"], "mount": "CHASSIS" if mount == "body" else mount}
            members.append(member)
    return {"format": EXCHANGE_FORMAT, "version": 1, "points": points, "tires": tires,
            "deformable_members": members}


def _apply_hardpoints(data, points, metadata, source):
    from .project import validate

    data = copy.deepcopy(data)
    geometry = data["geometry"]
    arms = {}
    for corner, values in points.items():
        missing = set(hardpoints.POINTS) - values.keys()
        if missing:
            raise ValueError(f"{corner}：硬點缺少 {', '.join(sorted(missing))}。")
        arms[corner] = {name: values[name] for name in hardpoints.POINTS}
    geometry["hardpoints"] = {"enabled": True, "corners": arms}
    centers = {corner: values["WC"] for corner, values in points.items() if "WC" in values}
    data.pop("wheel_centers", None)
    if centers:
        require_fields(centers, hardpoints.CORNERS, "匯入輪心")
        data["wheel_centers"] = centers

    if "hydraulic_lines" in metadata:
        data["hydraulic_lines"] = metadata["hydraulic_lines"]
    mount_modes = metadata.get("pushrod_mounts", {})
    if not isinstance(mount_modes, dict):
        raise ValueError("CSV 的 pushrod_mounts 需要輪位與輪端安裝類型。")
    previous_pushrods = data.get("pushrods", {})
    data["pushrods"] = {}
    supplemental = copy.deepcopy(geometry.get(hardpoints.KINEMATICS_KEY, {}).get("corners", {}))
    for corner, values in points.items():
        extras = supplemental.setdefault(corner, {})
        push_points = hardpoints.PUSHROD_OUTBOARD | hardpoints.PUSHROD_INBOARD
        if values.keys() & push_points:
            rod = pushrod_from_hardpoints(values, mount_modes.get(corner))
            if rod is None:
                raise ValueError(f"{corner} · 推桿需要 PUSH_IN 和 PUSH_OUT。")
            if rod["inboard"] == "body":
                previous = previous_pushrods.get(corner, {})
                rod.update({key: previous.get(key, default) for key, default in COEFFICIENTS.items()})
                if "hydraulic" in previous:
                    rod["hydraulic"] = previous["hydraulic"]
            data["pushrods"][corner] = rod
            for key in push_points:
                extras.pop(key, None)
        extras.update({name: value for name, value in values.items()
                       if name in hardpoints.TIE_ROD_POINTS or name == "FORWARD"})
    geometry[hardpoints.KINEMATICS_KEY] = {"unit": "m", "source": source, "corners": supplemental}
    data["csv_rockers"] = [
        corner for corner, values in points.items()
        if hardpoints.PUSHROD_INBOARD <= values.keys() and len(hardpoints.PUSHROD_OUTBOARD & values.keys()) == 1
    ]
    data["dynamics"].pop("elements", None)
    records = metadata.get("deformable_members", [])
    automatic = {f"{corner}_PUSH_ROD" for corner, record in data["pushrods"].items() if record["inboard"] == "body"}
    if not isinstance(records, list) or any(
        not isinstance(record, dict) or not isinstance(record.get("name"), str) for record in records
    ):
        raise ValueError("deformable_members 必須是變形元件清單。")
    old = {record["name"]: record for record in data.get("rocker_dampers", ())}
    for record in records:
        if record["name"] in automatic:
            corner = record["name"].split("_")[0]
            rod = data["pushrods"][corner]
            rod.update({key: record[key] for key in COEFFICIENTS if key in record})
            if rod["kind"] == "hydraulic" and record.get("hydraulic") is not None:
                rod["hydraulic"] = record["hydraulic"]
            elif rod["kind"] != "hydraulic" or "hydraulic" in record:
                rod.pop("hydraulic", None)
    data["rocker_dampers"] = []
    for record in records:
        if record["name"] in automatic:
            continue
        previous = old.get(record["name"], {})
        values = {key: record.get(key, previous.get(key, default)) for key, default in COEFFICIENTS.items()}
        config = record.get("hydraulic", previous.get("hydraulic"))
        if values["kind"] == "hydraulic" and config is not None:
            values["hydraulic"] = config
        data["rocker_dampers"].append({"name": record["name"], "a": record.get("a"), "b": record.get("b"), **values})
    data = validate(data)
    set_imported_steering(data)
    return validate(data)


def set_imported_steering(data):
    """Initialize the axle offsets from the source's absolute tie-rod coordinates."""
    from .geometry import static_geometry

    supplemental = data["geometry"].get(hardpoints.KINEMATICS_KEY, {}).get("corners", {})
    if not any(hardpoints.TIE_ROD_POINTS <= points.keys() for points in supplemental.values()):
        return
    _, _, static_centers = static_geometry(data["geometry"])
    centers = data.get("wheel_centers", static_centers)
    for axle, left, right in (("front", "FL", "FR"), ("rear", "RL", "RR")):
        for key, endpoint in (("TOE", "outer"), ("RACK", "inner")):
            offsets = [((np.asarray(supplemental[corner][key]) - centers[corner]) * [1, side, 1])
                       for corner, side in ((left, 1), (right, -1)) if key in supplemental.get(corner, {})]
            if offsets:
                if len(offsets) == 2 and not np.allclose(*offsets, atol=1e-8, rtol=0.):
                    raise ValueError(f"{axle}/{key}：目前轉向接點須以車軸左右鏡射。")
                data["steering"][axle][endpoint] = offsets[0].tolist()
