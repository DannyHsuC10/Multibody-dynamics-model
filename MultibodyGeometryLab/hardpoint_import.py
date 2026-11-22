"""Import hardpoint exports from StaticWireframe and JacobianKinematicsLab."""

import copy
import csv
import json
from pathlib import Path

import numpy as np

from ._geometry_support import hardpoints
from .dynamics_settings import require_fields
from .pushrod_settings import COEFFICIENTS, pushrod_from_hardpoints, validate_force

ALIASES = {"UFI": "UCA_F", "URI": "UCA_R", "LFI": "LCA_F", "LRI": "LCA_R",
           "WHEEL_CENTER": "WC"}
CSV_POINTS = set(hardpoints.POINTS) | hardpoints.TIE_ROD_POINTS | {"WC"} | (
    hardpoints.PUSHROD_INBOARD | hardpoints.PUSHROD_OUTBOARD)


def validate_rocker_dampers(records):
    if not isinstance(records, list):
        raise ValueError("搖臂避震器必須為清單。")
    names = set()
    result = copy.deepcopy(records)
    for record in result:
        if isinstance(record, dict):
            validate_force(record)
        require_fields(record, ("name", "a", "b", *COEFFICIENTS), "變形元件")
        name = record["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("搖臂避震器需要唯一的名稱。")
        names.add(name)
        for key in ("a", "b"):
            endpoint = record[key]
            require_fields(endpoint, ("xyz", "mount"), f"{name}/{key}")
            if endpoint["mount"] == "CHASSIS":
                endpoint["mount"] = "body"
            if endpoint["mount"] not in (*hardpoints.CORNERS, "body"):
                raise ValueError(f"{name}/{key}：無法辨識搖臂或車身接點。")
            hardpoints._coordinate(endpoint["xyz"], f"{name}/{key}")
        if {record["a"]["mount"], record["b"]["mount"]} == {"body"}:
            raise ValueError(f"{name}：至少一端須接在搖臂上。")
    return result


def import_hardpoint_csv(data, path):
    """Overlay the geometry in a CSV onto a multibody project."""
    from .project import validate

    path = Path(path)
    data = copy.deepcopy(data)
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

    geometry = data["geometry"]
    arms = {}
    for corner, values in points.items():
        missing = set(hardpoints.POINTS) - values.keys()
        if missing:
            raise ValueError(f"{corner}：硬點 CSV 缺少 {', '.join(sorted(missing))}。")
        arms[corner] = {name: values[name] for name in hardpoints.POINTS}
    geometry["hardpoints"] = {"enabled": True, "corners": arms}
    centers = {corner: values["WC"] for corner, values in points.items() if "WC" in values}
    data.pop("wheel_centers", None)
    if centers:
        require_fields(centers, hardpoints.CORNERS, "CSV 輪心")
        data["wheel_centers"] = centers

    metadata = metadata or {}
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
                raise ValueError(f"{corner} · CSV 推桿需要 PUSH_IN 和 PUSH_OUT。")
            if rod["inboard"] == "body":
                rod.update({key: previous_pushrods.get(corner, {}).get(key, default)
                            for key, default in COEFFICIENTS.items()})
            data["pushrods"][corner] = rod
            for key in push_points:
                extras.pop(key, None)
        extras.update({name: value for name, value in values.items() if name in hardpoints.TIE_ROD_POINTS})
    geometry[hardpoints.KINEMATICS_KEY] = {"unit": "m", "source": path.name, "corners": supplemental}
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
            data["pushrods"][corner].update({key: record[key] for key in COEFFICIENTS if key in record})
    data["rocker_dampers"] = [
        {"name": record["name"], "a": record.get("a"), "b": record.get("b"),
         **{key: record.get(key, old.get(record["name"], {}).get(key, default)) for key, default in COEFFICIENTS.items()}}
        for record in records if record["name"] not in automatic
    ]
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
