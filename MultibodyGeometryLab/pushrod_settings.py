"""Pushrod hardpoints in the static chassis coordinate frame, stored in metres."""

import copy

import numpy as np

from ._geometry_support.hardpoints import CORNERS, _coordinate
from .dynamics_settings import bounded, require_fields
from .hydraulic_circuits import validate_cylinder


OUTBOARD = ("LCA", "UCA", "UPRIGHT")
OUTBOARD_KEYS = {"LCA": "PUSH_OUT", "UCA": "PUSH_OUT_UCA", "UPRIGHT": "PUSH_OUT_UPRIGHT"}
AXIS_POINTS = ("ROCKER_A", "ROCKER_B")
COEFFICIENTS = {"kind": "spring_damper", "spring_n_m": 30000., "damper_ns_m": 1300., "hydraulic_force_n": 0.}


def validate_force(record):
    for key, default in COEFFICIENTS.items():
        record.setdefault(key, default)
    if record["kind"] not in ("spring_damper", "hydraulic"):
        raise ValueError("變形元件須為彈簧阻尼或液壓缸。")
    if "hydraulic" in record:
        if record["kind"] != "hydraulic":
            raise ValueError("只有液壓缸可設定油口。")
        record["hydraulic"] = validate_cylinder(record["hydraulic"])
    for key, label, low, high in (("spring_n_m", "彈簧剛性 (N/m)", 0., 500_000.),
                                  ("damper_ns_m", "阻尼 (N·s/m)", 0., 5000.),
                                  ("hydraulic_force_n", "液壓缸軸力 (N)", -100_000., 100_000.)):
        record[key] = bounded(record[key], label, low, high)


def validate_pushrods(records):
    if not isinstance(records, dict) or records.keys() - set(CORNERS):
        raise ValueError("推桿設定只接受 FL、FR、RL、RR。")
    result = copy.deepcopy(records)
    for name, record in result.items():
        if not isinstance(record, dict):
            raise ValueError(f"{name} · 推桿設定無效。")
        mode = record.get("inboard")
        if mode not in ("body", "rocker"):
            raise ValueError(f"{name} · 推桿內側須連接車體或搖臂。")
        axes = AXIS_POINTS if mode == "rocker" else ()
        if mode == "body":
            validate_force(record)
        force_fields = (*COEFFICIENTS, "hydraulic") if "hydraulic" in record else tuple(COEFFICIENTS)
        require_fields(record, ("inboard", "outboard", "PUSH_IN", "PUSH_OUT",
                                *(axes or force_fields)), f"{name} · 推桿")
        if record["outboard"] not in OUTBOARD:
            raise ValueError(f"{name} · 推桿輪端須連接下 A 臂、上 A 臂或 upright。")
        for key in ("PUSH_IN", "PUSH_OUT", *axes):
            record[key] = _coordinate(record[key], f"{name}/{key}").tolist()
        pairs = (("PUSH_IN", "PUSH_OUT"), ("ROCKER_A", "ROCKER_B")) if axes else (("PUSH_IN", "PUSH_OUT"),)
        for a, b in pairs:
            if np.linalg.norm(np.subtract(record[a], record[b])) <= 1e-8:
                raise ValueError(f"{name} · {a} 與 {b} 不可重合。")
    return result


def pushrod_from_hardpoints(points, outboard=None):
    coordinates = {kind: points[key] for kind, key in OUTBOARD_KEYS.items() if key in points}
    if len(coordinates) != 1 or "PUSH_IN" not in points:
        return None
    stored_mount, pickup = next(iter(coordinates.items()))
    outboard = stored_mount if outboard is None else outboard
    if outboard not in OUTBOARD:
        return None
    record = {"inboard": "rocker" if any(key in points for key in AXIS_POINTS) else "body",
              "outboard": outboard, "PUSH_IN": points["PUSH_IN"], "PUSH_OUT": pickup}
    if record["inboard"] == "rocker":
        record.update({key: points[key] for key in AXIS_POINTS if key in points})
    return record
