"""Build the reference mechanism from the project's static suspension geometry."""

from dataclasses import dataclass, replace
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from ._geometry_support import hardpoints, project as geometry_project
from ._geometry_support.static.fv.wireframe_geometry import build_wireframe_scene
from .elements import parse_member_endpoint
from .project import validate
from .pushrod_settings import OUTBOARD_KEYS

if TYPE_CHECKING:
    from .elements import Member
    from .rocker import Rocker, RockerDamper


FIXED = ("UCA_F", "UCA_R", "LCA_F", "LCA_R", "RACK")
MOVING = ("UBJ", "LBJ", "WC", "TOE")
POINTS = FIXED + MOVING
ARM_LINKS = (("UBJ", "UCA_F"), ("UBJ", "UCA_R"), ("LBJ", "LCA_F"), ("LBJ", "LCA_R"))
UPRIGHT_LINKS = tuple(combinations(MOVING, 2))
LINKS = ARM_LINKS + UPRIGHT_LINKS + (("TOE", "RACK"),)


@dataclass(frozen=True)
class CornerGeometry:
    name: str
    axle: str
    reference: tuple
    lengths: tuple
    wheel_radius: float
    wheel_width: float
    members: tuple["Member", ...] | None = None
    body_mount: tuple[float, float, float] | None = None
    rocker: "Rocker | None" = None
    pushrod_configured: bool = False


@dataclass(frozen=True)
class VehicleGeometry:
    corners: tuple[CornerGeometry, ...]
    cg: tuple[float, float, float]
    chassis_edges: tuple
    rocker_dampers: tuple["RockerDamper", ...] = ()
    csv_mechanism: bool = False
    hydraulic_network: object | None = None
    hydraulic_lines: dict | None = None
    hydraulic_cylinders: tuple = ()
    hydraulic_design: dict | None = None


def static_geometry(geometry):
    """Return the common hardpoints, car and four static wheel centers."""
    geometry = geometry_project.validate(geometry)
    car, front, rear = geometry_project.static_objects(geometry)
    if hardpoints.enabled(geometry):
        scene, _ = hardpoints.build_scene(geometry, car, front, rear)
        points = scene.geometry["hardpoints"]
    else:
        scene = build_wireframe_scene(car, front, rear, mount_x=geometry.get("arm_mount_x"),
                                      enforce_mount_clearance=False)
        for axle, checks in scene.geometry["mount_position_checks"].items():
            for name, check in checks.items():
                if check["requested_x"] is not None and check["fallback"]:
                    raise ValueError(f"{axle} · {geometry_project.MOUNT_X_FIELDS[name]}：{check['message']}")
        points = hardpoints.from_scene(scene)
    centers = {key: wheel["center"] for key, wheel in scene.geometry["wheel_projection_geometry"].items()}
    return points, car, centers


def chassis_edges(corners):
    """Construct the fixed chassis envelope from the inboard hinge midpoints."""
    perimeter = ("FL", "FR", "RR", "RL", "FL")
    rings = []
    for kind in ("UCA", "LCA"):
        ring = [tuple(np.mean([corners[name][kind + "_F"], corners[name][kind + "_R"]], axis=0))
                for name in perimeter]
        rings.append(ring)
    upper, lower = rings
    return tuple(zip(upper, upper[1:])) + tuple(zip(lower, lower[1:])) + tuple(zip(upper[:4], lower[:4]))


def build_vehicle(data):
    from .hydraulic_circuits import HydraulicNetwork
    from .rocker import Rocker, RockerDamper, RockerEndpoint, WheelPickup

    data = validate(data)
    corners, car, centers = static_geometry(data["geometry"])
    if hardpoints.enabled(data["geometry"]):
        centers = data.get("wheel_centers", centers)
    records = data.get("rocker_dampers", ())
    dampers = []
    cylinders = []
    rocker_mounts = {record[key]["mount"] for record in records for key in ("a", "b")
                     if record[key]["mount"] in hardpoints.CORNERS}
    reference_points = {}
    pushrods = data.get("pushrods", {})
    legacy_rockers = rocker_mounts | set(data.get("csv_rockers", ()))
    for name in rocker_mounts & pushrods.keys():
        if pushrods[name]["inboard"] != "rocker":
            raise ValueError(f"{name} · 變形元件仍連接此搖臂，請先調整該元件的端點。")
    supplemental = data["geometry"].get(hardpoints.KINEMATICS_KEY, {}).get("corners", {})
    model = []
    for name in hardpoints.CORNERS:
        axle = "front" if name.startswith("F") else "rear"
        points = dict(corners[name])
        points["WC"] = np.asarray(centers[name])
        mirror = np.array([1., 1. if name.endswith("L") else -1., 1.])
        for key, endpoint in (("TOE", "outer"), ("RACK", "inner")):
            points[key] = points["WC"] + mirror * data["steering"][axle][endpoint]
        reference_points[name] = points
        reference = np.array([points[point] for point in POINTS])
        lengths = tuple(float(np.linalg.norm(np.asarray(points[a]) - points[b])) for a, b in LINKS)
        if not np.isfinite(reference).all() or not np.isfinite(lengths).all() or min(lengths) < 1e-6:
            raise ValueError(f"{name}：接點重合或桿長無效，請檢查硬點及轉向拉桿設定。")
        relative = reference[len(FIXED) + 1:] - reference[len(FIXED)]
        if np.linalg.svd(relative, compute_uv=False)[-1] < 1e-6:
            raise ValueError(f"{name}：羊角、輪心及拉桿外接點共平面，請調整拉桿外接點偏移。")
        rod = pushrods.get(name)
        rocker = None
        if rod is not None:
            key = OUTBOARD_KEYS[rod["outboard"]]
            if rod["inboard"] == "rocker":
                rocker_points = {point: rod[point] for point in ("PUSH_IN", "ROCKER_A", "ROCKER_B")}
                rocker_points[key] = rod["PUSH_OUT"]
                rocker = Rocker.from_points(name, points, rocker_points)
            else:
                pickup = WheelPickup.from_pickup(name, points, key, rod["PUSH_OUT"])
                endpoints = (RockerEndpoint("body", np.asarray(rod["PUSH_IN"])),
                             RockerEndpoint(name, pickup.pickup, pickup))
                damper = RockerDamper.from_record({**rod, "name": f"{name}_PUSH_ROD"}, endpoints=endpoints)
                if rod.get("hydraulic") is not None:
                    damper = replace(damper, hydraulic_source=("direct", name))
                dampers.append(damper)
                cylinders.append(rod.get("hydraulic"))
        elif name in legacy_rockers:
            rocker = Rocker.from_points(name, points, supplemental.get(name, {}))
        model.append(CornerGeometry(name, axle, tuple(map(tuple, reference)), lengths,
                                    float(points["WC"][2]), float(car.tire_width), rocker=rocker,
                                    pushrod_configured=rod is not None))
    for record in records:
        endpoints = []
        for key in ("a", "b"):
            endpoint = record[key]
            if parse_member_endpoint(endpoint["mount"]) is not None:
                endpoints.append(RockerEndpoint(endpoint["mount"], np.asarray(endpoint["xyz"])))
                continue
            mount, _, part = endpoint["mount"].partition(":")
            xyz = np.asarray(endpoint["xyz"])
            pickup = WheelPickup.from_pickup(mount, reference_points[mount], OUTBOARD_KEYS[part], xyz) if part else None
            endpoints.append(RockerEndpoint(mount, xyz, pickup))
        damper = RockerDamper.from_record(record, endpoints=tuple(endpoints))
        if record.get("hydraulic") is not None:
            damper = replace(damper, hydraulic_source=("csv", record["name"]))
        dampers.append(damper)
        cylinders.append(record.get("hydraulic"))
    lines = data.get("hydraulic_lines", {})
    network = HydraulicNetwork.build(lines, cylinders) if any(cylinders) else None
    vehicle = data["geometry"]["vehicle"]
    names = [damper.name for damper in dampers]
    if len(set(names)) != len(names):
        raise ValueError("變形元件與直連推桿名稱必須唯一。")
    return VehicleGeometry(tuple(model), (-vehicle["cg_to_front"], 0., vehicle["cg_height"]),
                           chassis_edges(corners), tuple(dampers), "csv_rockers" in data, network, lines,
                           tuple(cylinders), data.get("hydraulic_design"))
