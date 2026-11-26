"""Project fields for the four elastic elements, road input and CG loading."""

import math

from ._geometry_support.hardpoints import CORNERS
from .elements import validate_elements


ATTACHMENTS = ("WC", "UBJ", "LBJ", "TOE")
MOUNT_REFERENCES = ("attachment", "wheel_center", "body_coordinates")
PUSHROD_FIELDS = {"z_mm": "車身端 Z", "y_mm": "車身端 Y（左側）"}
LEGACY_MOUNT_FIELDS = ("mount_mm", "attachment", "mount_reference")
TARGETS = (*CORNERS, "all")
CORNER_LIMITS = {
    "mass_kg": ("輪胎與輪端質量 (kg)", 1., 150.),
    "spring_n_m": ("彈簧剛性 (N/m)", 100., 500_000.),
    "damper_ns_m": ("避震器阻尼 (N·s/m)", 0., 5000.),
}
SIMULATION_LIMITS = {
    "bump_mm": ("路面擾動高度 (mm)", -100., 100.),
    "bump_seconds": ("擾動持續時間 (s)", .02, 12.),
    "duration_seconds": ("模擬時間 (s)", .5, 12.),
}
HEAVE_LIMITS = {
    "force_down_n": ("重心向下力 (N)", -5000., 5000.),
    "ramp_seconds": ("施力上升時間 (s)", .02, 12.),
    "duration_seconds": ("模擬時間 (s)", .5, 12.),
}
LATERAL_LIMITS = {
    "force_y_n": ("重心側向力 Fy (N)", -5000., 5000.),
    "ramp_seconds": HEAVE_LIMITS["ramp_seconds"],
    "duration_seconds": HEAVE_LIMITS["duration_seconds"],
}
INERTIA_LIMITS = {
    "ixx": ("Ixx · 側傾", .01, 1_000_000.),
    "iyy": ("Iyy · 俯仰", .01, 1_000_000.),
    "izz": ("Izz · 偏航", .01, 1_000_000.),
}
AERO_FIELDS = {
    "air_density_kg_m3": ("空氣密度", "kg/m³", .1, 2.),
    "drag_area_m2": ("阻力面積 CdA", "m²", 0., 10.),
    "downforce_area_m2": ("下壓力面積 ClA", "m²", 0., 20.),
    "front_balance_percent": ("前軸空力平衡", "%", 0., 100.),
    "pressure_center_height_m": ("壓力中心離地高", "m", 0., 3.),
}
DRIVE_LIMITS = {
    "torque_nm": ("每顆馬達扭矩上限 (N·m)", 0., 1000.),
    "target_slip_ratio": ("目標縱向滑移率（0.12 = 12%）", .01, .5),
    "slip_kp": ("滑移率比例增益 Kp (N·m)", 1., 2000.),
    "steer_deg": ("前輪平均轉角 (°)", -30., 30.),
    "initial_speed_m_s": ("初始車速 (m/s)", 0., 50.),
    "gear_ratio": ("減速比", 1., 20.),
    "gear_efficiency": ("傳動效率", .1, 1.),
    "wheel_inertia": ("單輪轉動慣量 (kg·m²)", .1, 10.),
    "duration_seconds": ("模擬時間 (s)", .5, 12.),
}
DEFAULT_SLIP_CONTROL = {"target_slip_ratio": .12, "slip_kp": 400.}
TEST_LIMITS = {
    "speed_m_s": ("車速 (m/s)", .5, 30.),
    "start_seconds": ("輸入開始時間 (s)", 0., 30.),
    "duration_seconds": ("模擬時間上限 (s)", .5, 120.),
    "steer_deg": ("前輪平均轉角 (°)", -25., 25.),
    "rise_seconds": ("轉向時間 (s)", .2, 20.),
    "frequency_hz": ("轉向頻率 (Hz)", .05, 3.),
    "throttle_percent": ("油門開度 (%)", 0., 100.),
    "rolling_resistance": ("滾動阻力係數", .005, .1),
    "radius_m": ("目標圓半徑 (m)", 4., 200.),
    "road_seed": ("路面種子", 0., 2**32 - 1),
}
TEST_PRESETS = {
    "steer_step": ("1. 定速轉向躍階", dict(speed_m_s=8., start_seconds=1., steer_deg=4., duration_seconds=5.)),
    "steer_ramp": ("2. 定速轉向漸增", dict(speed_m_s=8., start_seconds=1., steer_deg=8., rise_seconds=3.,
                                      duration_seconds=6.)),
    "coast": ("3. 鬆油門滑行至停止", dict(speed_m_s=5., rolling_resistance=.025, duration_seconds=40.)),
    "steer_sine": ("4. 定速正弦轉向", dict(speed_m_s=8., start_seconds=1., steer_deg=4., frequency_hz=.5,
                                      duration_seconds=6.)),
    "throttle_step": ("5. 起步油門躍階", dict(start_seconds=.5, throttle_percent=60., duration_seconds=4.)),
    "cruise_step": ("6. 定速後油門躍階", dict(speed_m_s=8., start_seconds=2., throttle_percent=80.,
                                        duration_seconds=5.)),
    "acceleration": ("7. 直線加速", dict(throttle_percent=100., duration_seconds=6.)),
    "circle": ("8. 定圓測試", dict(speed_m_s=7., radius_m=25., rise_seconds=1., duration_seconds=10.)),
    "iso_road": ("9. ISO 8608 路面", dict(speed_m_s=8., road_class="A", road_seed=1., duration_seconds=6.)),
}


def default_vehicle_test(preset="steer_step"):
    return {"preset": preset, **TEST_PRESETS[preset][1]}


def validate_vehicle_test(data):
    preset = data.get("preset") if isinstance(data, dict) else None
    if not isinstance(preset, str) or preset not in TEST_PRESETS:
        raise ValueError("請選擇車輛測試工況。")
    require_fields(data, default_vehicle_test(preset), "車輛測試設定")
    result = {"preset": preset}
    for key, value in data.items():
        if key in TEST_LIMITS:
            result[key] = bounded(value, *TEST_LIMITS[key])
    if "start_seconds" in result and result["start_seconds"] >= result["duration_seconds"]:
        raise ValueError("輸入開始時間須早於模擬結束時間。")
    if "rise_seconds" in result and result.get("start_seconds", 0.) + result["rise_seconds"] > result["duration_seconds"]:
        raise ValueError("模擬時間須涵蓋完整的轉向輸入。")
    if preset == "iso_road":
        if data["road_class"] not in ("A", "B", "C") or not result["road_seed"].is_integer():
            raise ValueError("ISO 路面須選擇 A、B 或 C 級，並填入整數種子。")
        result.update(road_class=data["road_class"], road_seed=int(result["road_seed"]))
    return result


def default_drive():
    return {"torque_nm": 15., "steer_deg": 3., "initial_speed_m_s": 2.,
            "gear_ratio": 11.2, "gear_efficiency": .95, "wheel_inertia": .45, "duration_seconds": 2.5,
            **DEFAULT_SLIP_CONTROL}


def validate_drive(data):
    if isinstance(data, dict):
        data = {**DEFAULT_SLIP_CONTROL, **data}
    require_fields(data, DRIVE_LIMITS, "馬達行駛設定")
    return {key: bounded(data[key], *limits) for key, limits in DRIVE_LIMITS.items()}


def default_chassis_inertia():
    return {"ixx": 55., "iyy": 95., "izz": 125.}


def default_aero():
    return {"air_density_kg_m3": 1.225, "drag_area_m2": 0., "downforce_area_m2": 0.,
            "front_balance_percent": 50., "pressure_center_height_m": .3}


def validate_aero(data):
    require_fields(data, AERO_FIELDS, "空力設定")
    return {key: bounded(data[key], f"{label} ({unit})", low, high)
            for key, (label, unit, low, high) in AERO_FIELDS.items()}


def default_heave():
    return {"force_down_n": 600., "ramp_seconds": .25, "duration_seconds": 2.5}


def default_lateral():
    return {"force_y_n": 600., "ramp_seconds": .25, "duration_seconds": 2.5}


def parallel_pushrod_mount():
    return {"attachment": "UBJ", "mount_reference": "attachment", "mount_mm": [0., 0., 250.]}


def default_dynamics():
    return {
        "corners": {
            name: {"mass_kg": 10., "spring_n_m": 30000., "damper_ns_m": 1300., **parallel_pushrod_mount()}
            for name in CORNERS
        },
        "tire_damping_ns_m": 100.,
        "simulation": {"target": "FL", "bump_mm": 20., "bump_seconds": .25, "duration_seconds": 2.5},
        "heave": default_heave(),
        "lateral": default_lateral(),
        "drive": default_drive(),
        "aero": default_aero(),
        "vehicle_test": default_vehicle_test(),
        "chassis_inertia": default_chassis_inertia(),
    }


def require_fields(data, keys, label):
    if not isinstance(data, dict) or data.keys() != set(keys):
        raise ValueError(f"{label}的欄位不完整或含未知欄位。")


def bounded(value, label, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label}須為 {low:g} 到 {high:g}。")
    return float(value)


def validate_heave(data):
    return _validate_cg_input(data, HEAVE_LIMITS, "重心 Heave 設定")


def validate_lateral(data):
    return _validate_cg_input(data, LATERAL_LIMITS, "重心側向力設定")


def _validate_cg_input(data, limits, label):
    require_fields(data, limits, label)
    settings = {key: bounded(data[key], *bounds) for key, bounds in limits.items()}
    if settings["ramp_seconds"] > settings["duration_seconds"]:
        raise ValueError("施力上升時間不可超過模擬時間。")
    return settings


def validate_chassis_inertia(data):
    require_fields(data, INERTIA_LIMITS, "車身轉動慣量")
    values = {key: bounded(data[key], f"{label} (kg·m²)", low, high)
              for key, (label, low, high) in INERTIA_LIMITS.items()}
    largest = max(values.values())
    if 2 * largest > sum(values.values()) + 1e-10 * largest:
        raise ValueError("車身轉動慣量中，任一軸的數值不可大於另外兩軸的總和。")
    return values


def validate_pushrod_mounts(data):
    require_fields(data, ("front", "rear"), "Push Rod 車身接點")
    result = {}
    for axle, label in (("front", "前軸"), ("rear", "後軸")):
        values = data[axle]
        require_fields(values, PUSHROD_FIELDS, f"{label} Push Rod 車身接點")
        for key, field in PUSHROD_FIELDS.items():
            if type(values[key]) not in (int, float) or not math.isfinite(values[key]):
                raise ValueError(f"{label} · {field} (mm) 需要有限數值。")
        if values["y_mm"] < 0:
            raise ValueError(f"{label} · 左側車身端 Y 必須大於或等於 0 mm。")
        result[axle] = {key: float(values[key]) for key in PUSHROD_FIELDS}
    return result


def _validate_legacy_mount(corner, values):
    if values["attachment"] not in ATTACHMENTS:
        raise ValueError(f"{corner} · 未知的避震器輪端接點。")
    if values["mount_reference"] not in MOUNT_REFERENCES:
        raise ValueError(f"{corner} · 未知的避震器車身接點基準。")
    mount = values["mount_mm"]
    if not isinstance(mount, list) or len(mount) != 3:
        raise ValueError(f"{corner} · 車身接點需要三個 XYZ 數值。")
    if values["mount_reference"] == "body_coordinates":
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in mount):
            raise ValueError(f"{corner} · 車身接點 XYZ 座標 (mm) 需要有限數值。")
        mount_mm = [float(value) for value in mount]
    else:
        mount_mm = [bounded(value, f"{corner} · 車身接點偏移 (mm)", -2000, 2000) for value in mount]
    return {"mount_mm": mount_mm, "attachment": values["attachment"], "mount_reference": values["mount_reference"]}


def validate_dynamics(data):
    fields = ("corners", "tire_damping_ns_m", "simulation", "heave", "lateral", "chassis_inertia",
              "drive", "aero", "vehicle_test")
    if isinstance(data, dict):
        data = {"heave": default_heave(), "lateral": default_lateral(), "drive": default_drive(),
                "chassis_inertia": default_chassis_inertia(), "aero": default_aero(),
                "vehicle_test": default_vehicle_test(), **data}
    mounts = None
    if isinstance(data, dict) and "pushrod_mounts" in data:
        fields = (*fields, "pushrod_mounts")
        mounts = validate_pushrod_mounts(data["pushrod_mounts"])
    elements = None
    if isinstance(data, dict) and "elements" in data:
        fields = (*fields, "elements")
        elements = validate_elements(data["elements"])
    require_fields(data, fields, "動力學設定")
    require_fields(data["corners"], CORNERS, "動力學輪位")
    corners = {}
    for corner in CORNERS:
        values = data["corners"][corner]
        corner_fields = tuple(CORNER_LIMITS)
        if mounts is None:
            corner_fields = (*corner_fields, *LEGACY_MOUNT_FIELDS)
            if isinstance(values, dict) and "mount_reference" not in values:
                values = {**values, "mount_reference": "wheel_center"}
        elif isinstance(values, dict):
            values = {key: value for key, value in values.items() if key not in LEGACY_MOUNT_FIELDS}
        require_fields(values, corner_fields, corner)
        corner_data = {key: bounded(values[key], f"{corner} · {label}", low, high)
                       for key, (label, low, high) in CORNER_LIMITS.items()}
        if mounts is None:
            corner_data.update(_validate_legacy_mount(corner, values))
        corners[corner] = corner_data

    tire_damping = bounded(data["tire_damping_ns_m"], "輪胎垂直阻尼 (N·s/m)", 0., 5000.)
    simulation = data["simulation"]
    require_fields(simulation, ("target", *SIMULATION_LIMITS), "模擬設定")
    if simulation["target"] not in TARGETS:
        raise ValueError("未知的路面擾動輪位。")
    inputs = {key: bounded(simulation[key], *limits) for key, limits in SIMULATION_LIMITS.items()}
    if inputs["bump_seconds"] > inputs["duration_seconds"]:
        raise ValueError("擾動持續時間不可超過模擬時間。")
    settings = {"corners": corners, "tire_damping_ns_m": tire_damping,
                "simulation": {"target": simulation["target"], **inputs}, "heave": validate_heave(data["heave"]),
                "lateral": validate_lateral(data["lateral"]),
                "drive": validate_drive(data["drive"]),
                "aero": validate_aero(data["aero"]),
                "vehicle_test": validate_vehicle_test(data["vehicle_test"]),
                "chassis_inertia": validate_chassis_inertia(data["chassis_inertia"])}
    if mounts is not None:
        settings["pushrod_mounts"] = mounts
    if elements is not None:
        settings["elements"] = elements
    return settings
