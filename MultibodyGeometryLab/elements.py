"""Editable two-point suspension elements and their project representation."""

from dataclasses import asdict, dataclass
from math import isfinite

from ._geometry_support.hardpoints import CORNERS


DAMPER_POINT = "DAMPER"
PUSH_ROD = "Push Rod"
DEFORMABLE_KINDS = ("spring_damper", "hydraulic")
KINDS = ("rigid", *DEFORMABLE_KINDS)


def member_endpoint_mount(scope, name, end):
    return f"member:{scope}:{end}:{name}"


def parse_member_endpoint(mount):
    """Return the element scope, name and end for a linked mount."""
    if not isinstance(mount, str) or not mount.startswith("member:"):
        return None
    _, _, suffix = mount.partition(":")
    scope, _, rest = suffix.partition(":")
    end, _, name = rest.partition(":")
    if scope not in ("csv", "direct", *CORNERS) or end not in ("a", "b") or not name.strip():
        raise ValueError("變形元件的參照端點不完整。")
    return scope, name, end


@dataclass(frozen=True)
class Member:
    name: str
    a: str
    b: str
    kind: str
    stiffness_n_m: float | None = None
    damping_ns_m: float | None = None
    hydraulic_force_n: float = 0.
    hydraulic: dict | None = None


def default_elements(*, csv_mechanism=False):
    from .geometry import LINKS

    rows = [Member(f"{a} → {b}", a, b, "rigid") for a, b in LINKS]
    if not csv_mechanism:
        rows.append(Member(PUSH_ROD, DAMPER_POINT, "UBJ", "spring_damper"))
    return {name: [asdict(row) for row in rows] for name in CORNERS}


def validate_elements(data):
    from .geometry import POINTS

    if not isinstance(data, dict) or set(data) != set(CORNERS):
        raise ValueError("機構元件必須包含 FL、FR、RL、RR 四個輪位。")
    points = (*POINTS, DAMPER_POINT)
    permitted = Member.__dataclass_fields__.keys()
    result = {}
    for corner, records in data.items():
        if not isinstance(records, list) or not 1 <= len(records) <= 64:
            raise ValueError(f"{corner} · 需要 1 至 64 個機構元件。")
        names = set()
        rows = []
        for record in records:
            if (not isinstance(record, dict) or not {"name", "a", "b", "kind"} <= record.keys()
                    or record.keys() - permitted):
                raise ValueError(f"{corner} · 元件欄位不完整或含未知設定。")
            member = Member(**record)
            if (not isinstance(member.name, str) or not member.name.strip()
                    or len(member.name) > 64 or member.name in names):
                raise ValueError(f"{corner} · 元件名稱須為 1 至 64 字，且不可重複。")
            names.add(member.name)
            if member.a not in points or member.b not in points or member.a == member.b:
                raise ValueError(f"{corner} · {member.name} 需要兩個不同的既有接點。")
            if member.kind not in KINDS:
                raise ValueError(f"{corner} · {member.name} 的元件類型無效。")
            if member.name == PUSH_ROD and {member.a, member.b} != {DAMPER_POINT, "UBJ"}:
                raise ValueError(f"{corner} · Push Rod 必須連接 DAMPER 與 UBJ，可變更其元件類型。")
            for value, low, high, label in ((member.stiffness_n_m, 0., 500_000., "剛性"),
                                            (member.damping_ns_m, 0., 5000., "阻尼"),
                                            (member.hydraulic_force_n, -100_000., 100_000., "液壓缸軸力")):
                if value is not None and (type(value) not in (int, float) or not isfinite(value)
                                          or not low <= value <= high):
                    raise ValueError(f"{corner} · {member.name} 的{label}須在 {low:g} 至 {high:g} 之間，或留空沿用輪位值。")
            if member.hydraulic_force_n is None:
                raise ValueError(f"{corner} · {member.name} 的液壓缸軸力需要數值。")
            row = asdict(member)
            if member.hydraulic is not None and member.kind == "hydraulic":
                from .hydraulic_circuits import validate_cylinder
                row["hydraulic"] = validate_cylinder(member.hydraulic)
            else:
                row["hydraulic"] = None
            rows.append(row)
        result[corner] = rows
    return result
