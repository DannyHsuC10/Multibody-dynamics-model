"""Shared oil-line pressures for double-acting cylinders in the multibody solver."""

from dataclasses import dataclass
from math import pi
import re

import numpy as np

from .dynamics_settings import bounded, require_fields


def line_name(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 64:
        raise ValueError("管路名稱須為 1 至 64 字。")
    return value.strip()


def validate_lines(lines):
    if not isinstance(lines, dict):
        raise ValueError("液壓管路設定需要名稱與參數。")
    result = {}
    for name, values in lines.items():
        name = line_name(name)
        if name in result:
            raise ValueError(f"液壓管路名稱重複：{name}")
        require_fields(values, ("pressure_mpa", "compliance_cm3_mpa"), f"管路 {name}")
        result[name] = {
            "pressure_mpa": bounded(values["pressure_mpa"], f"{name} · 初始壓力 (MPa)", 0., 100.),
            "compliance_cm3_mpa": bounded(values["compliance_cm3_mpa"], f"{name} · 順應性 (cm³/MPa)", .001, 10000.),
        }
    return result


def validate_cylinder(values):
    require_fields(values, ("bore_mm", "rod_mm", "port_a", "port_b"), "雙作用液壓缸")
    bore = bounded(values["bore_mm"], "缸徑 (mm)", .1, 1000.)
    rod = bounded(values["rod_mm"], "桿徑 (mm)", .1, 1000.)
    if rod >= bore:
        raise ValueError("液壓缸的桿徑須小於缸徑。")
    a, b = ("" if values[key] == "" else line_name(values[key]) for key in ("port_a", "port_b"))
    if not (a or b):
        raise ValueError("請為液壓缸至少指定一條管路。")
    return dict(bore_mm=bore, rod_mm=rod, port_a=a, port_b=b)


CORNERS_PATTERN = re.compile(r"(FL|FR|RL|RR):(UCA|LCA|UPRIGHT)")


@dataclass(frozen=True)
class HydraulicNetwork:
    """A and B chamber areas are positive and negative volume derivatives, respectively."""

    names: tuple[str, ...]
    areas: np.ndarray
    initial_pressures: np.ndarray
    inverse_compliances: np.ndarray

    @classmethod
    def build(cls, lines, cylinders):
        lookup = {name: index for index, name in enumerate(lines)}
        areas = np.zeros((len(lines), len(cylinders)))
        for index, values in enumerate(cylinders):
            if values is None:
                continue
            cap = pi / 4 * (values["bore_mm"] * .001)**2
            annular = cap - pi / 4 * (values["rod_mm"] * .001)**2
            for port, area in (("port_a", cap), ("port_b", -annular)):
                name = values[port]
                if name:
                    if name not in lookup:
                        raise ValueError(f"液壓缸 {port[-1].upper()} 油口引用了未定義的管路 {name}。")
                    areas[lookup[name], index] += area
        return cls(tuple(lines), areas, np.array([line["pressure_mpa"] * 1e6 for line in lines.values()]),
                   np.array([1e12 / line["compliance_cm3_mpa"] for line in lines.values()]))

    def response(self, strokes):
        """Common line pressures, cylinder forces, and positive stroke stiffness in SI."""
        raw_pressure = self.initial_pressures - self.inverse_compliances * (self.areas @ strokes)
        pressure = np.maximum(0., raw_pressure)
        force = self.areas.T @ pressure
        active = np.where(raw_pressure >= 0., self.inverse_compliances, 0.)
        stiffness = self.areas.T @ (active[:, None] * self.areas)
        return pressure, force, stiffness
