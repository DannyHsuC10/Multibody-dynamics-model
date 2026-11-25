"""Edit four direct-acting dampers and a road displacement event."""

import tkinter as tk
from tkinter import ttk

from ._geometry_support.hardpoints import CORNERS
from .dynamics_settings import (DRIVE_LIMITS, HEAVE_LIMITS, LATERAL_LIMITS, SIMULATION_LIMITS,
                                validate_drive, validate_heave, validate_lateral)


TARGET_LABELS = {**dict(zip(CORNERS, CORNERS)), "all": "四輪"}


def add_entry(parent, row, label, variable):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
    entry = ttk.Entry(parent, textvariable=variable, width=12, justify="right")
    entry.grid(row=row, column=1, sticky="w")
    return entry


class RoadEditor(ttk.Frame):
    def __init__(self, parent, mark_dirty, start):
        super().__init__(parent, padding=10)
        self.target = tk.StringVar(self)
        self.fields = {key: tk.StringVar(self) for key in SIMULATION_LIMITS}
        for variable in (self.target, *self.fields.values()):
            variable.trace_add("write", mark_dirty)

        inputs = ttk.Frame(self)
        inputs.pack(fill="x")
        ttk.Label(inputs, text="路面擾動輪位").grid(row=0, column=0, sticky="w")
        target = ttk.Combobox(inputs, textvariable=self.target, values=tuple(TARGET_LABELS.values()),
                              width=8, state="readonly")
        target.grid(row=0, column=1, sticky="w")
        for row, (key, (label, _, _)) in enumerate(SIMULATION_LIMITS.items(), 1):
            add_entry(inputs, row, label, self.fields[key])
        self.start_button = ttk.Button(self, text="模擬路面擾動", command=start)
        self.start_button.pack(fill="x", pady=(8, 0))

    def populate(self, settings):
        self.target.set(TARGET_LABELS[settings["target"]])
        for key, variable in self.fields.items():
            variable.set(str(settings[key]))

    def draft(self):
        try:
            target = next(key for key, label in TARGET_LABELS.items() if label == self.target.get())
            return {"target": target, **{key: float(variable.get()) for key, variable in self.fields.items()}}
        except (StopIteration, ValueError) as error:
            raise ValueError("路面擾動需要有效的輪位、高度及時間。") from error


class _CGForceEditor(ttk.Frame):
    limits = HEAVE_LIMITS
    validate = staticmethod(validate_heave)
    description = "平坦地面 · 車身只沿 Z 軸移動"
    directions = "正值向下施力，負值向上。"
    note = "車身質量採車輛總質量扣除四輪輪端質量。"
    button_label = "模擬重心 Heave"

    def __init__(self, parent, mark_dirty, start):
        super().__init__(parent, padding=10)
        ttk.Label(self, text=self.description).pack(anchor="w")
        inputs = ttk.Frame(self)
        inputs.pack(fill="x", pady=(8, 6))
        self.fields = {}
        for row, (name, (label, _, _)) in enumerate(self.limits.items()):
            variable = self.fields[name] = tk.StringVar(self)
            variable.trace_add("write", mark_dirty)
            add_entry(inputs, row, label, variable)
        ttk.Label(self, text=self.directions + "外力在上升時間內平滑增加，之後維持。",
                  wraplength=310).pack(anchor="w", pady=(2, 7))
        ttk.Label(self, text=self.note,
                  wraplength=310).pack(anchor="w", pady=(0, 8))
        self.start_button = ttk.Button(self, text=self.button_label, command=start)
        self.start_button.pack(fill="x")

    def populate(self, settings):
        for name, variable in self.fields.items():
            variable.set(str(settings[name]))

    def draft(self):
        try:
            parameters = {name: float(variable.get()) for name, variable in self.fields.items()}
        except ValueError as error:
            raise ValueError("重心外力、施力時間與模擬時間需要有效數值。") from error
        return self.validate(parameters)


class HeaveEditor(_CGForceEditor):
    pass


class LateralEditor(_CGForceEditor):
    limits = LATERAL_LIMITS
    validate = staticmethod(validate_lateral)
    description = "平坦地面 · 簧上重心沿 Y 方向受力"
    directions = "正值向左施力，負值向右。"
    note = "地面側向反力依各輪正向輪荷分配，適用於抓地條件下的側傾分析。"
    button_label = "模擬重心側向力"


class DriveEditor(ttk.Frame):
    def __init__(self, parent, mark_dirty, start):
        super().__init__(parent, padding=10)
        self.fields = {key: tk.StringVar(self) for key in DRIVE_LIMITS}
        for row, (key, (label, _, _)) in enumerate(DRIVE_LIMITS.items()):
            variable = self.fields[key]
            variable.trace_add("write", mark_dirty)
            add_entry(self, row, label, variable)
        ttk.Label(self, text="單馬達後驅 · Hoosier W8 Pacejka 輪胎。正轉角向左，齒條在模擬期間固定。",
                  wraplength=312).grid(row=len(self.fields), columnspan=2, sticky="w", pady=8)
        self.start_button = ttk.Button(self, text="模擬馬達行駛", command=start)
        self.start_button.grid(row=len(self.fields) + 1, columnspan=2, sticky="ew")

    def populate(self, settings):
        for key, variable in self.fields.items():
            variable.set(str(settings[key]))

    def draft(self):
        try:
            settings = {key: float(variable.get()) for key, variable in self.fields.items()}
        except ValueError as error:
            raise ValueError("馬達行駛欄位需要有效數值。") from error
        return validate_drive(settings)
