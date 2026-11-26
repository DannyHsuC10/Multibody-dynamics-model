"""Input forms for the supported dynamic simulations."""

import tkinter as tk
from tkinter import ttk

from ._geometry_support.hardpoints import CORNERS
from .dynamics_settings import (DRIVE_LIMITS, HEAVE_LIMITS, LATERAL_LIMITS, SIMULATION_LIMITS,
                                TEST_LIMITS, TEST_PRESETS, validate_drive,
                                validate_heave, validate_lateral, validate_vehicle_test)


TARGET_LABELS = {**dict(zip(CORNERS, CORNERS)), "all": "四輪"}


def add_entry(parent, row, label, variable):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
    entry = ttk.Entry(parent, textvariable=variable, width=12, justify="right")
    entry.grid(row=row, column=1, sticky="w")
    return entry


class RoadEditor(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.target = tk.StringVar(self)
        self.fields = {key: tk.StringVar(self) for key in SIMULATION_LIMITS}

        inputs = ttk.Frame(self)
        inputs.pack(fill="x")
        ttk.Label(inputs, text="路面擾動輪位").grid(row=0, column=0, sticky="w")
        target = ttk.Combobox(inputs, textvariable=self.target, values=tuple(TARGET_LABELS.values()),
                              width=8, state="readonly")
        target.grid(row=0, column=1, sticky="w")
        for row, (key, (label, _, _)) in enumerate(SIMULATION_LIMITS.items(), 1):
            add_entry(inputs, row, label, self.fields[key])

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

    def __init__(self, parent):
        super().__init__(parent, padding=10)
        ttk.Label(self, text=self.description).pack(anchor="w")
        inputs = ttk.Frame(self)
        inputs.pack(fill="x", pady=(8, 6))
        self.fields = {}
        for row, (name, (label, _, _)) in enumerate(self.limits.items()):
            variable = self.fields[name] = tk.StringVar(self)
            add_entry(inputs, row, label, variable)
        ttk.Label(self, text=self.directions + "外力在上升時間內平滑增加，之後維持。",
                  wraplength=310).pack(anchor="w", pady=(2, 7))
        ttk.Label(self, text=self.note,
                  wraplength=310).pack(anchor="w", pady=(0, 8))

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
    note = "由接地輪胎分擔側向反力，聯立 Fy 與 Mz 維持車頭方向，並計算側傾與輪荷轉移。"


class DriveEditor(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.fields = {key: tk.StringVar(self) for key in DRIVE_LIMITS}
        for row, (key, (label, _, _)) in enumerate(DRIVE_LIMITS.items()):
            variable = self.fields[key]
            add_entry(self, row, label, variable)
        ttk.Label(self, text="各輪依滑移率誤差調節扭矩。起步油門倍率在 0.25 秒內從 0 平滑升至 1。正轉角向左。",
                  wraplength=312).grid(row=len(self.fields), columnspan=2, sticky="w", pady=8)

    def populate(self, settings):
        for key, variable in self.fields.items():
            variable.set(str(settings[key]))

    def draft(self):
        try:
            settings = {key: float(variable.get()) for key, variable in self.fields.items()}
        except ValueError as error:
            raise ValueError("馬達行駛欄位需要有效數值。") from error
        return validate_drive(settings)


class VehicleTestEditor(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.preset_labels = {label: key for key, (label, _) in TEST_PRESETS.items()}
        self.preset = tk.StringVar(self)
        ttk.Label(self, text="測試工況").grid(row=0, column=0, sticky="w", padx=(0, 8))
        selector = ttk.Combobox(self, textvariable=self.preset, values=tuple(self.preset_labels),
                                state="readonly", width=24)
        selector.grid(row=0, column=1, sticky="w")
        selector.bind("<<ComboboxSelected>>", self.show_fields)
        self.inputs = ttk.Frame(self)
        self.inputs.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.variables = {
            preset: {key: tk.StringVar(self, value=str(value)) for key, value in defaults.items()}
            for preset, (_, defaults) in TEST_PRESETS.items()
        }
        ttk.Label(self, text="沿用專案的馬達扭矩上限、滑移率控制、輪胎及空力設定。",
                  wraplength=315).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def show_fields(self, _event=None):
        for child in self.inputs.winfo_children():
            child.destroy()
        fields = self.variables[self.preset_labels[self.preset.get()]]
        for row, (key, variable) in enumerate(fields.items()):
            if key == "road_class":
                ttk.Label(self.inputs, text="ISO 路面級別").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
                ttk.Combobox(self.inputs, textvariable=variable, values=("A", "B", "C"),
                             state="readonly", width=10).grid(row=row, column=1, sticky="w")
            else:
                add_entry(self.inputs, row, TEST_LIMITS[key][0], variable)

    def populate(self, settings):
        preset = settings["preset"]
        for key, variable in self.variables[preset].items():
            variable.set(str(settings[key]))
        self.preset.set(TEST_PRESETS[preset][0])
        self.show_fields()

    def draft(self):
        preset = self.preset_labels[self.preset.get()]
        settings = {"preset": preset}
        for key, variable in self.variables[preset].items():
            try:
                settings[key] = variable.get() if key == "road_class" else float(variable.get())
            except ValueError as error:
                raise ValueError(f"{TEST_LIMITS[key][0]}需要有效數值。") from error
        return validate_vehicle_test(settings)
