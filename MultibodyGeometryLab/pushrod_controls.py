"""Edit per-corner pushrod mounts and chassis-frame hardpoints."""

import tkinter as tk
from tkinter import ttk

from ._geometry_support.hardpoints import CORNERS
from .pushrod_settings import AXIS_POINTS, COEFFICIENTS, OUTBOARD


INBOARD_LABELS = {"project": "沿用專案", "body": "直連車體", "rocker": "連接搖臂"}
OUTBOARD_LABELS = dict(zip(OUTBOARD, ("下 A 臂", "上 A 臂", "upright")))
POINTS = ("PUSH_OUT", "PUSH_IN", *AXIS_POINTS)


def xyz_inputs(parent, row, label, variables):
    ttk.Label(parent, text=label).grid(row=row, column=0, padx=(0, 5), pady=3, sticky="w")
    entries = []
    for column, variable in enumerate(variables, 1):
        entry = ttk.Entry(parent, textvariable=variable, width=9, justify="right")
        entry.grid(row=row, column=column, padx=2, sticky="ew")
        entries.append(entry)
    return entries


def read_xyz(variables):
    return [float(variable.get()) / 1000. for variable in variables]


class PushrodEditor(ttk.Frame):
    def __init__(self, parent, mark_dirty, sync_members):
        super().__init__(parent, padding=10)
        self.sync_members = sync_members
        self.corner = tk.StringVar(self, value="FL")
        self.fields = {}
        self.saved = {}
        self.legacy_rockers = set()
        for corner in CORNERS:
            variables = {key: tk.StringVar(self) for key in ("inboard", "outboard")}
            variables.update({key: tuple(tk.StringVar(self) for _ in range(3)) for key in POINTS})
            for variable in (variables["inboard"], variables["outboard"], *(axis for key in POINTS for axis in variables[key])):
                variable.trace_add("write", mark_dirty)
            self.fields[corner] = variables

        selection = ttk.Frame(self)
        selection.pack(fill="x")
        ttk.Label(selection, text="輪位").pack(side="left")
        selector = ttk.Combobox(selection, textvariable=self.corner, values=CORNERS, width=8, state="readonly")
        selector.pack(side="left", padx=6)
        selector.bind("<<ComboboxSelected>>", self.show_corner)
        self.form = ttk.Frame(self)
        self.form.pack(fill="x", pady=5)
        self.status = ttk.Label(self, wraplength=320)
        self.status.pack(anchor="w")

    def populate(self, records, fallback, legacy_rockers):
        self.saved = records
        self.legacy_rockers = set(legacy_rockers)
        for name, variables in self.fields.items():
            record = records.get(name) or fallback[name]
            variables["inboard"].set(INBOARD_LABELS[record["inboard"] if name in records else "project"])
            variables["outboard"].set(OUTBOARD_LABELS[record["outboard"]])
            for key in POINTS:
                values = record[key] if key in record else fallback[name].get(key)
                for index, variable in enumerate(variables[key]):
                    variable.set("" if values is None else f"{1000 * values[index]:.15g}")
        self.show_corner()

    def mode(self, corner):
        label = self.fields[corner]["inboard"].get()
        return next((key for key, value in INBOARD_LABELS.items() if value == label), "project")

    def rocker_corners(self):
        return tuple(name for name in CORNERS if self.mode(name) == "rocker" or (
            self.mode(name) == "project" and name in self.legacy_rockers))

    def explicit_corners(self):
        return {name for name in CORNERS if self.mode(name) != "project"}

    def direct_records(self):
        result = {}
        for name in CORNERS:
            if self.mode(name) == "body":
                saved = self.saved.get(name, {})
                result[name] = {"name": f"{name}_PUSH_ROD",
                                **{key: saved.get(key, default) for key, default in COEFFICIENTS.items()}}
        return result

    def show_corner(self, _event=None):
        name = self.corner.get()
        variables = self.fields[name]
        for widget in self.form.winfo_children():
            widget.destroy()
        mode = self.mode(name)
        for row, (key, label, choices) in enumerate((
            ("inboard", "車體側", INBOARD_LABELS), ("outboard", "輪端固定於", OUTBOARD_LABELS)
        )):
            ttk.Label(self.form, text=label).grid(row=row, column=0, sticky="w")
            state = "disabled" if key == "outboard" and mode == "project" else "readonly"
            values = tuple(value for code, value in choices.items() if code != "project" or name not in self.saved)
            combo = ttk.Combobox(self.form, values=values, textvariable=variables[key], state=state, width=17)
            combo.grid(row=row, column=1, columnspan=3, sticky="ew", pady=2)
            if key == "inboard":
                combo.bind("<<ComboboxSelected>>", self.change_mode)
        for col, label in enumerate(("X (mm)", "Y (mm)", "Z (mm)"), 1):
            ttk.Label(self.form, text=label).grid(row=2, column=col)
        shown = POINTS if mode == "rocker" or (mode == "project" and name in self.legacy_rockers) else POINTS[:2]
        for row, key in enumerate(shown, 3):
            entries = xyz_inputs(self.form, row, key, variables[key])
            if mode == "project":
                for entry in entries:
                    entry.configure(state="disabled")
        message = {"project": "選擇安裝方式後可編輯座標。", "body": "直連推桿的力學設定見 deformable_members。",
                   "rocker": "ROCKER_A 和 ROCKER_B 定義搖臂旋轉軸。"}
        self.status.configure(text=message[mode])

    def change_mode(self, _event=None):
        self.show_corner()
        self.sync_members()

    def draft(self):
        result = {}
        direct = self.direct_records()
        for name in self.explicit_corners():
            mode = self.mode(name)
            variables = self.fields[name]
            outboard = next(key for key, label in OUTBOARD_LABELS.items() if label == variables["outboard"].get())
            points = POINTS if mode == "rocker" else POINTS[:2]
            try:
                record = {key: read_xyz(variables[key]) for key in points}
            except ValueError as error:
                raise ValueError(f"{name} · 請輸入推桿及搖臂所需的 XYZ 座標。") from error
            record.update(inboard=mode, outboard=outboard)
            if mode == "body":
                record.update({key: direct[name][key] for key in COEFFICIENTS})
            result[name] = record
        return result
