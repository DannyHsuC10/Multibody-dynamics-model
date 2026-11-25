"""Edit shared hydraulic lines and the two ports of a cylinder."""

import tkinter as tk
from tkinter import simpledialog, ttk

from .dynamics_controls import add_entry
from .hydraulic_circuits import validate_cylinder, validate_lines


class LineDialog(simpledialog.Dialog):
    def __init__(self, parent, lines):
        self.lines = lines
        self.rows = {}
        super().__init__(parent, title="液壓管路")

    def body(self, master):
        self.table = ttk.Frame(master)
        self.table.pack()
        for col, label in enumerate(("管路", "初始壓力 (MPa)", "順應性 (cm³/MPa)")):
            ttk.Label(self.table, text=label).grid(row=0, column=col, padx=4)
        for name, line in self.lines.items():
            self.add_row(name, line["pressure_mpa"], line["compliance_cm3_mpa"])
        ttk.Button(master, text="新增管路", command=self.add_row).pack(anchor="w", pady=4)
        ttk.Label(master, text="讓多支缸的油口選同一管路即可分岔。").pack(anchor="w")
        self.error = ttk.Label(master, foreground="#9e3829", wraplength=350)
        self.error.pack(anchor="w")

    def add_row(self, name="", pressure=0., compliance=5.):
        row = max(self.rows, default=0) + 1
        values = tuple(tk.StringVar(self, value=value) for value in (name, pressure, compliance))
        widgets = [ttk.Entry(self.table, textvariable=value, width=14) for value in values]
        widgets.append(ttk.Button(self.table, text="移除", command=lambda: self.remove_row(row)))
        self.rows[row] = (values, widgets)
        for col, widget in enumerate(widgets):
            widget.grid(row=row, column=col, padx=4, pady=2)

    def remove_row(self, row):
        _, widgets = self.rows.pop(row)
        for widget in widgets:
            widget.destroy()

    def validate(self):
        try:
            lines = {}
            for (name, pressure, compliance), _ in self.rows.values():
                key = name.get().strip()
                if key in lines:
                    raise ValueError(f"液壓管路名稱重複：{key}")
                lines[key] = {"pressure_mpa": float(pressure.get()), "compliance_cm3_mpa": float(compliance.get())}
            self.result = validate_lines(lines)
        except ValueError as error:
            self.error.configure(text=str(error))
            return False
        return True


class CylinderDialog(simpledialog.Dialog):
    def __init__(self, parent, settings, lines):
        self.settings = settings or {"bore_mm": 25., "rod_mm": 12., "port_a": "", "port_b": ""}
        self.lines = tuple(lines)
        super().__init__(parent, title="雙作用缸油口")

    def body(self, master):
        self.fields = {key: tk.StringVar(master, value=value) for key, value in self.settings.items()}
        first = add_entry(master, 0, "缸徑 (mm)", self.fields["bore_mm"])
        add_entry(master, 1, "桿徑 (mm)", self.fields["rod_mm"])
        for row, (key, label) in enumerate((("port_a", "A 油口（無桿腔）"), ("port_b", "B 油口（有桿腔）")), 2):
            ttk.Label(master, text=label).grid(row=row, column=0, sticky="w")
            ttk.Combobox(master, textvariable=self.fields[key], values=("", *self.lines), state="readonly",
                         width=18).grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Label(master, text="空白油口視為零表壓。").grid(columnspan=2, sticky="w")
        self.error = ttk.Label(master, foreground="#9e3829", wraplength=350)
        self.error.grid(columnspan=2, sticky="w")
        return first

    def validate(self):
        try:
            settings = {key: variable.get() for key, variable in self.fields.items()}
            for key in ("bore_mm", "rod_mm"):
                settings[key] = float(settings[key])
            result = validate_cylinder(settings)
            if any(result[key] and result[key] not in self.lines for key in ("port_a", "port_b")):
                raise ValueError("請選擇已定義的管路。")
            self.result = result
        except ValueError as error:
            self.error.configure(text=str(error))
            return False
        return True
