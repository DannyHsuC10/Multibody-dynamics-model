"""Edit axial force models for deformable members and vertical tire contact."""

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from .dynamics_controls import add_entry
from .elements import DEFORMABLE_KINDS, PUSH_ROD
from .hardpoint_import import validate_rocker_dampers
from .pushrod_controls import read_xyz, xyz_inputs
from .pushrod_settings import COEFFICIENTS


KINDS = {"彈簧／阻尼": "spring_damper", "液壓缸": "hydraulic"}
KIND_LABELS = {value: key for key, value in KINDS.items()}
AXIAL_COEFFICIENTS = (("stiffness_n_m", "spring_n_m", 30000.), ("damping_ns_m", "damper_ns_m", 1300.))


class DeformableEditor(ttk.Frame):
    def __init__(self, parent, mark_dirty, tire_stiffness, rocker_corners):
        super().__init__(parent, padding=10)
        self.mark_dirty = mark_dirty
        self.tire_damping = self.variable(tk.StringVar, "")
        self.choices = tk.StringVar(self)
        self.items = {}
        self.labels = {}
        self.corners = {}
        self.rockers = []
        self.direct = {}
        self.members = {}
        self.explicit_corners = set()
        self.rocker_corners = rocker_corners

        tire = ttk.LabelFrame(self, text="輪胎垂直接地（四輪共用）", padding=8)
        tire.pack(fill="x")
        add_entry(tire, 0, "垂直剛性 (N/m)", tire_stiffness)
        add_entry(tire, 1, "垂直阻尼 (N·s/m)", self.tire_damping)

        group = ttk.LabelFrame(self, text="變形元件", padding=8)
        group.pack(fill="x", pady=(8, 0))
        self.selector = ttk.Combobox(group, textvariable=self.choices, state="readonly", width=28)
        self.selector.pack(fill="x")
        self.selector.bind("<<ComboboxSelected>>", self.show_member)
        tools = ttk.Frame(group)
        tools.pack(fill="x", pady=(5, 0))
        ttk.Button(tools, text="新增座標元件", command=self.add_member).pack(side="left")
        self.edit_button = ttk.Button(tools, text="編輯端點", command=self.edit_member)
        self.edit_button.pack(side="left", padx=4)
        self.remove_button = ttk.Button(tools, text="移除", command=self.remove_member)
        self.remove_button.pack(side="left")
        self.member_fields = ttk.Frame(group)
        self.member_fields.pack(fill="x", pady=(5, 0))

    def variable(self, kind, value):
        variable = kind(self, value=value)
        variable.trace_add("write", self.mark_dirty)
        return variable

    def populate(self, dynamics, rockers, members, direct, explicit_corners):
        self.corners = dynamics["corners"]
        self.rockers = list(rockers)
        self.tire_damping.set(str(dynamics["tire_damping_ns_m"]))
        self.items.clear()
        self.sync_members(members, direct, explicit_corners)

    def sync_members(self, members, direct, explicit_corners):
        self.members, self.direct, self.explicit_corners = members, direct, explicit_corners
        self.labels = {}
        for record in self.rockers:
            key = ("csv", record["name"])
            self.labels[record["name"]] = key
            if key not in self.items:
                self.items[key] = self.controls(record, None)
        for corner, record in direct.items():
            key = ("direct", record["name"])
            self.labels[f"{corner} · 直連推桿"] = key
            if key not in self.items:
                self.items[key] = self.controls(record, None)
        for corner, records in members.items():
            for record in records:
                if record["kind"] in DEFORMABLE_KINDS and not (
                    corner in explicit_corners and record["name"] == PUSH_ROD
                ):
                    key = (corner, record["name"])
                    self.labels[f"{corner} · {record['name']}"] = key
                    if key not in self.items:
                        self.items[key] = self.controls(record, corner)
        labels = tuple(self.labels)
        self.selector.configure(values=labels, state="readonly" if labels else "disabled")
        if self.choices.get() not in self.labels:
            self.choices.set(labels[0] if labels else "")
        self.show_member()

    def controls(self, record, corner):
        model = self.variable(tk.StringVar, KIND_LABELS[record["kind"]])
        fields = {}
        for local, imported, default in AXIAL_COEFFICIENTS:
            value = record[imported] if corner is None else record.get(local)
            if value is None:
                value = self.corners[corner][imported]
            fields[imported] = (self.variable(tk.BooleanVar, bool(value)),
                                self.variable(tk.StringVar, str(value or default)))
        force = self.variable(tk.StringVar, str(record.get("hydraulic_force_n", 0.)))
        return model, fields, force

    def show_member(self, _event=None):
        panel = self.member_fields
        for widget in panel.winfo_children():
            widget.destroy()
        key = self.labels.get(self.choices.get())
        enabled = "normal" if key and key[0] == "csv" else "disabled"
        self.edit_button.configure(state=enabled)
        self.remove_button.configure(state=enabled)
        if key is None:
            ttk.Label(panel, text="可在「機構元件」將選定桿件加入此處。", wraplength=280).pack(anchor="w")
            return
        model, fields, force = self.items[key]
        selector = ttk.Combobox(panel, textvariable=model, values=tuple(KINDS), state="readonly", width=14)
        selector.grid(row=0, column=0, sticky="w", pady=3)
        selector.bind("<<ComboboxSelected>>", self.show_member)
        if KINDS[model.get()] == "hydraulic":
            add_entry(panel, 1, "固定軸力 (N)", force)
            ttk.Label(panel, text="正值推開兩端，負值拉近兩端。").grid(row=2, column=0, columnspan=2, sticky="w")
            return
        for row, (field, label) in enumerate((("spring_n_m", "彈簧剛性 (N/m)"), ("damper_ns_m", "阻尼 (N·s/m)")), 1):
            enabled, value = fields[field]
            entry = ttk.Entry(panel, textvariable=value, width=12, justify="right",
                              state="normal" if enabled.get() else "disabled")
            ttk.Checkbutton(panel, text=label, variable=enabled,
                            command=lambda entry=entry, enabled=enabled:
                            entry.configure(state="normal" if enabled.get() else "disabled")).grid(row=row, column=0, sticky="w")
            entry.grid(row=row, column=1, sticky="w")

    def update_record(self, record, corner=None, *, direct=False):
        key = ("direct" if direct else (corner or "csv"), record["name"])
        model, fields, force = self.items[key]
        snapshot = {**record, "kind": KINDS[model.get()]}
        try:
            if snapshot["kind"] == "hydraulic":
                snapshot["hydraulic_force_n"] = float(force.get())
            else:
                for local, imported, _ in AXIAL_COEFFICIENTS:
                    enabled, variable = fields[imported]
                    value = float(variable.get()) if enabled.get() else 0.
                    if corner is None:
                        snapshot[imported] = value
                    elif record.get(local) is not None or value != self.corners[corner][imported]:
                        snapshot[local] = value
        except ValueError as error:
            raise ValueError(f"{record['name']} · 變形元件需要有效的剛性、阻尼或軸力。") from error
        return snapshot


    def update_direct(self, corner, record):
        current = self.update_record({**record, "name": f"{corner}_PUSH_ROD"}, direct=True)
        current.pop("name")
        return current

    def draft_members(self):
        return validate_rocker_dampers([self.update_record(record) for record in self.rockers])

    def selected_record(self):
        key = self.labels.get(self.choices.get())
        if key and key[0] == "csv":
            return next((item for item in self.rockers if item["name"] == key[1]), None)
        return None

    def add_member(self):
        mounts = self.rocker_corners()
        if not mounts:
            messagebox.showerror("新增座標元件", "請先在「推桿」定義至少一側搖臂。", parent=self)
            return
        record = {"name": "", "a": {"xyz": [0., 0., 0.], "mount": "body"},
                  "b": {"xyz": [0., 0., 0.], "mount": mounts[0]}, **COEFFICIENTS}
        self.edit_geometry(record, existing=False)

    def edit_member(self):
        record = self.selected_record()
        if record is not None:
            self.edit_geometry(record, existing=True)

    def edit_geometry(self, record, *, existing):
        names = {item["name"] for item in self.rockers if item is not record}
        names.update(item["name"] for item in self.direct.values())
        dialog = EndpointDialog(self, record, self.rocker_corners(), names)
        if dialog.result is None:
            return
        new = dialog.result
        if existing:
            self.rockers[self.rockers.index(record)] = new
            self.items["csv", new["name"]] = self.items.pop(("csv", record["name"]))
        else:
            self.rockers.append(new)
        self.choices.set(new["name"])
        self.mark_dirty()
        self.sync_members(self.members, self.direct, self.explicit_corners)

    def remove_member(self):
        record = self.selected_record()
        if record is not None:
            self.rockers.remove(record)
            self.items.pop(("csv", record["name"]))
            self.mark_dirty()
            self.sync_members(self.members, self.direct, self.explicit_corners)


class EndpointDialog(simpledialog.Dialog):
    def __init__(self, parent, record, mounts, names):
        self.record, self.mounts, self.names = record, mounts, names
        super().__init__(parent, title="變形元件端點")

    def body(self, master):
        self.name = tk.StringVar(master, value=self.record["name"])
        name_entry = add_entry(master, 0, "元件名稱", self.name)
        choices = {"body": "車體", **{name: f"{name} 搖臂" for name in self.mounts}}
        self.endpoints = {}
        for end in ("a", "b"):
            endpoint = self.record[end]
            frame = ttk.LabelFrame(master, text=f"{end.upper()} 端 · 靜態車身座標", padding=6)
            frame.grid(sticky="ew", columnspan=4, pady=3)
            mount = tk.StringVar(frame, value=choices.get(endpoint["mount"], ""))
            combo = ttk.Combobox(frame, state="readonly", textvariable=mount, values=tuple(choices.values()))
            combo.grid(row=0, column=1, columnspan=3, sticky="ew")
            ttk.Label(frame, text="安裝於").grid(row=0, column=0, sticky="w")
            variables = tuple(tk.StringVar(frame, value=f"{xyz * 1000:.15g}") for xyz in endpoint["xyz"])
            xyz_inputs(frame, 1, "X / Y / Z (mm)", variables)
            self.endpoints[end] = (mount, variables)
        self.mount_labels = {label: code for code, label in choices.items()}
        self.error = ttk.Label(master, wraplength=340, foreground="#9e3829")
        self.error.grid(sticky="w", pady=4, columnspan=4)
        return name_entry

    def validate(self):
        name = self.name.get().strip()
        try:
            if name in self.names:
                raise ValueError("元件名稱已使用。")
            ends = {end: {"mount": self.mount_labels[mount.get()], "xyz": read_xyz(fields)}
                    for end, (mount, fields) in self.endpoints.items()}
            self.result = validate_rocker_dampers([{**self.record, "name": name, **ends}])[0]
        except (KeyError, ValueError) as error:
            self.error.configure(text="請選擇端點安裝位置。" if isinstance(error, KeyError) else str(error))
            return False
        return True
