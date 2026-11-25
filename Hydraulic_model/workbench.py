"""Native editor for constructing and simulating hydraulic circuits."""

from dataclasses import replace
from itertools import count
import tkinter as tk
from tkinter import messagebox, ttk

if __package__:
    from .design import (CylinderDesign, DamperDesign, Hose, Port, SpringEnd, SpringTarget,
                         assemble_design, import_project_circuit, pressure_groups,
                         project_cylinder_records, project_endpoint_labels, solve_design)
    from .transient_controls import TransientPane
    from .transient_network import simulate_transient
    from .vehicle_coupling import DEFAULT_COMPLIANCE, read_vehicle_design, vehicle_design
else:
    from design import (CylinderDesign, DamperDesign, Hose, Port, SpringEnd, SpringTarget,
                        assemble_design, import_project_circuit, pressure_groups,
                        project_cylinder_records, project_endpoint_labels, solve_design)
    from transient_controls import TransientPane
    from transient_network import simulate_transient


BODY_WIDTH = 174
BODY_HEIGHT = 68
SPRING_WIDTH = 142
SPRING_HEIGHT = 70
CHASSIS_XY = (686, 426)
CHASSIS_TERMINAL = ("chassis", 0, "")
CANVAS_BG = "#f6f8fb"
INK = "#17324a"
MUTED = "#52667b"
CAP_COLOR = "#246d9a"
ROD_COLOR = "#a65436"
MECHANICAL = "#604489"
PIPE_COLORS = ("#267c9e", "#a5543b", "#7556a7", "#2b8269", "#926b2b")
FONT = "Microsoft JhengHei UI"
CYLINDER_FIELDS = (
    ("name", "名稱"),
    ("bore_mm", "缸徑 (mm)"),
    ("rod_mm", "活塞桿徑 (mm)"),
    ("compression_force_n", "外力 (N)"),
    ("prescribed_compression_mm", "指定位移 (mm)"),
)
DAMPER_FIELDS = (
    ("name", "名稱"),
    ("stiffness_n_mm", "彈簧剛性 (N/mm)"),
    ("damping_n_s_mm", "阻尼係數 (N·s/mm)"),
)
FIELDS = dict((*CYLINDER_FIELDS, *DAMPER_FIELDS))


def short_name(name, limit=14):
    return name if len(name) <= limit else name[:limit - 1] + "…"


class HydraulicWorkbench(ttk.Frame):
    def __init__(self, parent, *, embedded=False, apply_vehicle_design=None, reload_vehicle=None):
        super().__init__(parent, padding=(12, 8) if embedded else (18, 14))
        if not embedded:
            parent.title("液壓設計平台")
            parent.geometry("1180x740")
            parent.minsize(920, 560)
            parent.option_add("*Font", (FONT, 10))
        self.pack(fill="both", expand=True)
        self.cylinders: dict[int, CylinderDesign] = {}
        self.positions: dict[int, tuple[float, float]] = {}
        self.hoses: set[Hose] = set()
        self.line_anchors: dict[Port, str] = {}
        self.project_lines = {}
        self.springs: dict[int, DamperDesign] = {}
        self.spring_positions: dict[int, tuple[float, float]] = {}
        self.spring_links: dict[SpringEnd, SpringTarget] = {}
        self.next_id = 1
        self.next_spring_id = 1
        self.selection = None
        self.pending_port: Port | None = None
        self.pending_mechanical = None
        self.drag = None
        self.pressures = {}
        self.compression = {}
        self.required_forces = {}
        self.apply_vehicle_design = apply_vehicle_design
        self.reload_vehicle = reload_vehicle
        self.default_compliance = DEFAULT_COMPLIANCE if apply_vehicle_design is not None else None
        self.status = tk.StringVar(self, "新增液壓缸開始設計，或載入三缸範例。")
        self.editor_title = tk.StringVar(self)
        self.fields = {key: tk.StringVar(self) for key in FIELDS}
        self.build_layout()
        if not embedded:
            parent.bind("<Escape>", self.cancel_connection)
        self.populate_editor()
        self.draw_network()
        self.project_circuit_snapshot = self.circuit_snapshot()

    def build_layout(self):
        ttk.Label(self, text="液壓設計平台", font=(FONT, 20, "bold")).pack(anchor="w")
        ttk.Label(self, text="液壓油路 · 靜力與瞬態分析", foreground=MUTED).pack(anchor="w")
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(10, 10))
        for text, command in (
            ("新增液壓缸", self.add_cylinder), ("新增避震器", self.add_spring),
            ("刪除選取", self.delete_selection),
            ("載入三缸範例", self.load_example),
        ):
            ttk.Button(toolbar, text=text, command=command).pack(side="left", padx=(0, 7))
        ttk.Button(toolbar, text="求解靜力", command=self.solve).pack(side="right")
        if self.apply_vehicle_design is not None:
            ttk.Button(toolbar, text="套用至多體", command=self.apply_coupling).pack(side="left", padx=(0, 7))
            self.disconnect_button = ttk.Button(toolbar, text="解除連動", command=self.disconnect_coupling)
            self.disconnect_button.pack(side="left", padx=(0, 7))
            ttk.Button(toolbar, text="重新匯入車輛油路", command=self.reload_vehicle).pack(side="left")

        content = ttk.Frame(self)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)
        ttk.Label(content, text="圓形油口接油管 · 方形接點接避震器 · 拖曳元件移動 · Esc 取消連接",
                  foreground=MUTED).grid(row=0, column=0, sticky="w", pady=(0, 5))
        workspace = ttk.Frame(content)
        workspace.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(workspace, background=CANVAS_BG, highlightthickness=1,
                                highlightbackground="#c7d3df", width=500, height=450)
        vertical = ttk.Scrollbar(workspace, orient="vertical", command=self.canvas.yview)
        horizontal = ttk.Scrollbar(workspace, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<Configure>", lambda _: self.draw_network())
        self.canvas.bind("<Button-1>", self.canvas_press)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)

        self.tabs = ttk.Notebook(content, width=345)
        self.tabs.grid(row=0, column=1, rowspan=2, sticky="nsew")
        self.editor = ttk.Frame(self.tabs, padding=12)
        self.results = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.editor, text="元件設定")
        self.tabs.add(self.results, text="求解結果")
        self.transient = TransientPane(self.tabs, self.run_transient)
        self.tabs.add(self.transient, text="瞬態模擬")
        if self.apply_vehicle_design is not None:
            response = ttk.Frame(self.tabs, padding=12)
            self.vehicle_response_label = ttk.Label(response, text="尚未套用至多體。")
            self.vehicle_response_label.pack(anchor="w")
            self.vehicle_response = ttk.Treeview(response, columns=("cylinder", "stroke", "force"), show="headings")
            for key, label, width in (("cylinder", "車載液壓缸", 142), ("stroke", "伸長 (mm)", 84),
                                      ("force", "伸張力 (N)", 86)):
                self.vehicle_response.heading(key, text=label)
                self.vehicle_response.column(key, width=width, anchor="center")
            self.vehicle_response.pack(fill="both", expand=True, pady=(6, 0))
            self.tabs.add(response, text="多體回饋")
            self.set_vehicle_coupling(False)
        self.project_line_page = ttk.Frame(self.tabs, padding=12)
        ttk.Label(self.project_line_page, text="專案初始油路設定").pack(anchor="w")
        self.line_table = ttk.Treeview(self.project_line_page, columns=("name", "pressure", "compliance", "ports"),
                                       show="headings")
        for column, label, width in (("name", "油路", 68), ("pressure", "表壓 (MPa)", 80),
                                      ("compliance", "cm³/MPa", 90), ("ports", "接管數", 55)):
            self.line_table.heading(column, text=label)
            self.line_table.column(column, width=width, anchor="center")
        self.line_table.pack(fill="both", expand=True, pady=(6, 0))
        self.build_editor()
        self.build_results()
        ttk.Label(self, textvariable=self.status, foreground=INK, wraplength=880).pack(
            fill="x", pady=(9, 0), anchor="w")

    def build_editor(self):
        self.editor.columnconfigure(1, weight=1)
        ttk.Label(self.editor, textvariable=self.editor_title, font=(FONT, 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        self.entries = {}
        self.labels = {}
        for row, (key, label) in enumerate(FIELDS.items(), 1):
            field_label = ttk.Label(self.editor, text=label)
            field_label.grid(row=row, column=0, sticky="w", pady=5)
            entry = ttk.Entry(self.editor, textvariable=self.fields[key], width=16)
            entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=5)
            self.entries[key] = entry
            self.labels[key] = field_label
        self.apply_button = ttk.Button(self.editor, text="套用設定", command=self.apply_editor)
        self.fields["prescribed_compression_mm"].trace_add("write", self.update_force_field)
        self.project_attachment = ttk.LabelFrame(self.editor, text="活塞桿連接 · 車輛專案", padding=(7, 4))
        self.project_attachment.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.project_attachment.columnconfigure(1, weight=1)
        self.endpoint_values = tuple(tk.StringVar(self) for _ in range(2))
        for row, (end, value) in enumerate(zip(("A 端", "B 端"), self.endpoint_values)):
            ttk.Label(self.project_attachment, text=end).grid(row=row, column=0, sticky="nw", padx=(0, 8))
            ttk.Label(self.project_attachment, textvariable=value, wraplength=220).grid(
                row=row, column=1, sticky="w")
        self.project_attachment.grid_remove()
        self.apply_button.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(12, 10))
        self.input_help = ttk.Label(
            self.editor, foreground=MUTED, wraplength=285,
        )
        self.input_help.grid(row=10, column=0, columnspan=2, sticky="w", pady=(0, 12))
        ttk.Label(self.editor, text="●  左側 · 無桿腔", foreground=CAP_COLOR).grid(
            row=11, column=0, columnspan=2, sticky="w")
        ttk.Label(self.editor, text="●  右側 · 有桿腔", foreground=ROD_COLOR).grid(
            row=12, column=0, columnspan=2, sticky="w", pady=(3, 10))
        ttk.Label(self.editor, text="■  活塞桿、避震器端點與車體固定點", foreground=MECHANICAL).grid(
            row=13, column=0, columnspan=2, sticky="w")
        ttk.Label(self.editor, text="避震器可接活塞桿或車體固定點。固定點相對車體的位移與速度為零。未接管的油口採零表壓。",
                  foreground=MUTED, wraplength=285).grid(row=14, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def build_results(self):
        self.results.columnconfigure(0, weight=1)
        self.results.rowconfigure(3, weight=1)
        ttk.Label(self.results, text="油路表壓與所需外力", font=(FONT, 11, "bold")).grid(row=0, column=0, sticky="w")
        self.result_table = ttk.Treeview(self.results, columns=("item", "value", "unit"), show="headings", height=5)
        for column, label, width in (("item", "項目", 135), ("value", "數值", 100), ("unit", "單位", 55)):
            self.result_table.heading(column, text=label)
            self.result_table.column(column, width=width, anchor="center")
        scrollbar = ttk.Scrollbar(self.results, orient="vertical", command=self.result_table.yview)
        self.result_table.configure(yscrollcommand=scrollbar.set)
        self.result_table.grid(row=1, column=0, sticky="ew", pady=(5, 12))
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(5, 12))
        ttk.Label(self.results, text="活塞桿位移 (mm)", font=(FONT, 11, "bold")).grid(
            row=2, column=0, sticky="w")
        self.chart = tk.Canvas(self.results, background=CANVAS_BG, highlightthickness=0, height=168, width=295)
        chart_scroll = ttk.Scrollbar(self.results, orient="vertical", command=self.chart.yview)
        self.chart.configure(yscrollcommand=chart_scroll.set)
        self.chart.grid(row=3, column=0, sticky="nsew", pady=(5, 5))
        chart_scroll.grid(row=3, column=1, sticky="ns", pady=(5, 5))
        self.chart.bind("<Configure>", lambda _: self.draw_chart())
        ttk.Label(self.results, text="位移與外力皆以壓縮為正", foreground=MUTED).grid(
            row=4, column=0, sticky="w")

    def selected_component(self):
        if self.selection is not None:
            kind, key = self.selection
            if kind == "cylinder":
                return self.cylinders, key, CYLINDER_FIELDS
            if kind == "spring":
                return self.springs, key, DAMPER_FIELDS
        return {}, None, ()

    def populate_editor(self):
        components, key, fields = self.selected_component()
        item = components.get(key)
        self.editor_title.set(item.name if item is not None else "請選擇液壓缸或避震器")
        visible = dict(fields)
        for field, variable in self.fields.items():
            if field in visible:
                value = getattr(item, field)
                variable.set("" if value is None else str(value))
                self.entries[field].grid()
                self.labels[field].grid()
            else:
                self.entries[field].grid_remove()
                self.labels[field].grid_remove()
        self.apply_button.configure(state="normal" if item is not None else "disabled")
        if isinstance(item, CylinderDesign):
            help_text = "指定位移留白時使用外力輸入。填入位移後，求解所需外力。正值皆為壓縮。"
            if item.project_source is not None and self.apply_vehicle_design is not None:
                help_text = "多體連動時，車載缸的位移由多體模型指定。欄位中的外力與位移供液壓頁面獨立求解使用。"
            self.input_help.configure(text=help_text)
            self.input_help.grid()
        else:
            self.input_help.grid_remove()
        self.show_project_attachments(item)
        self.update_force_field()

    def show_project_attachments(self, item):
        if isinstance(item, CylinderDesign) and item.project_endpoints is not None:
            for value, label in zip(self.endpoint_values, item.project_endpoints):
                value.set(label)
            self.project_attachment.grid()
        else:
            self.project_attachment.grid_remove()

    def refresh_vehicle_endpoints(self, data):
        if not any(cylinder.project_source for cylinder in self.cylinders.values()):
            return
        records = {source: record for source, _, record in project_cylinder_records(data)
                   if record["kind"] == "hydraulic" and record.get("hydraulic") is not None}
        for key, cylinder in self.cylinders.items():
            source = cylinder.project_source
            if source is None:
                continue
            endpoints = (project_endpoint_labels(source, records[source])
                         if source in records else ("目前專案找不到來源元件", "—"))
            self.cylinders[key] = replace(cylinder, project_endpoints=endpoints)
        items, key, _ = self.selected_component()
        self.show_project_attachments(items.get(key))

    def update_force_field(self, *_):
        prescribed = self.fields["prescribed_compression_mm"].get().strip()
        self.entries["compression_force_n"].configure(state="disabled" if prescribed else "normal")

    def commit_editor(self):
        components, key, _ = self.selected_component()
        if key is None:
            return True
        try:
            name = self.fields["name"].get().strip()
            try:
                if isinstance(components[key], CylinderDesign):
                    prescribed = self.fields["prescribed_compression_mm"].get().strip()
                    values = {
                        "bore_mm": float(self.fields["bore_mm"].get()),
                        "rod_mm": float(self.fields["rod_mm"].get()),
                        "prescribed_compression_mm": float(prescribed) if prescribed else None,
                        "compression_force_n": (components[key].compression_force_n if prescribed
                                                else float(self.fields["compression_force_n"].get())),
                    }
                else:
                    values = {field: float(self.fields[field].get()) for field, _ in DAMPER_FIELDS[1:]}
            except ValueError as error:
                raise ValueError("請在參數欄位輸入數值。") from error
            updated = replace(components[key], name=name, **values)
            if any(item.name == name for other, item in components.items() if other != key):
                raise ValueError("同類型元件的名稱不可重複。")
        except ValueError as error:
            self.status.set(str(error))
            self.tabs.select(self.editor)
            return False
        if updated != components[key]:
            components[key] = updated
            self.editor_title.set(name)
            self.invalidate_results()
            self.draw_network()
        return True

    def apply_editor(self):
        components, key, _ = self.selected_component()
        if key is not None and self.commit_editor():
            self.status.set(f"已套用 {components[key].name} 的設定。")

    def invalidate_results(self):
        self.clear_static_results()
        self.transient.clear()
        self.refresh_project_lines()

    def refresh_project_lines(self):
        if not self.project_lines:
            if str(self.project_line_page) in self.tabs.tabs():
                self.tabs.hide(self.project_line_page)
            return
        self.tabs.add(self.project_line_page, text="專案油路")
        self.line_table.delete(*self.line_table.get_children())
        groups = tuple(self.network_groups().values())
        for name, line in self.project_lines.items():
            self.line_table.insert("", "end", values=(name, f"{line.pressure_mpa:g}",
                                                      f"{line.compliance_cm3_mpa:g}", groups.count(name)))

    def clear_static_results(self):
        self.pressures = {}
        self.compression = {}
        self.required_forces = {}
        self.refresh_results()

    def run_transient(self, settings):
        if not self.commit_editor():
            return None
        system, cylinders, forces, prescribed = assemble_design(
            self.cylinders, self.hoses, self.springs, self.spring_links, require_connected=False,
            anchors=self.line_anchors, project_lines=self.project_lines,
            default_compliance_cm3_mpa=self.default_compliance)
        trace = simulate_transient(system, forces, prescribed, settings)
        names = {cylinder.rod: self.cylinders[key].name for key, cylinder in cylinders.items()}
        self.status.set("已完成瞬態模擬。可切換查看活塞桿位移、油壓變化或位移控制所需外力。")
        return trace, names

    def available_position(self):
        occupied = (*self.positions.values(), *self.spring_positions.values())
        for index in count():
            x = 80 + 272 * (index % 3)
            y = 62 + 114 * (index // 3)
            chassis_x, chassis_y = CHASSIS_XY
            if x - 20 <= chassis_x <= x + BODY_WIDTH + 20 and y - 45 <= chassis_y <= y + BODY_HEIGHT + 20:
                continue
            if all(abs(x - a) >= BODY_WIDTH + 45 or abs(y - b) >= BODY_HEIGHT + 28
                   for a, b in occupied):
                return x, y

    def add_cylinder(self):
        if not self.commit_editor():
            return
        names = {cylinder.name for cylinder in self.cylinders.values()}
        key = self.next_id
        name = f"缸 {key}"
        while name in names:
            key += 1
            name = f"缸 {key}"
        self.next_id = key + 1
        self.positions[key] = self.available_position()
        self.cylinders[key] = CylinderDesign(name)
        self.show_new_component("cylinder", key, self.positions[key])
        self.status.set(f"已新增 {name}。使用左右油口接油管，或用下方的方形接點接避震器。")

    def add_spring(self):
        if not self.commit_editor():
            return
        names = {spring.name for spring in self.springs.values()}
        key = self.next_spring_id
        name = f"避震器 {key}"
        while name in names:
            key += 1
            name = f"避震器 {key}"
        self.next_spring_id = key + 1
        self.spring_positions[key] = self.available_position()
        self.springs[key] = DamperDesign(name)
        self.show_new_component("spring", key, self.spring_positions[key])
        self.status.set(f"已新增 {name}。依序點選避震器端點和活塞桿或車體固定點，完成 A、B 兩端連接。")

    def show_new_component(self, kind, key, position):
        self.selection = (kind, key)
        self.pending_port = self.pending_mechanical = None
        self.populate_editor()
        self.tabs.select(self.editor)
        self.invalidate_results()
        self.draw_network()
        self.reveal_component(position)

    def reveal_component(self, position):
        x, y = position
        _, _, right, bottom = map(float, self.canvas.cget("scrollregion").split())
        if x < self.canvas.canvasx(0) or x + BODY_WIDTH > self.canvas.canvasx(self.canvas.winfo_width()):
            self.canvas.xview_moveto(max(0., x - 50) / right)
        if y < self.canvas.canvasy(0) or y + BODY_HEIGHT > self.canvas.canvasy(self.canvas.winfo_height()):
            self.canvas.yview_moveto(max(0., y - 40) / bottom)

    def load_example(self):
        if (self.cylinders or self.springs) and not messagebox.askyesno(
            "載入三缸範例", "要用三缸範例取代目前畫布上的設計嗎？", parent=self.winfo_toplevel()
        ):
            return
        self.cylinders = {1: CylinderDesign("A", rod_mm=8., compression_force_n=100.),
                          2: CylinderDesign("B", rod_mm=16., compression_force_n=180.),
                          3: CylinderDesign("C", rod_mm=12.)}
        self.positions = {key: (210, 70 + (key - 1) * 124) for key in self.cylinders}
        self.hoses = {((key, side), (key + 1, side)) for key in (1, 2) for side in ("cap", "rod")}
        self.line_anchors = {}
        self.project_lines = {}
        self.springs = {1: DamperDesign("避震器 K")}
        self.spring_positions = {1: (473, 391)}
        self.spring_links = {(1, "a"): 3, (1, "b"): None}
        self.next_id = 4
        self.next_spring_id = 2
        self.selection = ("cylinder", 1)
        self.pending_port = self.pending_mechanical = None
        self.drag = None
        self.populate_editor()
        self.tabs.select(self.editor)
        self.invalidate_results()
        self.draw_network()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.status.set("已載入三缸範例。避震器 K 的 A 端接 C 缸活塞桿，B 端接車體固定點。")

    def circuit_snapshot(self):
        return (self.cylinders.copy(), self.positions.copy(), self.hoses.copy(), self.line_anchors.copy(),
                self.springs.copy(), self.spring_positions.copy(), self.spring_links.copy())

    def has_circuit_edits(self):
        if self.circuit_snapshot() != self.project_circuit_snapshot:
            return True
        items, selected, fields = self.selected_component()
        if selected is None:
            return False
        for key, _ in fields:
            value = getattr(items[selected], key)
            if self.fields[key].get() != ("" if value is None else str(value)):
                return True
        return False

    def set_vehicle_coupling(self, enabled):
        if self.apply_vehicle_design is not None:
            self.disconnect_button.configure(state="normal" if enabled else "disabled")

    def apply_coupling(self):
        if not self.commit_editor():
            return
        design = vehicle_design(self.cylinders, self.hoses, self.line_anchors, self.project_lines,
                                self.springs, self.spring_links)
        if self.apply_vehicle_design(design):
            self.project_circuit_snapshot = self.circuit_snapshot()
            self.set_vehicle_coupling(True)
            self.status.set(f"已套用至多體。新油路的預設順應性為 {DEFAULT_COMPLIANCE:g} cm³/MPa。")

    def disconnect_coupling(self):
        if self.apply_vehicle_design(None):
            self.set_vehicle_coupling(False)
            self.status.set("已解除連動。多體將使用專案原有的油路設定。")

    def show_vehicle_response(self, readings, time):
        if self.apply_vehicle_design is None:
            return
        self.vehicle_response.delete(*self.vehicle_response.get_children())
        if not readings:
            self.vehicle_response_label.configure(text="尚未套用至多體。")
            return
        self.vehicle_response_label.configure(text="靜態參考姿態" if time is None else f"多體時間 {time:.3f} s")
        for name, reading in readings:
            self.vehicle_response.insert("", "end", values=(name, f"{reading.deflection * 1000:+.3f}",
                                                            f"{reading.force:+.3f}"))

    def load_vehicle_project(self, data):
        cylinders, self.hoses, self.line_anchors, self.project_lines, legacy_count = import_project_circuit(data)
        design = data.get("hydraulic_design")
        if design is not None:
            cylinders, self.hoses, self.line_anchors, self.project_lines, springs, links = read_vehicle_design(design)
        else:
            springs, links = {}, {}
        self.cylinders = cylinders
        self.positions = {key: (220, 62 + index * 114) for index, key in enumerate(cylinders)}
        self.springs, self.spring_links = springs, links
        self.spring_positions = {key: (520, 62 + index * 114) for index, key in enumerate(springs)}
        self.next_id, self.next_spring_id = max(cylinders, default=0) + 1, max(springs, default=0) + 1
        self.selection = ("cylinder", next(iter(cylinders))) if cylinders else None
        self.pending_port = self.pending_mechanical = self.drag = None
        self.project_circuit_snapshot = self.circuit_snapshot()
        self.populate_editor()
        self.refresh_vehicle_endpoints(data)
        self.set_vehicle_coupling(design is not None)
        self.tabs.select(self.editor)
        self.invalidate_results()
        self.draw_network()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        if design is not None:
            message = "已載入儲存的液壓連動設定。修改油路後請再次套用至多體。"
        elif cylinders or self.project_lines or legacy_count:
            message = f"已從車輛專案載入 {len(cylinders)} 支液壓缸與 {len(self.project_lines)} 條管路。"
            if legacy_count:
                message += f"另有 {legacy_count} 支固定軸力元件缺少缸徑與油口設定。"
        else:
            message = "此專案尚無液壓缸或油路設定。"
        self.status.set(message)

    def network_groups(self):
        return pressure_groups(self.hoses, self.line_anchors, reserved_names=self.project_lines)

    def delete_selection(self):
        if self.selection is None:
            self.status.set("請先在畫布上選取要刪除的元件或連線。")
            return
        kind, key = self.selection
        if kind == "cylinder":
            groups = self.network_groups()
            for port, name in tuple(self.line_anchors.items()):
                if port[0] == key:
                    self.line_anchors.pop(port)
                    remaining = [other for other, group in groups.items() if other[0] != key and group == name]
                    if remaining:
                        self.line_anchors[remaining[0]] = name
            name = self.cylinders.pop(key).name
            self.positions.pop(key)
            self.hoses = {hose for hose in self.hoses if all(port[0] != key for port in hose)}
            self.spring_links = {end: target for end, target in self.spring_links.items() if target != key}
            self.status.set(f"已刪除 {name} 和相關連線。")
        elif kind == "spring":
            name = self.springs.pop(key).name
            self.spring_positions.pop(key)
            self.spring_links = {end: target for end, target in self.spring_links.items() if end[0] != key}
            self.status.set(f"已刪除 {name} 和相關連線。")
        elif kind == "hose":
            self.hoses.remove(key)
            self.status.set("已刪除選取的油管。")
        elif kind == "line_port":
            self.line_anchors.pop(key)
            self.status.set("已解除此油口的管路。")
        else:
            self.spring_links.pop(key)
            self.status.set("已刪除選取的避震器連線。")
        self.selection = self.pending_port = self.pending_mechanical = None
        self.populate_editor()
        self.invalidate_results()
        self.draw_network()

    def port_name(self, port):
        key, side = port
        return f"{self.cylinders[key].name} · {'左側' if side == 'cap' else '右側'}"

    def connect_port(self, port):
        self.pending_mechanical = None
        if self.pending_port is None:
            self.pending_port = port
            self.status.set(f"已選 {self.port_name(port)}，請點選另一個油口。")
        elif self.pending_port == port:
            self.cancel_connection()
            return
        else:
            first, second = self.pending_port, port
            hose = tuple(sorted((first, second)))
            self.pending_port = None
            if hose in self.hoses:
                self.status.set("這兩個油口已直接相連。")
            else:
                try:
                    pressure_groups(self.hoses | {hose}, self.line_anchors, reserved_names=self.project_lines)
                except ValueError as error:
                    self.status.set(str(error))
                    self.draw_network()
                    return
                self.hoses.add(hose)
                self.status.set(f"已接管：{self.port_name(first)} ↔ {self.port_name(second)}。")
                self.invalidate_results()
        self.draw_network()

    def mechanical_name(self, terminal):
        kind, key, end = terminal
        if terminal == CHASSIS_TERMINAL:
            return "車體固定點"
        if kind == "rod":
            return f"{self.cylinders[key].name} · 活塞桿"
        return f"{self.springs[key].name} · {end.upper()} 端"

    def connect_mechanical(self, terminal):
        self.pending_port = None
        if self.pending_mechanical is None:
            self.pending_mechanical = terminal
            self.status.set(f"已選 {self.mechanical_name(terminal)}，請點選另一個方形接點。")
        elif self.pending_mechanical == terminal:
            self.cancel_connection()
            return
        else:
            first, second = self.pending_mechanical, terminal
            self.pending_mechanical = None
            if (first[0] == "spring") == (second[0] == "spring"):
                self.status.set("請將避震器的一端接到活塞桿或車體固定點。")
            else:
                spring, target = (first, second) if first[0] == "spring" else (second, first)
                spring_key, end = spring[1:]
                link = (spring_key, end)
                destination = None if target == CHASSIS_TERMINAL else target[1]
                other = (spring_key, "b" if end == "a" else "a")
                if link in self.spring_links:
                    self.status.set("這個避震器端點已有連線。請選取連線後刪除，再重新連接。")
                elif other in self.spring_links and self.spring_links[other] == destination:
                    self.status.set("避震器的 A、B 兩端須接到不同的機械接點。")
                else:
                    self.spring_links[link] = destination
                    self.invalidate_results()
                    self.status.set(f"已連接 {self.mechanical_name(spring)} ↔ {self.mechanical_name(target)}。")
        self.draw_network()

    def cancel_connection(self, _event=None):
        if self.pending_port is not None or self.pending_mechanical is not None:
            self.pending_port = self.pending_mechanical = None
            self.status.set("已取消連接。")
            self.draw_network()

    def canvas_press(self, event):
        tags = self.canvas.gettags("current")
        if not tags or not self.commit_editor():
            return
        if "port" in tags:
            key, side = next(tag for tag in tags if tag.startswith("port_"))[5:].split("/")
            self.connect_port((int(key), side))
            return
        if "mechanical" in tags:
            kind, key, end = next(tag for tag in tags if tag.startswith("mechanical_"))[11:].split("/")
            self.connect_mechanical((kind, int(key), end))
            return
        if "hose" in tags:
            hose = self.hose_items[next(tag for tag in tags if tag.startswith("hose_"))]
            self.selection = ("hose", hose)
            self.status.set("已選取油管。可按「刪除選取」解除這段連接。")
        elif "line_port" in tags:
            key, side = next(tag for tag in tags if tag.startswith("line_port_"))[10:].split("/")
            self.selection = ("line_port", (int(key), side))
            self.status.set("已選取單一油口的管路。可按「刪除選取」解除連接。")
        elif "spring_link" in tags:
            key, end = next(tag for tag in tags if tag.startswith("link_"))[5:].split("/")
            self.selection = ("link", (int(key), end))
            self.status.set("已選取避震器連線。可按「刪除選取」解除連接。")
        elif "cylinder" in tags or "spring" in tags:
            kind = "cylinder" if "cylinder" in tags else "spring"
            key = int(next(tag for tag in tags if tag.startswith(kind + "_"))[len(kind) + 1:])
            self.selection = (kind, key)
            self.tabs.select(self.editor)
            positions = self.positions if kind == "cylinder" else self.spring_positions
            x, y = positions[key]
            self.drag = (positions, key, self.canvas.canvasx(event.x) - x, self.canvas.canvasy(event.y) - y)
        self.populate_editor()
        self.draw_network()

    def canvas_drag(self, event):
        if self.drag is not None:
            positions, key, dx, dy = self.drag
            positions[key] = (max(55, self.canvas.canvasx(event.x) - dx),
                              max(54, self.canvas.canvasy(event.y) - dy))
            self.draw_network()

    def canvas_release(self, _event):
        self.drag = None

    def port_xy(self, port):
        key, side = port
        x, y = self.positions[key]
        return x + (BODY_WIDTH if side == "rod" else 0), y + BODY_HEIGHT / 2

    def hose_points(self, hose):
        first, second = hose
        x1, y1 = self.port_xy(first)
        x2, y2 = self.port_xy(second)
        lead1 = x1 + (-34 if first[1] == "cap" else 34)
        lead2 = x2 + (-34 if second[1] == "cap" else 34)
        if abs(lead1 - lead2) < 1.:
            return (x1, y1, lead1, y1, lead1, y2, x2, y2)
        if abs(y1 - y2) < 1. and (x1 < lead1 <= lead2 < x2 or x2 < lead2 <= lead1 < x1):
            return (x1, y1, x2, y2)
        middle = min(y1, y2) - 48 if abs(y1 - y2) < BODY_HEIGHT + 24 else (y1 + y2) / 2
        return (x1, y1, lead1, y1, lead1, middle, lead2, middle, lead2, y2, x2, y2)

    def mechanical_xy(self, terminal):
        kind, key, end = terminal
        if terminal == CHASSIS_TERMINAL:
            return CHASSIS_XY
        if kind == "rod":
            x, y = self.positions[key]
            return x + BODY_WIDTH / 2, y + BODY_HEIGHT
        x, y = self.spring_positions[key]
        return x + (SPRING_WIDTH if end == "b" else 0), y + SPRING_HEIGHT / 2

    def draw_mechanical_port(self, terminal, connected):
        x, y = self.mechanical_xy(terminal)
        pending = terminal == self.pending_mechanical
        color = "#e09a21" if pending else MECHANICAL
        radius = 8 if pending else 6
        kind, key, end = terminal
        self.canvas.create_rectangle(x - radius, y - radius, x + radius, y + radius,
                                     fill=color if connected or pending else "#ffffff", outline=color,
                                     width=2, tags=("mechanical", f"mechanical_{kind}/{key}/{end}"))

    def draw_springs(self):
        canvas = self.canvas
        for key, spring in self.springs.items():
            x, y = self.spring_positions[key]
            tags = ("spring", f"spring_{key}")
            selected = self.selection == ("spring", key)
            canvas.create_rectangle(x, y, x + SPRING_WIDTH, y + SPRING_HEIGHT, fill="#faf6ff",
                                    outline=MECHANICAL if selected else "#baa9cf",
                                    width=2 if selected else 1, tags=tags)
            center = x + SPRING_WIDTH / 2
            canvas.create_text(center, y + 15, text=short_name(spring.name), fill=INK,
                               font=(FONT, 11, "bold"), tags=tags)
            zigzag = [coordinate for index in range(9)
                      for coordinate in (center - 29 + index * 7, y + 35 + (index % 2 * 2 - 1) * 4)]
            canvas.create_line(*zigzag, fill=MECHANICAL, width=2, tags=tags)
            canvas.create_text(center, y + 55, text=f"K {spring.stiffness_n_mm:.4g}  ·  C {spring.damping_n_s_mm:.4g}",
                               fill=MUTED, font=(FONT, 9), tags=tags)
            for end, offset in (("a", 13), ("b", SPRING_WIDTH - 13)):
                canvas.create_text(x + offset, y + SPRING_HEIGHT / 2, text=end.upper(),
                                   font=(FONT, 8, "bold"), fill=MECHANICAL, tags=tags)
                self.draw_mechanical_port(("spring", key, end), (key, end) in self.spring_links)
        if self.springs:
            x, y = CHASSIS_XY
            canvas.create_line(x, y, x, y + 15, fill=MECHANICAL, width=2)
            canvas.create_line(x - 19, y + 15, x + 19, y + 15, fill=MECHANICAL, width=2)
            for offset in (-14, -7, 0, 7, 14):
                canvas.create_line(x + offset, y + 15, x + offset - 5, y + 21, fill=MECHANICAL)
            canvas.create_text(x, y + 34, text="車體固定點", fill=MECHANICAL, font=(FONT, 9))
            self.draw_mechanical_port(CHASSIS_TERMINAL, None in self.spring_links.values())

    def draw_network(self):
        canvas = self.canvas
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        positions = (*self.positions.values(), *self.spring_positions.values())
        right = max([width, 740 if self.springs else 600, *(x + BODY_WIDTH + 90 for x, _ in positions)])
        bottom = max([height, 490 if self.springs else 390, *(y + BODY_HEIGHT + 70 for _, y in positions)])
        canvas.configure(scrollregion=(0, 0, right, bottom))
        if not self.cylinders and not self.springs:
            x, y = width / 2, height / 2
            canvas.create_text(x, y - 15, text="畫布尚無元件", fill=INK, font=(FONT, 16, "bold"))
            canvas.create_text(x, y + 19, text="新增液壓缸與避震器，或載入三缸範例。", fill=MUTED, font=(FONT, 10))
            return

        groups = self.network_groups()
        colors = {name: PIPE_COLORS[index % len(PIPE_COLORS)]
                  for index, name in enumerate(dict.fromkeys(groups.values()))}
        labeled = set()
        self.hose_items = {}
        for index, hose in enumerate(sorted(self.hoses)):
            name = groups[hose[0]]
            color = colors[name]
            path = self.hose_points(hose)
            tag = f"hose_{index}"
            self.hose_items[tag] = hose
            canvas.create_line(*path, fill=color, width=6 if self.selection == ("hose", hose) else 4,
                               capstyle="round", joinstyle="round", tags=("hose", tag))
            if name not in labeled:
                labeled.add(name)
                points = list(zip(path[::2], path[1::2]))
                start, finish = max(zip(points, points[1:]),
                                    key=lambda segment: abs(segment[0][0] - segment[1][0])
                                    + abs(segment[0][1] - segment[1][1]))
                x, y = ((a + b) / 2 for a, b in zip(start, finish))
                canvas.create_text(x - 9, y - 10, text=name, fill=color, font=(FONT, 9, "bold"),
                                   anchor="e", tags=("hose", tag))
        for port, name in self.line_anchors.items():
            if name in labeled:
                continue
            x, y = self.port_xy(port)
            offset = -1 if port[1] == "cap" else 1
            end_x = x + offset * 34
            color = colors[name]
            tags = ("line_port", f"line_port_{port[0]}/{port[1]}")
            canvas.create_line(x, y, end_x, y, fill=color,
                               width=6 if self.selection == ("line_port", port) else 4, tags=tags)
            canvas.create_text(end_x, y - 10, text=short_name(name), fill=color, font=(FONT, 9, "bold"),
                               anchor="e" if offset < 0 else "w", tags=tags)

        for link, destination in self.spring_links.items():
            key, end = link
            first = self.mechanical_xy(("spring", key, end))
            terminal = CHASSIS_TERMINAL if destination is None else ("rod", destination, "")
            second = self.mechanical_xy(terminal)
            canvas.create_line(*first, *second, fill=MECHANICAL, dash=(6, 4),
                               width=5 if self.selection == ("link", link) else 3,
                               tags=("spring_link", f"link_{key}/{end}"))

        connected_rods = set(self.spring_links.values())
        for key, cylinder in self.cylinders.items():
            x, y = self.positions[key]
            tags = ("cylinder", f"cylinder_{key}")
            selected = self.selection == ("cylinder", key)
            canvas.create_rectangle(x, y, x + BODY_WIDTH, y + BODY_HEIGHT, fill="#ffffff",
                                    outline="#246d9a" if selected else "#b1c4d4",
                                    width=2 if selected else 1, tags=tags)
            canvas.create_text(x + BODY_WIDTH / 2, y + 18, text=short_name(cylinder.name), fill=INK,
                               font=(FONT, 12, "bold"), tags=tags)
            canvas.create_text(x + BODY_WIDTH / 2, y + 39,
                               text=f"缸 Ø{cylinder.bore_mm:g}  /  桿 Ø{cylinder.rod_mm:g} mm",
                               fill=MUTED, font=(FONT, 9), tags=tags)
            for side, color in (("cap", CAP_COLOR), ("rod", ROD_COLOR)):
                port = (key, side)
                a, b = self.port_xy(port)
                pending = port == self.pending_port
                port_color = "#e09a21" if pending else color
                radius = 9 if pending else 7
                canvas.create_oval(a - radius, b - radius, a + radius, b + radius,
                                   fill=port_color if port in groups or pending else "#ffffff",
                                   outline=port_color, width=2, tags=("port", f"port_{key}/{side}"))
            canvas.create_text(x + BODY_WIDTH / 2 + 13, y + BODY_HEIGHT - 10, text="活塞桿",
                               anchor="w", fill=MECHANICAL, font=(FONT, 8), tags=tags)
            self.draw_mechanical_port(("rod", key, ""), key in connected_rods)
        self.draw_springs()

    def solve(self):
        if not self.commit_editor():
            return
        try:
            self.pressures, self.compression, self.required_forces = solve_design(
                self.cylinders, self.hoses, self.springs, self.spring_links,
                anchors=self.line_anchors, project_lines=self.project_lines,
                default_compliance_cm3_mpa=self.default_compliance)
        except (ValueError, ArithmeticError) as error:
            self.clear_static_results()
            message = str(error)
            if message.startswith("Static displacement or pressure is not unique."):
                message = "靜力解不唯一。請檢查液壓缸有效面積、油管連接或避震器剛性。"
            elif message.startswith("The applied forces cannot satisfy"):
                message = "目前外力無法同時滿足活塞桿力平衡與油量守恆。請調整外力或缸徑、桿徑。"
            elif message.startswith("The prescribed displacements and forces"):
                message = "指定位移與其餘外力無法同時滿足靜力平衡及油量守恆。請調整輸入或油路。"
            elif message.startswith("Free rod displacement is not unique."):
                message = "仍有活塞桿位移無唯一解。可再指定一支桿的位移，或接上具有彈簧剛性的避震器。"
            self.status.set(message)
            return
        self.refresh_results()
        self.tabs.select(self.results)
        if any(value is None for value in (*self.pressures.values(), *self.required_forces.values())):
            self.status.set("已求得活塞桿位移。表中「無唯一值」表示該項需要額外的壓力或負載條件。")
        else:
            self.status.set(f"已求解 {len(self.cylinders)} 支液壓缸、{len(self.springs)} 支避震器及 "
                            f"{len(self.pressures)} 條等壓油路。")

    def refresh_results(self):
        self.result_table.delete(*self.result_table.get_children())
        for key, force in self.required_forces.items():
            value = "無唯一值" if force is None else f"{force:+.3f}"
            self.result_table.insert("", "end", values=(f"{self.cylinders[key].name} 所需外力", value, "N"))
        for name, pressure in self.pressures.items():
            value = "無唯一值" if pressure is None else f"{pressure:.4f}"
            self.result_table.insert("", "end", values=(name, value, "MPa"))
        self.draw_chart()

    def draw_chart(self):
        chart = self.chart
        chart.delete("all")
        width, height = chart.winfo_width(), chart.winfo_height()
        bottom = max(height, 30 + 33 * len(self.compression))
        chart.configure(scrollregion=(0, 0, width, bottom))
        if not self.compression:
            chart.create_text(width / 2, max(height / 2, 25), text="求解後顯示各缸位移", fill=MUTED, font=(FONT, 9))
            return
        left, right = 92, max(94, width - 64)
        center = (left + right) / 2
        span = (right - left) / 2
        scale = max(map(abs, self.compression.values())) or 1.
        chart.create_line(center, 10, center, bottom - 10, fill="#8ba0b3")
        for index, (key, value) in enumerate(self.compression.items()):
            y = 23 + 33 * index
            end = center + span * value / scale
            color = CAP_COLOR if value >= 0. else ROD_COLOR
            chart.create_rectangle(min(center, end), y - 8, max(center, end), y + 8, fill=color, outline="")
            chart.create_text(8, y, text=short_name(self.cylinders[key].name, 6), anchor="w", fill=INK, font=(FONT, 9))
            label = "0.000" if abs(value) < .0005 else f"{value:+.3f}"
            chart.create_text(width - 8, y, text=label, anchor="e", fill=INK, font=("Segoe UI", 9, "bold"))


def main():
    root = tk.Tk()
    HydraulicWorkbench(root)
    root.mainloop()


if __name__ == "__main__":
    main()
