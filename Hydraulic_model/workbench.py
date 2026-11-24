"""Native editor for constructing and solving static hydraulic circuits."""

from itertools import count
import tkinter as tk
from tkinter import messagebox, ttk

if __package__:
    from .design import CylinderDesign, Hose, Port, SpringDesign, SpringEnd, SpringTarget, pressure_groups, solve_design
else:
    from design import CylinderDesign, Hose, Port, SpringDesign, SpringEnd, SpringTarget, pressure_groups, solve_design


BODY_WIDTH = 174
BODY_HEIGHT = 68
SPRING_WIDTH = 142
SPRING_HEIGHT = 70
GROUND_XY = (686, 426)
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
    ("compression_force_n", "外力 (N，正值壓縮)"),
)
SPRING_FIELDS = (("name", "名稱"), ("stiffness_n_mm", "彈簧剛性 (N/mm)"))
FIELDS = dict((*CYLINDER_FIELDS, *SPRING_FIELDS))


def short_name(name, limit=14):
    return name if len(name) <= limit else name[:limit - 1] + "…"


class HydraulicWorkbench(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, padding=(18, 14))
        root.title("液壓設計平台")
        root.geometry("1180x740")
        root.minsize(920, 560)
        root.option_add("*Font", (FONT, 10))
        self.pack(fill="both", expand=True)
        self.cylinders: dict[int, CylinderDesign] = {}
        self.positions: dict[int, tuple[float, float]] = {}
        self.hoses: set[Hose] = set()
        self.springs: dict[int, SpringDesign] = {}
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
        self.status = tk.StringVar(self, "新增液壓缸開始設計，或載入三缸範例。")
        self.editor_title = tk.StringVar(self)
        self.fields = {key: tk.StringVar(self) for key in FIELDS}
        self.build_layout()
        root.bind("<Escape>", self.cancel_connection)
        self.populate_editor()
        self.draw_network()

    def build_layout(self):
        ttk.Label(self, text="液壓設計平台", font=(FONT, 20, "bold")).pack(anchor="w")
        ttk.Label(self, text="封閉油路靜力分析", foreground=MUTED).pack(anchor="w")
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(10, 10))
        for text, command in (
            ("新增液壓缸", self.add_cylinder), ("新增彈簧", self.add_spring),
            ("刪除選取", self.delete_selection),
            ("載入三缸範例", self.load_example),
        ):
            ttk.Button(toolbar, text=text, command=command).pack(side="left", padx=(0, 7))
        ttk.Button(toolbar, text="求解靜力", command=self.solve).pack(side="right")

        content = ttk.Frame(self)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)
        ttk.Label(content, text="圓形油口接油管 · 方形接點接彈簧 · 拖曳元件移動 · Esc 取消連接",
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
        self.apply_button.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(12, 14))
        ttk.Label(self.editor, text="●  左側 · 無桿腔", foreground=CAP_COLOR).grid(
            row=7, column=0, columnspan=2, sticky="w")
        ttk.Label(self.editor, text="●  右側 · 有桿腔", foreground=ROD_COLOR).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(3, 10))
        ttk.Label(self.editor, text="■  活塞桿、彈簧端點與接地", foreground=MECHANICAL).grid(
            row=9, column=0, columnspan=2, sticky="w")
        ttk.Label(self.editor, text="彈簧 A、B 兩端各接一個活塞桿或接地。未接管的油口採零表壓。",
                  foreground=MUTED, wraplength=285).grid(row=10, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def build_results(self):
        self.results.columnconfigure(0, weight=1)
        self.results.rowconfigure(3, weight=1)
        ttk.Label(self.results, text="油路表壓", font=(FONT, 11, "bold")).grid(row=0, column=0, sticky="w")
        self.pressure_table = ttk.Treeview(self.results, columns=("line", "pressure"), show="headings", height=4)
        self.pressure_table.heading("line", text="油路")
        self.pressure_table.heading("pressure", text="表壓 (MPa)")
        self.pressure_table.column("line", width=90, anchor="center")
        self.pressure_table.column("pressure", width=170, anchor="center")
        scrollbar = ttk.Scrollbar(self.results, orient="vertical", command=self.pressure_table.yview)
        self.pressure_table.configure(yscrollcommand=scrollbar.set)
        self.pressure_table.grid(row=1, column=0, sticky="ew", pady=(5, 12))
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(5, 12))
        ttk.Label(self.results, text="活塞桿位移 (mm)", font=(FONT, 11, "bold")).grid(
            row=2, column=0, sticky="w")
        self.chart = tk.Canvas(self.results, background=CANVAS_BG, highlightthickness=0, height=168, width=295)
        chart_scroll = ttk.Scrollbar(self.results, orient="vertical", command=self.chart.yview)
        self.chart.configure(yscrollcommand=chart_scroll.set)
        self.chart.grid(row=3, column=0, sticky="nsew", pady=(5, 5))
        chart_scroll.grid(row=3, column=1, sticky="ns", pady=(5, 5))
        self.chart.bind("<Configure>", lambda _: self.draw_chart())
        ttk.Label(self.results, text="正值為壓縮 · 負值為伸出", foreground=MUTED).grid(
            row=4, column=0, sticky="w")

    def selected_component(self):
        if self.selection is not None:
            kind, key = self.selection
            if kind == "cylinder":
                return self.cylinders, key, CYLINDER_FIELDS
            if kind == "spring":
                return self.springs, key, SPRING_FIELDS
        return {}, None, ()

    def populate_editor(self):
        components, key, fields = self.selected_component()
        item = components.get(key)
        self.editor_title.set(item.name if item is not None else "請選擇液壓缸或彈簧")
        visible = dict(fields)
        for field, variable in self.fields.items():
            if field in visible:
                variable.set(str(getattr(item, field)))
                self.entries[field].grid()
                self.labels[field].grid()
            else:
                self.entries[field].grid_remove()
                self.labels[field].grid_remove()
        self.apply_button.configure(state="normal" if item is not None else "disabled")

    def commit_editor(self):
        components, key, fields = self.selected_component()
        if key is None:
            return True
        try:
            name = self.fields["name"].get().strip()
            try:
                values = {field: float(self.fields[field].get()) for field, _ in fields[1:]}
            except ValueError as error:
                raise ValueError("請在參數欄位輸入數值。") from error
            updated = type(components[key])(name, **values)
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
        self.pressures = {}
        self.compression = {}
        self.refresh_results()

    def available_position(self):
        occupied = (*self.positions.values(), *self.spring_positions.values())
        for index in count():
            x = 80 + 272 * (index % 3)
            y = 62 + 114 * (index // 3)
            gx, gy = GROUND_XY
            if x - 20 <= gx <= x + BODY_WIDTH + 20 and y - 45 <= gy <= y + BODY_HEIGHT + 20:
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
        self.status.set(f"已新增 {name}。使用左右油口接油管，或用下方的方形接點接彈簧。")

    def add_spring(self):
        if not self.commit_editor():
            return
        names = {spring.name for spring in self.springs.values()}
        key = self.next_spring_id
        name = f"彈簧 {key}"
        while name in names:
            key += 1
            name = f"彈簧 {key}"
        self.next_spring_id = key + 1
        self.spring_positions[key] = self.available_position()
        self.springs[key] = SpringDesign(name)
        self.show_new_component("spring", key, self.spring_positions[key])
        self.status.set(f"已新增 {name}。依序點選彈簧端點和活塞桿或接地，完成 A、B 兩端連接。")

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
        self.springs = {1: SpringDesign("彈簧 K")}
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
        self.status.set("已載入三缸範例。彈簧 K 的 A 端接 C 缸活塞桿，B 端接地。")

    def delete_selection(self):
        if self.selection is None:
            self.status.set("請先在畫布上選取要刪除的元件或連線。")
            return
        kind, key = self.selection
        if kind == "cylinder":
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
        else:
            self.spring_links.pop(key)
            self.status.set("已刪除選取的彈簧連線。")
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
                self.hoses.add(hose)
                self.status.set(f"已接管：{self.port_name(first)} ↔ {self.port_name(second)}。")
                self.invalidate_results()
        self.draw_network()

    def mechanical_name(self, terminal):
        kind, key, end = terminal
        if kind == "ground":
            return "接地"
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
                self.status.set("請將彈簧的一端接到活塞桿或接地。")
            else:
                spring, target = (first, second) if first[0] == "spring" else (second, first)
                spring_key, end = spring[1:]
                link = (spring_key, end)
                destination = None if target[0] == "ground" else target[1]
                other = (spring_key, "b" if end == "a" else "a")
                if link in self.spring_links:
                    self.status.set("這個彈簧端點已有連線。請選取連線後刪除，再重新連接。")
                elif other in self.spring_links and self.spring_links[other] == destination:
                    self.status.set("彈簧的 A、B 兩端須接到不同的機械接點。")
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
        elif "spring_link" in tags:
            key, end = next(tag for tag in tags if tag.startswith("link_"))[5:].split("/")
            self.selection = ("link", (int(key), end))
            self.status.set("已選取彈簧連線。可按「刪除選取」解除連接。")
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
        if kind == "ground":
            return GROUND_XY
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
            canvas.create_text(center, y + 55, text=f"K {spring.stiffness_n_mm:g} N/mm",
                               fill=MUTED, font=(FONT, 9), tags=tags)
            for end, offset in (("a", 13), ("b", SPRING_WIDTH - 13)):
                canvas.create_text(x + offset, y + SPRING_HEIGHT / 2, text=end.upper(),
                                   font=(FONT, 8, "bold"), fill=MECHANICAL, tags=tags)
                self.draw_mechanical_port(("spring", key, end), (key, end) in self.spring_links)
        if self.springs:
            x, y = GROUND_XY
            canvas.create_line(x, y, x, y + 15, fill=MECHANICAL, width=2)
            for indent, level in ((12, 15), (8, 20), (4, 25)):
                canvas.create_line(x - indent, y + level, x + indent, y + level, fill=MECHANICAL, width=2)
            canvas.create_text(x, y + 36, text="接地", fill=MECHANICAL, font=(FONT, 9))
            self.draw_mechanical_port(("ground", 0, ""), None in self.spring_links.values())

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
            canvas.create_text(x, y + 19, text="新增液壓缸與彈簧，或載入三缸範例。", fill=MUTED, font=(FONT, 10))
            return

        groups = pressure_groups(self.hoses)
        labeled = set()
        self.hose_items = {}
        for index, hose in enumerate(sorted(self.hoses)):
            name = groups[hose[0]]
            color = PIPE_COLORS[(int(name[1:]) - 1) % len(PIPE_COLORS)]
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

        for link, destination in self.spring_links.items():
            key, end = link
            first = self.mechanical_xy(("spring", key, end))
            terminal = ("ground", 0, "") if destination is None else ("rod", destination, "")
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
            self.pressures, self.compression = solve_design(self.cylinders, self.hoses,
                                                            self.springs, self.spring_links)
        except (ValueError, ArithmeticError) as error:
            self.invalidate_results()
            message = str(error)
            if message.startswith("Static displacement or pressure is not unique."):
                message = "靜力解不唯一。請檢查液壓缸有效面積、油管連接或彈簧負載。"
            elif message.startswith("The applied forces cannot satisfy"):
                message = "目前外力無法同時滿足活塞桿力平衡與油量守恆。請調整外力或缸徑、桿徑。"
            self.status.set(message)
            return
        self.refresh_results()
        self.tabs.select(self.results)
        self.status.set(f"已求解 {len(self.cylinders)} 支液壓缸、{len(self.springs)} 支彈簧及 "
                        f"{len(self.pressures)} 條等壓油路。")

    def refresh_results(self):
        self.pressure_table.delete(*self.pressure_table.get_children())
        for name, pressure in self.pressures.items():
            self.pressure_table.insert("", "end", values=(name, f"{pressure:.4f}"))
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
