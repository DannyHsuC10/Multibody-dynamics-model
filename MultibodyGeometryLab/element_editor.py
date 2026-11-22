"""Tk editor for the members connecting the suspension hardpoints."""

from copy import deepcopy
import tkinter as tk
from tkinter import ttk

from ._geometry_support.hardpoints import CORNERS
from .elements import DAMPER_POINT, DEFORMABLE_KINDS, PUSH_ROD, validate_elements
from .geometry import POINTS


ENDPOINTS = (*POINTS, DAMPER_POINT)
KIND_LABELS = {"rigid": "剛性桿", **dict.fromkeys(DEFORMABLE_KINDS, "deformable_members")}
RIGID_LABEL, DEFORMABLE_LABEL = KIND_LABELS["rigid"], KIND_LABELS["spring_damper"]


class ElementEditor(ttk.Frame):
    """Edit per-corner member definitions before applying the mechanism."""

    def __init__(self, parent, mark_dirty, apply):
        super().__init__(parent, padding=8)
        self.mark_dirty = mark_dirty
        self.elements = {name: [] for name in CORNERS}
        self.points = {}
        self.rocker_corners = set()
        self.active_corner = "FL"
        self.selected = None
        self.syncing = False

        self.corner = tk.StringVar(self, value=self.active_corner)
        self.kind = tk.StringVar(self)
        self.end_a = tk.StringVar(self, value=ENDPOINTS[0])
        self.end_b = tk.StringVar(self, value=ENDPOINTS[-1])
        self.detail = tk.StringVar(self)
        self.error = tk.StringVar(self)

        header = ttk.Frame(self)
        header.pack(fill="x")
        ttk.Label(header, text="輪位").pack(side="left")
        wheel = ttk.Combobox(header, textvariable=self.corner, values=CORNERS, state="readonly", width=5)
        wheel.pack(side="left", padx=(5, 9))
        wheel.bind("<<ComboboxSelected>>", self.change_corner)
        ttk.Label(header, text="點線條或清單選元件").pack(side="left")

        self.preview = tk.Canvas(self, width=250, height=164, background="#f6f8fa", highlightthickness=0)
        self.preview.pack(fill="both", expand=True, pady=(6, 5))
        self.preview.bind("<Configure>", self.draw)
        self.preview.bind("<Button-1>", self.choose_line)

        table = ttk.Frame(self)
        table.pack(fill="x")
        self.tree = ttk.Treeview(table, columns=("name", "kind"), show="headings", height=5, selectmode="browse")
        self.tree.heading("name", text="元件")
        self.tree.heading("kind", text="類型")
        self.tree.column("name", width=162, stretch=True)
        self.tree.column("kind", width=130, stretch=False)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind("<<TreeviewSelect>>", self.change_row)

        self.build_member_controls()
        self.build_connection_controls()
        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=(6, 0))
        self.rigid_button = ttk.Button(buttons, text="全車設為剛性", command=self.make_rigid)
        self.rigid_button.pack(side="left")
        ttk.Button(buttons, text="套用機構", command=apply).pack(side="right")
        ttk.Label(self, textvariable=self.error, wraplength=280, foreground="#9e3829").pack(fill="x", pady=(3, 0))
        self.show_member(None)

    def build_member_controls(self):
        panel = ttk.Frame(self)
        panel.pack(fill="x", pady=(4, 0))
        ttk.Label(panel, textvariable=self.detail).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(panel, text="類型").grid(row=1, column=0, sticky="w")
        self.role = ttk.Combobox(panel, width=20, textvariable=self.kind, values=(RIGID_LABEL, DEFORMABLE_LABEL), state="readonly")
        self.role.grid(row=1, column=1, sticky="w", padx=(3, 7))
        self.role.bind("<<ComboboxSelected>>", self.change_kind)
        self.remove = ttk.Button(panel, text="移除所選", command=self.remove_member)
        self.remove.grid(row=1, column=2, columnspan=2, sticky="e")

    def build_connection_controls(self):
        panel = ttk.Frame(self)
        panel.pack(fill="x", pady=(5, 0))
        for variable in (self.end_a, self.end_b):
            ttk.Combobox(panel, textvariable=variable, values=ENDPOINTS, state="readonly", width=7).pack(
                side="left", padx=(0, 4))
        self.add_button = ttk.Button(panel, text="新增連線", command=self.add_member)
        self.add_button.pack(side="left")

    def populate(self, elements, points, rocker_corners=()):
        self.elements = deepcopy(elements)
        self.points = points
        self.rocker_corners = set(rocker_corners)
        self.error.set("")
        self.refresh_rows(0)

    def set_points(self, points, rocker_corners):
        self.points = points
        if self.rocker_corners != set(rocker_corners):
            self.rocker_corners = set(rocker_corners)
            self.refresh_rows(0)
        else:
            self.draw()

    def draft(self):
        return validate_elements(self.elements)

    def members(self):
        return self.elements[self.active_corner]

    def refresh_rows(self, selected):
        rocker = self.active_corner in self.rocker_corners
        self.syncing = True
        try:
            self.tree.delete(*self.tree.get_children())
            for index, item in enumerate(self.members()):
                self.tree.insert("", "end", iid=str(index), values=(item["name"], KIND_LABELS[item["kind"]]))
            if rocker:
                self.tree.insert("", "end", iid="rocker", values=("推桿（搖臂）", RIGID_LABEL))
            chosen = selected if self.members() or selected == "rocker" else None
            if chosen is not None:
                self.tree.selection_set(str(chosen))
                self.tree.see(str(chosen))
        finally:
            self.syncing = False
        self.show_member(chosen)

    def show_member(self, index):
        self.selected = index
        item = self.members()[index] if isinstance(index, int) else None
        if index == "rocker":
            self.kind.set(RIGID_LABEL)
            self.detail.set("推桿維持固定長度以定位搖臂。")
        else:
            self.kind.set(KIND_LABELS[item["kind"]] if item else "")
            self.detail.set(f"{item['a']} → {item['b']}" if item else "選取元件以編輯")
        self.role.configure(state="readonly" if item else "disabled")
        self.remove.configure(state="normal" if item and item["name"] != PUSH_ROD else "disabled")
        self.draw()

    def change_corner(self, _event=None):
        self.active_corner = self.corner.get()
        self.refresh_rows(0)

    def change_row(self, _event=None):
        selection = self.tree.selection()
        if self.syncing or not selection:
            return
        index = "rocker" if selection[0] == "rocker" else int(selection[0])
        if index != self.selected:
            self.show_member(index)

    def change_kind(self, _event=None):
        if not isinstance(self.selected, int):
            return
        item = self.members()[self.selected]
        kind = "rigid" if self.kind.get() == RIGID_LABEL else "spring_damper"
        if KIND_LABELS[item["kind"]] != KIND_LABELS[kind]:
            item["kind"] = kind
            self.mark_and_refresh(self.selected)

    def mark_and_refresh(self, selection):
        self.error.set("")
        self.mark_dirty()
        self.refresh_rows(selection)

    def make_rigid(self):
        for members in self.elements.values():
            for item in members:
                item["kind"] = "rigid"
        self.mark_and_refresh(0)

    def add_member(self):
        a, b = self.end_a.get(), self.end_b.get()
        if a == b:
            self.error.set("請選擇兩個不同的端點。")
            return
        label = f"{a} → {b}"
        name, suffix = label, 2
        names = {item["name"] for item in self.members()}
        while name in names:
            name = f"{label} ({suffix})"
            suffix += 1
        self.members().append({"name": name, "a": a, "b": b, "kind": "rigid",
                               "stiffness_n_m": None, "damping_ns_m": None})
        self.mark_and_refresh(len(self.members()) - 1)

    def remove_member(self):
        index = self.selected
        if not isinstance(index, int) or self.members()[index]["name"] == PUSH_ROD:
            return
        self.members().pop(index)
        self.mark_and_refresh(min(index, len(self.members()) - 1))

    def draw(self, _event=None):
        canvas = self.preview
        canvas.delete("all")
        points = self.points.get(self.active_corner)
        if not points:
            return
        width, height = max(canvas.winfo_width(), 120), max(canvas.winfo_height(), 100)
        projection = {
            name: (.78 * x + .62 * y, -.26 * x + .32 * y + .91 * z)
            for name, (x, y, z) in points.items()
        }
        xs, ys = zip(*projection.values())
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
        scale = min((width - 56) / max(x_max - x_min, 1e-9), (height - 32) / max(y_max - y_min, 1e-9))
        mapped = {
            name: (width / 2 + scale * (x - (x_min + x_max) / 2),
                   height / 2 - scale * (y - (y_min + y_max) / 2))
            for name, (x, y) in projection.items()
        }
        for index, item in enumerate(self.members()):
            a, b = item["a"], item["b"]
            if a not in mapped or b not in mapped:
                continue
            selected = index == self.selected
            canvas.create_line(*mapped[a], *mapped[b], width=4 if selected else 2,
                               fill="#2169a0" if selected else ("#617688" if item["kind"] == "rigid" else "#be6234"),
                               dash=(4, 2) if item["kind"] in DEFORMABLE_KINDS else (),
                               tags=("member", f"line:{index}"))
        if "PUSH_OUT" in mapped and "PUSH_IN" in mapped:
            if self.active_corner in self.rocker_corners:
                canvas.create_line(*mapped["PUSH_OUT"], *mapped["PUSH_IN"], width=2, fill="#617688",
                                   tags=("member", "line:rocker"))
            else:
                canvas.create_line(*mapped["PUSH_OUT"], *mapped["PUSH_IN"], width=2, fill="#be6234", dash=(4, 2))
        if self.selected is not None:
            canvas.tag_raise(f"line:{self.selected}")
        for name, (x, y) in mapped.items():
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#263848", outline="")
            canvas.create_text(x + 3, y - 3, text=name, anchor="sw", font=("TkDefaultFont", 7), fill="#354555")

    def choose_line(self, event):
        canvas = self.preview
        nearby = canvas.find_overlapping(event.x - 3, event.y - 3, event.x + 3, event.y + 3)
        for item in reversed(nearby):
            tags = canvas.gettags(item)
            if "member" in tags:
                index = next(tag.split(":")[1] for tag in tags if tag.startswith("line:"))
                self.tree.selection_set(index)
                self.tree.see(index)
                return
