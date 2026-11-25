"""Native project editor and four-corner kinematics viewer."""

import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Thread
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import numpy as np

from ._geometry_support import hardpoints, project as geometry_project
from .animation_controls import AnimationPlayer
from .deformable import (DamperResult, DynamicsModel, build_dynamics_model, sample_dampers,
                         sample_element_springs, with_axle_pushrod_mounts)
from .deformable_controls import DeformableEditor
from .dynamics_controls import DriveEditor, HeaveEditor, LateralEditor, RoadEditor, TARGET_LABELS
from .dynamics_settings import INERTIA_LIMITS
from .element_editor import ElementEditor
from .elements import DAMPER_POINT, DEFORMABLE_KINDS, PUSH_ROD, default_elements
from .geometry import LINKS, POINTS, VehicleGeometry, build_vehicle, static_geometry
from .rocker import rocker_states
from .pushrod_controls import PushrodEditor
from .pushrod_settings import pushrod_from_hardpoints
from .hardpoint_import import import_hardpoint_csv
from .heave import simulate_heave
from .project import (COMPONENTS, DRIVERS, default_project, read_project,
                      save_project, validate)
from .rigid_body import simulate_rigid_drive, simulate_rigid_lateral, simulate_rigid_road
from .signal_plot import MOUNT_LABELS, SignalView
from .solver import Unreachable, VehicleResult, solve_vehicle
from .viewport import MechanismView
from .widgets import ScrollFrame


PROJECTS = Path(__file__).with_name("projects")
IMPORTS = PROJECTS.parents[1] / "new"
HARDPOINT_LABELS = {**hardpoints.POINTS, "WC": "輪心", "RACK": "轉向拉桿內側", "TOE": "轉向拉桿外側"}
STEERING_ENDPOINTS = {"RACK": "inner", "TOE": "outer"}
ROD_ROTATION_LABELS = {"upper_arm": "上 A 臂", "lower_arm": "下 A 臂",
                       "upright": "羊角剛體", "axis_only": "桿軸方向"}
HINGE_LABELS = {"UCA": "上 A 臂", "LCA": "下 A 臂"}


def show_number(value):
    return f"{value:.5f}".rstrip("0").rstrip(".") if abs(value) >= .000005 else "0"


def show_vector(vector):
    return ", ".join(map(show_number, vector))


def make_table(parent, headings, widths, height):
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True)
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    table = ttk.Treeview(frame, columns=tuple(range(len(headings))), show="headings", height=height)
    for index, (label, width) in enumerate(zip(headings, widths)):
        table.heading(index, text=label)
        table.column(index, width=width, minwidth=width//2, anchor="center")
    vertical = ttk.Scrollbar(frame, orient="vertical", command=table.yview)
    horizontal = ttk.Scrollbar(frame, orient="horizontal", command=table.xview)
    table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
    table.grid(row=0, column=0, sticky="nsew")
    vertical.grid(row=0, column=1, sticky="ns")
    horizontal.grid(row=1, column=0, sticky="ew")
    return table


def fill_table(table, rows):
    items = table.get_children()
    if len(items) == len(rows):
        for item, row in zip(items, rows):
            table.item(item, values=row)
    else:
        table.delete(*items)
        for row in rows:
            table.insert("", "end", values=row)


@dataclass(frozen=True)
class PreparedModel:
    geometry: VehicleGeometry
    dynamics: DynamicsModel
    neutral: VehicleResult
    neutral_dampers: dict[str, DamperResult]
    problem: str | None = None
    neutral_springs: dict | None = None


def solve_run_job(results, model, data, mode):
    try:
        if mode == "heave":
            frames = simulate_heave(model, data["dynamics"]["heave"], data["geometry"]["vehicle"]["mass"])
        elif mode == "road":
            frames = simulate_rigid_road(model, data["dynamics"], data["geometry"]["vehicle"]["mass"])
        elif mode == "lateral":
            frames = simulate_rigid_lateral(model, data["dynamics"], data["geometry"]["vehicle"]["mass"])
        elif mode == "drive":
            frames = simulate_rigid_drive(model, data["dynamics"], data["geometry"]["vehicle"]["mass"])
        else:
            raise ValueError(f"未知的模擬模式：{mode}")
    except Exception as error:
        results.put((None, error))
    else:
        results.put((frames, None))


class Workbench(ttk.Frame):
    def __init__(self, root, data=None, path=None):
        super().__init__(root, padding=(14, 10))
        self.root = root
        self.path = path
        self.dirty = False
        self.loading = False
        self.revision = 0
        self.animation_job = None
        self.signal_window = self.signal_view = None
        self.data = None
        self.edit_base = None
        self.edit_history = []
        self.edit_index = 0
        self.saved_edit = None
        self.pending_edit = None
        self.hardpoint_draft = None
        self.wheel_center_draft = None
        self.steering_draft = None
        self.pose = self.model = self.neutral = None
        self.selected_hardpoint = None
        self.hardpoint_rows = {}
        self.fields = {}
        self.active_corner = tk.StringVar(value="FL")
        self.pose_mode = tk.StringVar(value="static")
        self.hardpoint_mode = tk.BooleanVar(value=False)
        self.name = self.new_variable()
        self.caption = tk.StringVar()
        self.status = tk.StringVar()
        self.result_label = tk.StringVar()
        self.angular_label = tk.StringVar()

        root.title("雙 A 臂多體幾何工作台")
        root.geometry("1360x840")
        root.minsize(960, 660)
        if "clam" in ttk.Style(root).theme_names():
            ttk.Style(root).theme_use("clam")
        root.option_add("*Font", "{Microsoft JhengHei UI} 10")
        self.pack(fill="both", expand=True)
        self.workspace_tabs = ttk.Notebook(self)
        self.workspace_tabs.pack(fill="both", expand=True)
        self.geometry_workspace = ttk.Frame(self.workspace_tabs, padding=8)
        self.hydraulic_workspace = ttk.Frame(self.workspace_tabs)
        self.workspace_tabs.add(self.geometry_workspace, text="多體幾何")
        self.workspace_tabs.add(self.hydraulic_workspace, text="液壓設計")
        self.hydraulic_workbench = None
        self.hydraulic_source = None
        self.workspace_tabs.bind("<<NotebookTabChanged>>", self.open_hydraulic_workspace)

        self.build_toolbar(self.geometry_workspace)
        panes = ttk.Panedwindow(self.geometry_workspace, orient="horizontal")
        panes.pack(fill="both", expand=True, pady=(10, 6))
        editor = ttk.Frame(panes, width=400)
        display = ttk.Frame(panes)
        panes.add(editor, weight=0)
        panes.add(display, weight=1)
        self.build_editor(editor)
        self.build_display(display)
        ttk.Label(self.geometry_workspace, textvariable=self.status, wraplength=910).pack(anchor="w")
        root.bind("<Control-o>", lambda _event: self.geometry_shortcut(self.open_file))
        root.bind("<Control-s>", lambda _event: self.geometry_shortcut(self.save))
        root.bind("<Control-z>", lambda _event: self.geometry_shortcut(self.step_history, -1))
        root.bind("<Control-y>", lambda _event: self.geometry_shortcut(self.step_history, 1))
        root.bind("<Escape>", self.cancel_hydraulic_connection, add="+")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.install(data if data is not None else default_project(), path)

    def open_hydraulic_workspace(self, event):
        if event.widget is not self.workspace_tabs or self.workspace_tabs.select() != str(self.hydraulic_workspace):
            return
        if self.hydraulic_workbench is None:
            if self.hydraulic_workspace.winfo_children():
                return
            try:
                from Hydraulic_model.workbench import HydraulicWorkbench
            except ModuleNotFoundError as error:
                if error.name != "Hydraulic_model":
                    raise
                ttk.Label(self.hydraulic_workspace, padding=16,
                          text="液壓設計需要同一專案中的 Hydraulic_model 資料夾。").pack(anchor="nw")
                return
            self.hydraulic_workbench = HydraulicWorkbench(self.hydraulic_workspace, embedded=True)
            self.hydraulic_workbench.load_vehicle_project(self.hydraulic_source)
        try:
            data = self.draft()
        except (ValueError, ArithmeticError):
            data = self.hydraulic_source
        self.hydraulic_workbench.refresh_vehicle_endpoints(data)

    def cancel_hydraulic_connection(self, event):
        if self.hydraulic_workbench is not None and self.workspace_tabs.select() == str(self.hydraulic_workspace):
            return self.hydraulic_workbench.cancel_connection(event)

    def geometry_shortcut(self, command, *args):
        if self.workspace_tabs.select() == str(self.geometry_workspace):
            command(*args)
            return "break"

    def new_variable(self):
        variable = tk.StringVar()
        variable.trace_add("write", self.mark_dirty)
        return variable

    def mark_dirty(self, *_):
        if not self.loading:
            self.revision += 1
            self.dirty = True
            self.update_caption()
            if self.pending_edit is None:
                self.pending_edit = self.after_idle(self.record_edit)

    def edit_variables(self):
        yield self.name
        yield self.hardpoint_mode
        yield from self.fields.values()
        yield self.road_editor.target
        for editor in (self.road_editor, self.heave_editor, self.lateral_editor, self.drive_editor):
            yield from editor.fields.values()
        for fields in self.pushrod_editor.fields.values():
            for field in fields.values():
                if isinstance(field, tuple):
                    yield from field
                else:
                    yield field

    def edit_snapshot(self):
        # Preserve raw field text so undo also works while a value is incomplete.
        return (tuple(variable.get() for variable in self.edit_variables()),
                copy.deepcopy((self.hardpoint_draft, self.wheel_center_draft, self.steering_draft)),
                copy.deepcopy(self.element_editor.elements),
                self.deformable_editor.edit_snapshot())

    def record_edit(self):
        self.pending_edit = None
        snapshot = self.edit_snapshot()
        if not self.edit_history or snapshot != self.edit_history[self.edit_index]:
            del self.edit_history[self.edit_index + 1:]
            self.edit_history.append(snapshot)
            del self.edit_history[:-100]
            self.edit_index = len(self.edit_history) - 1
        self.dirty = snapshot != self.saved_edit
        self.update_caption()

    def flush_edit(self):
        if self.pending_edit is not None:
            self.after_cancel(self.pending_edit)
            self.record_edit()

    def step_history(self, direction):
        self.flush_edit()
        index = self.edit_index + direction
        if not 0 <= index < len(self.edit_history):
            return "break"
        self.edit_index = index
        snapshot = self.edit_history[index]
        values, hardpoints_snapshot, elements, deformable = snapshot
        self.loading = True
        try:
            for variable, value in zip(self.edit_variables(), values):
                variable.set(value)
            self.hardpoint_draft, self.wheel_center_draft, self.steering_draft = copy.deepcopy(hardpoints_snapshot)
            self.element_editor.populate(elements, self.element_editor.points, self.element_editor.rocker_corners)
            self.pushrod_editor.show_corner()
            self.deformable_editor.restore_edits(deformable)
            self.sync_deformables()
        finally:
            self.loading = False

        self.dirty = snapshot != self.saved_edit
        self.update_caption()
        self.revision += 1
        label = "已復原" if direction < 0 else "已重做"
        try:
            data = self.draft()
            prepared = self.prepare_for_editing(data)
        except (ValueError, ArithmeticError):
            self.player.clear()
            if self.signal_view is not None:
                self.signal_view.set_frames(())
            self.refresh_results()
            self.refresh_hardpoints()
            self.status.set(f"{label}設定 · 目前欄位尚無法解算，預覽為上一個有效結果。")
        else:
            self.data = data
            self.show_prepared(prepared)
            self.status.set(f"{label}設定 · {self.status.get()}")
        return "break"

    def update_caption(self):
        file_name = self.path.name if self.path else "尚未儲存"
        self.caption.set(file_name + ("  *" if self.dirty else ""))

    def build_toolbar(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill="x")
        ttk.Label(bar, text="雙 A 臂多體幾何", font=("Microsoft JhengHei UI", 17, "bold")).pack(side="left")
        ttk.Label(bar, textvariable=self.caption, foreground="#466079").pack(side="left", padx=20)
        for text, command in (("另存 JSON", lambda: self.save(True)), ("儲存 JSON", self.save),
                              ("匯入硬點 CSV", self.import_csv),
                              ("開啟 JSON", self.open_file)):
            ttk.Button(bar, text=text, command=command).pack(side="right", padx=(5, 0))

    def field(self, parent, key, label, unit=""):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=(8, 10), pady=2)
        ttk.Label(row, text=f"{label} ({unit})" if unit else label, wraplength=215).pack(side="left")
        variable = self.fields[key] = self.new_variable()
        entry = ttk.Entry(row, width=13, textvariable=variable, justify="right")
        entry.pack(side="right", padx=(6, 0))

    def build_editor(self, parent):
        name_row = ttk.Frame(parent)
        name_row.pack(fill="x", padx=(0, 7), pady=(2, 8))
        ttk.Label(name_row, text="專案名稱").pack(side="left", padx=(0, 10))
        ttk.Entry(name_row, textvariable=self.name).pack(side="left", fill="x", expand=True)
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True, padx=(0, 7))
        self.editor_tabs = notebook
        vehicle = ScrollFrame(notebook)
        notebook.add(vehicle, text="車輛")
        ttk.Label(vehicle.content, text="車身轉動慣量（繞簧上重心）", padding=(8, 8, 0, 4)).pack(anchor="w")
        for key, (label, _, _) in INERTIA_LIMITS.items():
            self.field(vehicle.content, ("inertia", "chassis", key), label, "kg·m²")
        ttk.Separator(vehicle.content).pack(fill="x", padx=8, pady=8)
        ttk.Label(vehicle.content, text="輪胎與輪端質量", padding=(8, 4)).pack(anchor="w")
        for corner in hardpoints.CORNERS:
            self.field(vehicle.content, ("mass", corner, "mass_kg"), corner, "kg")
        ttk.Separator(vehicle.content).pack(fill="x", padx=8, pady=8)
        for key, (label, unit) in geometry_project.FIELDS["vehicle"].items():
            if key in ("rim_gap", "clearance_steer_deg"):
                continue
            if key == "tire_stiffness":
                self.fields["geometry", "vehicle", key] = self.new_variable()
            else:
                self.field(vehicle.content, ("geometry", "vehicle", key), label, unit)

        points = ttk.Frame(notebook, padding=6)
        self.hardpoint_tab = points
        notebook.add(points, text="硬點")
        ttk.Checkbutton(points, text="用硬點 XYZ 定義幾何", variable=self.hardpoint_mode,
                        command=self.toggle_hardpoints).pack(anchor="w")
        ttk.Button(points, text="由專案參數重建 A 臂硬點", command=self.generate_hardpoints).pack(anchor="w", pady=4)
        ttk.Label(points, text="雙擊編輯 · 左右鏡射 · 單位 mm").pack(anchor="w", pady=(0, 4))
        self.hardpoint_table = make_table(points, ("輪位", "硬點", "X", "Y", "Z"), (50, 70, 63, 63, 63), 16)
        self.hardpoint_table.bind("<Double-1>", self.edit_hardpoint)
        notebook.bind("<<NotebookTabChanged>>", self.reveal_selected_hardpoint)

        dynamics_tabs = ttk.Notebook(notebook)
        notebook.add(dynamics_tabs, text="輸入")
        self.road_editor = RoadEditor(dynamics_tabs, self.mark_dirty, self.start_dynamics)
        dynamics_tabs.add(self.road_editor, text="路面擾動")
        self.heave_editor = HeaveEditor(dynamics_tabs, self.mark_dirty, self.start_heave)
        dynamics_tabs.add(self.heave_editor, text="重心 Heave")
        self.lateral_editor = LateralEditor(dynamics_tabs, self.mark_dirty, lambda: self.start_run("lateral"))
        dynamics_tabs.add(self.lateral_editor, text="重心側向力")
        self.drive_editor = DriveEditor(dynamics_tabs, self.mark_dirty, lambda: self.start_run("drive"))
        dynamics_tabs.add(self.drive_editor, text="馬達行駛")
        self.pushrod_editor = PushrodEditor(notebook, self.mark_dirty, self.sync_deformables)
        notebook.add(self.pushrod_editor, text="推桿")
        self.element_editor = ElementEditor(notebook, self.on_member_edit, self.apply)
        notebook.add(self.element_editor, text="機構元件")
        self.deformable_editor = DeformableEditor(
            notebook, self.mark_dirty, self.fields["geometry", "vehicle", "tire_stiffness"],
            self.pushrod_editor.rocker_corners)
        notebook.add(self.deformable_editor, text="deformable_members")

        buttons = ttk.Frame(parent)
        buttons.pack(fill="x", padx=(0, 7), pady=(8, 0))
        ttk.Button(buttons, text="解算靜態", command=self.apply).pack(side="left", fill="x", expand=True)

    def build_display(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(1, 5))
        ttk.Label(bar, textvariable=self.result_label, font=("Microsoft JhengHei UI", 11, "bold")).pack(side="left")
        ttk.Button(bar, text="重設視角", command=lambda: self.viewport.reset()).pack(side="right", padx=(4, 0))
        projection = ttk.Combobox(bar, values=("3D", "前視", "側視", "上視"), width=6, state="readonly")
        projection.set("3D")
        projection.pack(side="right")
        projection.bind("<<ComboboxSelected>>", lambda _event: self.viewport.change_view(projection.get()))
        ttk.Button(bar, text="2D 訊號圖", command=self.open_signal_view).pack(side="right", padx=(0, 6))
        self.viewport = MechanismView(parent, self.select_hardpoint)
        self.viewport.pack(fill="both", expand=True)
        self.player = AnimationPlayer(parent, self.refresh_results)
        self.player.pack(fill="x", pady=(6, 0))

        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(8, 4))
        ttk.Label(bar, text="結果輪位").pack(side="left")
        selector = ttk.Combobox(bar, textvariable=self.active_corner, values=hardpoints.CORNERS,
                                width=5, state="readonly")
        selector.pack(side="left", padx=6)
        selector.bind("<<ComboboxSelected>>", lambda _event: self.refresh_results())
        ttk.Button(bar, text="回到靜態", command=self.select_pose).pack(side="left")
        tabs = ttk.Notebook(parent)
        tabs.pack(fill="x")
        summary, points, links, angular, hinges, dampers, loads = (ttk.Frame(tabs, padding=3) for _ in range(7))
        for tab, title in ((summary, "四輪摘要"), (points, "節點運動"), (links, "桿長／誤差"),
                           (angular, "桿件角運動"), (hinges, "A 臂鉸鏈"), (dampers, "避震器"),
                           (loads, "車身接點力")):
            tabs.add(tab, text=title)
        self.summary = make_table(summary, ("輪角", "行程 mm", "轉向角 °", "外傾 °", "輪心 Vz mm/s",
                                            "輪心 Az mm/s²", "Jacobian 條件數"), (56, 83, 88, 83, 115, 125, 130), 4)
        self.point_result = make_table(points, ("節點", "X mm", "Y mm", "Z mm", "Vx mm/s", "Vy mm/s",
                                                 "Vz mm/s", "Ax mm/s²", "Ay mm/s²", "Az mm/s²"),
                                       (62, 67, 67, 67, 80, 80, 80, 83, 83, 83), 5)
        self.link_result = make_table(links, ("端點 A", "端點 B", "靜態長度 mm", "目前長度 mm", "誤差 mm"),
                                      (94, 94, 126, 126, 126), 5)
        self.rod_result = make_table(angular, ("A → B", "轉動來源", "方位 °", "仰角 °", "距靜態 °",
                                              "ω X, Y, Z (rad/s)", "α X, Y, Z (rad/s²)"),
                                     (128, 86, 74, 74, 78, 170, 175), 5)
        ttk.Label(angular, text="方位由 +X 朝 +Y，仰角向上為正。轉向拉桿只顯示桿軸方向的轉動。").pack(
            anchor="w", pady=(3, 0))
        self.hinge_result = make_table(hinges, ("A 臂", "鉸鏈軸 X, Y, Z", "轉角 °", "角速度 rad/s", "角加速度 rad/s²"),
                                       (80, 186, 94, 116, 126), 2)
        ttk.Label(hinges, text="轉角相對靜態零位，正向沿車身前接點指向後接點。").pack(anchor="w", pady=(3, 0))
        self.damper_result = make_table(dampers, ("輪位／元件", "長度 mm", "伸長 mm", "傾角 °", "長度速率 mm/s",
                                                "長度加速度 mm/s²", "彈簧阻尼力 N", "車身端 XYZ 力 (N)", "輪胎正向力 N"),
                                       (128, 75, 72, 58, 111, 127, 90, 184, 98), 4)
        ttk.Label(dampers, text="伸長以靜態為零。傾角相對車身垂直軸。XYZ 為車身座標的彈簧阻尼力。").pack(
            anchor="w", pady=(3, 0))
        self.mount_result = make_table(loads, ("車身接點", "Fx (N)", "Fy (N)", "Fz (N)"),
                                      (162, 121, 121, 121), 5)
        ttk.Label(loads, text="接點施加在車身上的力，以地面座標表示，+Z 向上。").pack(anchor="w", pady=(3, 0))
        ttk.Label(parent, textvariable=self.angular_label, padding=(0, 6, 0, 0)).pack(anchor="w")

    def open_signal_view(self):
        if self.signal_window is not None:
            self.signal_window.deiconify()
            self.signal_window.lift()
            return
        window = tk.Toplevel(self.root)
        window.title("多體幾何 · 2D 訊號圖")
        window.geometry("920x505")
        window.minsize(620, 320)
        self.signal_window = window
        self.signal_view = SignalView(window, self.player.seek_time)
        self.signal_view.pack(fill="both", expand=True)
        self.signal_view.set_geometry(self.model)
        self.signal_view.set_frames(self.player.frames)
        frame = self.player.frame
        self.signal_view.set_time(frame.time if frame is not None else None)
        window.protocol("WM_DELETE_WINDOW", self.close_signal_view)

    def close_signal_view(self):
        self.signal_window.destroy()
        self.signal_window = self.signal_view = None

    def toggle_hardpoints(self):
        if self.hardpoint_mode.get() and self.hardpoint_draft is None and not self.generate_hardpoints():
            self.hardpoint_mode.set(False)
            return
        self.refresh_hardpoints()
        self.mark_dirty()

    def generate_hardpoints(self):
        try:
            points, _, _ = static_geometry(self.read_geometry(direct=False))
        except (ValueError, ArithmeticError) as error:
            self.report_error(error)
            return False
        self.hardpoint_draft = copy.deepcopy(points)
        self.hardpoint_mode.set(True)
        self.refresh_hardpoints()
        self.mark_dirty()
        return True

    def refresh_hardpoints(self):
        source = self.hardpoint_coordinates()
        rows = [(corner, key, *(show_number(value * 1000) for value in source[corner][key]))
                for corner in hardpoints.CORNERS for key in HARDPOINT_LABELS]
        fill_table(self.hardpoint_table, rows)
        self.hardpoint_rows = {row[:2]: item for row, item in zip(rows, self.hardpoint_table.get_children())}
        self.reveal_selected_hardpoint()

    def select_hardpoint(self, hardpoint):
        self.selected_hardpoint = hardpoint
        self.viewport.highlight_hardpoint(hardpoint)
        self.reveal_selected_hardpoint()

    def reveal_selected_hardpoint(self, _event=None):
        row = self.hardpoint_rows.get(self.selected_hardpoint)
        if row is not None and self.editor_tabs.select() == str(self.hardpoint_tab):
            self.hardpoint_table.selection_set(row)
            self.hardpoint_table.focus(row)
            self.hardpoint_table.see(row)

    def hardpoint_coordinates(self):
        direct = self.hardpoint_mode.get()
        source = {corner.name: dict(zip(POINTS, corner.reference)) for corner in self.model.corners}
        for corner, points in source.items():
            if direct and self.hardpoint_draft is not None:
                points.update(self.hardpoint_draft[corner])
            if direct and self.wheel_center_draft is not None:
                points["WC"] = self.wheel_center_draft[corner]
            axle = "front" if corner.startswith("F") else "rear"
            mirror = [1., 1. if corner.endswith("L") else -1., 1.]
            for key, endpoint in STEERING_ENDPOINTS.items():
                points[key] = np.asarray(points["WC"]) + np.multiply(mirror, self.steering_draft[axle][endpoint])
        return source

    def edit_hardpoint(self, event):
        row = self.hardpoint_table.identify_row(event.y)
        if not row:
            return
        corner, key, *_ = self.hardpoint_table.item(row, "values")
        source = self.hardpoint_coordinates()
        current = source[corner][key]
        value = simpledialog.askstring("編輯硬點", f"{corner} · {HARDPOINT_LABELS[key]}\n輸入 X, Y, Z（mm）",
                                       initialvalue=", ".join(f"{v*1000:.12g}" for v in current), parent=self.root)
        if value is None:
            return
        try:
            coordinates = [float(item) / 1000 for item in value.replace("，", ",").split(",")]
            if len(coordinates) != 3 or not np.isfinite(coordinates).all():
                raise ValueError("請輸入三個有限的 XYZ 座標。")
            direct = self.hardpoint_mode.get()
            if key == "WC" and coordinates[1] * (1. if corner.endswith("L") else -1.) <= 0.:
                raise ValueError("左輪輪心 Y 必須大於 0，右輪輪心 Y 必須小於 0。")
            if key in STEERING_ENDPOINTS:
                axle = "front" if corner.startswith("F") else "rear"
                center = source[corner]["WC"]
                mirror = [1., 1. if corner.endswith("L") else -1., 1.]
                self.steering_draft[axle][STEERING_ENDPOINTS[key]] = (
                    (np.asarray(coordinates) - center) * mirror).tolist()
            else:
                arms = self.hardpoint_draft
                if not direct:
                    arms, _, centers = static_geometry(self.read_geometry(direct=False))
                if key in hardpoints.POINTS:
                    hardpoints.set_point(arms, corner, key, coordinates)
                else:
                    reference = {name: point["WC"] for name, point in source.items()} if direct else centers
                    edited = {name: list(map(float, point)) for name, point in reference.items()}
                    edited[corner] = coordinates
                    x, y, z = coordinates
                    edited[hardpoints.MIRROR[corner]] = [x, -y, z]
                    self.wheel_center_draft = edited
                self.hardpoint_draft = arms
                self.hardpoint_mode.set(True)
        except ValueError as error:
            self.report_error(error)
            return
        self.refresh_hardpoints()
        self.mark_dirty()

    def number(self, key, label):
        try:
            return float(self.fields[key].get())
        except ValueError as error:
            raise ValueError(f"{label}需要有效數值。") from error

    def read_geometry(self, *, direct):
        geometry = copy.deepcopy(self.data["geometry"])
        geometry["name"] = self.name.get().strip()
        for key, (label, _unit) in geometry_project.FIELDS["vehicle"].items():
            field = ("geometry", "vehicle", key)
            if field in self.fields:
                geometry["vehicle"][key] = self.number(field, label)
        if self.hardpoint_draft is not None:
            geometry["hardpoints"] = {"enabled": direct, "corners": copy.deepcopy(self.hardpoint_draft)}
        return geometry

    def on_member_edit(self):
        self.mark_dirty()
        self.sync_deformables()

    def sync_deformables(self):
        self.deformable_editor.sync_members(self.element_editor.elements, self.pushrod_editor.direct_records(),
                                            self.pushrod_editor.explicit_corners())

    def draft(self):
        data = copy.deepcopy(self.edit_base)
        data["geometry"] = self.read_geometry(direct=self.hardpoint_mode.get())
        data["steering"] = copy.deepcopy(self.steering_draft)
        if self.wheel_center_draft is not None:
            data["wheel_centers"] = copy.deepcopy(self.wheel_center_draft)
        inertia = {key: self.number(("inertia", "chassis", key), label)
                   for key, (label, _, _) in INERTIA_LIMITS.items()}
        dynamics = data["dynamics"]
        for corner, values in dynamics["corners"].items():
            values["mass_kg"] = self.number(("mass", corner, "mass_kg"), f"{corner} 輪端質量")
        try:
            tire_damping = float(self.deformable_editor.tire_damping.get())
        except ValueError as error:
            raise ValueError("輪胎垂直阻尼需要有效數值。") from error
        dynamics.update(chassis_inertia=inertia, simulation=self.road_editor.draft(),
                        heave=self.heave_editor.draft(), lateral=self.lateral_editor.draft(),
                        drive=self.drive_editor.draft(), tire_damping_ns_m=tire_damping)
        pushrods = self.pushrod_editor.draft()
        for corner, record in pushrods.items():
            if record["inboard"] == "body":
                pushrods[corner] = self.deformable_editor.update_direct(corner, record)
        if pushrods or "pushrods" in data:
            data["pushrods"] = pushrods
        members = self.deformable_editor.draft_members()
        if members or "rocker_dampers" in data:
            data["rocker_dampers"] = members
        if self.deformable_editor.lines or "hydraulic_lines" in data:
            data["hydraulic_lines"] = self.deformable_editor.lines
        elements = self.element_editor.draft()
        for corner, records in elements.items():
            elements[corner] = [self.deformable_editor.update_record(record, corner)
                                if record["kind"] in DEFORMABLE_KINDS and not (corner in pushrods and record["name"] == PUSH_ROD)
                                else record for record in records]
        if dynamics.get("elements") is not None or elements != self.initial_members:
            dynamics["elements"] = elements
        return validate(data)

    def prepare(self, data, *, geometry=None):
        model = build_vehicle(data) if geometry is None else geometry
        dynamics = build_dynamics_model(model, data["geometry"]["vehicle"], data["dynamics"])
        model = dynamics.geometry
        zero = {name: dict.fromkeys(COMPONENTS, 0.) for name in DRIVERS}
        neutral = solve_vehicle(model, zero)
        rocker_states(model, neutral.corners)
        neutral_dampers = sample_dampers(dynamics, neutral)
        neutral_springs = sample_element_springs(dynamics, neutral)
        return PreparedModel(model, dynamics, neutral, neutral_dampers, neutral_springs=neutral_springs)

    def prepare_for_editing(self, data, geometry=None):
        try:
            return self.prepare(data, geometry=geometry)
        except Unreachable as error:
            if "elements" not in data["dynamics"]:
                raise
            preview = {**data, "dynamics": {key: value for key, value in data["dynamics"].items() if key != "elements"}}
            prepared = self.prepare(preview, geometry=geometry)
            return replace(prepared, problem=f"使用基準機構預覽。{error}")

    @staticmethod
    def element_points(prepared):
        points = {}
        for corner in prepared.geometry.corners:
            name = corner.name
            pose = prepared.neutral.corners[name]
            points[name] = dict(zip(POINTS, pose.position))
            if corner.members is None or any(DAMPER_POINT in (member.a, member.b) for member in corner.members):
                points[name][DAMPER_POINT] = prepared.neutral_dampers[name].fixed_point
            if corner.rocker is not None:
                rocker = corner.rocker
                points[name].update(PUSH_OUT=rocker.pickup, PUSH_IN=rocker.pin,
                                    ROCKER_A=rocker.pivot, ROCKER_B=rocker.axis_end,
                                    ROCKER_CENTER=rocker.pin - rocker.radial)
        return points

    def populate(self, prepared):
        geometry = self.data["geometry"]
        self.loading = True
        try:
            self.name.set(geometry["name"])
            self.hardpoint_mode.set(hardpoints.enabled(geometry))
            self.hardpoint_draft = copy.deepcopy(geometry.get("hardpoints", {}).get("corners"))
            self.wheel_center_draft = copy.deepcopy(self.data.get("wheel_centers"))
            self.steering_draft = copy.deepcopy(self.data["steering"])
            dynamics = self.data["dynamics"]
            self.road_editor.populate(dynamics["simulation"])
            self.heave_editor.populate(dynamics["heave"])
            self.lateral_editor.populate(dynamics["lateral"])
            self.drive_editor.populate(dynamics["drive"])
            self.initial_members = dynamics.get("elements")
            if self.initial_members is None:
                self.initial_members = default_elements(csv_mechanism=prepared.geometry.csv_mechanism)
                for corner in prepared.geometry.corners:
                    if corner.members is not None:
                        self.initial_members[corner.name] = [asdict(member) for member in corner.members]
            self.element_editor.populate(self.initial_members, self.element_points(prepared),
                                         (corner.name for corner in prepared.geometry.corners if corner.rocker is not None))
            pushrods = self.data.get("pushrods", {})
            supplemental = geometry.get(hardpoints.KINEMATICS_KEY, {}).get("corners", {})
            fallback = {}
            legacy_rockers = set()
            for corner in prepared.geometry.corners:
                name = corner.name
                if name not in pushrods and corner.rocker is not None:
                    fallback[name] = pushrod_from_hardpoints(supplemental.get(name, {}))
                    legacy_rockers.add(name)
                else:
                    spec = prepared.dynamics.corners[name]
                    fallback[name] = {"inboard": "body", "outboard": "UCA",
                                      "PUSH_IN": spec.fixed_point, "PUSH_OUT": corner.reference[spec.attachment_index]}
            self.pushrod_editor.populate(pushrods, fallback, legacy_rockers)
            self.deformable_editor.populate(dynamics, self.data.get("rocker_dampers", ()), self.initial_members,
                                           self.pushrod_editor.direct_records(), self.pushrod_editor.explicit_corners(),
                                           self.data.get("hydraulic_lines"))
            for key, var in self.fields.items():
                group, channel, field = key
                if group == "geometry":
                    value = geometry[channel][field]
                elif group == "inertia":
                    value = self.data["dynamics"]["chassis_inertia"][field]
                elif group == "mass":
                    value = dynamics["corners"][channel][field]
                var.set(str(value))
        finally:
            self.loading = False

    def show_prepared(self, prepared):
        self.revision += 1
        self.player.clear()
        if self.signal_view is not None:
            self.signal_view.set_frames(())
        self.model, self.neutral = prepared.geometry, prepared.neutral
        if self.signal_view is not None:
            self.signal_view.set_geometry(self.model)
        self.neutral_dampers = prepared.neutral_dampers
        self.neutral_springs = prepared.neutral_springs
        self.element_editor.set_points(self.element_points(prepared),
                                       tuple(corner.name for corner in prepared.geometry.corners if corner.rocker is not None))
        self.pose_mode.set("static")
        self.refresh_results()
        self.refresh_hardpoints()
        mode = "硬點模式" if hardpoints.enabled(self.data["geometry"]) else "懸吊參數模式"
        if prepared.problem:
            self.status.set(prepared.problem)
        elif prepared.dynamics.unbalanced_corners:
            names = "、".join(prepared.dynamics.unbalanced_corners)
            self.status.set(f"{mode} · {names} 的變形元件無法平衡匯入姿態的輪荷。")
        else:
            self.status.set(f"{mode} · X 向前、Y 向左、Z 向上 · 拖曳旋轉，滾輪縮放")

    def apply(self):
        try:
            data = self.draft()
            prepared = self.prepare(data)
        except (ValueError, ArithmeticError) as error:
            self.report_error(error)
            return False
        self.data = data
        self.show_prepared(prepared)
        return prepared.problem is None

    def install(self, data, path, *, unsaved=False, load_hydraulics=True):
        data = validate(data)
        geometry = build_vehicle(data)
        data["dynamics"] = with_axle_pushrod_mounts(geometry, data["dynamics"])
        prepared = self.prepare_for_editing(data, geometry)
        if self.pending_edit is not None:
            self.after_cancel(self.pending_edit)
            self.pending_edit = None
        self.data = self.edit_base = data
        self.path, self.dirty = path, unsaved
        self.selected_hardpoint = self.viewport.selected_hardpoint = None
        self.hardpoint_table.selection_remove(*self.hardpoint_table.selection())
        self.populate(prepared)
        self.edit_history = [self.edit_snapshot()]
        self.edit_index = 0
        self.saved_edit = None if unsaved else self.edit_history[0]
        self.update_caption()
        self.show_prepared(prepared)
        if load_hydraulics:
            self.hydraulic_source = data
            if self.hydraulic_workbench is not None:
                self.hydraulic_workbench.load_vehicle_project(data)

    def start_dynamics(self):
        self.start_run("road")

    def start_heave(self):
        self.start_run("heave")

    def start_run(self, mode):
        if self.animation_job is not None:
            return
        try:
            data = self.draft()
            prepared = self.prepare(data)
        except (ValueError, ArithmeticError) as error:
            self.report_error(error)
            return
        self.player.pause()
        self.animation_job = SimpleQueue()
        self.set_run_buttons("disabled")
        messages = {"heave": "正在解算車身與四輪的垂直運動…",
                    "road": "正在解算路面擾動下的車身平移、轉動與輪端運動…",
                    "lateral": "正在解算重心側向力下的車身姿態與輪端運動…",
                    "drive": "正在聯立馬達、輪胎與車身運動…"}
        self.status.set(messages[mode])
        job, revision = self.animation_job, self.revision
        Thread(target=solve_run_job, args=(job, prepared.dynamics, data, mode), daemon=True).start()
        self.after(25, self.finish_animation, revision, data, prepared, mode)

    def set_run_buttons(self, state):
        self.road_editor.start_button.configure(state=state)
        self.heave_editor.start_button.configure(state=state)
        self.lateral_editor.start_button.configure(state=state)
        self.drive_editor.start_button.configure(state=state)

    def finish_animation(self, revision, data, prepared, mode):
        try:
            frames, error = self.animation_job.get_nowait()
        except Empty:
            self.after(25, self.finish_animation, revision, data, prepared, mode)
            return
        self.animation_job = None
        self.set_run_buttons("normal")
        if revision != self.revision:
            if self.status.get().startswith("正在"):
                self.status.set("設定已變更，請重新解算。")
            return
        if error is not None:
            self.report_error(error)
            return
        self.data = data
        self.show_prepared(prepared)
        if mode == "heave":
            label = "重心受力 · Heave"
        elif mode == "lateral":
            label = "重心側向力"
        elif mode == "drive":
            label = "馬達行駛 · Pacejka"
        else:
            target = TARGET_LABELS[data["dynamics"]["simulation"]["target"]]
            label = f"路面擾動 · {target}"
        self.player.load(frames, False, label)
        if self.signal_view is not None:
            self.signal_view.set_frames(frames)
            self.signal_view.set_time(frames[0].time)
        self.player.toggle()

    def select_pose(self):
        self.player.leave()
        self.status.set("X 向前、Y 向左、Z 向上 · 拖曳旋轉，滾輪縮放")
        self.refresh_results()

    def refresh_results(self):
        frame = self.player.frame
        if frame is not None:
            if self.pose_mode.get() != "animation":
                self.status.set("3D、車身姿態與接點力採地面座標，節點與桿件運動採車身座標。"
                                if frame.heave is not None else
                                "3D 以地面為基準，輪端結果採車身座標。拖曳時間軸可暫停檢查。")
            self.pose_mode.set("animation")
            self.pose = frame.result
            self.result_label.set(self.player.preset_label)
        else:
            self.pose_mode.set("static")
            self.pose = self.neutral
            self.result_label.set("靜態全車")
        if self.pose is None:
            return
        dampers = frame.dampers if frame is not None else self.neutral_dampers
        springs = frame.element_springs if frame is not None else self.neutral_springs
        rocker_readings = {name: reading for readings in (springs or {}).values()
                           for name, reading in readings.items() if reading.endpoints is not None}
        self.viewport.set_result(self.model, self.pose, frame, dampers, rocker_readings,
                                 self.player.frames if frame is not None and frame.drive is not None else (),
                                 hardpoints_clickable=frame is None)
        if self.signal_view is not None:
            self.signal_view.set_time(frame.time if frame is not None else None)
        wheel = POINTS.index("WC")
        rows = []
        for corner in self.model.corners:
            p = self.pose.corners[corner.name]
            jounce = (p.position[wheel, 2] - corner.reference[wheel][2]) * 1000
            numbers = (jounce, p.toe_deg, p.camber_deg, p.velocity[wheel, 2] * 1000,
                       p.acceleration[wheel, 2] * 1000, p.condition)
            rows.append((corner.name, *map(show_number, numbers)))
        fill_table(self.summary, rows)
        name = self.active_corner.get()
        corner = next(item for item in self.model.corners if item.name == name)
        pose = self.pose.corners[name]
        rigid_links = pose.rigid_links if corner.members is not None else LINKS
        rigid_lengths = pose.rigid_lengths if corner.members is not None else corner.lengths
        fill_table(self.point_result, [
            (point, *(show_number(component * 1000) for vec in (position, velocity, acceleration) for component in vec))
            for point, position, velocity, acceleration in zip(POINTS, pose.position, pose.velocity, pose.acceleration)
        ])
        fill_table(self.link_result, [
            (a, b, show_number(length * 1000), show_number((length + error) * 1000), f"{error * 1000:.2e}")
            for (a, b), length, error in zip(rigid_links, rigid_lengths, pose.length_error)
        ])
        rod_names = ({link: f"{link[0]} → {link[1]}" for link in LINKS} if corner.members is None
                     else {(member.a, member.b): member.name for member in corner.members})
        fill_table(self.rod_result, [
            (rod_names[link], ROD_ROTATION_LABELS[rod.rotation_model],
             "—" if rod.azimuth_deg is None else show_number(rod.azimuth_deg),
             show_number(rod.elevation_deg), show_number(rod.from_static_deg),
             show_vector(rod.angular_velocity), show_vector(rod.angular_acceleration))
            for link, rod in pose.rod_motion.items() if link in rod_names
        ])
        fill_table(self.hinge_result, [
            (HINGE_LABELS[name], show_vector(hinge.axis), show_number(hinge.angle_deg),
             show_number(hinge.velocity), show_number(hinge.acceleration))
            for name, hinge in pose.arm_motion.items()
        ])
        tire_only = {item.name for item in self.model.corners
                     if item.rocker is not None or item.pushrod_configured or self.model.csv_mechanism}
        damper_rows = [
            (name, show_number(1000 * element.length), show_number(1000 * element.deflection),
             show_number(element.inclination_deg), show_number(1000 * element.velocity),
             show_number(1000 * element.acceleration), show_number(element.force),
             show_vector(element.force_on_chassis),
             "—" if element.tire_force is None else show_number(element.tire_force))
            for name, element in (dampers or {}).items() if name not in tire_only
        ]
        axial_readings = [(name, reading) for name, reading in rocker_readings.items()]
        axial_readings.extend((f"{corner} · {name}", reading) for corner, readings in (springs or {}).items()
                              for name, reading in readings.items()
                              if reading.endpoints is None and (name != PUSH_ROD or corner in tire_only))
        for name, reading in axial_readings:
            damper_rows.append((name, show_number(1000 * reading.length), show_number(1000 * reading.deflection),
                                "—", show_number(1000 * reading.velocity), show_number(1000 * reading.acceleration),
                                show_number(reading.force), "—", "—"))
        for name, element in (dampers or {}).items():
            if name in tire_only and element.tire_force is not None:
                damper_rows.append((f"{name} 輪胎", *("—",) * 7, show_number(element.tire_force)))
        fill_table(self.damper_result, damper_rows)
        mounts = frame.mount_loads[name] if frame is not None and frame.mount_loads else None
        mount_rows = []
        if mounts is not None:
            mount_rows = [(MOUNT_LABELS[key], *map(show_number, force)) for key, force in mounts.by_mount.items()
                          if key != DAMPER_POINT or name not in tire_only or any(force)]
            mount_rows.append(("本輪接點合力", *map(show_number, mounts.total)))
        fill_table(self.mount_result, mount_rows)
        omega = show_vector(pose.angular_velocity)
        alpha = show_vector(pose.angular_acceleration)
        self.angular_label.set(f"{name} 羊角角速度 ({omega}) rad/s    角加速度 ({alpha}) rad/s²")

    def report_error(self, error):
        self.status.set(str(error))
        messagebox.showerror("無法套用設定", str(error), parent=self.root)

    def confirm_discard(self):
        self.flush_edit()
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("未儲存設定", "是否先儲存目前設定？", parent=self.root)
        return self.save() if answer is True else answer is False

    def open_file(self):
        file = filedialog.askopenfilename(parent=self.root, title="開啟專案",
                                          initialdir=IMPORTS if IMPORTS.is_dir() else PROJECTS,
                                          filetypes=[("JSON 專案", "*.json")])
        if not file or not self.confirm_discard():
            return
        try:
            path = Path(file)
            data, native = read_project(path)
            hydraulic = self.hydraulic_workbench
            if hydraulic is not None and hydraulic.has_circuit_edits() and not messagebox.askyesno(
                "開啟專案", "開啟 JSON 會以新專案的設定取代目前液壓設計。要繼續嗎？", parent=self.root
            ):
                return
            self.install(data, path if native else None)
        except (ValueError, ArithmeticError, OSError) as error:
            self.report_error(error)

    def import_csv(self):
        file = filedialog.askopenfilename(parent=self.root, title="匯入硬點 CSV",
                                          initialdir=IMPORTS if IMPORTS.is_dir() else PROJECTS,
                                          filetypes=[("硬點 CSV", "*.csv")])
        if not file:
            return
        try:
            self.install(import_hardpoint_csv(self.draft(), file), None, unsaved=True, load_hydraulics=False)
        except (ValueError, ArithmeticError, OSError) as error:
            self.report_error(error)
        else:
            self.status.set(f"已匯入 {Path(file).name} · {self.status.get()}")

    def save(self, save_as=False):
        path = self.path
        if path is None or save_as:
            file = filedialog.asksaveasfilename(parent=self.root, title="儲存專案", initialdir=PROJECTS,
                                                defaultextension=".json", initialfile=path.name if path else "",
                                                filetypes=[("JSON 專案", "*.json")])
            if not file:
                return False
            path = Path(file)
        try:
            data = self.draft()
            prepared = self.prepare_for_editing(data)
            save_project(path, data)
        except (ValueError, ArithmeticError, OSError) as error:
            self.report_error(error)
            return False
        self.flush_edit()
        self.saved_edit = self.edit_snapshot()
        self.data = data
        self.path, self.dirty = path, False
        self.update_caption()
        self.show_prepared(prepared)
        self.status.set(f"已儲存 {path.name} · {self.status.get()}")
        return True

    def close(self):
        if self.confirm_discard():
            self.root.destroy()
