"""Tk canvas for the mechanism with a stationary ground reference."""

import tkinter as tk
import numpy as np

from .elements import DAMPER_POINT
from .geometry import ARM_LINKS, LINKS, POINTS
from .rocker import rocker_states


INDEX = {name: i for i, name in enumerate(POINTS)}
PICK_RADIUS = 8


class MechanismView(tk.Canvas):
    def __init__(self, parent, on_hardpoint=None):
        super().__init__(parent, background="#101d2b", highlightthickness=0, height=360)
        self.model = self.result = self.drag = None
        self.frame = None
        self.drive_frames = ()
        self.on_hardpoint = on_hardpoint
        self.hardpoints_clickable = False
        self.selected_hardpoint = None
        self.hardpoint_pixels = {}
        self.press_origin = None
        self.dampers = {}
        self.rocker_readings = {}
        self.view = "3D"
        self.azimuth, self.elevation, self.zoom = .72, .48, 1.
        self.bind("<Configure>", self.draw)
        self.bind("<ButtonPress-1>", self.start_drag)
        self.bind("<B1-Motion>", self.rotate)
        self.bind("<ButtonRelease-1>", self.finish_drag)
        self.bind("<MouseWheel>", self.wheel)
        self.bind("<Double-Button-1>", self.double_click)

    def set_result(self, model, result, frame=None, dampers=None, rocker_readings=None, drive_frames=(),
                   *, hardpoints_clickable=False):
        self.model, self.result, self.frame = model, result, frame
        self.drive_frames = drive_frames
        self.dampers = dampers or {}
        self.rocker_readings = rocker_readings or {}
        self.hardpoints_clickable = hardpoints_clickable and self.on_hardpoint is not None
        self.draw()

    def change_view(self, view):
        self.view = view
        self.reset()

    def reset(self):
        self.azimuth, self.elevation, self.zoom = .72, .48, 1.
        self.draw()

    def start_drag(self, event):
        self.press_origin = event.x, event.y
        self.drag = None

    def rotate(self, event):
        if self.press_origin is not None and self.moved(event):
            self.drag = self.press_origin
            self.press_origin = None
        if self.view == "3D" and self.drag is not None:
            x, y = self.drag
            self.azimuth += (event.x - x) * .009
            self.elevation = float(np.clip(self.elevation + (event.y - y) * .009, -1.55, 1.55))
            self.drag = event.x, event.y
            self.draw()

    def moved(self, event):
        x, y = self.press_origin
        return (event.x - x)**2 + (event.y - y)**2 > 16

    def pick_candidates(self, event):
        distances = [(float((x - event.x)**2 + (y - event.y)**2), key)
                     for key, (x, y) in self.hardpoint_pixels.items()]
        nearest = min((distance for distance, _ in distances), default=float("inf"))
        if nearest > PICK_RADIUS**2:
            return []
        # Exact overlaps occur in orthographic projections. Repeated clicks
        # at the same spot cycle through those points.
        return [key for distance, key in sorted(distances) if distance <= min(nearest + 1., PICK_RADIUS**2)]

    def finish_drag(self, event):
        is_click = self.press_origin is not None and not self.moved(event)
        self.press_origin = self.drag = None
        if is_click:
            candidates = self.pick_candidates(event)
            if candidates:
                selected = self.selected_hardpoint
                index = candidates.index(selected) + 1 if selected in candidates else 0
                self.on_hardpoint(candidates[index % len(candidates)])

    def double_click(self, event):
        if not self.pick_candidates(event):
            self.reset()
        self.start_drag(event)

    def highlight_hardpoint(self, hardpoint):
        self.selected_hardpoint = hardpoint
        self.draw()

    def wheel(self, event):
        self.zoom = float(np.clip(self.zoom * (1.12 ** (event.delta / 120)), .55, 5.))
        self.draw()

    def draw(self, _event=None):
        self.delete("all")
        self.hardpoint_pixels.clear()
        if self.result is None:
            return
        if self.view == "前視":
            axes = np.array([[0., -1., 0.], [0., 0., 1.]])
        elif self.view == "側視":
            axes = np.array([[1., 0., 0.], [0., 0., 1.]])
        elif self.view == "上視":
            axes = np.array([[0., -1., 0.], [1., 0., 0.]])
        else:
            az, el = self.azimuth, self.elevation
            axes = np.array([[np.cos(az), -np.sin(az), 0.],
                             [np.sin(el)*np.sin(az), np.sin(el)*np.cos(az), np.cos(el)]])

        segments, labels = [], []

        def add(points, color, width=1, dash=()):
            segments.append((np.asarray(points), color, width, dash))

        x0 = min(c.reference[INDEX["WC"]][0] for c in self.model.corners) - .30
        road_half_width = max(.8, *(abs(c.reference[INDEX["WC"]][1]) + c.wheel_width / 2 + .08
                                     for c in self.model.corners))
        for y in np.linspace(-road_half_width, road_half_width, 9):
            add([(x0, y, 0), (.35, y, 0)], "#253a4c")
        for x in np.linspace(x0, .35, 11):
            add([(x, -road_half_width, 0), (x, road_half_width, 0)], "#253a4c")
        add([(x0, 0., 0.), (.35, 0., 0.)], "#5c7085", dash=(4, 5))
        ground_segment_count = len(segments)
        for corner in self.model.corners:
            element = self.dampers.get(corner.name)
            if element is not None and element.road_height:
                x, y, _ = corner.reference[INDEX["WC"]]
                z, w = element.road_height, corner.wheel_width * .62
                add([(x-.13, y-w, z), (x+.13, y-w, z), (x+.13, y+w, z),
                     (x-.13, y+w, z), (x-.13, y-w, z)], "#cc9cef", 2)

        if self.frame is not None:
            for edge in self.model.chassis_edges:
                add(edge, "#43586c", 1, (4, 3))
        first_vehicle_segment = len(segments)
        for edge in self.model.chassis_edges:
            add(edge, "#d7e6f5" if self.frame is not None else "#60778b", 2)

        theta = np.linspace(0, 2 * np.pi, 49)
        rockers = rocker_states(self.model, self.result.corners)
        for corner in self.model.corners:
            pose = self.result.corners[corner.name]
            positions = pose.position
            color = "#56c6ff" if corner.axle == "front" else "#ffac76"
            element = self.dampers.get(corner.name) if corner.members is None else None
            if element is not None:
                upper_mount = (positions[INDEX["UCA_F"]] + positions[INDEX["UCA_R"]]) / 2
                add([upper_mount, element.fixed_point], "#8aa1b5", 1.3)
            if corner.members is None:
                for a, b in ARM_LINKS + (("UCA_F", "UCA_R"), ("LCA_F", "LCA_R"), ("UBJ", "LBJ"), ("LBJ", "TOE")):
                    add(positions[[INDEX[a], INDEX[b]]], color, 2)
                add(positions[[INDEX[a] for a in LINKS[-1]]], "#d7e87f", 2)
                if element is not None:
                    add(coil_line(element.fixed_point, element.moving_point), "#f0bb68", 2)
            else:
                points = dict(zip(POINTS, positions))
                points[DAMPER_POINT] = np.asarray(corner.body_mount)
                for member in corner.members:
                    if member.hydraulic is not None:
                        continue
                    a, b = points[member.a], points[member.b]
                    if member.kind == "spring_damper":
                        add(coil_line(a, b), "#f0bb68", 2)
                    elif member.kind == "hydraulic":
                        add([a, b], "#f0bb68", 4)
                    else:
                        add([a, b], color, 2)
            if corner.rocker is not None:
                rocker = rockers[corner.name]
                add([rocker.pickup, rocker.pin], color, 2)
                add([rocker.rocker.pivot, rocker.pin], "#8aa1b5", 2)
            center = positions[INDEX["WC"]]
            axis = pose.wheel_axis
            forward = np.cross(axis, [0, 0, 1])
            length = np.linalg.norm(forward)
            forward = forward / length if length > 1e-9 else np.array([1., 0., 0.])
            upward = np.cross(forward, axis)
            radial = np.cos(theta)[:, None] * upward + np.sin(theta)[:, None] * forward
            for side in (-1, 1):
                add(center + corner.wheel_radius * radial + side * corner.wheel_width / 2 * axis,
                    color, 1.35)
            if corner.members is None:
                add([positions[INDEX["LBJ"]], center], color, 1, (3, 3))
            labels.append((center + [0., 0., corner.wheel_radius + .035], corner.name, color))
        for damper in self.model.rocker_dampers:
            reading = self.rocker_readings.get(damper.name)
            if reading is None:
                continue
            for endpoint, position in zip(damper.endpoints, reading.endpoints):
                if endpoint.pickup is None and endpoint.mount in rockers:
                    add([rockers[endpoint.mount].rocker.pivot, position], "#8aa1b5", 2)
            if damper.is_hydraulic:
                add(reading.endpoints, "#f0bb68", 4)
            else:
                add(coil_line(*reading.endpoints), "#f0bb68", 2)
        for left, right in (("FL", "FR"), ("RL", "RR")):
            add([self.result.corners[name].position[INDEX["RACK"]] for name in (left, right)],
                "#d7e87f", 1, (4, 4))
        cg = self.frame.pivot if self.frame is not None and self.frame.body is not None else self.model.cg
        labels.append((cg, "CG", "#e5edf7"))

        if self.frame is not None:
            frame = self.frame

            def to_ground(points):
                translation = frame.translation
                if frame.drive is not None:
                    translation = (0., 0., translation[2])
                return (np.asarray(points) - frame.pivot) @ frame.rotation.T + frame.pivot + translation

            # Fit once to the neutral geometry in the current view, so the
            # ground does not pan or zoom as the frame changes.
            anchors = [points for points, *_ in segments[:ground_segment_count]]
            anchors.extend(np.asarray(edge) for edge in self.model.chassis_edges)
            for corner in self.model.corners:
                center = np.asarray(corner.reference[INDEX["WC"]])
                half_extents = np.array([corner.wheel_radius, corner.wheel_width / 2, corner.wheel_radius]) + .08
                anchors.append(np.array([center + half_extents * [x, y, z]
                                         for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]))
            anchors.extend(element.fixed_point[None, :] for element in self.dampers.values())
            anchors.extend(np.array([endpoint.xyz for endpoint in damper.endpoints])
                           for damper in self.model.rocker_dampers)
            projected = np.vstack(anchors) @ axes.T
            segments[first_vehicle_segment:] = [(to_ground(points), color, weight, dash)
                                                for points, color, weight, dash in segments[first_vehicle_segment:]]
            labels = [(to_ground(point), name, color) for point, name, color in labels]
        else:
            projected = np.vstack([points @ axes.T for points, *_ in segments] +
                                  [np.array([point for point, *_ in labels]) @ axes.T])
        low, high = projected.min(axis=0), projected.max(axis=0)
        width, height = self.winfo_width(), self.winfo_height()
        scale = max(1., min((width-76) / max(high[0]-low[0], .01),
                            (height-76) / max(high[1]-low[1], .01))) * self.zoom
        mid = (high + low) / 2
        origin = np.array([width/2, height/2])

        def project(points):
            return origin + (np.asarray(points) @ axes.T - mid) * [scale, -scale]

        for points, color, weight, dash in segments:
            self.create_line(*project(points).ravel(), fill=color, width=weight, dash=dash)
        for point, name, color in labels:
            x, y = project(point)
            if name == "CG":
                self.create_oval(x-3, y-3, x+3, y+3, fill=color, outline="")
                x += 16
            self.create_text(x, y, text=name, fill=color, font=("Segoe UI", 10, "bold"))
        if self.hardpoints_clickable:
            self.draw_hardpoints(project)
        self.create_text(14, 14, anchor="nw", text="前軸  /  後軸  /  轉向拉桿", fill="#b4cadc",
                         font=("Microsoft JhengHei UI", 9))
        if self.dampers:
            self.create_text(14, height-12, anchor="sw", text="彈簧阻尼", fill="#f0bb68",
                             font=("Microsoft JhengHei UI", 9))
        if self.frame is not None:
            heave = self.frame.heave
            body = self.frame.body
            if body is not None:
                roll, pitch, yaw = body.angles_deg
                readout = (f"簧上重心 ΔZ {body.displacement[2] * 1000:+.1f} mm"
                           f"\nRoll {roll:+.2f}°  Pitch {pitch:+.2f}°  Yaw {yaw:+.2f}°")
                if self.frame.drive is not None:
                    drive = self.frame.drive
                    readout += (f"\n車速 {body.planar_speed * 3.6:.1f} km/h"
                                f"    馬達 {drive.motor_torque_nm:.1f} N·m / {drive.motor_rpm:.0f} rpm")
            elif heave is not None:
                readout = (f"重心 ΔZ {heave.displacement * 1000:+.1f} mm    "
                           f"重心外力 Fz {heave.force_z:+.0f} N    (+Z 向上)")
            else:
                rotation = self.frame.rotation
                height_change = self.frame.translation[2] * 1000
                roll = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
                readout = f"Heave {height_change:+.1f} mm    Roll {roll:+.2f}°"
            if self.frame.mount_loads:
                mount_z = sum(load.total[2] for load in self.frame.mount_loads.values())
                readout += f"\n四輪接點合力 Fz {mount_z:+.0f} N"
            self.create_text(width-14, 14, anchor="ne", text=readout, justify="right",
                             fill="#d7e6f5", font=("Segoe UI", 9))
        if self.drive_frames and width > 360 and height > 240:
            self.draw_drive_path(width, height)

    def draw_hardpoints(self, project):
        for corner, pose in self.result.corners.items():
            for name, (x, y) in zip(POINTS, project(pose.position)):
                self.hardpoint_pixels[corner, name] = (x, y)
                self.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#101d2b", outline="#a1bed4")
        if self.selected_hardpoint in self.hardpoint_pixels:
            x, y = self.hardpoint_pixels[self.selected_hardpoint]
            self.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#ffd36b", width=2)
            self.create_text(x + 9, y - 9, anchor="sw", text=" · ".join(self.selected_hardpoint),
                             fill="#ffd36b", font=("Segoe UI", 9, "bold"))

    def draw_drive_path(self, width, height):
        origin = np.array([width - 202., height - 120.])
        dimensions = np.array([188., 106.])
        self.create_rectangle(*origin, *(origin + dimensions), fill="#101d2b", outline="#43586c")
        self.create_text(*(origin + [9., 8.]), anchor="nw", text="俯視軌跡 · +X 向右 / +Y 向上",
                         fill="#d7e6f5", font=("Microsoft JhengHei UI", 8))
        path = np.array([frame.translation[:2] for frame in self.drive_frames])
        low, high = path.min(axis=0), path.max(axis=0)
        scale = min((dimensions - [22., 34.]) / np.maximum(high - low, .1))
        middle = origin + dimensions / 2 + [0., 9.]
        points = middle + (path - (high + low) / 2) * [scale, -scale]
        self.create_line(*points.ravel(), fill="#43586c", width=2)
        current = middle + (self.frame.translation[:2] - (high + low) / 2) * [scale, -scale]
        self.create_oval(*(current - 4), *(current + 4), fill="#ffac76", outline="")


def coil_line(start, end):
    delta = end - start
    axis = delta / np.linalg.norm(delta)
    reference = (1., 0., 0.) if abs(axis[0]) < .95 else (0., 1., 0.)
    side = np.cross(axis, reference)
    side *= .013 / np.linalg.norm(side)
    fractions = np.linspace(.16, .84, 13)
    offsets = np.r_[0., np.tile([1., -1.], 5), 1., 0.]
    return np.vstack((start, start + fractions[:, None] * delta + offsets[:, None] * side, end))
