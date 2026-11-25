"""Vehicle attitude, contact loads and CG motion on the animation timeline."""

import tkinter as tk

import numpy as np

from ._geometry_support.hardpoints import CORNERS
from .rigid_body import _angles_deg
from .signal_plot import AXIS, BACKGROUND, CURSOR, GRID, TEXT, finite_runs, number


CHARTS = (
    ("車身姿態 (°)", ("Roll", "Pitch")),
    ("四輪正向輪荷 (N)", CORNERS),
    ("重心平面車速 (km/h)", ("車速",)),
    ("重心加速度 (m/s²) · 地面座標", ("Ax", "Ay", "Az")),
)
COLORS = ("#56c6ff", "#ffac76", "#65d5bd", "#cc9cef")


def vehicle_readings(frame, dampers):
    loads = []
    for name in CORNERS:
        reading = (dampers or {}).get(name)
        load = reading.tire_force if reading is not None else None
        loads.append(load if load is not None else np.nan)
    if frame is None:
        return (0., 0.), loads, (0.,), (0., 0., 0.)
    if frame.body is not None:
        body = frame.body
        return body.angles_deg[:2], loads, (body.planar_speed * 3.6,), body.acceleration
    angles = _angles_deg(frame.rotation)[:2]
    if frame.heave is not None:
        return angles, loads, (0.,), (0., 0., frame.heave.acceleration)
    return angles, loads, (np.nan,), (np.nan,) * 3


class VehicleTelemetryView(tk.Canvas):
    def __init__(self, parent, seek):
        super().__init__(parent, background=BACKGROUND, highlightthickness=0)
        self.seek = seek
        self.frames = ()
        self.times = np.array([])
        self.series = ()
        self.time = None
        self.plots = []
        self.bind("<Configure>", self.draw)
        self.bind("<Button-1>", self.scrub)
        self.bind("<B1-Motion>", self.scrub)

    def set_frames(self, frames):
        if frames is self.frames:
            return
        self.frames = frames
        self.times = np.asarray([frame.time for frame in frames])
        readings = [vehicle_readings(frame, frame.dampers) for frame in frames]
        self.series = tuple(np.asarray(group).T for group in zip(*readings))
        self.time = None
        self.draw()

    def set_time(self, time):
        if time != self.time:
            self.time = time
            self.draw_cursor()

    def label(self, x, y, content, *, anchor="center", color=TEXT, size=9):
        self.create_text(x, y, text=content, anchor=anchor, fill=color, font=("Microsoft JhengHei UI", size))

    def draw(self, _event=None):
        self.delete("all")
        self.plots.clear()
        width, height = self.winfo_width(), self.winfo_height()
        if width < 450 or height < 260:
            return
        if not self.frames:
            self.label(width / 2, height / 2, "完成模擬後顯示車輛時序資料", size=12)
            return
        for index, ((title, names), series) in enumerate(zip(CHARTS, self.series)):
            column, row = index % 2, index // 2
            left = column * width / 2 + 62
            right = (column + 1) * width / 2 - 19
            top = row * height / 2 + 58
            bottom = (row + 1) * height / 2 - 40
            self.draw_plot(title, names, series, left, top, right, bottom)
        self.draw_cursor()

    def draw_plot(self, title, names, series, left, top, right, bottom):
        self.label(left, top - 43, title, anchor="w", size=11)
        for index, (name, color) in enumerate(zip(names, COLORS)):
            x = left + index * 57
            self.create_line(x, top - 19, x + 14, top - 19, fill=color, width=2)
            self.label(x + 18, top - 19, name, anchor="w")
        finite = series[np.isfinite(series)]
        if not len(finite):
            self.label((left + right) / 2, (top + bottom) / 2, "此工況無資料")
            return
        low, high = float(finite.min()), float(finite.max())
        margin = (high - low) * .08 if high > low else max(abs(low) * .05, .1)
        low, high = low - margin, high + margin
        start, end = self.times[0], max(self.times[-1], self.times[0] + 1e-9)
        self.plots.append((left, top, right, bottom))
        xs = np.interp(self.times, (start, end), (left, right))
        for value in np.linspace(low, high, 4):
            y = np.interp(value, (low, high), (bottom, top))
            self.create_line(left, y, right, y, fill=GRID)
            self.label(left - 7, y, number(value), anchor="e")
        for time in np.linspace(start, end, 5):
            x = np.interp(time, (start, end), (left, right))
            self.create_line(x, top, x, bottom, fill=GRID)
            self.label(x, bottom + 12, number(time))
        self.create_line(left, top, left, bottom, right, bottom, fill=AXIS)
        self.label((left + right) / 2, bottom + 28, "時間 (s)")
        for values, color in zip(series, COLORS):
            ys = np.interp(values, (low, high), (bottom, top))
            for first, last in finite_runs(values):
                if last - first > 1:
                    points = np.column_stack((xs[first:last], ys[first:last]))
                    self.create_line(*points.ravel(), fill=color, width=2)
                else:
                    x, y = xs[first], ys[first]
                    self.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")

    def draw_cursor(self):
        self.delete("cursor")
        if self.time is None:
            return
        for left, top, right, bottom in self.plots:
            x = np.interp(self.time, (self.times[0], max(self.times[-1], self.times[0] + 1e-9)), (left, right))
            self.create_line(x, top, x, bottom, fill=CURSOR, dash=(4, 3), tags="cursor")

    def scrub(self, event):
        for left, top, right, bottom in self.plots:
            if left <= event.x <= right and top <= event.y <= bottom:
                self.seek(float(np.interp(event.x, (left, right), (self.times[0], self.times[-1]))))
                return
