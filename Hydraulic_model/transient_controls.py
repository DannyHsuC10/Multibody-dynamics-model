"""Transient input panel and time-series chart for the workbench."""

import tkinter as tk
from tkinter import ttk

if __package__:
    from .transient_network import TransientSettings
else:
    from transient_network import TransientSettings


FIELDS = (
    ("duration_s", "模擬時間 (s)"),
    ("ramp_s", "輸入上升時間 (s)"),
    ("mass_kg", "每支桿等效質量 (kg)"),
    ("damping_n_s_mm", "每支桿本體阻尼 (N·s/mm)"),
    ("bulk_modulus_mpa", "油液體積模數 (MPa)"),
    ("line_volume_ml", "每路參考油量 (mL)"),
)


class TransientPane(ttk.Frame):
    def __init__(self, parent, simulate):
        super().__init__(parent, padding=12)
        self.simulate = simulate
        self.curves = {}
        self.times = ()
        self.selection = tk.StringVar(self)
        self.message = tk.StringVar(self)
        defaults = TransientSettings()
        self.fields = {name: tk.StringVar(self, str(getattr(defaults, name))) for name, _ in FIELDS}
        self.line_delay = tk.BooleanVar(self, defaults.line_delay_enabled)
        self.line_tau = tk.StringVar(self, str(defaults.line_time_constant_s))

        form = ttk.Frame(self)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        for row, (name, label) in enumerate(FIELDS):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(form, textvariable=self.fields[name], width=12).grid(
                row=row, column=1, sticky="ew", padx=(8, 0), pady=2)
        self.delay_check = ttk.Checkbutton(form, text="油路一階延遲", variable=self.line_delay,
                                           command=self.update_delay_field)
        self.delay_check.grid(row=len(FIELDS), column=0, columnspan=2, sticky="w", pady=(5, 2))
        ttk.Label(form, text="延遲時間常數 τ (s)").grid(
            row=len(FIELDS) + 1, column=0, sticky="w", pady=2)
        self.tau_entry = ttk.Entry(form, textvariable=self.line_tau, width=12, state="disabled")
        self.tau_entry.grid(row=len(FIELDS) + 1, column=1, sticky="ew", padx=(8, 0), pady=2)
        self.start_button = ttk.Button(self, text="執行瞬態模擬", command=self.run)
        self.start_button.pack(fill="x", pady=(8, 6))
        ttk.Label(self, text="輸入從零平滑升至設定值。勾選延遲後可比較同一油路的各油口壓力變化。",
                  foreground="#52667b", wraplength=300).pack(anchor="w")

        self.signal = ttk.Combobox(self, state="disabled", textvariable=self.selection)
        self.signal.pack(fill="x", pady=(8, 4))
        self.signal.bind("<<ComboboxSelected>>", lambda _: self.draw())
        self.chart = tk.Canvas(self, height=130, width=275, background="#f6f8fb", highlightthickness=0)
        self.chart.pack(fill="both", expand=True)
        self.chart.bind("<Configure>", lambda _: self.draw())
        ttk.Label(self, textvariable=self.message, foreground="#17324a", wraplength=300).pack(
            fill="x", anchor="w", pady=(5, 0))
        for field in self.fields.values():
            field.trace_add("write", lambda *_: self.clear())
        self.line_tau.trace_add("write", lambda *_: self.clear())
        self.clear()

    def update_delay_field(self):
        self.tau_entry.configure(state="normal" if self.line_delay.get() else "disabled")
        self.clear()

    def clear(self):
        self.curves = {}
        self.times = ()
        self.signal.configure(values=(), state="disabled")
        self.selection.set("")
        self.message.set("")
        self.draw()

    def run(self):
        try:
            parameters = {key: float(field.get()) for key, field in self.fields.items()}
            if self.line_delay.get():
                parameters.update(line_delay_enabled=True, line_time_constant_s=float(self.line_tau.get()))
            settings = TransientSettings(**parameters)
            result = self.simulate(settings)
        except (ValueError, ArithmeticError) as error:
            self.clear()
            message = str(error)
            if message.startswith("could not convert string"):
                message = "請在瞬態參數欄位輸入數值。"
            self.message.set(message)
            return
        if result is None:
            return

        trace, names = result
        self.times = trace.time_s
        self.curves = {f"{names[rod]} · 壓縮位移": ("mm", -values)
                       for rod, values in trace.rod_extension_mm.items()}
        self.curves.update({f"{name} · 平均油壓變化": ("MPa", values)
                            for name, values in trace.pressure_change_mpa.items()})
        self.curves.update({f"{line} · {names[rod]} {'左側' if side == 'cap' else '右側'}油壓變化": ("MPa", values)
                            for (line, rod, side), values in trace.port_pressure_change_mpa.items()})
        self.curves.update({f"{names[rod]} · 所需壓縮力": ("N", -values)
                            for rod, values in trace.required_rod_forces_n.items()})
        options = tuple(self.curves)
        self.signal.configure(values=options, state="readonly")
        self.selection.set(options[0])
        self.draw()

    def draw(self):
        canvas = self.chart
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 110 or height < 70:
            return
        selected = self.curves.get(self.selection.get())
        if selected is None:
            canvas.create_text(width / 2, height / 2, text="求解後顯示時間曲線", fill="#52667b")
            return
        unit, values = selected
        low, high = float(values.min()), float(values.max())
        scale = max(abs(low), abs(high)) or 1.
        scaled_low, scaled_high = low / scale, high / scale
        padding = .08 * (scaled_high - scaled_low) if high != low else .1
        lower, upper = scaled_low - padding, scaled_high + padding
        left, top, right, bottom = 62, 18, width - 14, height - 29
        duration = self.times[-1]
        for index in range(5):
            fraction = index / 4
            x = left + fraction * (right - left)
            y = bottom - fraction * (bottom - top)
            canvas.create_line(left, y, right, y, fill="#dbe4ee")
            canvas.create_line(x, top, x, bottom, fill="#dbe4ee")
            tick = (lower + fraction * (upper - lower)) * scale
            canvas.create_text(left - 4, y, text=f"{tick:.3g}",
                               anchor="e", fill="#52667b", font=("Segoe UI", 8))
            canvas.create_text(x, bottom + 8, text=f"{fraction * duration:.3g}",
                               anchor="n", fill="#52667b", font=("Segoe UI", 8))
        canvas.create_text(left - 4, top - 4, text=unit, anchor="se", fill="#52667b", font=("Segoe UI", 8))
        canvas.create_text(right, height - 1, text="s", anchor="se", fill="#52667b", font=("Segoe UI", 8))

        # Preserve extrema in each pixel-scale group so short pressure peaks stay visible.
        stride = max(1, (len(values) + int(right - left) - 1) // int(right - left))
        indices = {0, len(values) - 1}
        for start in range(0, len(values), stride):
            section = values[start:start + stride]
            indices.update((start + int(section.argmin()), start + int(section.argmax())))
        coordinates = []
        for index in sorted(indices):
            x = left + float(self.times[index] / duration) * (right - left)
            y = bottom - float((values[index] / scale - lower) / (upper - lower)) * (bottom - top)
            coordinates.extend((x, y))
        canvas.create_line(*coordinates, width=2, fill="#246d9a")
        self.message.set(f"末值 {values[-1]:+.3g} {unit}　範圍 {low:.3g} ～ {high:.3g} {unit}")
