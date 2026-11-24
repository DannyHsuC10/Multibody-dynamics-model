"""Plot three cylinders sharing P1 and P2 under two rod forces and a spring."""

import argparse

if __package__:
    from .model import GROUND, HydraulicCylinder, LinearSpring, SpringElement
    from .static_network import PressureLine, StaticHydraulicSystem
else:
    from model import GROUND, HydraulicCylinder, LinearSpring, SpringElement
    from static_network import PressureLine, StaticHydraulicSystem


def plot_displacement(compression_mm, f1, f2, stiffness, destination):
    import matplotlib

    if destination is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    figure.suptitle("Three-cylinder static response", fontsize=17, fontweight="bold")
    axis.set_title(f"F1 = {f1:g} N   |   F2 = {f2:g} N   |   K = {stiffness:g} N/mm", fontsize=11)
    colors = ["#23699b" if value >= 0. else "#ab5722" for value in compression_mm]
    bars = axis.bar(("A · F1", "B · F2", "C · spring"), compression_mm, width=.54, color=colors)
    labels = ["0.000" if abs(value) < .0005 else f"{value:+.3f}" for value in compression_mm]
    axis.bar_label(bars, labels=labels, padding=6, fontsize=11, fontweight="bold")
    axis.axhline(0., color="#252c35", linewidth=1)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=.23)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylabel("Rod displacement (mm)")
    axis.set_xlabel("Positive = compression   |   Negative = extension", labelpad=12)
    axis.margins(y=.22)

    if destination is None:
        plt.show()
    else:
        figure.savefig(destination, dpi=170)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f1-n", type=float, default=100., help="Compressive force on A (default: 100 N)")
    parser.add_argument("--f2-n", type=float, default=180., help="Compressive force on B (default: 180 N)")
    parser.add_argument("--spring-n-mm", type=float, default=100., help="Spring stiffness on C (default: 100 N/mm)")
    parser.add_argument("--bores-mm", nargs=3, type=float, metavar=("A", "B", "C"), default=(20., 20., 20.),
                        help="Cylinder bores (default: 20 20 20 mm)")
    parser.add_argument("--rods-mm", nargs=3, type=float, metavar=("A", "B", "C"), default=(8., 16., 12.),
                        help="Rod diameters (default: 8 16 12 mm)")
    plotting = parser.add_mutually_exclusive_group()
    plotting.add_argument("--save-plot", metavar="PATH", help="Export the bar chart instead of opening a window")
    plotting.add_argument("--no-plot", action="store_true", help="Print the static results only")
    args = parser.parse_args()

    try:
        cylinders = tuple(HydraulicCylinder(bore, rod_diameter_mm=rod)
                          for bore, rod in zip(args.bores_mm, args.rods_mm))
        a, b, c = cylinders
        p1 = PressureLine("P1", tuple(cylinder.cap_end for cylinder in cylinders))
        p2 = PressureLine("P2", tuple(cylinder.rod_end for cylinder in cylinders))
        spring = SpringElement(LinearSpring(args.spring_n_mm), c.rod, GROUND)
        system = StaticHydraulicSystem((p1, p2), springs=(spring,))
        result = system.solve({a.rod: -args.f1_n, b.rod: -args.f2_n})
    except ValueError as error:
        parser.error(str(error))

    compression = [-result.rod_extension_mm[cylinder.rod] for cylinder in cylinders]
    for name, pressure in result.pressures_mpa.items():
        print(f"{name}: {pressure:.4f} MPa (gauge)")
    for name, value in zip(("A", "B", "C"), compression):
        print(f"{name}: {value:+.3f} mm (positive = compression)")
    if not args.no_plot:
        plot_displacement(compression, args.f1_n, args.f2_n, args.spring_n_mm, args.save_plot)


if __name__ == "__main__":
    main()
