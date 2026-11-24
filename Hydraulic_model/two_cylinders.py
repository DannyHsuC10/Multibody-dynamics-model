"""Double-acting master cylinder driving a spring-loaded slave.

The cap chambers share a supply line and the rod chambers share a closed
return line at the zero-gauge reference. Both rod diameters are half the
respective bore diameter, so the oil volumes balance in both lines.
"""

import argparse
from math import isfinite

if __package__:
    from .model import HydraulicCylinder, HydraulicLine, HydraulicSystem, LinearSpring
else:
    from model import HydraulicCylinder, HydraulicLine, HydraulicSystem, LinearSpring


def plot_response(responses, destination):
    import matplotlib

    if destination is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (travel, oil, force) = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    figure.suptitle("Two-cylinder hydraulic response")
    masters = [response.master for response in responses]
    slaves = [response.slaves[0] for response in responses]
    strokes = [master.stroke_mm for master in masters]
    travel.plot(strokes, [slave.stroke_mm for slave in slaves], color="#007f86", marker="o")
    travel.set_ylabel("Slave stroke (mm)")
    oil.plot(strokes, [response.pressure_mpa for response in responses], color="#7364aa", marker="o")
    oil.set_ylabel("Supply pressure (MPa)")
    force.plot(strokes, [master.force_n for master in masters], color="#2878b5", marker="o", label="Master input")
    force.plot(strokes, [slave.force_n for slave in slaves], color="#b85c00", marker="s", label="Slave output")
    force.set_ylabel("Axial force (N)")
    force.legend()
    for axis in (travel, oil, force):
        axis.set_xlabel("Master stroke (mm)")
        axis.grid(alpha=.25)

    if destination is None:
        plt.show()
    else:
        figure.savefig(destination, dpi=160)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--master-diameter-mm", type=float, default=20.,
                        help="Master cylinder bore diameter (default: 20 mm)")
    parser.add_argument("--slave-diameter-mm", type=float, default=40.,
                        help="Slave cylinder bore diameter (default: 40 mm)")
    drive = parser.add_mutually_exclusive_group()
    drive.add_argument("--master-stroke-mm", type=float, default=20.,
                       help="Maximum inward master stroke (default: 20 mm)")
    drive.add_argument("--master-force-n", type=float,
                       help="Force on the master rod instead of stroke; negative retracts the rod")
    parser.add_argument("--slave-force-n", type=float, default=0.,
                        help="Additional force on the slave rod; positive acts toward extension")
    parser.add_argument("--spring-n-mm", type=float, default=100.,
                        help="Slave return spring stiffness (default: 100 N/mm)")
    plotting = parser.add_mutually_exclusive_group()
    plotting.add_argument("--save-plot", metavar="PATH", help="Save the figure instead of opening a window")
    plotting.add_argument("--no-plot", action="store_true", help="Print the numeric table only")
    args = parser.parse_args()
    for name in ("master_diameter_mm", "slave_diameter_mm", "master_stroke_mm", "spring_n_mm"):
        value = getattr(args, name)
        if not isfinite(value) or value <= 0.:
            parser.error(f"--{name.replace('_', '-')} must be a finite positive number")
    if not isfinite(args.slave_force_n) or (args.master_force_n is not None and not isfinite(args.master_force_n)):
        parser.error("Rod forces must be finite.")

    try:
        master = HydraulicCylinder(args.master_diameter_mm, rod_diameter_mm=args.master_diameter_mm / 2.)
        slave = HydraulicCylinder(args.slave_diameter_mm, rod_diameter_mm=args.slave_diameter_mm / 2.)
        supply = HydraulicLine(master.cap_end, (slave.cap_end,))
        return_line = HydraulicLine(master.rod_end, (slave.rod_end,))
        system = HydraulicSystem(supply, return_line=return_line,
                                 rod_loads={slave.rod: LinearSpring(args.spring_n_mm)})
        rod_forces = {slave.rod: args.slave_force_n}
        fractions = (0., .25, .5, .75, 1.)
        if args.master_force_n is None:
            responses = [system.solve(args.master_stroke_mm * fraction, rod_forces_n=rod_forces)
                         for fraction in fractions]
        else:
            responses = [system.solve_force(args.master_force_n * fraction, rod_forces_n=rod_forces)
                         for fraction in fractions]
    except ValueError as error:
        parser.error(str(error))

    print(f"{'Master (mm)':>12} {'Slave (mm)':>12} {'Pressure (MPa)':>16}"
          f" {'Master force (N)':>17} {'Slave force (N)':>16}")
    for response in responses:
        master_state, (slave_state,) = response.master, response.slaves
        print(f"{master_state.stroke_mm:12.3f} {slave_state.stroke_mm:12.3f} {response.pressure_mpa:16.3f}"
              f" {master_state.force_n:17.3f} {slave_state.force_n:16.3f}")

    if not args.no_plot:
        try:
            plot_response(responses, args.save_plot)
        except ModuleNotFoundError as error:
            if error.name != "matplotlib":
                raise
            parser.error("plotting requires Matplotlib; install it or use --no-plot")


if __name__ == "__main__":
    main()
