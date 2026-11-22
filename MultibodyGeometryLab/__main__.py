import argparse
from pathlib import Path
import tkinter as tk

from .gui import Workbench
from .hardpoint_import import import_hardpoint_csv
from .project import default_project, read_project


def main():
    parser = argparse.ArgumentParser(description="Four-corner multibody suspension geometry workbench.")
    parser.add_argument("--project", type=Path, help="Open a MultibodyGeometryLab or SuspensionGeometryLab JSON file.")
    parser.add_argument("--hardpoints", type=Path, help="Overlay a hardpoint CSV on the selected project.")
    args = parser.parse_args()
    root = None
    try:
        data, native = read_project(args.project) if args.project else (default_project(), False)
        if args.hardpoints:
            data = import_hardpoint_csv(data, args.hardpoints)
            native = False
        root = tk.Tk()
        workbench = Workbench(root, data, args.project if native else None)
        if args.hardpoints:
            workbench.mark_dirty()
    except (ValueError, ArithmeticError, OSError) as error:
        if root is not None:
            root.destroy()
        parser.error(str(error))
    root.mainloop()


if __name__ == "__main__":
    main()
