"""
run.py — Master launcher for the VRP QAOA project.

Presents an interactive menu and runs the selected script.
"""

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    {
        "name": "QAOA Solver (notebook-exact)",
        "file": "scripts/QAOA.py",
        "desc": (
            "Recursive QAOA VRP solver, identical to the notebook.\n"
            "  Loads an instance, runs QAOA at leaf nodes, saves plots and reports."
        ),
    },
    {
        "name": "QAOA Solver + Inter-route Polish",
        "file": "scripts/QAOA_polished.py",
        "desc": (
            "Same as above but applies Or-opt + 2-opt* after every recursion level.\n"
            "  Reduces total distance; may increase route-length std deviation."
        ),
    },
    {
        "name": "Classical VRP Benchmark",
        "file": "scripts/classical_vrp_benchmark.py",
        "desc": (
            "Runs Nearest-Neighbour, Clarke-Wright, Sweep, OR-Tools, and\n"
            "  recursive QAOA side-by-side. Saves a benchmark CSV."
        ),
    },
    {
        "name": "Inter-route Optimizer (smoke test)",
        "file": "scripts/inter_route_opt.py",
        "desc": (
            "Standalone Or-opt + 2-opt* local search module.\n"
            "  Runs the built-in smoke test on a tiny 6-node toy instance."
        ),
    },
    {
        "name": "Convert VRP files to arrays",
        "file": "scripts/vrptoarray.py",
        "desc": (
            "Converts .vrp instance files in Instances/ into Python array\n"
            "  files (dist_matrix, node_coords) in arrays/."
        ),
    },
    {
        "name": "Classic VRP Solver (main.py)",
        "file": "scripts/main.py",
        "desc": (
            "Reads a .vrp file and benchmarks Recursive, NN, Clarke-Wright,\n"
            "  Sweep, and OR-Tools. Accepts a .vrp path as argument."
        ),
    },
    {
        "name": "Analyze Benchmark Results",
        "file": "scripts/analyze_benchmark_results.py",
        "desc": (
            "Reads saved benchmark CSVs from results/ and prints a summary\n"
            "  analysis across instances and algorithm variants."
        ),
    },
    {
        "name": "Generate Dashboard",
        "file": "scripts/dashboard_generator.py",
        "desc": (
            "Creates a visual HTML/text dashboard from all results in results/."
        ),
    },
    {
        "name": "Find Best Vehicle Count",
        "file": "scripts/find_best_vehicles.py",
        "desc": (
            "Sweeps different values of K (number of vehicles) and reports\n"
            "  which value gives the best total distance."
        ),
    },
    {
        "name": "Reorganize Results",
        "file": "scripts/reorganize_results.py",
        "desc": (
            "Reorganizes any flat results files into the\n"
            "  results/instance/K_vehicles/[plots|data|logs] hierarchy."
        ),
    },
]


def _banner():
    print()
    print("=" * 60)
    print("  VRP QAOA Project — Master Launcher")
    print("=" * 60)


def _menu():
    _banner()
    print()
    for i, s in enumerate(SCRIPTS, 1):
        print(f"  [{i:2d}]  {s['name']}")
        for line in s["desc"].splitlines():
            print(f"        {line}")
        print()
    print("  [ 0]  Exit")
    print()


def _pick() -> int:
    while True:
        try:
            raw = input("Select a script to run [0-{}]: ".format(len(SCRIPTS))).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if raw == "":
            continue
        try:
            choice = int(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if 0 <= choice <= len(SCRIPTS):
            return choice
        print(f"  Please enter a number between 0 and {len(SCRIPTS)}.")


def _run(script_entry: dict, extra_args: list[str]) -> None:
    path = os.path.join(ROOT, script_entry["file"])
    if not os.path.exists(path):
        print(f"\n  ERROR: {script_entry['file']} not found at {path}")
        return
    print(f"\n  Running: {script_entry['file']}")
    print("  " + "-" * 56)
    cmd = [sys.executable, path] + extra_args
    result = subprocess.run(cmd, cwd=ROOT)
    print("  " + "-" * 56)
    if result.returncode != 0:
        print(f"  Script exited with code {result.returncode}.")
    else:
        print("  Done.")


def main():
    # Forward any extra CLI args (e.g. --instance 2) to the selected script.
    extra_args = sys.argv[1:]

    _menu()
    choice = _pick()

    if choice == 0:
        print("  Bye.")
        return

    _run(SCRIPTS[choice - 1], extra_args)


if __name__ == "__main__":
    main()
