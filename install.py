#!/usr/bin/env python3
"""
install.py — one-shot setup for vrptheo

Installs all required Python packages.  Run once before using run.py.

    python install.py            # core packages only
    python install.py --qaoa     # core + Qiskit/QAOA packages
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CORE = [
    "numpy>=1.24",
    "matplotlib>=3.7",
    "ortools>=9.7",
]

QAOA = [
    "qiskit>=1.0",
    "qiskit-aer>=0.13",
    "qiskit-algorithms>=0.3",
    "qiskit-optimization>=0.6",
]


def pip_install(packages: list[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    print(f"\n  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode == 0


def check(package_name: str) -> bool:
    import importlib
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser(description="Install vrptheo dependencies.")
    ap.add_argument(
        "--qaoa", action="store_true",
        help="Also install Qiskit packages needed for QAOA algorithms.",
    )
    args = ap.parse_args()

    print("=" * 60)
    print("  vrptheo — dependency installer")
    print("=" * 60)

    print("\n[1/2] Installing core packages …")
    ok = pip_install(CORE)
    if not ok:
        print("\n  ERROR: pip failed for core packages. Check the output above.")
        sys.exit(1)

    if args.qaoa:
        print("\n[2/2] Installing Qiskit/QAOA packages …")
        ok = pip_install(QAOA)
        if not ok:
            print("\n  ERROR: pip failed for Qiskit packages. Check the output above.")
            sys.exit(1)
    else:
        print("\n[2/2] Skipping Qiskit (run with --qaoa to include it).")

    print("\n" + "=" * 60)
    print("  Verifying installs …")
    checks = {
        "numpy":      "numpy",
        "matplotlib": "matplotlib",
        "ortools":    "ortools",
    }
    if args.qaoa:
        checks.update({
            "qiskit":              "qiskit",
            "qiskit-aer":         "qiskit_aer",
            "qiskit-algorithms":  "qiskit_algorithms",
            "qiskit-optimization":"qiskit_optimization",
        })

    all_ok = True
    for name, mod in checks.items():
        status = "OK" if check(mod) else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"    {name:<26} {status}")

    print("=" * 60)
    if all_ok:
        print("  All packages installed successfully.")
        print("  You can now run:  python run.py")
    else:
        print("  Some packages are missing — see above.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
