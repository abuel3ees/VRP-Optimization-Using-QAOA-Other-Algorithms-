#!/usr/bin/env python3
"""
Benchmark Results Analyzer

Analyze completed benchmark runs and generate comprehensive reports.

Usage:
    python analyze_benchmark_results.py                     # Analyze latest run
    python analyze_benchmark_results.py --run 2026-04-18_182543
    python analyze_benchmark_results.py --all              # Analyze all runs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from openpyxl import Workbook
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def find_latest_run() -> Path | None:
    """Find the most recent timestamped run folder."""
    if not RESULTS_DIR.exists():
        return None
    # Only match timestamped runs (format: YYYY-MM-DD_HHMMSS)
    import re
    timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{6}$')
    runs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir() and timestamp_pattern.match(d.name)])
    return runs[-1] if runs else None


def analyze_run(run_dir: Path):
    """Generate comprehensive analysis for a benchmark run."""
    sep = "="*80
    print(f"\n{sep}")
    print(f"Analyzing: {run_dir.name}")
    print(f"{sep}\n")
    
    # Collect all results
    all_results = []
    instance_dirs = [d for d in run_dir.iterdir() if d.is_dir() and "_k" in d.name]
    
    for inst_dir in sorted(instance_dirs):
        data_dir = inst_dir / "data"
        # Find the benchmark_report.json file (may have prefix)
        json_files = list(data_dir.glob("*benchmark_report.json"))
        for json_file in json_files:
            try:
                data = json.loads(json_file.read_text())
                print(f"✓ {inst_dir.name}: {len(data['results'])} algorithms")
                all_results.extend(data["results"])
            except Exception as e:
                print(f"✗ Error reading {json_file}: {e}")
    
    if not all_results:
        print("⚠ No results found in this run")
        return
    
    print(f"\n📊 Generating run overview...")
    generate_run_overview(run_dir, all_results)
    
    if len(instance_dirs) > 1:
        print(f"\n📈 Generating cross-instance comparison...")
        generate_cross_instance_comparison(run_dir, all_results, instance_dirs)
    
    print(f"\n✓ Analysis complete!")


def generate_run_overview(run_dir: Path, results: List[dict]):
    """Generate overview report for the entire run."""
    report_path = run_dir / "RUN_OVERVIEW.md"
    
    valid = [r for r in results if r["valid"]]
    by_instance = {}
    for r in results:
        inst = r["instance"]
        if inst not in by_instance:
            by_instance[inst] = []
        by_instance[inst].append(r)
    
    lines = [
        "# Benchmark Run Overview",
        "",
        f"**Run ID:** {run_dir.name}",
        "",
        "## Summary",
        "",
        f"- **Total Instances:** {len(by_instance)}",
        f"- **Total Algorithms Run:** {len(results)}",
        f"- **Valid Solutions:** {len(valid)} ({100*len(valid)/len(results):.1f}%)",
        "",
        "## Instances",
        "",
    ]
    
    for inst_name in sorted(by_instance.keys()):
        inst_results = by_instance[inst_name]
        inst_valid = [r for r in inst_results if r["valid"]]
        if inst_valid:
            best = min(inst_valid, key=lambda r: r["total"])
            lines.append(f"### {inst_name}")
            lines.append(f"- Algorithms: {len(inst_results)}")
            lines.append(f"- Valid: {len(inst_valid)}")
            lines.append(f"- Best Distance: {best['total']:.2f} ({best['algo']})")
            lines.append("")
    
    lines.extend([
        "## Best Performers (Overall)",
        "",
        "| Category | Algorithm | Value | Instance |",
        "|----------|-----------|-------|----------|",
    ])
    
    if valid:
        best_dist = min(valid, key=lambda r: r["total"])
        best_fair = min(valid, key=lambda r: r["std"])
        best_time = min(valid, key=lambda r: r["time"])
        
        lines.extend([
            f"| Distance | {best_dist['algo'][:40]} | {best_dist['total']:.2f} | {best_dist['instance']} |",
            f"| Fairness | {best_fair['algo'][:40]} | {best_fair['std']:.2f} | {best_fair['instance']} |",
            f"| Speed | {best_time['algo'][:40]} | {best_time['time']:.4f}s | {best_time['instance']} |",
        ])
    
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n")
    print(f"  ✓ {report_path.relative_to(ROOT)}")


def generate_cross_instance_comparison(run_dir: Path, results: List[dict], instance_dirs: List[Path]):
    """Compare algorithm performance across instances."""
    
    if not HAS_MATPLOTLIB:
        return
    
    try:
        # Collect scores by algorithm
        algo_scores = {}
        for r in results:
            if not r["valid"]:
                continue
            algo = r["algo"]
            if algo not in algo_scores:
                algo_scores[algo] = {"distances": [], "times": [], "fairnesses": []}
            algo_scores[algo]["distances"].append(r["total"])
            algo_scores[algo]["times"].append(r["time"])
            algo_scores[algo]["fairnesses"].append(r["std"])
        
        # Plot: Average scores by algorithm
        if len(algo_scores) > 5:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            algos = list(algo_scores.keys())
            avg_dists = [np.mean(algo_scores[a]["distances"]) for a in algos]
            avg_times = [np.mean(algo_scores[a]["times"]) for a in algos]
            avg_fairs = [np.mean(algo_scores[a]["fairnesses"]) for a in algos]
            
            # Sort by distance
            sorted_idx = sorted(range(len(algos)), key=lambda i: avg_dists[i])[:12]
            sorted_algos = [algos[i] for i in sorted_idx]
            sorted_dists = [avg_dists[i] for i in sorted_idx]
            
            axes[0].barh(sorted_algos, sorted_dists, color="steelblue")
            axes[0].set_xlabel("Avg Total Distance")
            axes[0].set_title("Distance Comparison")
            
            sorted_times = [avg_times[i] for i in sorted_idx]
            axes[1].barh(sorted_algos, sorted_times, color="coral")
            axes[1].set_xlabel("Avg Time (s)")
            axes[1].set_title("Speed Comparison")
            
            sorted_fairs = [avg_fairs[i] for i in sorted_idx]
            axes[2].barh(sorted_algos, sorted_fairs, color="seagreen")
            axes[2].set_xlabel("Avg Fairness Std")
            axes[2].set_title("Fairness Comparison")
            
            plt.tight_layout()
            plot_path = run_dir / "00_cross_instance_comparison.png"
            fig.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  ✓ {plot_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"  ⚠ Comparison plot failed: {e}")


def main():
    ap = argparse.ArgumentParser(description="Analyze benchmark results")
    ap.add_argument("--run", help="Analyze specific run (e.g. 2026-04-18_182543)")
    ap.add_argument("--all", action="store_true", help="Analyze all runs")
    args = ap.parse_args()
    
    if args.all:
        runs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir() and "_" in d.name])
        for run_dir in runs:
            analyze_run(run_dir)
    elif args.run:
        run_dir = RESULTS_DIR / args.run
        if not run_dir.exists():
            print(f"Error: Run '{args.run}' not found")
            sys.exit(1)
        analyze_run(run_dir)
    else:
        latest = find_latest_run()
        if latest:
            analyze_run(latest)
        else:
            print("No runs found in results/")
            sys.exit(1)


if __name__ == "__main__":
    main()
