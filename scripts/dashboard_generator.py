#!/usr/bin/env python3
"""
Quick Benchmark Dashboard Generator

Creates a visual dashboard summarizing all benchmark runs.

Usage:
    python dashboard_generator.py
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def gather_all_runs() -> Dict[str, List[dict]]:
    """Gather all benchmark results by run."""
    runs_data = {}
    
    if not RESULTS_DIR.exists():
        return runs_data
    
    # Only match timestamped runs (format: YYYY-MM-DD_HHMMSS)
    import re
    timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{6}$')
    run_dirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir() and timestamp_pattern.match(d.name)])
    
    for run_dir in run_dirs:
        results = []
        for inst_dir in run_dir.iterdir():
            if not inst_dir.is_dir() or "_k" not in inst_dir.name:
                continue
            data_dir = inst_dir / "data"
            # Find the benchmark_report.json file (may have prefix)
            json_files = list(data_dir.glob("*benchmark_report.json"))
            for json_file in json_files:
                try:
                    data = json.loads(json_file.read_text())
                    results.extend(data["results"])
                except Exception:
                    pass
        if results:
            runs_data[run_dir.name] = results
    
    return runs_data


def create_timeline_dashboard(runs_data: Dict[str, List[dict]]):
    """Create a timeline showing how algorithm performance changes across runs."""
    if not HAS_MATPLOTLIB or not runs_data:
        return
    
    try:
        # Group by algorithm, track best distance over runs
        algo_history = defaultdict(list)
        run_names = sorted(runs_data.keys())
        
        for run_name in run_names:
            results = runs_data[run_name]
            valid = [r for r in results if r["valid"]]
            by_algo = {}
            for r in valid:
                if r["algo"] not in by_algo or r["total"] < by_algo[r["algo"]]["total"]:
                    by_algo[r["algo"]] = r
            
            for algo, best_result in by_algo.items():
                algo_history[algo].append(best_result["total"])
        
        # Plot top algorithms' progression
        fig, ax = plt.subplots(figsize=(14, 8))
        top_algos = sorted(algo_history.keys(), key=lambda a: algo_history[a][-1])[:8]
        
        for algo in top_algos:
            distances = algo_history[algo]
            x = list(range(len(distances)))
            ax.plot(x, distances, marker="o", label=algo[:30], linewidth=2, markersize=6)
        
        ax.set_xticks(range(len(run_names)))
        ax.set_xticklabels([name[:10] for name in run_names], rotation=45)
        ax.set_xlabel("Run ID", fontweight="bold")
        ax.set_ylabel("Best Distance Found", fontweight="bold")
        ax.set_title("Algorithm Performance Timeline", fontweight="bold", fontsize=14)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = RESULTS_DIR / "DASHBOARD_timeline.png"
        fig.savefig(plot_path, dpi=150)
        plt.close()
        print(f"✓ Timeline: {plot_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"⚠ Timeline failed: {e}")


def create_algorithm_scorecard(runs_data: Dict[str, List[dict]]):
    """Create a scorecard ranking algorithms across all runs."""
    if not runs_data:
        return
    
    algo_stats = {}
    
    for run_results in runs_data.values():
        valid = [r for r in run_results if r["valid"]]
        for r in valid:
            algo = r["algo"]
            if algo not in algo_stats:
                algo_stats[algo] = {
                    "distances": [],
                    "times": [],
                    "counts": 0,
                    "wins": 0
                }
            algo_stats[algo]["distances"].append(r["total"])
            algo_stats[algo]["times"].append(r["time"])
            algo_stats[algo]["counts"] += 1
    
    # Rank by average distance
    ranked = sorted(
        algo_stats.items(),
        key=lambda x: np.mean(x[1]["distances"]) if x[1]["distances"] else float("inf")
    )
    
    # Create report
    report_lines = [
        "# Benchmark Scorecard",
        "",
        f"Algorithm ranking across all {len(runs_data)} runs.",
        "",
        "| Rank | Algorithm | Avg Dist | Std Dev | Avg Time | Run Count |",
        "|------|-----------|----------|---------|----------|----------|",
    ]
    
    for rank, (algo, stats) in enumerate(ranked[:20], 1):
        if stats["distances"]:
            avg_dist = np.mean(stats["distances"])
            std_dist = np.std(stats["distances"])
            avg_time = np.mean(stats["times"])
            report_lines.append(
                f"| {rank} | {algo[:45]} | {avg_dist:.2f} | {std_dist:.2f} | {avg_time:.4f}s | {stats['counts']} |"
            )
    
    report_path = RESULTS_DIR / "SCORECARD.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"✓ Scorecard: {report_path.relative_to(ROOT)}")


def create_summary_index():
    """Create a master summary index of all data."""
    import re
    timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{6}$')
    runs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir() and timestamp_pattern.match(d.name)])
    
    lines = [
        "# Dashboard - Benchmark Results",
        "",
        f"**Last Updated:** {Path(__file__).stat().st_mtime}",
        "",
        "## Quick Navigation",
        "",
        f"- **Latest Run:** {runs[-1].name if runs else 'None'}",
        f"- **Total Runs:** {len(runs)}",
        "- **Analysis Files:**",
        "  - [SCORECARD.md](SCORECARD.md) - Algorithm ranking across all runs",
        "  - [DASHBOARD_timeline.png](DASHBOARD_timeline.png) - Performance over time",
        "",
        "## Recent Runs",
        "",
    ]
    
    for run_dir in runs[-5:]:
        manifest = run_dir / "manifest.md"
        if manifest.exists():
            lines.append(f"### {run_dir.name}")
            lines.append(f"[View Run Results]({run_dir.name}/manifest.md)")
            lines.append("")
    
    lines.extend([
        "## All Runs",
        "",
        "| Run ID | Instances | Folder |",
        "|--------|-----------|--------|",
    ])
    
    for run_dir in sorted(runs, reverse=True):
        instances = len([d for d in run_dir.iterdir() if d.is_dir() and "_k" in d.name])
        lines.append(f"| {run_dir.name} | {instances} | [{run_dir.name}]({run_dir.name}/) |")
    
    index_path = RESULTS_DIR / "DASHBOARD.md"
    index_path.write_text("\n".join(lines) + "\n")
    print(f"✓ Dashboard: {index_path.relative_to(ROOT)}")


def main():
    sep = "="*80
    print(f"\n{sep}")
    print("Dashboard Generator")
    print(f"{sep}\n")
    
    if not RESULTS_DIR.exists():
        print("⚠ No results/ directory found")
        return
    
    print("📈 Gathering all runs...")
    runs_data = gather_all_runs()
    print(f"  Found {len(runs_data)} runs\n")
    
    print("Creating dashboard files...")
    create_summary_index()
    create_algorithm_scorecard(runs_data)
    if HAS_MATPLOTLIB:
        create_timeline_dashboard(runs_data)
    
    print(f"\n✓ Dashboard complete!")
    print(f"View at: results/DASHBOARD.md")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
