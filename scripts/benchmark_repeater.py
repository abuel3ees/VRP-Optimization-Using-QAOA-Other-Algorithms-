#!/usr/bin/env python3
"""
benchmark_repeater.py

Runs benchmark_seed.py N times, then produces a PDF with averaged metrics
across all runs.  Because benchmark_seed uses randomised QAOA seeds, each
run gives slightly different results; this script surfaces the mean ± σ of
every algorithm over N repetitions.

Usage (interactive — will ask how many runs):
    python scripts/benchmark_repeater.py

Usage (non-interactive):
    python scripts/benchmark_repeater.py --runs 5
    python scripts/benchmark_repeater.py --runs 3 --instance 50_0 --k 7
    python scripts/benchmark_repeater.py --runs 5 --algos Nearest Sweep
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Force line-buffered stdout so every print appears in the GUI immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── load benchmark_seed as a live module ────────────────────────────────────

def _load_benchmark_seed():
    spec = importlib.util.spec_from_file_location(
        "benchmark_seed",
        ROOT / "scripts" / "benchmark_seed.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_seed"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── averaging ────────────────────────────────────────────────────────────────

def average_results(all_runs: List[List[dict]]) -> List[dict]:
    """Average numeric metrics across runs, grouped by (instance, algo)."""
    buckets: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for run in all_runs:
        for r in run:
            buckets[(r["instance"], r["algo"])].append(r)

    averaged = []
    for (instance, algo), rows in buckets.items():
        totals = [r["total"] for r in rows]
        stds   = [r["std"]   for r in rows]
        wfs    = [r.get("weighted_fairness", 0) for r in rows]
        times  = [r["time"]  for r in rows]
        gaps   = [r.get("gap", 0) for r in rows]

        averaged.append({
            "instance":    instance,
            "algo":        algo,
            "k_used":      rows[0]["k_used"],
            "n_runs":      len(rows),
            "valid_runs":  sum(1 for r in rows if r["valid"]),
            "avg_total":   float(np.mean(totals)),
            "sd_total":    float(np.std(totals)),
            "min_total":   float(np.min(totals)),
            "max_total":   float(np.max(totals)),
            "avg_std":     float(np.mean(stds)),
            "avg_wf":      float(np.mean(wfs)),
            "avg_time":    float(np.mean(times)),
            "avg_gap":     float(np.mean(gaps)),   # recalculated below
        })
    return averaged


def _recompute_gaps(averaged: List[dict]) -> None:
    """Recalculate avg_gap relative to the best avg_wf per instance."""
    by_inst: Dict[str, List[dict]] = defaultdict(list)
    for r in averaged:
        by_inst[r["instance"]].append(r)
    for rows in by_inst.values():
        valid = [r for r in rows if r["valid_runs"] > 0]
        if not valid:
            continue
        best = min(r["avg_wf"] for r in valid)
        for r in rows:
            r["avg_gap"] = 100.0 * (r["avg_wf"] - best) / best if best > 0 else 0.0


# ── PDF generation ───────────────────────────────────────────────────────────

def generate_averaged_pdf(
    averaged: List[dict],
    all_runs: List[List[dict]],
    n_runs: int,
    out_path: Path,
    stamp: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as e:
        print(f"  PDF generation requires matplotlib: {e}")
        return

    instances = sorted({r["instance"] for r in averaged})
    BLUE      = "#4472c4"
    GREEN     = "#00b050"
    GREEN_BG  = "#c6efce"
    RED_BG    = "#ffc7ce"
    HDR_BG    = "#2e4057"

    with PdfPages(out_path) as pdf:

        # ── PAGE 1: Cover ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.set_axis_off()
        ax.axhline(0.82, color="#333333", linewidth=3, xmin=0.05, xmax=0.95)
        ax.axhline(0.16, color="#333333", linewidth=1, xmin=0.05, xmax=0.95)
        ax.text(0.5, 0.91, "VRP Benchmark — Averaged Results",
                ha="center", fontsize=26, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.78, f"{n_runs} independent run{'s' if n_runs != 1 else ''} averaged",
                ha="center", fontsize=16, color="#1a1a8c",
                transform=ax.transAxes)
        meta = [
            ("Runs",       str(n_runs)),
            ("Instances",  ", ".join(instances)),
            ("Algorithms", str(len({r["algo"] for r in averaged}))),
            ("Generated",  stamp),
        ]
        for i, (label, val) in enumerate(meta):
            y = 0.62 - i * 0.08
            ax.text(0.32, y, label + ":", ha="right", fontsize=13,
                    color="#555555", transform=ax.transAxes)
            ax.text(0.35, y, val, ha="left", fontsize=13,
                    fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.04, "Generated by benchmark_repeater.py",
                ha="center", fontsize=9, color="#888888",
                transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── PER-INSTANCE PAGES ─────────────────────────────────────────────
        for inst in instances:
            rows = sorted(
                [r for r in averaged if r["instance"] == inst],
                key=lambda r: r["avg_gap"],
            )
            valid_rows  = [r for r in rows if r["valid_runs"] > 0]
            best_wf_avg  = min((r["avg_wf"]    for r in valid_rows), default=None)
            worst_wf_avg = max((r["avg_wf"]    for r in valid_rows), default=None)
            best_avg     = min((r["avg_total"] for r in valid_rows), default=None)

            # ── Averaged results table ──────────────────────────────────
            # W_Fair is col 5 — highlighted gold
            WF_COL = 5
            col_labels = [
                "Algorithm", "k", "Avg Total", "±SD",
                "Avg Std", "Avg W_Fair ★", "Avg Time(s)",
                f"Valid/{n_runs}", "WF Gap%",
            ]
            table_data = []
            for r in rows:
                table_data.append([
                    r["algo"],
                    str(r["k_used"]),
                    f"{r['avg_total']:.1f}",
                    f"±{r['sd_total']:.1f}",
                    f"{r['avg_std']:.1f}",
                    f"{r['avg_wf']:.1f}",
                    f"{r['avg_time']:.3f}",
                    f"{r['valid_runs']}/{r['n_runs']}",
                    f"{r['avg_gap']:.1f}",
                ])

            fig_h = max(7, 1.2 + len(rows) * 0.32)
            fig2, ax2 = plt.subplots(figsize=(16, fig_h))
            ax2.set_axis_off()
            ax2.set_title(
                f"{inst} — Averaged Results ({n_runs} run{'s' if n_runs != 1 else ''})",
                fontsize=13, fontweight="bold", pad=10,
            )
            tbl = ax2.table(
                cellText=table_data, colLabels=col_labels,
                loc="center", cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7.5)
            tbl.scale(1.0, 1.6)
            for j in range(len(col_labels)):
                tbl[0, j].set_facecolor(HDR_BG)
                tbl[0, j].set_text_props(color="white", fontweight="bold")
            for ri, r in enumerate(rows):
                if best_wf_avg is not None and abs(r["avg_wf"] - best_wf_avg) < 1e-3:
                    fc = GREEN_BG
                elif worst_wf_avg is not None and abs(r["avg_wf"] - worst_wf_avg) < 1e-3:
                    fc = RED_BG
                else:
                    fc = "#ffffff"
                for j in range(len(col_labels)):
                    tbl[ri + 1, j].set_facecolor(fc)
            fig2.tight_layout()
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

            # ── Bar chart: avg total distance with error bars ────────────
            fig_h3 = max(5, len(rows) * 0.40 + 1.5)
            fig3, ax3 = plt.subplots(figsize=(12, fig_h3))
            labels = [r["algo"] for r in rows]
            avgs   = [r["avg_total"] for r in rows]
            errs   = [r["sd_total"]  for r in rows]
            colors = [
                GREEN if (best_wf_avg is not None and abs(r["avg_wf"] - best_wf_avg) < 1e-3) else BLUE
                for r in rows
            ]
            bars = ax3.barh(
                range(len(labels)), avgs, xerr=errs,
                color=colors, edgecolor="white", linewidth=0.5,
                error_kw={"ecolor": "#888888", "capsize": 3},
            )
            ax3.set_yticks(range(len(labels)))
            ax3.set_yticklabels(labels, fontsize=7)
            ax3.invert_yaxis()
            best_wf_row = min(valid_rows, key=lambda r: r["avg_wf"]) if valid_rows else None
            if best_wf_row is not None:
                ax3.axvline(best_wf_row["avg_total"], color=GREEN, linestyle="--",
                            linewidth=1.5,
                            label=f"Best-WF algo ({best_wf_row['algo']}): {best_wf_row['avg_total']:.1f}")
            ax3.set_xlabel("Average Total Distance", fontsize=10)
            ax3.set_title(
                f"{inst} — Avg Total Distance  (error bars = ±1σ across {n_runs} runs)",
                fontsize=12, fontweight="bold",
            )
            ax3.legend(fontsize=9)
            max_val = max(avgs) if avgs else 1
            for bar, val, err in zip(bars, avgs, errs):
                ax3.text(
                    val + err + max_val * 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=6,
                )
            ax3.grid(axis="x", alpha=0.3)
            fig3.tight_layout()
            pdf.savefig(fig3, bbox_inches="tight")
            plt.close(fig3)

            # ── Run-by-run trend for top-5 algorithms (only when n>1) ────
            if n_runs > 1:
                top5 = [r["algo"] for r in rows[:5]]
                run_vals: Dict[str, List[float]] = {a: [] for a in top5}
                for run in all_runs:
                    for r in run:
                        if r["instance"] == inst and r["algo"] in run_vals:
                            run_vals[r["algo"]].append(r["total"])

                fig4, ax4 = plt.subplots(figsize=(10, 5))
                x = list(range(1, n_runs + 1))
                for algo, vals in run_vals.items():
                    if vals:
                        ax4.plot(x[:len(vals)], vals, marker="o",
                                 linewidth=1.8, markersize=5, label=algo[:45])
                ax4.set_xlabel("Run #", fontsize=10)
                ax4.set_ylabel("Total Distance", fontsize=10)
                ax4.set_title(
                    f"{inst} — Top-5 Algorithms: Distance per Run",
                    fontsize=12, fontweight="bold",
                )
                ax4.legend(fontsize=7, loc="best", framealpha=0.85)
                ax4.grid(True, alpha=0.3)
                ax4.set_xticks(x)
                fig4.tight_layout()
                pdf.savefig(fig4, bbox_inches="tight")
                plt.close(fig4)

            # ── Fairness comparison bar charts ────────────────────────────
            fig_h5 = max(5, len(rows) * 0.38 + 2)
            fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, fig_h5))

            def _fairness_bar(ax, metric_key, title):
                vals = [r[metric_key] for r in rows]
                best_v = min(
                    (v for v, r in zip(vals, rows) if r["valid_runs"] > 0),
                    default=None,
                )
                cols = [
                    (GREEN if (best_v is not None and abs(v - best_v) < 1e-3) else BLUE)
                    for v in vals
                ]
                ax.barh(range(len(rows)), vals, color=cols,
                        edgecolor="white", linewidth=0.5)
                ax.set_yticks(range(len(rows)))
                ax.set_yticklabels([r["algo"] for r in rows], fontsize=7)
                ax.invert_yaxis()
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.grid(axis="x", alpha=0.3)

            _fairness_bar(ax5a, "avg_std",
                          "Avg Std Dev of Routes (lower = fairer)")
            _fairness_bar(ax5b, "avg_wf",
                          "Avg Weighted Fairness (lower = better)")
            fig5.suptitle(f"{inst} — Fairness Metrics ({n_runs} runs)",
                          fontsize=12, fontweight="bold")
            fig5.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig5, bbox_inches="tight")
            plt.close(fig5)

        # ── FINAL PAGE: Summary statistics table across all instances ──────
        fig6, ax6 = plt.subplots(figsize=(14, 7))
        ax6.set_axis_off()
        ax6.set_title("Summary — Average Metrics across All Instances",
                      fontsize=13, fontweight="bold", pad=10)

        # Pick best algorithm per instance by avg_wf (primary metric)
        summary_rows = []
        for inst in instances:
            inst_rows = [r for r in averaged if r["instance"] == inst]
            valid = [r for r in inst_rows if r["valid_runs"] > 0]
            if not valid:
                continue
            best = min(valid, key=lambda r: r["avg_wf"])
            fairest = min(valid, key=lambda r: r["avg_std"])
            summary_rows.append([
                inst,
                best["algo"],
                f"{best['avg_wf']:.2f}",
                fairest["algo"],
                f"{fairest['avg_std']:.1f}",
                str(n_runs),
            ])

        if summary_rows:
            col_labels6 = [
                "Instance", "Best W_Fairness (algo)",
                "Avg W_Fair ★", "Fairest Std (algo)",
                "Avg Std Dev", "Runs",
            ]
            tbl6 = ax6.table(
                cellText=summary_rows, colLabels=col_labels6,
                loc="center", cellLoc="center",
            )
            tbl6.auto_set_font_size(False)
            tbl6.set_fontsize(9)
            tbl6.scale(1.2, 2.2)
            for j in range(len(col_labels6)):
                tbl6[0, j].set_facecolor(HDR_BG)
                tbl6[0, j].set_text_props(color="white", fontweight="bold")
            for ri in range(len(summary_rows)):
                for j in range(len(col_labels6)):
                    tbl6[ri + 1, j].set_facecolor(
                        GREEN_BG if j in (1, 2) else
                        "#e8f4ff" if j in (3, 4) else "#ffffff"
                    )

        fig6.tight_layout()
        pdf.savefig(fig6, bbox_inches="tight")
        plt.close(fig6)

    print(f"  ✓ PDF saved: {out_path.relative_to(ROOT)}")


# ── interactive prompt ───────────────────────────────────────────────────────

def _ask_runs() -> int:
    while True:
        try:
            raw = input(
                "How many times should benchmark_seed.py run? [1-50]: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not raw:
            continue
        try:
            n = int(raw)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if 1 <= n <= 50:
            return n
        print("  Please enter a number between 1 and 50.")


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run benchmark_seed N times and produce a PDF with averaged results."
        )
    )
    ap.add_argument("--runs", type=int, default=None,
                    help="Number of repetitions (1-50). Asked interactively if omitted.")
    ap.add_argument("--instance", default=None,
                    help="e.g. 50_0  (RioClaroPostToy_50_0). Omit for all instances.")
    ap.add_argument("--k", type=int, default=None,
                    help="Number of vehicles. Default: MAX_VEHICLES from .vrp file.")
    ap.add_argument("--algos", nargs="+", metavar="ALGO", default=None,
                    help="Include only algorithms whose name contains these substrings.")
    args = ap.parse_args()

    n_runs = args.runs if args.runs is not None else _ask_runs()

    print("\nLoading benchmark_seed …")
    bs = _load_benchmark_seed()

    # Resolve instances
    all_names = bs.list_array_instances()
    if args.instance:
        target = f"RioClaroPostToy_{args.instance}"
        if target not in all_names:
            sys.exit(f"Instance '{target}' not in arrays/. Available: {all_names}")
        names = [target]
    else:
        names = all_names

    # Resolve algorithm filter
    if args.algos:
        filt = [s.lower() for s in args.algos]
        active_algos = [
            (n, f) for n, f in bs.ALGORITHMS
            if any(s in n.lower() for s in filt)
        ]
        if not active_algos:
            sys.exit(f"No algorithms matched {args.algos}.")
    else:
        active_algos = bs.ALGORITHMS

    stamp   = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "results" / f"averaged_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  benchmark_repeater — {n_runs} run{'s' if n_runs != 1 else ''}")
    print(f"  Instances : {', '.join(names)}")
    print(f"  Algorithms: {len(active_algos)}")
    print(f"  Output    : results/averaged_{stamp}/")
    print(f"{'='*70}\n")

    all_runs: List[List[dict]] = []

    for run_i in range(1, n_runs + 1):
        print(f"\n{'─'*70}")
        print(f"  RUN {run_i} / {n_runs}")
        print(f"{'─'*70}")

        # Reset per-run accumulator globals inside benchmark_seed
        bs.QAOA_STATS.clear()
        bs.QAOA_STATS.update({"success": 0, "fallback": 0})
        bs.QAOA_LOG.clear()
        bs.set_global_run_stamp()

        run_results: List[dict] = []
        for name in names:
            k = args.k or bs.read_vrp_k(name) or 7
            results = bs.run_one_instance(
                name, k,
                save_outputs=True,
                active_algorithms=active_algos,
            )
            run_results.extend(results)

        all_runs.append(run_results)

    # Average and write outputs
    print(f"\n{'='*70}")
    print(f"  Averaging {n_runs} run{'s' if n_runs != 1 else ''} …")

    averaged = average_results(all_runs)
    _recompute_gaps(averaged)

    # CSV
    csv_path = out_dir / f"averaged_{n_runs}runs.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Instance", "Algorithm", "k", "N_runs", "Valid_runs",
            "Avg_Total", "SD_Total", "Min_Total", "Max_Total",
            "Avg_Std", "Avg_WF", "Avg_Time_s", "Avg_Gap_pct",
        ])
        for r in sorted(averaged, key=lambda x: (x["instance"], x["avg_gap"])):
            w.writerow([
                r["instance"], r["algo"], r["k_used"],
                r["n_runs"], r["valid_runs"],
                f"{r['avg_total']:.2f}", f"{r['sd_total']:.2f}",
                f"{r['min_total']:.2f}", f"{r['max_total']:.2f}",
                f"{r['avg_std']:.2f}", f"{r['avg_wf']:.2f}",
                f"{r['avg_time']:.4f}", f"{r['avg_gap']:.2f}",
            ])
    print(f"  ✓ CSV:  {csv_path.relative_to(ROOT)}")

    # PDF
    pdf_path = out_dir / f"averaged_{n_runs}runs_report.pdf"
    generate_averaged_pdf(averaged, all_runs, n_runs, pdf_path, stamp)

    print(f"\n{'='*70}")
    print(f"  Done!  {n_runs} run{'s' if n_runs != 1 else ''} complete.")
    print(f"  Folder: results/averaged_{stamp}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
