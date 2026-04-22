# 📊 Benchmark Results Organization & Analysis Guide

## Overview

Your benchmark system now includes comprehensive organization, Excel export, and data analysis capabilities. Every run is automatically timestamped with analysis reports, visualizations, and Excel sheets.

---

## 📁 New Folder Structure

Each benchmark run gets a unique timestamped folder:

```
results/
├── RUNS_INDEX.md                    ← Master index of all runs
├── DASHBOARD.md                     ← Quick navigation dashboard
├── SCORECARD.md                     ← Algorithm ranking
├── DASHBOARD_timeline.png           ← Performance trends
├── 2026-04-18_182543/               ← Each run has timestamp
│   ├── manifest.md                  ← Run details
│   ├── RioClaroPostToy_50_0_k7/    ← Instance folder
│   │   ├── data/
│   │   │   ├── benchmark_summary.md           ← Quick table
│   │   │   ├── benchmark_algorithms.csv       ← Raw data
│   │   │   ├── benchmark_report.json          ← Full results
│   │   │   ├── benchmark_results.xlsx         ← 📊 Excel sheets
│   │   │   └── ANALYSIS_REPORT.md             ← 📈 Analysis
│   │   └── plots/
│   │       ├── 01_distance_vs_fairness.png   ← Bubble chart
│   │       ├── 02_algorithm_comparison.png   ← Top algorithms
│   │       ├── 03_performance_heatmap.png    ← Performance matrix
│   │       └── (individual algorithm plots)
│   └── RioClaroPostToy_100_0_k30/
│       └── ...
└── 2026-04-19_103015/
    └── ...
```

---

## 📊 Auto-Generated Files

### 1. Excel Workbook: `benchmark_results.xlsx`

**Multi-sheet workbook with formatting:**

| Sheet | Content |
|-------|---------|
| **Summary** | Best algorithms, total counts, valid solutions |
| **Detailed Results** | All algorithms sorted by gap %, with metrics |
| **Statistics** | Mean/std/min/max for distance, fairness, time |

### 2. Analysis Report: `ANALYSIS_REPORT.md`

Includes:
- Executive summary
- Best performers by category
- Performance statistics
- Gap analysis (% within best)
- Top 20 ranked results

### 3. Visualizations (3 Auto-Generated Plots)

**01_distance_vs_fairness.png**
- Bubble chart: X=total distance, Y=fairness std
- Size = execution time
- Color = gap from best (green/orange/red)

**02_algorithm_comparison.png**
- Top 10 by distance
- Top 10 by fairness
- Top 10 by speed
- Gap % distribution histogram

**03_performance_heatmap.png**
- Top 15 algorithms × 3 metrics heatmap
- Normalized scores (red=bad, green=good)

---

## 🚀 Usage

### Basic Benchmark Run

```bash
python classical_vrp_benchmark.py --instance 50_0 --k 7
```

**Output:**
```
================================================================================
VRP BENCHMARK
================================================================================
Run ID: 2026-04-18_182543
✓ Weighted Fairness metric: ((total_distance/k) + distance_std) / 2
✓ Recursive QAOA solver: uses Qiskit for small leaves (≤5 nodes)
✓ Results: results/2026-04-18_182543/
================================================================================

=== RioClaroPostToy_50_0  (n=50, k=7) ===
Algorithm                                  k   Total        Std  W_Fair    Time_s  Valid   Gap%
----...
...
✓ Solution found in 687.45s

  📊 Generating analysis...
  ✓ Excel: benchmark_results.xlsx
  ✓ Analysis: ANALYSIS_REPORT.md
  ✓ Plot: distance_vs_fairness.png
  ✓ Plot: algorithm_comparison.png
  ✓ Plot: performance_heatmap.png
  ✓ Saved to: results/2026-04-18_182543/RioClaroPostToy_50_0_k7/data
```

---

## 🔍 Analysis Tools

### 1. Analyze a Specific Run

```bash
python analyze_benchmark_results.py --run 2026-04-18_182543
```

Generates:
- `RUN_OVERVIEW.md` - Summary across all instances in that run
- `00_cross_instance_comparison.png` - Algorithm comparison

### 2. Analyze Latest Run

```bash
python analyze_benchmark_results.py
```

Automatically finds and analyzes the most recent run.

### 3. Analyze All Runs

```bash
python analyze_benchmark_results.py --all
```

Creates overview for every run in `results/`

### 4. Generate Master Dashboard

```bash
python dashboard_generator.py
```

Generates:
- **DASHBOARD.md** - Master navigation page
- **SCORECARD.md** - Algorithm ranking across all runs
- **DASHBOARD_timeline.png** - Performance trends over time

---

## 📈 What Gets Tracked

Per algorithm per instance:

- ✅ Total distance
- ✅ Distance std (fairness metric)
- ✅ Weighted fairness score
- ✅ Execution time
- ✅ Validity (valid solution or not)
- ✅ Gap % from best solution

Per run:

- ✅ Run timestamp
- ✅ Instances tested
- ✅ Manifest with file structure
- ✅ Cross-instance analysis

---

## 🎯 Quick Navigation

After a run completes:

1. **Quick Look:** `results/<RUN_ID>/RioClaroPostToy_50_0_k7/data/benchmark_summary.md`
2. **Deep Dive:** `results/<RUN_ID>/RioClaroPostToy_50_0_k7/data/ANALYSIS_REPORT.md`
3. **Visualizations:** `results/<RUN_ID>/RioClaroPostToy_50_0_k7/plots/*.png`
4. **Excel Analysis:** `results/<RUN_ID>/RioClaroPostToy_50_0_k7/data/benchmark_results.xlsx`

After multiple runs:

1. **Algorithm Ranking:** `results/SCORECARD.md`
2. **Performance Trends:** `results/DASHBOARD.md` + `results/DASHBOARD_timeline.png`
3. **Master Index:** `results/RUNS_INDEX.md`

---

## 📦 Dependencies

Optional (auto-detected):

```bash
pip install openpyxl matplotlib pandas
```

If not installed:
- ✓ Excel export skipped (but CSV/JSON work)
- ✓ Analysis plots skipped (but reports work)

---

## 💡 Pro Tips

1. **Find latest run:** `ls -t results/ | grep -E '^[0-9]{4}-[0-9]{2}' | head -1`
2. **View timestamp folder:** `open results/2026-04-18_182543/`
3. **Quick stats:** Check `SCORECARD.md` for algorithm rankings
4. **Compare across runs:** Use `DASHBOARD.md` navigation
5. **Excel analysis:** Open `benchmark_results.xlsx` in Excel/Sheets for pivot tables

---

## 🎨 Color Coding in Visualizations

- 🟢 **Green** = Good (within 5% of best)
- 🟡 **Orange** = Okay (5-15% from best)
- 🔴 **Red** = Poor (>15% from best)

---

## Example Files

### manifest.md
Shows what instances were run in each timestamped run folder

### RUNS_INDEX.md
Historical index: when each run happened, what instances, best result

### DASHBOARD.md
Landing page for quick navigation to recent runs

### SCORECARD.md
Algorithm rankings across all runs (average performance)

---

**Enjoy your organized, analyzed, beautiful benchmark results! 🎉**
