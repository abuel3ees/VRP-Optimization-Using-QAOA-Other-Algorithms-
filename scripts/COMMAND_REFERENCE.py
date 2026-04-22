#!/usr/bin/env python3
"""
Quick Command Reference - Run this to see all available commands
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                     📊 BENCHMARK COMMAND REFERENCE                        ║
╚════════════════════════════════════════════════════════════════════════════╝

🚀 RUN BENCHMARKS
═══════════════════════════════════════════════════════════════════════════

All instances, default k:
  $ python classical_vrp_benchmark.py

Specific instance:
  $ python classical_vrp_benchmark.py --instance 50_0

Specific k vehicles:
  $ python classical_vrp_benchmark.py --instance 50_0 --k 7

Only certain algorithms (uses substring matching):
  $ python classical_vrp_benchmark.py --algos recursive qaoa
  $ python classical_vrp_benchmark.py --algos or-tools


📊 ANALYZE RESULTS
═══════════════════════════════════════════════════════════════════════════

Analyze latest run:
  $ python analyze_benchmark_results.py

Analyze specific run:
  $ python analyze_benchmark_results.py --run 2026-04-18_182543

Analyze all runs:
  $ python analyze_benchmark_results.py --all


🎯 GENERATE DASHBOARD
═══════════════════════════════════════════════════════════════════════════

Create master dashboard (scorecard + timeline):
  $ python dashboard_generator.py


📁 FILE ORGANIZATION
═══════════════════════════════════════════════════════════════════════════

After running a benchmark:

results/
  ├── RUNS_INDEX.md                    ← Master index (when runs happened)
  ├── DASHBOARD.md                     ← Quick navigation
  ├── SCORECARD.md                     ← Algorithm ranking
  ├── DASHBOARD_timeline.png           ← Trends over time
  │
  └── 2026-04-18_182543/               ← Timestamped run folder
      ├── manifest.md                  ← Run details
      ├── RUN_OVERVIEW.md              ← Cross-instance summary
      │
      └── RioClaroPostToy_50_0_k7/    ← Per-instance folder
          ├── data/
          │   ├── benchmark_summary.md           ← Quick table
          │   ├── benchmark_algorithms.csv       ← Raw CSV
          │   ├── benchmark_report.json          ← Full JSON
          │   ├── benchmark_results.xlsx         ← 📊 Excel sheets
          │   └── ANALYSIS_REPORT.md             ← 📈 Detailed analysis
          │
          └── plots/
              ├── 01_distance_vs_fairness.png    ← Bubble chart
              ├── 02_algorithm_comparison.png    ← Top algorithms
              ├── 03_performance_heatmap.png     ← Performance matrix
              └── (individual algorithm plots)


📈 KEY METRICS
═══════════════════════════════════════════════════════════════════════════

For each algorithm:
  • Total Distance    : Sum of all route lengths
  • Distance Std      : Fairness (lower = more fair)
  • Weighted Fairness : (avg_dist + std) / 2
  • Time (s)          : Execution time in seconds
  • Gap %             : % above best solution found


✨ OUTPUT FILES
═══════════════════════════════════════════════════════════════════════════

Per instance:

  summary.md              : Quick reference table
  algorithms.csv          : Parsed CSV data
  report.json             : Full structured data
  results.xlsx            : 📊 Excel with 3 sheets
  ANALYSIS_REPORT.md      : 📈 Detailed statistics & rankings

Visualizations (3 auto-generated):

  01_distance_vs_fairness.png    : Trade-off bubble chart
  02_algorithm_comparison.png    : Top 10 comparison grid
  03_performance_heatmap.png     : Performance matrix heatmap

Across runs:

  SCORECARD.md            : Algorithm ranking (avg across runs)
  DASHBOARD_timeline.png  : How performance changed over runs


💡 TIPS
═══════════════════════════════════════════════════════════════════════════

View latest results:
  $ ls -t results/ | grep '^[0-9]' | head -1 | xargs -I {} cat results/{}/manifest.md

Quick open latest run folder:
  $ open results/$(ls -t results/ | grep '^[0-9]' | head -1)/

Find slowest algorithms:
  $ grep "Slowest" results/*/*/data/ANALYSIS_REPORT.md

Compare Excel sheets:
  $ open results/2026-04-18_182543/RioClaroPostToy_50_0_k7/data/benchmark_results.xlsx

Export to CSV for external analysis:
  $ cat results/2026-04-18_182543/RioClaroPostToy_50_0_k7/data/benchmark_algorithms.csv

View dashboard in browser:
  $ cat results/DASHBOARD.md


📚 DETAILED GUIDE
═══════════════════════════════════════════════════════════════════════════

For full documentation, see:
  → BENCHMARK_ANALYSIS_GUIDE.md


═══════════════════════════════════════════════════════════════════════════
""")
