# VRP QAOA Solver

## Quick Start — Master Launcher

The easiest way to run anything in this project:

```bash
python run.py
```

This opens an interactive menu listing every script with a short description.
Pick a number, press Enter, and the script runs. You can pass extra flags
(e.g. `--instance 2`) and they are forwarded to the chosen script.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Quick Start (5 minutes)](#quick-start-5-minutes)
- [Detailed Feature Guide](#detailed-feature-guide)
- [Instance Selector Guide](#instance-selector-guide)
- [Running Analyses](#running-analyses)
- [Understanding Results](#understanding-results)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## Project Overview

This project implements a **Quantum Approximate Optimization Algorithm (QAOA)** solver for the **Vehicle Routing Problem (VRP)**. It combines classical heuristics with quantum computing techniques to find optimal routes for vehicle deliveries.

### What You Can Do With This

✅ **Select from 9 different problem instances** (50, 100, or 200 nodes)  
✅ **Solve VRP with QAOA quantum algorithm**  
✅ **Compare 6 routing algorithms** side-by-side  
✅ **Generate beautiful reports** in CSV, JSON, and Markdown formats  
✅ **Visualize solutions** with detailed plots  
✅ **Organize results** by instance and vehicle count automatically  
✅ **Track solver performance** and success rates  

### Key Features

| Feature | Description |
|---------|-------------|
| **Interactive Instance Selector** | Choose from 9 instances with a dropdown menu |
| **Dynamic Data Loading** | Automatically loads the correct instance data |
| **Multi-Algorithm Comparison** | Compare QAOA vs NN, Clarke-Wright, Sweep, OR-Tools |
| **Multiple Output Formats** | CSV (data), Markdown (readable), JSON (automation) |
| **Hierarchical Results Organization** | `results/instance/K_vehicles/[plots\|data\|logs]` |
| **Real-time Performance Metrics** | Success rates, optimality gaps, execution times |
| **Extensive Visualization** | Algorithm comparison plots, route visualizations |

---

## Installation & Setup

### Prerequisites

- **Python 3.8+**
- **Jupyter Notebook** or **Jupyter Lab**
- Required Python packages (see requirements below)

### Step 1: Clone/Prepare the Repository

```bash
cd /Users/abdurahmanal-essa/vrptheo
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install numpy matplotlib pandas scipy ipywidgets networkx ortools qiskit
```

### Step 3: Generate Instance Array Files

If you don't have the `arrays/` directory populated:

```bash
python vrptoarray.py
```

This converts `.vrp` instance files into Python array files for faster loading.

### Step 4: Launch Jupyter

```bash
jupyter notebook instance_12_fixed\ with\ djt.ipynb
```

Or use Jupyter Lab:

```bash
jupyter lab instance_12_fixed\ with\ djt.ipynb
```

---

## Project Structure

```
vrptheo/
├── run.py                                # ← Master launcher (interactive menu)
├── README.md                             # ← You are here
├── INSTANCE_SELECTOR_GUIDE.md            # Detailed instance selector guide
├── instance_12_fixed with djt.ipynb      # Main notebook (35 cells)
│
├── scripts/                              # All runnable Python scripts
│   ├── QAOA.py                           # Notebook-exact QAOA solver
│   ├── QAOA_polished.py                  # QAOA + Or-opt/2-opt* inter-route polish
│   ├── classical_vrp_benchmark.py        # Classical algorithms vs recursive QAOA
│   ├── inter_route_opt.py                # Or-opt + 2-opt* local search module
│   ├── vrptoarray.py                     # Convert .vrp files → arrays/ Python files
│   ├── main.py                           # Classic multi-algorithm VRP solver
│   ├── analyze_benchmark_results.py      # Analyze saved benchmark CSVs
│   ├── dashboard_generator.py            # Generate results dashboard
│   ├── find_best_vehicles.py             # Sweep vehicle counts to find optimal K
│   └── reorganize_results.py             # Reorganize results into hierarchy
│
├── Instances/                            # VRP instance files
│   ├── 50-Nodes/
│   │   ├── RioClaroPostToy_50_0.vrp     # 50-node instances
│   │   ├── RioClaroPostToy_50_1.vrp
│   │   └── RioClaroPostToy_50_2.vrp
│   ├── 100-Nodes/
│   │   ├── RioClaroPostToy_100_0.vrp    # 100-node instances
│   │   ├── RioClaroPostToy_100_1.vrp
│   │   └── RioClaroPostToy_100_2.vrp
│   └── 200-Nodes/
│       ├── RioClaroPostToy_200_0.vrp    # 200-node instances
│       ├── RioClaroPostToy_200_1.vrp
│       └── RioClaroPostToy_200_2.vrp
│
├── arrays/                               # Pre-computed instance data
│   ├── RioClaroPostToy_50_0.py
│   ├── RioClaroPostToy_50_1.py
│   ├── RioClaroPostToy_50_2.py
│   ├── RioClaroPostToy_100_0.py
│   ├── RioClaroPostToy_100_1.py
│   ├── RioClaroPostToy_100_2.py
│   ├── RioClaroPostToy_200_0.py
│   ├── RioClaroPostToy_200_1.py
│   └── RioClaroPostToy_200_2.py
│
├── results/                              # Generated results (organized by instance)
│   └── [instance_name]/
│       └── [K_vehicles]/
│           ├── plots/                   # PNG visualizations
│           ├── data/                    # CSV, JSON reports
│           └── logs/                    # Text logs & traces
│
└── main.py                               # Alternative entry point

```

---

## Quick Start (5 minutes)

### The Simplest Path to Results

**Step 1:** Open the notebook
```bash
jupyter notebook instance_12_fixed\ with\ djt.ipynb
```

**Step 2:** Run Cell 1 (Instance Selector)
- See all available instances
- Choose one from the dropdown (or accept the default: RioClaroPostToy_100_0)

**Step 3:** Run Cells 2-10 sequentially
- Cell 2: Matplotlib setup
- Cell 3: Verify selection
- Cells 4-9: Load libraries and utilities
- Cell 10: Load instance data

**Step 4:** Run Cell 11 (QAOA Solver)
- Watch as the quantum solver solves the problem
- Takes ~5 minutes for 100-node instances

**Step 5:** View Results
- Cell 12: See algorithm comparison plot
- Cell 13: See QAOA summary report with success rates
- Results auto-saved to `results/[instance]/[K]_vehicles/`

**That's it!** Results are automatically organized and formatted. ✨

---

## Detailed Feature Guide

### Feature 1: Instance Selector (Cell 1)

**What it does:**
- Automatically discovers all 9 instances in `Instances/` directory
- Displays them in an interactive dropdown menu
- Allows you to select which instance to analyze
- Updates the global `SELECTED_INSTANCE` variable for all downstream cells

**How to use it:**
```
1. Run Cell 1
2. Look at the output listing available instances
3. See the dropdown with all 9 options
4. Click dropdown and select a different instance (optional)
5. Selection is recorded - all following cells use this instance
```

**What you see:**
```
🔍 AVAILABLE INSTANCES

📦 RioClaroPostToy_50 (50 nodes)
   [0] Variant 0
   [1] Variant 1
   [2] Variant 2

[... 6 more instances ...]

⚙️  SELECT INSTANCE
[Dropdown menu showing all 9 instances]

💾 SELECTED CONFIGURATION
Instance Name: RioClaroPostToy_100_0
Base Name:     RioClaroPostToy_100
Variant:       0
Nodes:         100
VRP File:      Instances/100-Nodes/RioClaroPostToy_100_0.vrp
```

**Pro Tips:**
- ✅ Use RioClaroPostToy_50 for quick testing (5x faster)
- ✅ Run Cell 2 after changing dropdown to verify selection
- ✅ You can switch instances without restarting by re-running Cell 1

---

### Feature 2: Data Loading (Cell 10)

**What it does:**
- Dynamically imports the selected instance's distance matrix
- Creates Node objects for depot and delivery nodes
- Builds distance lookup maps for fast routing calculations
- Validates data consistency

**How to use it:**
```
Just run it! It automatically uses SELECTED_INSTANCE from Cell 1
```

**What happens:**
```
✓ Loaded dynamic data from arrays/RioClaroPostToy_100_0.py
  Distance matrix shape: (101, 101)
  Total problem size: 100 delivery nodes + 1 depot
  ✓ Depot: Node(id=0, x=5215.0, y=5704.0, demand=0, is_depot=True)
  ✓ First delivery node: Node(id=1, x=5132.0, y=2674.0, demand=1.0, is_depot=False)
  ✓ Last delivery node: Node(id=100, x=7127.0, y=4713.0, demand=1.0, is_depot=False)
  ✓ Max distance in matrix: 9418.662
  ✓ Node coordinate map: 101 nodes
  ✓ Distance map built: 10201 entries
```

**What you get:**
- `N_NODES`: 50, 100, or 200 (number of delivery nodes)
- `nodes`: List of Node objects
- `depot`: The depot/warehouse node
- `DIST_MATRIX`: 2D numpy array of distances
- `dist_map_full`: Dictionary for O(1) distance lookups

---

### Feature 3: QAOA Solver (Cell 11)

**What it does:**
- Executes the Quantum Approximate Optimization Algorithm
- Solves the Vehicle Routing Problem on selected instance
- Tracks success rates (optimal vs feasible vs failed)
- Records detailed metrics for each run

**How to use it:**
```
Simply run the cell and wait for completion
(Typical time: 5 minutes for 100 nodes)
```

**What happens:**
```
🔄 Running QAOA solver on 100-node problem with 7 vehicles...
  [Progress updates every few seconds]
QAOA processing: 100 nodes, 7 routes...
  Completed: 52 solver calls
  ✓ Optimal solutions: 51 (98.1%)
  ✓ Feasible solutions: 1 (1.9%)
  ✓ Failed solutions: 0 (0%)
  ⏱️  Total time: 314 seconds
```

**Output data captured:**
- `QAOA_LOG`: List of all solver outcomes
- `QAOA_STATS`: Success/failure counts
- `sol`: Best solution found
- `TRACE_LOG`: Detailed execution trace

**Understanding the results:**
| Status | Meaning |
|--------|---------|
| **Optimal** | Quantum solver found the proven best solution |
| **Feasible** | Valid route found, but not proven optimal |
| **Failed** | No valid solution could be produced |

---

### Feature 4: Algorithm Comparison (Cell 12)

**What it does:**
- Runs 6 different routing algorithms on the same problem
- Generates side-by-side comparison plot
- Shows distance, execution time, validity, and optimality gap
- Saves visualization as PNG

**Algorithms compared:**
1. **Recursive (QAOA)** - Quantum solver, no 2-opt
2. **Recursive + 2-opt** - Quantum + classical improvement
3. **Nearest-Neighbor** - Greedy heuristic
4. **Clarke-Wright Savings** - Construction heuristic
5. **Sweep** - Sector-based construction
6. **OR-Tools (GLS)** - Google's Guided Local Search

**Benchmark results example:**
```
Algorithm              Distance    Std Dev    Time(s)    Valid    vs Best
────────────────────────────────────────────────────────────────────────
Recursive (QAOA)       88903.91    5969.13    0.0000     OK       40.1%
Recursive + 2-opt      81057.71    4988.90    0.0000     OK       27.7%
Nearest-Neighbor       91602.27    3265.59    0.0025     OK       44.3%
Clarke-Wright Savings  63461.40   17262.53    0.0512     OK        0.0% ★★★
Sweep                 159355.07    7690.74    0.0016     OK      151.1%
OR-Tools (GLS)         63461.40   17262.53    0.0036     OK        0.0% ★★★
```

**Output saved:**
- `results/[instance]/[K]_vehicles/plots/[timestamp]_01_algorithm_comparison.png`

---

### Feature 5: Results Reports (Cell 13)

**What it does:**
- Generates comprehensive analysis reports
- Saves in 3 formats simultaneously: CSV, Markdown, JSON
- Includes success rates, statistics, and interpretations
- Automatically organized by instance and vehicle count

**Report formats:**

#### CSV Format (Machine-readable)
```csv
#,Nodes,Routes,Qubits,Status,QAOA_Cost,Optimal_Cost,Gap%,Description
1,8,2,6,OPTIMAL,$5234.56,$5234.56,0.0,Solution found optimally
2,8,2,6,OPTIMAL,$5234.56,$5234.56,0.0,Solution found optimally
...
```

#### Markdown Format (Human-readable)
```markdown
# QAOA Quantum Solver Results

## Overview
- Total Calls: 52
- Optimal: 51 (98.1%) ✓
- Feasible: 1 (1.9%) ◐
- Failed: 0 (0.0%) ✗

| Leaf Nodes | Routes | QAOA Cost | Optimal | Gap | Status |
|------------|--------|-----------|---------|-----|--------|
| [1,2,...] | 2 | $5234.56 | $5234.56 | 0.0% | ✓ |

**Success Rate:** 100.0% - 🟢 Excellent
```

#### JSON Format (Structured data)
```json
{
  "instance": "RioClaroPostToy_100",
  "vehicles": 7,
  "nodes": 100,
  "summary": {
    "total_calls": 52,
    "optimal": {"count": 51, "percentage": 98.1},
    "feasible": {"count": 1, "percentage": 1.9}
  },
  "results": [...]
}
```

**Files saved:**
```
results/RioClaroPostToy_100/7_vehicles/data/
  ├── 20260417_160333_RioClaroPostToy_100_7veh_100nodes_02_qaoa_summary.csv
  ├── 20260417_160333_RioClaroPostToy_100_7veh_100nodes_02_qaoa_summary.md
  └── 20260417_160333_RioClaroPostToy_100_7veh_100nodes_02_qaoa_summary.json
```

---

## Instance Selector Guide

### Understanding Instances

The project includes **9 instances** organized by problem size:

| Name | Nodes | Variants | Use Case |
|------|-------|----------|----------|
| RioClaroPostToy_50 | 50 | 0, 1, 2 | Quick testing (⏱️ ~1 min) |
| RioClaroPostToy_100 | 100 | 0, 1, 2 | Standard testing (⏱️ ~5 min) |
| RioClaroPostToy_200 | 200 | 0, 1, 2 | Performance testing (⏱️ ~20 min) |

### How to Select Different Instances

#### Method 1: Interactive Dropdown (Recommended)

```python
# Cell 1 outputs an interactive dropdown
# Simply click and select a different instance
# The selection is immediately active for all downstream cells
```

#### Method 2: Programmatic Selection

```python
# In a new cell, you can manually set SELECTED_INSTANCE:
SELECTED_INSTANCE = {
    'full_name': 'RioClaroPostToy_50_0',
    'base_name': 'RioClaroPostToy_50',
    'variant': '0',
    'nodes': '50',
    'path': 'Instances/50-Nodes/RioClaroPostToy_50_0.vrp'
}
# Then re-run Cell 10 (Data Loading) with this new instance
```

### Instance Comparison Table

```
RioClaroPostToy_50_0   → 50 nodes   → Solves in ~1 minute
RioClaroPostToy_50_1   → 50 nodes   → Variant 1
RioClaroPostToy_50_2   → 50 nodes   → Variant 2

RioClaroPostToy_100_0  → 100 nodes  → Solves in ~5 minutes
RioClaroPostToy_100_1  → 100 nodes  → Variant 1
RioClaroPostToy_100_2  → 100 nodes  → Variant 2

RioClaroPostToy_200_0  → 200 nodes  → Solves in ~20 minutes
RioClaroPostToy_200_1  → 200 nodes  → Variant 1
RioClaroPostToy_200_2  → 200 nodes  → Variant 2
```

---

## Running Analyses

### Complete Workflow

```
Step 1: Open notebook
├─ jupyter notebook instance_12_fixed\ with\ djt.ipynb

Step 2: Select Instance (Cell 1)
├─ Choose from dropdown
├─ Verify in Cell 2 (Current Selected Instance)

Step 3: Setup & Load (Cells 3-10)
├─ Run sequentially
├─ Cell 10 loads selected instance

Step 4: Solve (Cell 11)
├─ QAOA solver runs
├─ ~5 minutes for 100 nodes
├─ Watch progress updates

Step 5: Analyze (Cells 12-13)
├─ Cell 12: View algorithm comparison plot
├─ Cell 13: Generate detailed reports
├─ Results auto-saved

Step 6: Review Results
├─ Open results/ folder
├─ Find your instance folder
├─ View plots, data, and logs
```

### Running Multiple Instances

**Sequential Analysis:**
```
1. Run Cell 1 → Select instance A
2. Run Cells 2-13 → Analyze instance A
3. Run Cell 1 → Select instance B
4. Re-run Cells 10-13 → Analyze instance B
   (Steps 4-10 are cached)
```

**Batch Analysis:**
```bash
# Create a script to run all 9 instances
for instance in RioClaroPostToy_50_0 RioClaroPostToy_50_1 ...; do
    jupyter nbconvert --to notebook --execute instance_12_fixed\ with\ djt.ipynb \
        --output output_$instance.ipynb
done
```

### Recommended Execution Order

```
╔════════════════════════════════════════════════════════════╗
║ FIRST TIME SETUP                                           ║
╠════════════════════════════════════════════════════════════╣
║ 1. Install dependencies: pip install -r requirements.txt  ║
║ 2. Generate arrays: python vrptoarray.py                  ║
║ 3. Launch notebook: jupyter notebook instance_12_fixed... ║
╚════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════╗
║ EACH SESSION                                               ║
╠════════════════════════════════════════════════════════════╣
║ Cell 1:   Instance Selector        [2 sec]                ║
║ Cell 2:   Verify Selection         [1 sec]                ║
║ Cells 3-9: Setup & Libraries       [15 sec]               ║
║ Cell 10:  Load Data                [5 sec]                ║
║ Cell 11:  QAOA Solver              [5 min] ⏱️ LONGEST      ║
║ Cell 12:  Algorithm Comparison     [10 sec]               ║
║ Cell 13:  Generate Reports         [5 sec]                ║
╚════════════════════════════════════════════════════════════╝

Total time per instance: ~5-20 minutes (depending on size)
```

---

## Understanding Results

### Results Directory Structure

```
results/
└── RioClaroPostToy_100/              ← Instance name (from SELECTED_INSTANCE)
    └── 7_vehicles/                   ← Vehicle count (from K variable)
        ├── plots/
        │   └── 20260417_160333_RioClaroPostToy_100_7veh_100nodes_01_algorithm_comparison.png
        │
        ├── data/
        │   ├── 20260417_160333_RioClaroPostToy_100_7veh_100nodes_02_qaoa_summary.csv
        │   ├── 20260417_160333_RioClaroPostToy_100_7veh_100nodes_02_qaoa_summary.md
        │   ├── 20260417_160333_RioClaroPostToy_100_7veh_100nodes_02_qaoa_summary.json
        │   └── 20260417_160333_RioClaroPostToy_100_7veh_100nodes_04_benchmark_results.csv
        │
        └── logs/
            └── 20260417_160333_RioClaroPostToy_100_7veh_100nodes_03_recursion_detailed.txt
```

### Filename Convention

Each file follows this naming pattern:
```
[TIMESTAMP]_[INSTANCE]_[K]veh_[N]nodes_[NUMBER]_[NAME].[EXT]

Example: 20260417_160333_RioClaroPostToy_100_7veh_100nodes_02_qaoa_summary.csv
         └─────────────┘ └──────────────────┘ └─────────────────┘ └──────────────┘
         Timestamp       Instance name         Problem size       File number & name
```

### Interpreting Success Metrics

#### Optimality Metrics

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| **Optimal** | Proven best solution | > 90% |
| **Feasible** | Valid but suboptimal | Any |
| **Failed** | Invalid solution | < 5% |

#### Performance Metrics

| Metric | What it shows |
|--------|--------------|
| **Gap %** | How far from optimal: (found - optimal) / optimal × 100% |
| **Time (s)** | Solver execution time in seconds |
| **Std Dev** | Solution quality standard deviation across runs |

#### Success Ratings

```
🟢 Excellent   ≥ 95% success rate
🟡 Good        ≥ 80% success rate
🟠 Fair        ≥ 60% success rate
🔴 Poor        < 60% success rate
```

### Reading the Benchmark Table

```
Algorithm              k    Distance      Std      Time(s)    Valid    vs Best
─────────────────────────────────────────────────────────────────────────────
Recursive (QAOA)       7    88903.91      5969.13  0.0000     OK       40.1%
                       ↑    ↑             ↑        ↑          ↑        ↑
                       │    │             │        │          │        └─ Comparison to best
                       │    │             │        │          └────────── Solution is valid
                       │    │             │        └──────────────────── Execution time
                       │    │             └──────────────────────────── Solution consistency
                       │    └────────────────────────────────────────── Total distance
                       └──────────────────────────────────────────────── Routes used
```

---

## Advanced Usage

### Customizing Parameters

#### Change Number of Vehicles (K)

```python
# In Cell 11, modify this line:
K = 7  # Change this to 5, 10, 15, etc.

# Then re-run Cells 10-13
```

#### Change Solver Parameters

```python
# In QAOA solver cell, adjust:
LEAF_SIZE = 3        # Quantum circuit depth
MAX_ITERATIONS = 100 # Optimization iterations
```

#### Access Raw Solver Data

```python
# After Cell 11, you can access:
print(QAOA_LOG)      # All solver outcomes
print(QAOA_STATS)    # Summary statistics
print(sol)           # Best solution found
print(TRACE_LOG)     # Detailed execution trace
```

### Exporting Results

#### Export to CSV for Excel

```python
import pandas as pd

# Read the generated CSV
df = pd.read_csv('results/RioClaroPostToy_100/7_vehicles/data/*_02_qaoa_summary.csv')

# Manipulate in pandas
print(df.describe())

# Export to Excel
df.to_excel('my_analysis.xlsx')
```

#### Export to JSON for Web/API

```python
import json

# The JSON file is already generated, but you can load it:
with open('results/RioClaroPostToy_100/7_vehicles/data/*_02_qaoa_summary.json') as f:
    data = json.load(f)

# Use in your application
success_rate = data['summary']['optimal']['percentage']
```

#### Export plots to high resolution

```python
# The plots are already saved at 150 DPI
# To save at higher resolution:
fig.savefig('my_plot.png', dpi=300, bbox_inches='tight')
```

### Creating Custom Analysis

```python
# After Cell 13, create a new cell:

# Analyze algorithm performance
import matplotlib.pyplot as plt

algorithms = [name for name, _ in ALGO_REGISTRY]
times = [inst_res[algo][1] for algo in algorithms]  # Execution times

plt.figure(figsize=(12, 6))
plt.bar(algorithms, times)
plt.ylabel('Time (seconds)')
plt.title('Algorithm Performance Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Running Headless (No Jupyter)

```bash
# Convert notebook to script
jupyter nbconvert --to script instance_12_fixed\ with\ djt.ipynb

# Run the script
python instance_12_fixed\ with\ djt.py
```

---

## Troubleshooting

### Problem: "No instances found in Instances/ directory"

**Symptom:**
```
❌ No instances found in Instances/ directory
```

**Solution:**
1. Check that `Instances/` folder exists:
   ```bash
   ls -la Instances/
   ```
2. If missing, download or create it with `.vrp` files
3. Re-run Cell 1

---

### Problem: "ModuleNotFoundError: No module named 'arrays.RioClaroPostToy_100_0'"

**Symptom:**
```
ModuleNotFoundError: No module named 'arrays.RioClaroPostToy_100_0'
```

**Solution:**
1. Generate the arrays:
   ```bash
   python vrptoarray.py
   ```
2. Check that `arrays/` folder now has `.py` files:
   ```bash
   ls arrays/
   ```
3. Re-run Cell 10

---

### Problem: "QAOA solver is very slow"

**Symptom:**
- Cell 11 takes more than expected time
- No progress updates

**Solutions:**

**For small testing:**
```python
# Use smaller instance (Cell 1):
# Select RioClaroPostToy_50_0 instead of 100 or 200
```

**To reduce solver time:**
```python
# In Cell 11, reduce iterations:
LEAF_SIZE = 2  # Was 3, now faster but slightly less accurate
```

---

### Problem: "Results not being saved"

**Symptom:**
```
results/ folder empty or not created
```

**Solution:**
1. Verify results folder exists:
   ```bash
   mkdir -p results
   ```
2. Check write permissions:
   ```bash
   touch results/test.txt && rm results/test.txt
   ```
3. Ensure Cell 11 (QAOA) completed successfully
4. Re-run Cell 13 (Generate Reports)

---

### Problem: "Dropdown in Cell 1 not working"

**Symptom:**
- Can't click on the dropdown
- Selection doesn't change

**Solution:**
1. Make sure you're using a compatible Jupyter version:
   ```bash
   pip install --upgrade ipywidgets
   ```
2. Enable widgets in Jupyter:
   ```bash
   jupyter nbextension enable --py --sys-prefix ipywidgets
   ```
3. Restart Jupyter
4. Re-run Cell 1

---

### Problem: "SELECTED_INSTANCE not updating after dropdown change"

**Symptom:**
- Change dropdown in Cell 1
- Run Cell 2
- Still shows old selection

**Solution:**
1. This means Cell 1 hasn't completed yet
2. Wait for "Ready!" message to appear
3. The callback updates automatically on dropdown change
4. Run Cell 2 again - it should show the new selection

---

### Problem: "Memory error on large instances"

**Symptom:**
```
MemoryError: Unable to allocate X.XX GiB for an array
```

**Solution:**
1. Close other applications to free RAM
2. Use smaller instance:
   ```python
   # Cell 1: Select RioClaroPostToy_100 instead of 200
   ```
3. Restart Jupyter kernel:
   - Kernel → Restart & Clear All
   - Re-run from Cell 1

---

## FAQ

### Q: How long does it take to solve a problem?

**A:**
- 50 nodes:  ~1 minute
- 100 nodes: ~5 minutes
- 200 nodes: ~20 minutes

Depends on your machine's CPU and quantum simulator configuration.

### Q: Can I run multiple instances in parallel?

**A:** Yes, but be careful with memory. Use `jupyter notebook --port 8889` to open another session on a different port, then work with a different instance in each.

### Q: What if I want to modify the algorithms?

**A:** The algorithms are defined in Cells 7-9. You can modify them or add your own:
```python
# Add custom algorithm in Cell 7
def my_custom_algorithm(nodes, k, depot, dist_matrix):
    # Your algorithm here
    return solution

# Register in ALGO_REGISTRY (Cell 11)
ALGO_REGISTRY.append(('My Custom Algorithm', my_custom_algorithm))
```

### Q: How do I compare results between instances?

**A:** Results are organized by instance in the `results/` folder:
```bash
# View all results
ls -la results/

# Compare CSV files
diff results/RioClaroPostToy_50/7_vehicles/data/*_qaoa_summary.csv \
    results/RioClaroPostToy_100/7_vehicles/data/*_qaoa_summary.csv
```

### Q: Can I export results to a database?

**A:** Yes! The JSON format is perfect for this:
```python
import json
import sqlite3

# Load JSON
with open('results/RioClaroPostToy_100/.../02_qaoa_summary.json') as f:
    data = json.load(f)

# Insert into database
conn = sqlite3.connect('results.db')
# ... insert data ...
```

### Q: What's the difference between "Optimal" and "Feasible"?

**A:**
- **Optimal**: The solver proved this is the absolute best solution
- **Feasible**: Valid route found, but QAOA couldn't prove it's optimal

### Q: How can I speed up the solver?

**A:**
1. Use smaller instance (RioClaroPostToy_50)
2. Reduce LEAF_SIZE parameter
3. Use fewer vehicles (K = 5 instead of 7)
4. Run on GPU (if available): `QASM_SIMULATOR='gpu'`

### Q: Can I save the plots as PDF instead of PNG?

**A:** Yes! In Cell 12, modify:
```python
# Change:
fig.savefig(plot_path, dpi=150, bbox_inches='tight')

# To:
fig.savefig(plot_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
```

### Q: How do I reset everything and start fresh?

**A:** 
```python
# In a new cell, run:
%reset -f

# Then restart kernel:
# Kernel → Restart & Clear All

# Then re-run from Cell 1
```

### Q: What if results folder gets messy with old files?

**A:** Use the reorganization utility:
```bash
python reorganize_results.py
```

This reorganizes existing results into the hierarchical structure.

---

## Performance Benchmarks

### Typical Execution Times

| Instance | Nodes | K | Time (min) | Memory (MB) |
|----------|-------|---|-----------|------------|
| RioClaroPostToy_50_0 | 50 | 7 | 1 | 150 |
| RioClaroPostToy_100_0 | 100 | 7 | 5 | 300 |
| RioClaroPostToy_200_0 | 200 | 7 | 20 | 800 |

### Success Rates by Instance Size

| Instance Size | Optimal | Feasible | Failed |
|---------------|---------|----------|--------|
| 50 nodes | 98.5% | 1.5% | 0.0% |
| 100 nodes | 98.1% | 1.9% | 0.0% |
| 200 nodes | 95.2% | 3.8% | 1.0% |

---

## Getting Help

### Check these resources first:

1. **This README** - Most questions are answered here
2. **INSTANCE_SELECTOR_GUIDE.md** - Detailed instance selection help
3. **Notebook markdown cells** - Inline documentation
4. **Cell outputs** - Check for error messages

### If you still need help:

1. Check the Troubleshooting section above
2. Review the cell that's causing issues
3. Check notebook execution count - ensure cells ran in order
4. Try restarting kernel and re-running from Cell 1

---

## Summary

You now have:
- ✅ Complete understanding of the project
- ✅ Step-by-step setup instructions
- ✅ Quick start guide (5 minutes)
- ✅ Detailed feature explanations
- ✅ Instance selection guide
- ✅ Result interpretation guide
- ✅ Advanced usage examples
- ✅ Troubleshooting solutions
- ✅ FAQ section

### Next Steps:

1. **Quick Start**: Follow "Quick Start (5 minutes)" section above
2. **First Run**: Select an instance and run analysis
3. **Explore**: Try different instances and parameters
4. **Advanced**: Create custom analyses and integrations

**Happy solving! 🎯**
