# 📊 Instance Selector Guide

## How to Use

The notebook now has an **interactive instance selector** at the beginning that allows you to choose which VRP instance to analyze.

### Step 1: Run Cell 1 (Instance Selector)
Execute the first code cell to see available instances:

```
🔍 AVAILABLE INSTANCES
════════════════════════════════════════════════════════════════════════════════

📦 RioClaroPostToy_100 (100 nodes)
   [0] Variant 0
   [1] Variant 1
   [2] Variant 2

📦 RioClaroPostToy_200 (200 nodes)
   [0] Variant 0
   [1] Variant 1
   [2] Variant 2

📦 RioClaroPostToy_50 (50 nodes)
   [0] Variant 0
   [1] Variant 1
   [2] Variant 2
```

### Step 2: Select Instance from Dropdown
A dropdown menu will appear showing all 9 available instances:
- **RioClaroPostToy_50** (50 nodes) - Variants 0, 1, 2
- **RioClaroPostToy_100** (100 nodes) - Variants 0, 1, 2
- **RioClaroPostToy_200** (200 nodes) - Variants 0, 1, 2

### Step 3: Run Remaining Cells
Continue with the rest of the notebook - it will automatically use the selected instance:

- **Cell 2**: Imports and utility functions
- **Cell 3**: Instance data loading (uses `SELECTED_INSTANCE` from Cell 1)
- **Cells 4+**: Algorithm execution, analysis, and result generation

## Available Instances

| Instance | Nodes | Location | Array File |
|----------|-------|----------|-----------|
| RioClaroPostToy_50_0 | 50 | `Instances/50-Nodes/` | `arrays/RioClaroPostToy_50_0.py` |
| RioClaroPostToy_50_1 | 50 | `Instances/50-Nodes/` | `arrays/RioClaroPostToy_50_1.py` |
| RioClaroPostToy_50_2 | 50 | `Instances/50-Nodes/` | `arrays/RioClaroPostToy_50_2.py` |
| RioClaroPostToy_100_0 | 100 | `Instances/100-Nodes/` | `arrays/RioClaroPostToy_100_0.py` |
| RioClaroPostToy_100_1 | 100 | `Instances/100-Nodes/` | `arrays/RioClaroPostToy_100_1.py` |
| RioClaroPostToy_100_2 | 100 | `Instances/100-Nodes/` | `arrays/RioClaroPostToy_100_2.py` |
| RioClaroPostToy_200_0 | 200 | `Instances/200-Nodes/` | `arrays/RioClaroPostToy_200_0.py` |
| RioClaroPostToy_200_1 | 200 | `Instances/200-Nodes/` | `arrays/RioClaroPostToy_200_1.py` |
| RioClaroPostToy_200_2 | 200 | `Instances/200-Nodes/` | `arrays/RioClaroPostToy_200_2.py` |

## Result Organization

Results are automatically organized by:
- **Instance name** (e.g., `RioClaroPostToy_100`)
- **Vehicle count** (e.g., `7_vehicles`)
- **Result type** (plots, data, logs)

### Directory Structure
```
results/
└── RioClaroPostToy_100/
    └── 7_vehicles/
        ├── plots/          (PNG images, charts)
        ├── data/           (CSV, JSON reports)
        └── logs/           (Text logs, traces)
```

## Workflow

1. **Run Cell 1** → Select instance from dropdown
2. **Run Cells 2-8** → Setup & utilities (unchanged by instance)
3. **Run Cell 9** → Load selected instance's data dynamically
4. **Run Cell 10+** → Execute algorithms on selected instance
5. **Results saved** → Automatically organized in `results/` directory

## Technical Details

### Instance Selection Variables
After running Cell 1, the following variables are available:

```python
SELECTED_INSTANCE = {
    'full_name': 'RioClaroPostToy_100_0',      # Full instance identifier
    'base_name': 'RioClaroPostToy_100',        # Instance base name
    'variant': '0',                             # Variant number
    'nodes': '100',                             # Number of delivery nodes
    'path': 'Instances/100-Nodes/RioClaroPostToy_100_0.vrp'  # VRP file path
}
```

### Dynamic Data Loading
Cell 9 uses `SELECTED_INSTANCE` to dynamically import the correct array file:

```python
instance_array_name = SELECTED_INSTANCE['full_name']
array_module_path = f"arrays.{instance_array_name}"
array_module = importlib.import_module(array_module_path)
DIST_MATRIX_IMPORTED = array_module.dist_matrix
node_coords = array_module.node_coords
```

### Results Directory Setup
Cell 10 uses the selected instance to organize results:

```python
instance_name = SELECTED_INSTANCE['base_name']  # e.g., "RioClaroPostToy_100"
vehicle_dir = os.path.join(results_base, instance_name, f"{K}_vehicles")
```

## Tips

✅ **Default Selection**: If you don't change the dropdown, it defaults to `RioClaroPostToy_100_0`

✅ **Quick Testing**: Use `RioClaroPostToy_50` for fast testing (50 nodes = faster execution)

✅ **Batch Analysis**: Run the notebook multiple times with different instance selections to analyze all variants

✅ **No Restart Needed**: If you want to switch instances, just:
   1. Run Cell 1 again and select a different instance
   2. Run Cell 9 again to reload the new instance's data
   3. Continue from Cell 10 with the new instance

## Example Scenarios

### Scenario 1: Test with 50-node instance
1. Run Cell 1
2. Select: "RioClaroPostToy_50 - 50 nodes (var 0)"
3. Run remaining cells → Results saved to `results/RioClaroPostToy_50/7_vehicles/`

### Scenario 2: Compare multiple variants
1. Run Cell 1, select variant 0 → run all cells
2. Run Cell 1 again, select variant 1 → run relevant cells
3. Run Cell 1 again, select variant 2 → run relevant cells
4. Compare results in `results/` directory

### Scenario 3: Full benchmark (all instances)
Run the notebook 9 times, selecting each instance:
- RioClaroPostToy_50_0, 50_1, 50_2
- RioClaroPostToy_100_0, 100_1, 100_2
- RioClaroPostToy_200_0, 200_1, 200_2

Results will be organized automatically in hierarchical structure.
