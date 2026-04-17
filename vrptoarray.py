"""
extract_distance_matrix.py
===========================
Reads a .vrp file (FULL_MATRIX format) and writes the distance matrix
and node coordinates as a Python file inside an 'arrays' folder.

Usage:
    python extract_distance_matrix.py <instance.vrp>

Output:
    arrays/<instance_name>.py   — contains dist_matrix and coordinates
"""

import sys
import os
import numpy as np


def extract_data(path: str):
    with open(path) as f:
        lines = [ln.rstrip() for ln in f.readlines()]

    dimension = None
    for ln in lines:
        if ln.startswith('DIMENSION'):
            dimension = int(ln.split(':')[1].strip())
            break

    # Collect distance matrix values
    in_weights = False
    raw_values = []
    in_display = False
    coords = {}  # node_id -> (x, y)

    for ln in lines:
        if ln.startswith('EDGE_WEIGHT_SECTION'):
            in_weights = True
            in_display = False
            continue
        if ln.startswith('DISPLAY_DATA_SECTION'):
            in_weights = False
            in_display = True
            continue
        if ln.startswith('DEPOT_SECTION') or ln.strip() == 'EOF':
            in_weights = False
            in_display = False
            continue

        if in_weights:
            stripped = ln.strip()
            if stripped and stripped not in ('-1', 'EOF'):
                raw_values.extend(float(v) for v in stripped.split())
        elif in_display:
            parts = ln.strip().split()
            if len(parts) == 3:
                nid, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                coords[nid] = (x, y)

    # Build distance matrix — (DIMENSION+1) x (DIMENSION+1), node 0 = depot
    n = dimension + 1
    expected = n * n

    if len(raw_values) >= expected:
        mat = np.array(raw_values[:expected]).reshape(n, n)
    else:
        mat = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                mat[i][j] = raw_values[idx]
                mat[j][i] = raw_values[idx]
                idx += 1

    return mat, coords, dimension


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(base_dir, "instances")

    vrp_files = []
    for root, dirs, files in os.walk(instances_dir):
        for file in files:
            if file.endswith('.vrp'):
                vrp_files.append(os.path.join(root, file))

    if not vrp_files:
        print(f"No .vrp files found in {instances_dir}")
        sys.exit(1)

    out_dir = os.path.join(base_dir, "arrays")
    os.makedirs(out_dir, exist_ok=True)

    for vrp_path in vrp_files:
        mat, coords, dimension = extract_data(vrp_path)
        instance_name = os.path.splitext(os.path.basename(vrp_path))[0]
        out_path = os.path.join(out_dir, f"{instance_name}.py")

        with open(out_path, 'w') as f:
            f.write(f"# Extracted from: {vrp_path}\n")
            f.write(f"# {dimension} delivery nodes + 1 depot = {dimension + 1} total\n")
            f.write(f"# node 0 = depot, nodes 1..{dimension} = delivery nodes\n\n")

            # Distance matrix
            f.write("dist_matrix = [\n")
            for row in mat:
                formatted = ', '.join(f'{v:.3f}' for v in row)
                f.write(f"    [{formatted}],\n")
            f.write("]\n\n")

            # Coordinates as a dict: {node_id: (x, y)}
            f.write("# node_coords[node_id] = (x, y)\n")
            f.write("node_coords = {\n")
            for nid in sorted(coords):
                x, y = coords[nid]
                f.write(f"    {nid}: ({x:.3f}, {y:.3f}),\n")
            f.write("}\n")

        print(f"Saved: {out_path}")
        print(f"  dist_matrix : {mat.shape}")
        print(f"  node_coords : {len(coords)} nodes (0 = depot)")