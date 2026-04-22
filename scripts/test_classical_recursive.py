#!/usr/bin/env python3
"""Test classical recursive solver first."""

import warnings
warnings.filterwarnings('ignore')

from scripts.benchmark import (
    load_instance, build_dist_map, algo_recursive
)

# Load instance
instance_name = "RioClaroPostToy_50_0"
nodes, depot, dist_matrix = load_instance(instance_name)
dist_map = build_dist_map(nodes, depot, dist_matrix)
node_map = {n.id: n for n in nodes}

k = 7

print(f"\nTesting Classical Recursive solver on {instance_name} (n={len(nodes)}, k={k})")
print("=" * 70)

# Test classical recursive
print("\n1. Testing: Recursive (NO 2-opt) [Classical]")
import time
start = time.perf_counter()
try:
    sol = algo_recursive(nodes, k, depot, dist_map, node_map)
    elapsed = time.perf_counter() - start
    if sol:
        print(f"   ✓ Solution found in {elapsed:.2f}s!")
        print(f"   Total distance: {sol.total_distance:.2f}")
        print(f"   Distance std: {sol.distance_std:.2f}")
        print(f"   Weighted fairness: {sol.weighted_fairness:.2f}")
        print(f"   Vehicles used: {sol.num_vehicles_used}")
    else:
        print(f"   ✗ No solution returned after {elapsed:.2f}s")
except Exception as e:
    elapsed = time.perf_counter() - start
    print(f"   ✗ Error after {elapsed:.2f}s: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Classical recursive test completed!")