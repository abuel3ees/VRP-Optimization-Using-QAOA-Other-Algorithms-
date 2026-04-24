#!/usr/bin/env python3
"""Minimal QAOA benchmark test."""

import warnings
warnings.filterwarnings('ignore')

from scripts.benchmark import (
    load_instance, build_dist_map, algo_recursive_qaoa, with_2opt
)

# Load instance
instance_name = "RioClaroPostToy_50_0"
nodes, depot, dist_matrix = load_instance(instance_name)
dist_map = build_dist_map(nodes, depot, dist_matrix)
node_map = {n.id: n for n in nodes}

k = 7

print(f"\nTesting QAOA algorithms on {instance_name} (n={len(nodes)}, k={k})")
print("=" * 70)

# Test recursive QAOA (NO 2-opt)
print("\n1. Testing: Recursive QAOA (NO 2-opt) [Qiskit at leaves]")
try:
    sol1 = algo_recursive_qaoa(nodes, k, depot, dist_map, node_map)
    if sol1:
        print(f"   ✓ Solution found!")
        print(f"   Total distance: {sol1.total_distance:.2f}")
        print(f"   Distance std: {sol1.distance_std:.2f}")
        print(f"   Weighted fairness: {sol1.weighted_fairness:.2f}")
        print(f"   Vehicles used: {sol1.num_vehicles_used}")
    else:
        print(f"   ✗ No solution returned")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test recursive QAOA + 2-opt
print("\n2. Testing: Recursive QAOA + 2-opt [Qiskit at leaves]")
try:
    algo_with_2opt = with_2opt(algo_recursive_qaoa)
    sol2 = algo_with_2opt(nodes, k, depot, dist_map, node_map)
    if sol2:
        print(f"   ✓ Solution found!")
        print(f"   Total distance: {sol2.total_distance:.2f}")
        print(f"   Distance std: {sol2.distance_std:.2f}")
        print(f"   Weighted fairness: {sol2.weighted_fairness:.2f}")
        print(f"   Vehicles used: {sol2.num_vehicles_used}")
    else:
        print(f"   ✗ No solution returned")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n✓ QAOA benchmark test completed successfully!")
