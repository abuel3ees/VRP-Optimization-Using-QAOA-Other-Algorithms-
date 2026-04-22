#!/usr/bin/env python3
"""Test QAOA leaf solver directly on small instance."""

import warnings
warnings.filterwarnings('ignore')

import time
from classical_vrp_benchmark import (
    load_instance, build_dist_map, _leaf_solve_qaoa
)

# Load instance
instance_name = "RioClaroPostToy_50_0"
nodes, depot, dist_matrix = load_instance(instance_name)
dist_map = build_dist_map(nodes, depot, dist_matrix)

# Test 1: Direct QAOA on 3 nodes (small leaf)
print(f"\nTesting QAOA leaf solver")
print("=" * 70)

small_nodes = nodes[:3]  # Take first 3 nodes
node_ids = [n.id for n in small_nodes]

print(f"\nTest 1: QAOA on {len(node_ids)} nodes (leaf)")
print(f"  Node IDs: {node_ids}")

start = time.perf_counter()
try:
    result = _leaf_solve_qaoa(node_ids, depot.id, dist_map)
    elapsed = time.perf_counter() - start
    print(f"  ✓ Result: {result}")
    print(f"  Time: {elapsed:.3f}s")
except Exception as e:
    elapsed = time.perf_counter() - start
    print(f"  ✗ Error after {elapsed:.3f}s: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ QAOA leaf test completed!")