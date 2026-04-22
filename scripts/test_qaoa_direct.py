#!/usr/bin/env python3
"""Direct test of QAOA leaf solver."""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scripts.benchmark import _leaf_solve_qaoa, Node, route_distance

# Create a small test instance
depot = Node(0, 0, 0, 0)  # id, x, y, demand
nodes = [
    Node(1, 1.0, 0.0, 1),
    Node(2, 0.0, 1.0, 1),
    Node(3, -1.0, 0.0, 1),
    Node(4, 0.0, -1.0, 1),
]

# Create distance map
dist_map = {}
all_nodes = [depot] + nodes
for i, n1 in enumerate(all_nodes):
    for j, n2 in enumerate(all_nodes):
        dx = n1.x - n2.x
        dy = n1.y - n2.y
        dist_map[(n1.id, n2.id)] = np.sqrt(dx*dx + dy*dy)

print("Testing QAOA leaf solver on 4-node TSP...")
print(f"Distance map: {dist_map}")

node_ids = [n.id for n in nodes]
print(f"\nNode IDs: {node_ids}")

# Test QAOA
result = _leaf_solve_qaoa(node_ids, depot.id, dist_map)
print(f"QAOA result: {result}")

# Compute distance
dist = route_distance(result, depot.id, dist_map)
print(f"Route distance: {dist:.2f}")
print("\n✓ QAOA test completed successfully!")
