#!/usr/bin/env python3
"""
OR-Tools VRP Solver — Find Best Number of Vehicles

Runs OR-Tools on a selected instance with different K values,
finds the minimum number of vehicles that satisfies the route-length
constraint, and reports the total distance.

Usage:
    python find_best_vehicles.py                  # Interactive selector
    python find_best_vehicles.py 100 0            # Instance: 100 nodes, variant 0
"""

import sys
import os
import importlib
import re
from pathlib import Path
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


def list_available_instances():
    """Scan Instances/ directory and return list of (nodes, variant) tuples."""
    instances_dir = Path("Instances")
    if not instances_dir.exists():
        return []
    instances = []
    for category_dir in sorted(instances_dir.iterdir()):
        if category_dir.is_dir():
            for vrp_file in sorted(category_dir.glob("*.vrp")):
                stem = vrp_file.stem  # e.g., "RioClaroPostToy_100_0"
                parts = stem.rsplit('_', 2)
                if len(parts) == 3:
                    _, nodes, variant = parts
                    instances.append({
                        'nodes': int(nodes),
                        'variant': int(variant),
                        'full_name': stem,
                        'vrp_path': str(vrp_file),
                    })
    return instances


def parse_vrp_constraints(vrp_path):
    """Read MAX_VEHICLES and MAX_ALLOWED_ROUTE from the .vrp file header."""
    max_vehicles = None
    max_route = None
    with open(vrp_path, 'r') as f:
        for line in f:
            if line.startswith("MAX_VEHICLES"):
                max_vehicles = int(line.split(":")[1].strip())
            elif line.startswith("MAX_ALLOWED_ROUTE"):
                max_route = float(line.split(":")[1].strip())
            elif line.startswith("EDGE_WEIGHT_SECTION"):
                break
    return max_vehicles, max_route


def load_distance_matrix(instance_name):
    """Import the distance matrix from arrays/<instance_name>.py."""
    array_module = importlib.import_module(f"arrays.{instance_name}")
    return array_module.dist_matrix


def solve_ortools(dist_matrix, k, max_route_dist, time_limit=30):
    """Run OR-Tools VRP with k vehicles and a max route distance constraint.

    Returns (total_distance, routes) if feasible, else (None, None).
    """
    n = len(dist_matrix)
    SCALE = 1000

    def or_dist(i, j):
        return int(dist_matrix[i][j] * SCALE)

    manager = pywrapcp.RoutingIndexManager(n, k, 0)
    routing = pywrapcp.RoutingModel(manager)

    transit_cb_idx = routing.RegisterTransitCallback(
        lambda i, j: or_dist(manager.IndexToNode(i), manager.IndexToNode(j))
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # Distance constraint: each route's total distance must not exceed max_route_dist
    routing.AddDimension(
        transit_cb_idx,
        0,
        int(max_route_dist * SCALE),
        True,
        'Distance'
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = time_limit

    assignment = routing.SolveWithParameters(search_params)

    if not assignment:
        return None, None

    total_dist = 0.0
    routes = []
    for v in range(k):
        idx = routing.Start(v)
        route = []
        route_dist = 0.0
        while not routing.IsEnd(idx):
            node_idx = manager.IndexToNode(idx)
            if node_idx != 0:
                route.append(node_idx)
            next_idx = assignment.Value(routing.NextVar(idx))
            route_dist += dist_matrix[manager.IndexToNode(idx)][manager.IndexToNode(next_idx)]
            idx = next_idx
        if route:
            routes.append({'nodes': route, 'distance': route_dist})
            total_dist += route_dist

    return total_dist, routes


def find_minimum_vehicles(dist_matrix, max_vehicles, max_route_dist, time_limit=30):
    """Binary search for the minimum K that yields a feasible solution."""
    print(f"\n{'='*70}")
    print(f"Searching for minimum number of vehicles...")
    print(f"Max vehicles: {max_vehicles}, Max route distance: {max_route_dist:.2f}")
    print(f"{'='*70}\n")

    results = {}
    lo, hi = 1, max_vehicles
    best_k = None

    # Linear scan from low K upward (so we find the true minimum)
    for k in range(1, max_vehicles + 1):
        print(f"  Trying K={k:2d}... ", end="", flush=True)
        total_dist, routes = solve_ortools(dist_matrix, k, max_route_dist, time_limit)
        if total_dist is not None:
            # Check all routes respect max_route_dist
            longest = max(r['distance'] for r in routes) if routes else 0
            feasible = longest <= max_route_dist + 1e-6
            status = "✓ feasible" if feasible else f"✗ longest route {longest:.1f} > {max_route_dist:.1f}"
            print(f"{status}  total={total_dist:.2f}  routes={len(routes)}")
            results[k] = {'total': total_dist, 'routes': routes, 'feasible': feasible, 'longest': longest}
            if feasible and best_k is None:
                best_k = k
                # Found the minimum — keep going a few more to show the trend
                print(f"\n  ★ MINIMUM FEASIBLE K = {k}")
                # Optionally continue to find a better tradeoff
                for k2 in range(k + 1, min(k + 4, max_vehicles + 1)):
                    print(f"  Trying K={k2:2d}... ", end="", flush=True)
                    total_dist2, routes2 = solve_ortools(dist_matrix, k2, max_route_dist, time_limit)
                    if total_dist2 is not None:
                        longest2 = max(r['distance'] for r in routes2) if routes2 else 0
                        feas2 = longest2 <= max_route_dist + 1e-6
                        print(f"{'✓' if feas2 else '✗'}  total={total_dist2:.2f}  routes={len(routes2)}")
                        results[k2] = {'total': total_dist2, 'routes': routes2, 'feasible': feas2, 'longest': longest2}
                    else:
                        print("infeasible")
                break
        else:
            print("infeasible (no solution found)")

    return best_k, results


def main():
    args = sys.argv[1:]
    instances = list_available_instances()

    if not instances:
        print("❌ No instances found in Instances/ directory")
        return

    # Select instance
    selected = None
    if len(args) == 2:
        nodes_arg, variant_arg = int(args[0]), int(args[1])
        for inst in instances:
            if inst['nodes'] == nodes_arg and inst['variant'] == variant_arg:
                selected = inst
                break
        if selected is None:
            print(f"❌ Instance {nodes_arg}_{variant_arg} not found")
            return
    else:
        print("\n" + "=" * 70)
        print("AVAILABLE INSTANCES")
        print("=" * 70)
        for i, inst in enumerate(instances):
            print(f"  [{i}] {inst['full_name']}  ({inst['nodes']} nodes, variant {inst['variant']})")
        print()
        choice = input("Select instance number: ").strip()
        try:
            selected = instances[int(choice)]
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            return

    print(f"\n{'='*70}")
    print(f"SELECTED: {selected['full_name']}")
    print(f"{'='*70}")

    # Parse constraints
    max_vehicles, max_route = parse_vrp_constraints(selected['vrp_path'])
    print(f"MAX_VEHICLES:       {max_vehicles}")
    print(f"MAX_ALLOWED_ROUTE:  {max_route:.2f}")

    # Load distance matrix
    dist_matrix = load_distance_matrix(selected['full_name'])
    print(f"Distance matrix:    {len(dist_matrix)}x{len(dist_matrix)}")

    # Find minimum feasible K
    best_k, results = find_minimum_vehicles(dist_matrix, max_vehicles, max_route)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if best_k is None:
        print(f"❌ No feasible solution found with up to {max_vehicles} vehicles")
        return

    # Find best actual-routes-used across feasible solutions
    min_actual_routes = min(len(r['routes']) for r in results.values() if r['feasible'])
    best_by_distance = min(
        (k for k, r in results.items() if r['feasible']),
        key=lambda k: (len(results[k]['routes']), results[k]['total'])
    )

    print(f"\nResults table:")
    print(f"  {'K':>4} | {'Routes Used':>12} | {'Total Dist':>12} | {'Longest':>12} | {'Status':>10}")
    print(f"  {'-'*4}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    for k in sorted(results.keys()):
        r = results[k]
        status = "feasible" if r['feasible'] else "infeasible"
        print(f"  {k:>4} | {len(r['routes']):>12} | {r['total']:>12.2f} | {r['longest']:>12.2f} | {status:>10}")

    print(f"\n{'='*70}")
    print(f"★ RECOMMENDED K = {min_actual_routes}  (actual routes needed)")
    print(f"  - Min K tried by OR-Tools: {best_k}")
    print(f"  - Best-distance K: {best_by_distance} (total: {results[best_by_distance]['total']:.2f})")
    print(f"{'='*70}")
    print(f"→ Plug K = {min_actual_routes} into the notebook for instance {selected['full_name']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
