"""
vrp_solver.py
=============
Distance-Constrained VRP Solver with multiple algorithms.
Reads a .vrp file (FULL_MATRIX format) and runs:
  1. Recursive Cluster-then-Route (with optional 2-opt)
  2. Nearest Neighbour
  3. Clarke-Wright Savings
  4. Sweep
  5. OR-Tools (GLS)

Usage:
    python vrp_solver.py <instance.vrp> [--k K] [--leaf LEAF_SIZE] [--no-2opt]

Arguments:
    instance.vrp   Path to the .vrp file
    --k K          Number of vehicles (overrides MAX_VEHICLES in file)
    --leaf SIZE    Max nodes per leaf before brute-force (default: 6)
    --no-2opt      Skip 2-opt post-processing on the recursive solver
    --algo NAME    Run only one algorithm: recursive, nn, savings, sweep, ortools
    --quiet        Suppress per-algorithm verbose output
"""

import argparse
import math
import time
import itertools
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    id: int
    x: float
    y: float
    demand: float = 1.0
    is_depot: bool = False


@dataclass
class Route:
    node_ids: List[int]
    distance: float = 0.0
    load: float = 0.0


@dataclass
class VRPSolution:
    routes: List[Route]
    total_distance: float = 0.0
    num_vehicles_used: int = 0
    distance_std: float = 0.0

    def __post_init__(self):
        self.num_vehicles_used = len(self.routes)
        self.total_distance = sum(r.distance for r in self.routes)
        if len(self.routes) > 1:
            self.distance_std = float(np.std([r.distance for r in self.routes]))
        else:
            self.distance_std = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# VRP file parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_vrp(path: str):
    """
    Parse a .vrp file with FULL_MATRIX edge weights and TWOD_DISPLAY coords.
    Returns (dist_matrix, nodes, depot, max_vehicles, max_route, instance_name).
    The distance matrix is a 2-D numpy array indexed 0..DIMENSION-1.
    Index 0 is always the depot.
    """
    with open(path) as f:
        content = f.read()

    lines = [ln.rstrip() for ln in content.splitlines()]

    # ── Header ────────────────────────────────────────────────────────────────
    meta = {}
    for ln in lines:
        if ':' in ln and not ln[0].isdigit():
            k, _, v = ln.partition(':')
            meta[k.strip()] = v.strip()

    name         = meta.get('NAME', path)
    dimension    = int(meta.get('DIMENSION', 0))
    max_vehicles = int(meta.get('MAX_VEHICLES', 1))
    max_route    = float(meta.get('MAX_ALLOWED_ROUTE', float('inf')))
    # DIMENSION counts delivery nodes only; the matrix includes depot (node 0)
    # so total matrix size is (DIMENSION+1) x (DIMENSION+1)
    n_total = dimension + 1  # used for matrix reshape below

    # ── Distance matrix ───────────────────────────────────────────────────────
    in_weights = False
    raw_values = []
    in_display = False
    display_coords = {}  # node_id -> (x, y)
    depot_id = 0

    for ln in lines:
        if ln.startswith('EDGE_WEIGHT_SECTION'):
            in_weights = True
            in_display = False
            continue
        if ln.startswith('DISPLAY_DATA_SECTION'):
            in_weights = False
            in_display = True
            continue
        if ln.startswith('DEPOT_SECTION'):
            in_display = False
            in_weights = False
            continue
        # Only treat these as section terminators when NOT inside weights
        if not in_weights and ln in ('EOF', '-1', ''):
            in_display = False
            continue
        if ln in ('EOF',):
            in_weights = False
            in_display = False
            continue

        stripped = ln.strip()
        if in_weights:
            # Skip blank lines and section markers inside the weight block
            if stripped and stripped not in ('-1', 'EOF'):
                raw_values.extend(float(v) for v in stripped.split())
        elif in_display:
            parts = ln.split()
            if len(parts) >= 3:
                nid = int(parts[0])
                display_coords[nid] = (float(parts[1]), float(parts[2]))

    # The full matrix includes the depot as node 0.
    # Size is n_total x n_total (n_total = DIMENSION + 1).
    n = n_total
    expected = n * n
    if len(raw_values) < expected:
        # Lower-triangular format: build symmetric matrix
        tri_vals = raw_values
        mat = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                mat[i][j] = tri_vals[idx]
                mat[j][i] = tri_vals[idx]
                idx += 1
    else:
        mat = np.array(raw_values[:expected]).reshape(n, n)

    # ── Build node objects ────────────────────────────────────────────────────
    depot = Node(
        id=0,
        x=display_coords.get(0, (0.0, 0.0))[0],
        y=display_coords.get(0, (0.0, 0.0))[1],
        demand=0.0,
        is_depot=True,
    )
    nodes = []
    for nid in range(1, n):
        x, y = display_coords.get(nid, (0.0, 0.0))
        nodes.append(Node(id=nid, x=x, y=y, demand=1.0, is_depot=False))

    return mat, nodes, depot, max_vehicles, max_route, name


# ─────────────────────────────────────────────────────────────────────────────
# Core utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_dist_map(nodes: List[Node], depot: Node, dist_matrix) -> Dict[tuple, float]:
    all_nodes = [depot] + nodes
    dmap = {}
    for a in all_nodes:
        for b in all_nodes:
            dmap[(a.id, b.id)] = float(dist_matrix[a.id][b.id])
    return dmap


def route_distance(route_ids: List[int], depot_id: int,
                   dist_map: Dict[tuple, float]) -> float:
    if not route_ids:
        return 0.0
    d = dist_map[(depot_id, route_ids[0])]
    for i in range(len(route_ids) - 1):
        d += dist_map[(route_ids[i], route_ids[i + 1])]
    d += dist_map[(route_ids[-1], depot_id)]
    return d


def two_opt(route_ids: List[int], depot_id: int,
            dist_map: Dict[tuple, float]) -> List[int]:
    if len(route_ids) <= 2:
        return route_ids
    improved = True
    best = list(route_ids)
    best_dist = route_distance(best, depot_id, dist_map)
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                nr = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                nd = route_distance(nr, depot_id, dist_map)
                if nd < best_dist - 1e-10:
                    best = nr
                    best_dist = nd
                    improved = True
                    break
            if improved:
                break
    return best


def two_opt_routes(sol: VRPSolution, depot: Node,
                   dist_map: Dict[tuple, float]) -> VRPSolution:
    new_routes = []
    for r in sol.routes:
        improved_ids = two_opt(r.node_ids, depot.id, dist_map)
        new_routes.append(Route(
            node_ids=improved_ids,
            distance=route_distance(improved_ids, depot.id, dist_map),
            load=r.load,
        ))
    return VRPSolution(routes=new_routes)


def nearest_neighbor_route(node_ids: List[int], depot_id: int,
                            dist_map: Dict[tuple, float]) -> List[int]:
    if not node_ids:
        return []
    unvisited = set(node_ids)
    route = []
    current = depot_id
    while unvisited:
        nearest = min(unvisited, key=lambda nid: dist_map[(current, nid)])
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    return route


# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────

def cluster_nodes(nodes: List[Node], depot: Node, num_clusters: int) -> List[List[Node]]:
    if num_clusters <= 1:
        return [nodes]
    if len(nodes) < 2:
        return [nodes]
    num_clusters = min(num_clusters, len(nodes))

    def angle(n):
        return math.atan2(n.y - depot.y, n.x - depot.x)

    sorted_nodes = sorted(nodes, key=angle)
    clusters = [[] for _ in range(num_clusters)]
    for i, n in enumerate(sorted_nodes):
        clusters[i % num_clusters].append(n)

    node_xy = np.array([(n.x, n.y) for n in nodes])
    for _ in range(10):
        centroids = np.array([
            (np.mean([n.x for n in cl]), np.mean([n.y for n in cl]))
            if cl else (depot.x, depot.y)
            for cl in clusters
        ])
        dists = np.linalg.norm(node_xy[:, None, :] - centroids[None, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)
        new_clusters = [[] for _ in range(num_clusters)]
        for i, n in enumerate(nodes):
            new_clusters[assignments[i]].append(n)
        new_clusters = [c for c in new_clusters if c]

        while len(new_clusters) < num_clusters:
            largest = max(new_clusters, key=len)
            if len(largest) < 2:
                break
            lx = np.array([n.x for n in largest])
            ly = np.array([n.y for n in largest])
            order = np.argsort(lx) if np.var(lx) >= np.var(ly) else np.argsort(ly)
            sorted_cluster = [largest[int(idx)] for idx in order]
            mid = len(sorted_cluster) // 2
            new_clusters.remove(largest)
            new_clusters += [sorted_cluster[:mid], sorted_cluster[mid:]]

        if ({frozenset(n.id for n in cl) for cl in new_clusters} ==
                {frozenset(n.id for n in cl) for cl in clusters}):
            break
        clusters = new_clusters
    return [c for c in clusters if c]


def _split_routes_to_k(routes_list, k, n_nodes, depot_id=None,
                        dist_map=None, node_map=None):
    if n_nodes < k:
        return routes_list
    result = [list(r) for r in routes_list if r]

    def _split_one(route):
        if len(route) < 2:
            return route, []
        if dist_map is not None:
            edge_costs = [
                dist_map.get((route[i], route[i + 1]), 0.0)
                for i in range(len(route) - 1)
            ]
            if edge_costs:
                cut = int(max(range(len(edge_costs)), key=lambda i: edge_costs[i])) + 1
                return route[:cut], route[cut:]
        if node_map is not None:
            coords = [(nid, node_map[nid].x, node_map[nid].y)
                      for nid in route if nid in node_map]
            if len(coords) >= 2:
                cx = sum(x for _, x, _ in coords) / len(coords)
                cy = sum(y for _, _, y in coords) / len(coords)
                sorted_by_angle = sorted(
                    coords, key=lambda t: math.atan2(t[2] - cy, t[1] - cx))
                mid = len(sorted_by_angle) // 2
                return ([nid for nid, _, _ in sorted_by_angle[:mid]],
                        [nid for nid, _, _ in sorted_by_angle[mid:]])
        mid = len(route) // 2
        return route[:mid], route[mid:]

    while len(result) < k:
        idx = max(range(len(result)), key=lambda i: len(result[i]))
        r = result[idx]
        if len(r) < 2:
            break
        left, right = _split_one(r)
        if not right:
            break
        result[idx] = left
        result.append(right)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Clarke-Wright Savings heuristic
# ─────────────────────────────────────────────────────────────────────────────

def greedy_vrp_no_opt(nodes: List[Node], k: int, depot: Node,
                      dist_map: Dict[tuple, float]) -> VRPSolution:
    if not nodes:
        return VRPSolution(routes=[])
    node_map = {n.id: n for n in nodes}
    node_ids = [n.id for n in nodes]
    routes_dict = {nid: [nid] for nid in node_ids}
    route_of = {nid: nid for nid in node_ids}

    savings = sorted(
        [(dist_map[(depot.id, i)] + dist_map[(depot.id, j)] - dist_map[(i, j)], i, j)
         for i in node_ids for j in node_ids if i < j],
        reverse=True,
    )

    def route_load(r):
        return sum(node_map[nid].demand for nid in r)

    target = max(1, min(k, len(node_ids)))
    for s, i, j in savings:
        if len(routes_dict) <= target:
            break
        ri = route_of.get(i)
        rj = route_of.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        if ri not in routes_dict or rj not in routes_dict:
            continue
        ri_r = routes_dict[ri]
        rj_r = routes_dict[rj]
        if ri_r[-1] != i and ri_r[0] != i:
            continue
        if rj_r[-1] != j and rj_r[0] != j:
            continue
        if ri_r[-1] == i and rj_r[0] == j:
            merged = ri_r + rj_r
        elif ri_r[0] == i and rj_r[-1] == j:
            merged = rj_r + ri_r
        elif ri_r[-1] == i and rj_r[-1] == j:
            merged = ri_r + rj_r[::-1]
        elif ri_r[0] == i and rj_r[0] == j:
            merged = ri_r[::-1] + rj_r
        else:
            continue
        routes_dict[ri] = merged
        if rj in routes_dict:
            del routes_dict[rj]
        for nid in merged:
            route_of[nid] = ri

    final = list(routes_dict.values())
    while len(final) > target and len(final) > 1:
        final.sort(key=route_load)
        r1 = final.pop(0)
        r2 = final.pop(0)
        final.append(r1 + r2)

    if len(nodes) >= k:
        final = _split_routes_to_k(final, k, len(nodes),
                                    depot_id=depot.id, dist_map=dist_map,
                                    node_map=node_map)
    return VRPSolution(routes=[
        Route(node_ids=r,
              distance=route_distance(r, depot.id, dist_map),
              load=route_load(r))
        for r in final
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Brute-force leaf solver (classical, replaces QAOA for large-scale use)
# ─────────────────────────────────────────────────────────────────────────────

def solve_leaf_classical(nodes: List[Node], k: int, depot: Node,
                         dist_map: Dict[tuple, float]) -> VRPSolution:
    """Brute-force optimal for small leaves; Clarke-Wright fallback for larger ones."""
    if not nodes:
        return VRPSolution(routes=[])

    node_ids = [n.id for n in nodes]
    node_map = {n.id: n for n in nodes}
    eff_k = min(k, len(node_ids))

    # For very small leaves use brute-force
    if len(node_ids) <= 8 and eff_k == 1:
        best_dist = float('inf')
        best_perm = node_ids
        for perm in itertools.permutations(node_ids):
            d = route_distance(list(perm), depot.id, dist_map)
            if d < best_dist:
                best_dist = d
                best_perm = list(perm)
        load = sum(n.demand for n in nodes)
        return VRPSolution(routes=[Route(node_ids=best_perm,
                                         distance=best_dist, load=load)])

    # Otherwise use Clarke-Wright (fast, good quality)
    return greedy_vrp_no_opt(nodes, eff_k, depot, dist_map)


# ─────────────────────────────────────────────────────────────────────────────
# Recursive VRP solver
# ─────────────────────────────────────────────────────────────────────────────

LEAF_SIZE = 6   # nodes per leaf; change via --leaf argument


def _allocate_vehicles(cluster_sizes: List[int], k: int) -> List[int]:
    """Proportionally allocate k vehicles across clusters by size."""
    n = sum(cluster_sizes)
    if n == 0:
        return [1] * len(cluster_sizes)
    raw = [max(1, round(k * s / n)) for s in cluster_sizes]
    diff = k - sum(raw)
    # Adjust by giving/taking from the largest clusters first
    order = sorted(range(len(raw)), key=lambda i: cluster_sizes[i], reverse=True)
    i = 0
    while diff != 0:
        idx = order[i % len(order)]
        if diff > 0:
            raw[idx] += 1
            diff -= 1
        elif raw[idx] > 1:
            raw[idx] -= 1
            diff += 1
        i += 1
        if i > 10 * k:
            break
    return raw


def vrp_recursive(k: int, nodes: List[Node], depot: Node,
                  dist_map: Dict[tuple, float],
                  depth: int = 0) -> VRPSolution:
    """Recursive cluster-then-route solver."""
    if not nodes:
        return VRPSolution(routes=[])

    n = len(nodes)

    # ── Base case: leaf ───────────────────────────────────────────────────────
    if n <= LEAF_SIZE or k == 1:
        sol = solve_leaf_classical(nodes, k, depot, dist_map)
        # Ensure we have exactly k routes if possible
        if len(nodes) >= k and len(sol.routes) != k:
            raw = _split_routes_to_k(
                [r.node_ids for r in sol.routes], k, n,
                depot_id=depot.id, dist_map=dist_map,
                node_map={nd.id: nd for nd in nodes},
            )
            node_map = {nd.id: nd for nd in nodes}
            sol = VRPSolution(routes=[
                Route(node_ids=r,
                      distance=route_distance(r, depot.id, dist_map),
                      load=sum(node_map[nid].demand for nid in r))
                for r in raw
            ])
        return sol

    # ── Recursive case: cluster then recurse ─────────────────────────────────
    num_clusters = min(k, n)
    clusters = cluster_nodes(nodes, depot, num_clusters)
    allocs = _allocate_vehicles([len(c) for c in clusters], k)

    all_routes = []
    for cluster, alloc in zip(clusters, allocs):
        sub_sol = vrp_recursive(alloc, cluster, depot, dist_map, depth + 1)
        all_routes.extend(sub_sol.routes)

    return VRPSolution(routes=all_routes)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm wrappers
# ─────────────────────────────────────────────────────────────────────────────

def algo_recursive(nodes, k, depot, dist_map, apply_2opt=True):
    sol = vrp_recursive(k, nodes, depot, dist_map)
    if apply_2opt:
        sol = two_opt_routes(sol, depot, dist_map)
    return sol


def algo_nn(nodes, k, depot, dist_map):
    """Nearest-Neighbour heuristic: build k routes greedily."""
    all_ids = [n.id for n in nodes]
    node_map = {n.id: n for n in nodes}
    unvisited = set(all_ids)
    routes = []
    for i in range(k):
        if not unvisited:
            break
        remaining_routes = k - i
        share = math.ceil(len(unvisited) / remaining_routes)
        route = []
        current = depot.id
        for _ in range(share):
            if not unvisited:
                break
            nearest = min(unvisited, key=lambda nid: dist_map[(current, nid)])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        if route:
            routes.append(Route(
                node_ids=route,
                distance=route_distance(route, depot.id, dist_map),
                load=sum(node_map[nid].demand for nid in route),
            ))

    # Guarantee exactly k routes
    if len(routes) != k:
        raw = _split_routes_to_k([r.node_ids for r in routes], k, len(nodes),
                                   depot_id=depot.id, dist_map=dist_map,
                                   node_map=node_map)
        routes = [Route(node_ids=r,
                        distance=route_distance(r, depot.id, dist_map),
                        load=sum(node_map[nid].demand for nid in r))
                  for r in raw if r]
    return VRPSolution(routes=routes)


def algo_savings(nodes, k, depot, dist_map):
    return greedy_vrp_no_opt(nodes, k, depot, dist_map)


def algo_sweep(nodes, k, depot, dist_map):
    """Angular sweep: sort nodes by polar angle, split into k sectors."""
    if not nodes:
        return VRPSolution(routes=[])
    node_map = {n.id: n for n in nodes}

    def angle(n):
        return math.atan2(n.y - depot.y, n.x - depot.x)

    sorted_nodes = sorted(nodes, key=angle)
    chunk = math.ceil(len(sorted_nodes) / k)
    raw_routes = []
    for i in range(k):
        chunk_nodes = sorted_nodes[i * chunk:(i + 1) * chunk]
        if not chunk_nodes:
            continue
        ids = nearest_neighbor_route([n.id for n in chunk_nodes], depot.id, dist_map)
        raw_routes.append(ids)

    # Ensure exactly k routes
    if len(raw_routes) != k:
        raw_routes = _split_routes_to_k(raw_routes, k, len(nodes),
                                         depot_id=depot.id, dist_map=dist_map,
                                         node_map=node_map)

    routes = []
    for ids in raw_routes:
        if ids:
            routes.append(Route(
                node_ids=ids,
                distance=route_distance(ids, depot.id, dist_map),
                load=sum(node_map[nid].demand for nid in ids),
            ))
    return VRPSolution(routes=routes)


_ORTOOLS_AVAILABLE = None  # cached check

def _check_ortools():
    global _ORTOOLS_AVAILABLE
    if _ORTOOLS_AVAILABLE is None:
        try:
            from ortools.constraint_solver import routing_enums_pb2, pywrapcp
            _ORTOOLS_AVAILABLE = True
        except ImportError:
            _ORTOOLS_AVAILABLE = False
    return _ORTOOLS_AVAILABLE


def or_opt(sol: VRPSolution, depot: Node,
           dist_map: Dict[tuple, float],
           segment_sizes=(1, 2, 3)) -> VRPSolution:
    """
    Or-opt: try relocating segments of 1, 2, or 3 nodes from one route
    into another route. Improves cross-route quality after clustering.
    Runs until no improvement is found.
    """
    routes = [list(r.node_ids) for r in sol.routes]
    node_map = {nid: sol.routes[i].load / max(len(sol.routes[i].node_ids), 1)
                for i, r in enumerate(sol.routes) for nid in r.node_ids}

    improved = True
    while improved:
        improved = False
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for seg_len in segment_sizes:
                if len(routes[i]) < seg_len + 1:
                    continue
                for pos in range(len(routes[i]) - seg_len + 1):
                    segment = routes[i][pos:pos + seg_len]
                    # Cost of removing segment from route i
                    before_i = route_distance(routes[i], depot.id, dist_map)
                    new_i = routes[i][:pos] + routes[i][pos + seg_len:]
                    if not new_i:
                        continue
                    after_i = route_distance(new_i, depot.id, dist_map)
                    gain_remove = before_i - after_i

                    # Try inserting segment into each other route j
                    for j in range(len(routes)):
                        if j == i:
                            continue
                        before_j = route_distance(routes[j], depot.id, dist_map)
                        best_insert_cost = float('inf')
                        best_insert_pos = -1
                        for ins in range(len(routes[j]) + 1):
                            new_j = routes[j][:ins] + segment + routes[j][ins:]
                            cost_j = route_distance(new_j, depot.id, dist_map)
                            if cost_j < best_insert_cost:
                                best_insert_cost = cost_j
                                best_insert_pos = ins

                        gain_insert = before_j - best_insert_cost
                        if gain_remove + gain_insert > 1e-6:
                            # Accept the move
                            routes[i] = new_i
                            routes[j] = (routes[j][:best_insert_pos] +
                                         segment +
                                         routes[j][best_insert_pos:])
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    # Rebuild solution — preserve node-to-load mapping
    orig_node_map = {nid: None for r in sol.routes for nid in r.node_ids}
    all_nodes_demand = {nid: 1.0 for nid in orig_node_map}  # unit demand
    new_routes = []
    for r in routes:
        if r:
            new_routes.append(Route(
                node_ids=r,
                distance=route_distance(r, depot.id, dist_map),
                load=sum(all_nodes_demand.get(nid, 1.0) for nid in r),
            ))
    return VRPSolution(routes=new_routes)


def algo_ortools(nodes, k, depot, dist_map):
    """OR-Tools guided local search."""
    if not _check_ortools():
        return None  # signals caller to skip this algorithm
    try:
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
    except ImportError:
        return None

    node_map = {n.id: n for n in nodes}
    all_ids = [depot.id] + [n.id for n in nodes]
    idx_of = {nid: i for i, nid in enumerate(all_ids)}

    def dist_callback(from_idx, to_idx):
        return int(dist_map[(all_ids[from_idx], all_ids[to_idx])])

    manager = pywrapcp.RoutingIndexManager(len(all_ids), k, 0)
    routing = pywrapcp.RoutingModel(manager)
    cb_idx = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_params.time_limit.seconds = 30

    assignment = routing.SolveWithParameters(search_params)
    if not assignment:
        return VRPSolution(routes=[])

    routes = []
    for v in range(k):
        idx = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            node_idx = manager.IndexToNode(idx)
            if node_idx != 0:
                route.append(all_ids[node_idx])
            idx = assignment.Value(routing.NextVar(idx))
        if route:
            routes.append(Route(
                node_ids=route,
                distance=route_distance(route, depot.id, dist_map),
                load=sum(node_map[nid].demand for nid in route),
            ))
    return VRPSolution(routes=routes)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(sol: VRPSolution, nodes: List[Node], depot: Node,
             k: int, dist_map: Dict[tuple, float]) -> dict:
    issues = []
    all_ids = set(n.id for n in nodes)
    visited = set()
    for r in sol.routes:
        for nid in r.node_ids:
            if nid in visited:
                issues.append(f'Node {nid} visited more than once')
            visited.add(nid)
    missing = all_ids - visited
    if missing:
        issues.append(f'Unvisited nodes: {sorted(missing)}')
    extra = visited - all_ids
    if extra:
        issues.append(f'Unknown nodes in solution: {extra}')
    expected_k = min(k, len(nodes))
    if nodes and len(sol.routes) != expected_k:
        issues.append(
            f'Route count: got {len(sol.routes)}, expected {expected_k}')
    for i, r in enumerate(sol.routes):
        actual = route_distance(r.node_ids, depot.id, dist_map)
        if abs(actual - r.distance) > 1.0:
            issues.append(f'Route {i} distance mismatch: stored {r.distance:.2f}, computed {actual:.2f}')
    return {'valid': len(issues) == 0, 'issues': issues}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(dist_matrix, nodes, depot, k, instance_name,
                  apply_2opt=True, run_ortools=True, quiet=False,
                  algo_filter=None):
    """Run all algorithms and print a comparison table."""

    global LEAF_SIZE
    dist_map = build_dist_map(nodes, depot, dist_matrix)

    # Build algorithm list — OR-Tools is last and skipped if not installed
    _ortools_fn = algo_ortools  # capture after definition
    ALGORITHMS = [
        ('Recursive (no 2-opt)',  lambda: vrp_recursive(k, nodes, depot, dist_map)),
        ('Recursive + 2-opt',     lambda: algo_recursive(nodes, k, depot, dist_map, apply_2opt=True)),
        ('Recursive + or-opt',    lambda: or_opt(vrp_recursive(k, nodes, depot, dist_map),
                                                  depot, dist_map)),
        ('Nearest-Neighbour',     lambda: algo_nn(nodes, k, depot, dist_map)),
        ('Clarke-Wright Savings', lambda: algo_savings(nodes, k, depot, dist_map)),
        ('Sweep',                 lambda: algo_sweep(nodes, k, depot, dist_map)),
        ('OR-Tools (GLS)',        lambda: _ortools_fn(nodes, k, depot, dist_map)),
    ]

    if algo_filter:
        key = algo_filter.lower()
        ALGORITHMS = [(n, fn) for n, fn in ALGORITHMS if key in n.lower()]

    if not run_ortools:
        ALGORITHMS = [(n, fn) for n, fn in ALGORITHMS if 'ortools' not in n.lower()]

    print('=' * 110)
    print(f'  BENCHMARK — {instance_name}')
    print(f'  Nodes: {len(nodes)}   Vehicles (k): {k}   Leaf size: {LEAF_SIZE}')
    print('=' * 110)

    name_w, k_w, col_w, std_w = 28, 5, 14, 11
    hdr = (f'  {"Algorithm":{name_w}s}  {"k":>{k_w}s}  '
           f'{"Distance":>{col_w}s}  {"Std σ":>{std_w}s}  '
           f'{"Time(s)":>{col_w}s}  {"Valid":>6s}  {"vs Best":>{col_w}s}')
    div = '  ' + '-' * (len(hdr) - 2)
    print(hdr)
    print(div)

    results = {}
    for name, fn in ALGORITHMS:
        t0 = time.time()
        try:
            sol = fn()
        except Exception as e:
            if not quiet:
                print(f'  {name:{name_w}s}  ERROR: {e}')
            continue
        elapsed = time.time() - t0
        if sol is None:
            # Algorithm signalled it should be skipped (e.g. OR-Tools not installed)
            if not quiet:
                print(f'  {name:{name_w}s}  (skipped — dependency not available)')
            continue
        v = validate(sol, nodes, depot, k, dist_map)
        if not v['valid'] and not quiet:
            for issue in v['issues']:
                print(f'    !! {issue}')
        results[name] = (sol, elapsed, v['valid'])

    best_dist = min(
        sol.total_distance for sol, _, valid in results.values() if valid
    ) if any(v for _, _, v in results.values()) else float('inf')

    for name, fn in ALGORITHMS:
        if name not in results:
            continue
        sol, elapsed, valid = results[name]
        dist = sol.total_distance
        gap = (dist - best_dist) / best_dist * 100 if best_dist > 0 else 0.0
        marker = ' ***' if abs(dist - best_dist) < 1e-6 and valid else ''
        status = 'OK' if valid else 'FAIL'
        print(f'  {name:{name_w}s}  {sol.num_vehicles_used:{k_w}d}  '
              f'{dist:{col_w}.2f}  {sol.distance_std:{std_w}.2f}  '
              f'{elapsed:{col_w}.4f}  '
              f'{status:>6s}  {gap:{col_w - 1}.1f}%{marker}')

    print('=' * 110)

    # ── Per-route detail for best solution ───────────────────────────────────
    best_name = None
    for name, fn in ALGORITHMS:
        if name not in results:
            continue
        sol, _, valid = results[name]
        if valid and abs(sol.total_distance - best_dist) < 1e-6:
            best_name = name
            best_sol = sol
            break

    if best_name:
        print(f'\n  Best solution: {best_name}  (total distance = {best_dist:.2f})\n')
        print(f'  {"Route":>6s}  {"Nodes":>6s}  {"Distance":>12s}  Node IDs')
        print('  ' + '-' * 70)
        for i, r in enumerate(best_sol.routes, 1):
            ids_str = ' → '.join(str(nid) for nid in r.node_ids)
            if len(ids_str) > 60:
                ids_str = ids_str[:57] + '...'
            print(f'  {i:>6d}  {len(r.node_ids):>6d}  {r.distance:>12.2f}  {ids_str}')

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='VRP Solver — reads a .vrp FULL_MATRIX instance and benchmarks algorithms.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('instance', help='Path to the .vrp instance file')
    parser.add_argument('--k', type=int, default=None,
                        help='Number of vehicles (default: MAX_VEHICLES from file)')
    parser.add_argument('--leaf', type=int, default=6,
                        help='Max nodes per leaf for recursive solver (default: 6)')
    parser.add_argument('--no-2opt', action='store_true',
                        help='Skip 2-opt post-processing')
    parser.add_argument('--no-ortools', action='store_true',
                        help='Skip OR-Tools algorithm')
    parser.add_argument('--algo', type=str, default=None,
                        help='Run only one algorithm (substring match): '
                             'recursive, nn, savings, sweep, ortools')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress validation issue details')
    args = parser.parse_args()

    global LEAF_SIZE
    LEAF_SIZE = args.leaf

    print(f'\nLoading instance: {args.instance}')
    dist_matrix, nodes, depot, max_vehicles, max_route, name = parse_vrp(args.instance)
    k = args.k if args.k is not None else max_vehicles

    print(f'  Instance : {name}')
    print(f'  Nodes    : {len(nodes)} delivery + 1 depot = {len(nodes) + 1} total')
    print(f'  Vehicles : {k}  (file max: {max_vehicles})')
    print(f'  Max route: {max_route:.1f}')
    print(f'  Dist mat : {dist_matrix.shape}  max={dist_matrix.max():.1f}\n')

    run_benchmark(
        dist_matrix=dist_matrix,
        nodes=nodes,
        depot=depot,
        k=k,
        instance_name=name,
        apply_2opt=not args.no_2opt,
        run_ortools=not args.no_ortools,
        quiet=args.quiet,
        algo_filter=args.algo,
    )


if __name__ == '__main__':
    main()