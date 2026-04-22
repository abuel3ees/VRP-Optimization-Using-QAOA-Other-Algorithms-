#!/usr/bin/env python3
"""
Classical VRP Benchmark

Runs every classical VRP construction heuristic, local-search improver,
metaheuristic, OR-Tools strategy, and Recursive solver (from notebook) on 
the instances already extracted in arrays/, using distance, fairness (std of 
route distances), and weighted fairness metrics.

Metrics:
  - Total Distance: sum of all route distances
  - Distance Std: standard deviation of route distances (fairness)
  - Weighted Fairness: ((total_distance/k) + distance_std) / 2
  - Gap%: percentage above the best solution found

Usage:
    python classical_vrp_benchmark.py                       # all instances, k from .vrp
    python classical_vrp_benchmark.py --k 7                 # all instances, k=7
    python classical_vrp_benchmark.py --instance 50_0       # single instance
    python classical_vrp_benchmark.py --instance 50_0 --k 7
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import importlib.util
import itertools
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

ROOT = Path(__file__).resolve().parent.parent
ARRAYS_DIR = ROOT / "arrays"
INSTANCES_DIR = ROOT / "Instances"
sys.path.insert(0, str(ROOT))


# ─────────────────────────── core data model ────────────────────────────
@dataclass
class Node:
    id: int
    x: float
    y: float
    demand: float = 1.0


@dataclass
class Route:
    node_ids: List[int]
    distance: float = 0.0
    load: float = 0.0


@dataclass
class VRPSolution:
    routes: List[Route] = field(default_factory=list)
    total_distance: float = 0.0
    distance_std: float = 0.0
    num_vehicles_used: int = 0
    weighted_fairness: float = 0.0

    def __post_init__(self):
        self.total_distance = sum(r.distance for r in self.routes)
        self.num_vehicles_used = len([r for r in self.routes if r.node_ids])
        dists = [r.distance for r in self.routes if r.node_ids]
        self.distance_std = float(np.std(dists)) if dists else 0.0
        # weighted_fairness = ((avg_distance_per_vehicle) + (std)) / 2
        if self.num_vehicles_used > 0:
            avg_dist = self.total_distance / self.num_vehicles_used
            self.weighted_fairness = (avg_dist + self.distance_std) / 2.0
        else:
            self.weighted_fairness = 0.0


# ──────────────────────── distance helpers ──────────────────────────────
def build_dist_map(nodes: List[Node], depot: Node, dist_matrix=None
                   ) -> Dict[Tuple[int, int], float]:
    all_nodes = [depot] + nodes
    dmap: Dict[Tuple[int, int], float] = {}
    for a in all_nodes:
        for b in all_nodes:
            if dist_matrix is not None:
                dmap[(a.id, b.id)] = float(dist_matrix[a.id][b.id])
            else:
                dmap[(a.id, b.id)] = math.hypot(a.x - b.x, a.y - b.y)
    global GLOBAL_MAX_DIST
    GLOBAL_MAX_DIST = max(GLOBAL_MAX_DIST, max(dmap.values()) if dmap else 1.0)
    return dmap


def euclidean(a: Node, b: Node) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def route_distance(route_ids: List[int], depot_id: int,
                   dist_map: Dict[Tuple[int, int], float]) -> float:
    if not route_ids:
        return 0.0
    d = dist_map[(depot_id, route_ids[0])]
    for i in range(len(route_ids) - 1):
        d += dist_map[(route_ids[i], route_ids[i + 1])]
    d += dist_map[(route_ids[-1], depot_id)]
    return d


def solution_from_routes(routes_ids: List[List[int]], depot_id: int,
                         dist_map, node_map) -> VRPSolution:
    rs = []
    for r in routes_ids:
        if not r:
            continue
        rs.append(Route(
            node_ids=r,
            distance=route_distance(r, depot_id, dist_map),
            load=sum(node_map[nid].demand for nid in r),
        ))
    return VRPSolution(routes=rs)


# ──────────────────────── local search ──────────────────────────────────
def two_opt(route_ids: List[int], depot_id: int, dist_map) -> List[int]:
    if len(route_ids) <= 2:
        return route_ids
    best = list(route_ids)
    best_dist = route_distance(best, depot_id, dist_map)
    improved = True
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


def or_opt(route_ids: List[int], depot_id: int, dist_map) -> List[int]:
    """Move chunks of length 1..3 to a better position."""
    if len(route_ids) <= 3:
        return route_ids
    best = list(route_ids)
    best_dist = route_distance(best, depot_id, dist_map)
    improved = True
    while improved:
        improved = False
        for seg_len in (1, 2, 3):
            for i in range(len(best) - seg_len + 1):
                segment = best[i:i + seg_len]
                rest = best[:i] + best[i + seg_len:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    cand = rest[:j] + segment + rest[j:]
                    cd = route_distance(cand, depot_id, dist_map)
                    if cd < best_dist - 1e-10:
                        best, best_dist = cand, cd
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best


def apply_local_search(sol: VRPSolution, depot_id: int, dist_map,
                       node_map, improver=two_opt) -> VRPSolution:
    new_routes = []
    for r in sol.routes:
        improved_ids = improver(r.node_ids, depot_id, dist_map)
        new_routes.append(improved_ids)
    return solution_from_routes(new_routes, depot_id, dist_map, node_map)


# ─────────────────── construction: angular partition ────────────────────
def partition_by_angle(nodes: List[Node], depot: Node, k: int) -> List[List[int]]:
    eff_k = max(1, min(k, len(nodes)))
    sorted_nodes = sorted(nodes, key=lambda n: math.atan2(n.y - depot.y, n.x - depot.x))
    groups: List[List[int]] = [[] for _ in range(eff_k)]
    for i, n in enumerate(sorted_nodes):
        groups[i % eff_k].append(n.id)
    return [g for g in groups if g]


# ────────────────── construction heuristics ─────────────────────────────
def nn_route(node_ids: List[int], start_id: int, dist_map) -> List[int]:
    if not node_ids:
        return []
    unvisited = set(node_ids)
    route, current = [], start_id
    while unvisited:
        nxt = min(unvisited, key=lambda nid: dist_map[(current, nid)])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return route


def algo_nearest_neighbour(nodes, k, depot, dist_map, node_map):
    groups = partition_by_angle(nodes, depot, k)
    routes = [nn_route(g, depot.id, dist_map) for g in groups]
    return solution_from_routes(routes, depot.id, dist_map, node_map)


def algo_sweep(nodes, k, depot, dist_map, node_map):
    groups = partition_by_angle(nodes, depot, k)
    # within-group order is angular already (partition keeps angular order)
    return solution_from_routes(groups, depot.id, dist_map, node_map)


def _split_longest_to_k(routes_ids, k, depot_id, dist_map):
    """Split the longest route at its longest edge until we have k routes."""
    routes_ids = [r for r in routes_ids if r]
    while len(routes_ids) < k:
        # find longest route
        lens = [route_distance(r, depot_id, dist_map) for r in routes_ids]
        if not lens:
            break
        idx = int(np.argmax(lens))
        r = routes_ids[idx]
        if len(r) < 2:
            break
        # split at longest internal edge
        gaps = [dist_map[(r[i], r[i + 1])] for i in range(len(r) - 1)]
        cut = int(np.argmax(gaps)) + 1
        a, b = r[:cut], r[cut:]
        routes_ids.pop(idx)
        routes_ids.extend([a, b])
    return routes_ids


def algo_savings_parallel(nodes, k, depot, dist_map, node_map):
    """Classic Clarke-Wright (parallel) — merge highest savings until k routes."""
    node_ids = [n.id for n in nodes]
    routes_dict = {nid: [nid] for nid in node_ids}
    route_of = {nid: nid for nid in node_ids}
    savings = sorted(
        [(dist_map[(depot.id, i)] + dist_map[(depot.id, j)] - dist_map[(i, j)], i, j)
         for i in node_ids for j in node_ids if i < j],
        reverse=True,
    )
    target = max(1, min(k, len(node_ids)))
    for _, i, j in savings:
        if len(routes_dict) <= target:
            break
        ri, rj = route_of.get(i), route_of.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        if ri not in routes_dict or rj not in routes_dict:
            continue
        rr, sr = routes_dict[ri], routes_dict[rj]
        if rr[-1] != i and rr[0] != i:
            continue
        if sr[-1] != j and sr[0] != j:
            continue
        if rr[-1] == i and sr[0] == j:
            merged = rr + sr
        elif rr[0] == i and sr[-1] == j:
            merged = sr + rr
        elif rr[-1] == i and sr[-1] == j:
            merged = rr + sr[::-1]
        elif rr[0] == i and sr[0] == j:
            merged = rr[::-1] + sr
        else:
            continue
        routes_dict[ri] = merged
        del routes_dict[rj]
        for nid in merged:
            route_of[nid] = ri
    final = list(routes_dict.values())
    while len(final) > target and len(final) > 1:
        final.sort(key=lambda r: sum(node_map[n].demand for n in r))
        a, b = final.pop(0), final.pop(0)
        final.append(a + b)
    if len(final) < target:
        final = _split_longest_to_k(final, target, depot.id, dist_map)
    return solution_from_routes(final, depot.id, dist_map, node_map)


def algo_savings_sequential(nodes, k, depot, dist_map, node_map):
    """Sequential Clarke-Wright: extend one route at a time along best savings."""
    node_ids = [n.id for n in nodes]
    savings = sorted(
        [(dist_map[(depot.id, i)] + dist_map[(depot.id, j)] - dist_map[(i, j)], i, j)
         for i in node_ids for j in node_ids if i < j],
        reverse=True,
    )
    unassigned = set(node_ids)
    routes: List[List[int]] = []
    target = max(1, min(k, len(node_ids)))
    while unassigned and len(routes) < target:
        seed = next(iter(unassigned))
        route = [seed]
        unassigned.remove(seed)
        extended = True
        while extended:
            extended = False
            for _, i, j in savings:
                if (i in unassigned) ^ (j in unassigned):
                    inside, outside = (j, i) if i in unassigned else (i, j)
                    if route[0] == inside:
                        route.insert(0, outside)
                        unassigned.remove(outside)
                        extended = True
                        break
                    if route[-1] == inside:
                        route.append(outside)
                        unassigned.remove(outside)
                        extended = True
                        break
        routes.append(route)
    # dump any leftovers into the shortest route
    if unassigned:
        routes.sort(key=lambda r: route_distance(r, depot.id, dist_map))
        routes[0].extend(list(unassigned))
        unassigned.clear()
    # without vehicle capacity, one route can swallow everything — split to k
    if len(routes) < target:
        routes = _split_longest_to_k(routes, target, depot.id, dist_map)
    return solution_from_routes(routes, depot.id, dist_map, node_map)


def _insertion_base(nodes, k, depot, dist_map, node_map, pick_next):
    """Shared scaffolding for NN/cheapest/farthest insertion heuristics."""
    groups = partition_by_angle(nodes, depot, k)
    all_routes: List[List[int]] = []
    for group in groups:
        remaining = set(group)
        if not remaining:
            continue
        # seed: farthest-from-depot
        seed = max(remaining, key=lambda nid: dist_map[(depot.id, nid)])
        route = [seed]
        remaining.remove(seed)
        while remaining:
            nid = pick_next(route, remaining, dist_map, depot.id)
            # insert at best position
            best_pos, best_delta = 0, float("inf")
            for pos in range(len(route) + 1):
                prev = depot.id if pos == 0 else route[pos - 1]
                nxt = depot.id if pos == len(route) else route[pos]
                delta = (dist_map[(prev, nid)] + dist_map[(nid, nxt)]
                         - dist_map[(prev, nxt)])
                if delta < best_delta:
                    best_delta, best_pos = delta, pos
            route.insert(best_pos, nid)
            remaining.remove(nid)
        all_routes.append(route)
    return solution_from_routes(all_routes, depot.id, dist_map, node_map)


def _pick_nearest(route, remaining, dist_map, depot_id):
    return min(remaining,
               key=lambda nid: min(dist_map[(r, nid)] for r in route))


def _pick_farthest(route, remaining, dist_map, depot_id):
    return max(remaining,
               key=lambda nid: min(dist_map[(r, nid)] for r in route))


def _pick_cheapest(route, remaining, dist_map, depot_id):
    best_nid, best_cost = None, float("inf")
    for nid in remaining:
        for pos in range(len(route) + 1):
            prev = depot_id if pos == 0 else route[pos - 1]
            nxt = depot_id if pos == len(route) else route[pos]
            delta = (dist_map[(prev, nid)] + dist_map[(nid, nxt)]
                     - dist_map[(prev, nxt)])
            if delta < best_cost:
                best_cost, best_nid = delta, nid
    return best_nid


def algo_nearest_insertion(nodes, k, depot, dist_map, node_map):
    return _insertion_base(nodes, k, depot, dist_map, node_map, _pick_nearest)


def algo_farthest_insertion(nodes, k, depot, dist_map, node_map):
    return _insertion_base(nodes, k, depot, dist_map, node_map, _pick_farthest)


def algo_cheapest_insertion(nodes, k, depot, dist_map, node_map):
    return _insertion_base(nodes, k, depot, dist_map, node_map, _pick_cheapest)


# ───────────────────── metaheuristics ────────────────────────────────────
def _flatten(sol: VRPSolution) -> List[List[int]]:
    return [list(r.node_ids) for r in sol.routes if r.node_ids]


def _random_neighbor(routes: List[List[int]], rng: random.Random
                     ) -> List[List[int]]:
    routes = [list(r) for r in routes]
    move = rng.choice(["swap_within", "swap_between", "move_between", "reverse"])
    non_empty = [i for i, r in enumerate(routes) if r]
    if not non_empty:
        return routes
    if move == "swap_within":
        i = rng.choice(non_empty)
        if len(routes[i]) >= 2:
            a, b = rng.sample(range(len(routes[i])), 2)
            routes[i][a], routes[i][b] = routes[i][b], routes[i][a]
    elif move == "reverse":
        i = rng.choice(non_empty)
        if len(routes[i]) >= 2:
            a, b = sorted(rng.sample(range(len(routes[i])), 2))
            routes[i][a:b + 1] = routes[i][a:b + 1][::-1]
    elif move == "swap_between" and len(non_empty) >= 2:
        i, j = rng.sample(non_empty, 2)
        a = rng.randrange(len(routes[i]))
        b = rng.randrange(len(routes[j]))
        routes[i][a], routes[j][b] = routes[j][b], routes[i][a]
    elif move == "move_between" and len(non_empty) >= 2:
        i, j = rng.sample(non_empty, 2)
        if routes[i]:
            a = rng.randrange(len(routes[i]))
            node = routes[i].pop(a)
            pos = rng.randint(0, len(routes[j]))
            routes[j].insert(pos, node)
    return routes


def _cost(routes: List[List[int]], depot_id: int, dist_map) -> float:
    return sum(route_distance(r, depot_id, dist_map) for r in routes)


def algo_simulated_annealing(nodes, k, depot, dist_map, node_map,
                             iters=4000, T0=1000.0, alpha=0.995, seed=None):
    rng = random.Random(seed)
    init = algo_savings_parallel(nodes, k, depot, dist_map, node_map)
    current = _flatten(init)
    current_cost = _cost(current, depot.id, dist_map)
    best, best_cost = current, current_cost
    T = T0
    for _ in range(iters):
        cand = _random_neighbor(current, rng)
        cc = _cost(cand, depot.id, dist_map)
        if cc < current_cost or rng.random() < math.exp(-(cc - current_cost) / max(T, 1e-9)):
            current, current_cost = cand, cc
            if cc < best_cost:
                best, best_cost = cand, cc
        T *= alpha
    return solution_from_routes(best, depot.id, dist_map, node_map)


def algo_tabu_search(nodes, k, depot, dist_map, node_map,
                     iters=800, tenure=25, seed=None):
    rng = random.Random(seed)
    init = algo_savings_parallel(nodes, k, depot, dist_map, node_map)
    current = _flatten(init)
    current_cost = _cost(current, depot.id, dist_map)
    best, best_cost = current, current_cost
    tabu: List[Tuple] = []
    for _ in range(iters):
        candidates = [_random_neighbor(current, rng) for _ in range(30)]
        scored = sorted(
            ((_cost(c, depot.id, dist_map), c) for c in candidates),
            key=lambda t: t[0],
        )
        chosen = None
        for cc, c in scored:
            key = tuple(tuple(r) for r in c)
            if key in tabu and cc >= best_cost:
                continue
            chosen = (cc, c, key)
            break
        if chosen is None:
            continue
        current_cost, current, key = chosen
        tabu.append(key)
        if len(tabu) > tenure:
            tabu.pop(0)
        if current_cost < best_cost:
            best, best_cost = current, current_cost
    return solution_from_routes(best, depot.id, dist_map, node_map)


def algo_iterated_local_search(nodes, k, depot, dist_map, node_map,
                               iters=20, seed=None):
    rng = random.Random(seed)
    init = algo_savings_parallel(nodes, k, depot, dist_map, node_map)
    current = apply_local_search(init, depot.id, dist_map, node_map, two_opt)
    current_r = _flatten(current)
    best_r, best_cost = current_r, _cost(current_r, depot.id, dist_map)
    for _ in range(iters):
        # perturb: a few random moves
        cand = current_r
        for _ in range(4):
            cand = _random_neighbor(cand, rng)
        # local search via 2-opt on each route
        improved = [two_opt(r, depot.id, dist_map) for r in cand]
        cost_i = _cost(improved, depot.id, dist_map)
        if cost_i < best_cost:
            best_r, best_cost = improved, cost_i
            current_r = improved
    return solution_from_routes(best_r, depot.id, dist_map, node_map)


def algo_genetic(nodes, k, depot, dist_map, node_map,
                 pop_size=30, generations=80, mutation=0.2, seed=None):
    rng = random.Random(seed)

    def random_individual():
        ids = [n.id for n in nodes]
        rng.shuffle(ids)
        # split into k contiguous chunks
        sz = max(1, len(ids) // max(1, k))
        chunks = [ids[i:i + sz] for i in range(0, len(ids), sz)]
        while len(chunks) > k and len(chunks) > 1:
            smallest = min(range(len(chunks)), key=lambda i: len(chunks[i]))
            merged = chunks.pop(smallest)
            target = min(range(len(chunks)), key=lambda i: len(chunks[i]))
            chunks[target].extend(merged)
        return chunks

    pop = []
    # seed population with good constructions
    for f in (algo_savings_parallel, algo_nearest_neighbour, algo_sweep,
              algo_cheapest_insertion):
        pop.append(_flatten(f(nodes, k, depot, dist_map, node_map)))
    while len(pop) < pop_size:
        pop.append(random_individual())

    def fitness(ind): return _cost(ind, depot.id, dist_map)

    for _ in range(generations):
        pop.sort(key=fitness)
        elite = pop[:max(2, pop_size // 5)]
        children = list(elite)
        while len(children) < pop_size:
            p1, p2 = rng.sample(elite, 2) if len(elite) >= 2 else (elite[0], elite[0])
            # one-point crossover on flattened sequence
            flat1 = [nid for r in p1 for nid in r]
            flat2 = [nid for r in p2 for nid in r]
            cut = rng.randint(1, max(1, len(flat1) - 1))
            child_flat = flat1[:cut] + [nid for nid in flat2 if nid not in flat1[:cut]]
            # re-split to k chunks
            sz = max(1, len(child_flat) // max(1, k))
            child = [child_flat[i:i + sz] for i in range(0, len(child_flat), sz)]
            while len(child) > k and len(child) > 1:
                child[-2].extend(child.pop(-1))
            # mutate
            if rng.random() < mutation:
                child = _random_neighbor(child, rng)
            children.append(child)
        pop = children
    pop.sort(key=fitness)
    return solution_from_routes(pop[0], depot.id, dist_map, node_map)


# ───────────────────── OR-Tools wrappers ─────────────────────────────────
def _run_ortools(nodes, k, depot, dist_map, node_map,
                 first_solution, metaheuristic=None, time_limit=10,
                 hard_k=True, seed_from_savings=False):
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        return None

    if not nodes:
        return VRPSolution(routes=[])

    eff_k = min(k, len(nodes))
    all_nodes = [depot] + nodes
    n_total = len(all_nodes)
    SCALE = 1000

    def or_dist(i, j):
        return int(dist_map[(all_nodes[i].id, all_nodes[j].id)] * SCALE)

    manager = pywrapcp.RoutingIndexManager(n_total, eff_k, 0)
    routing = pywrapcp.RoutingModel(manager)
    cb_idx = routing.RegisterTransitCallback(
        lambda i, j: or_dist(manager.IndexToNode(i), manager.IndexToNode(j))
    )
    routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

    def demand_cb(from_index):
        node = manager.IndexToNode(from_index)
        return 0 if node == 0 else 1

    d_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        d_idx, 0, [len(nodes)] * eff_k, True, "Count"
    )
    count_dim = routing.GetDimensionOrDie("Count")
    if hard_k:
        # Hard lower bound: each vehicle must visit at least 1 delivery.
        for v in range(eff_k):
            count_dim.CumulVar(routing.End(v)).SetMin(1)
    else:
        # Soft penalty large enough to dominate any distance saving.
        big_penalty = int(max(dist_map.values()) * SCALE * 100)
        for v in range(eff_k):
            count_dim.SetCumulVarSoftLowerBound(routing.End(v), 1, big_penalty)

    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = first_solution
    if metaheuristic is not None:
        sp.local_search_metaheuristic = metaheuristic
    sp.time_limit.seconds = time_limit

    if seed_from_savings:
        # Seed with a k-route Clarke-Wright solution so the k-vehicle
        # constraint is baked in before refinement.
        seed_sol = algo_savings_parallel(nodes, k, depot, dist_map, node_map)
        id_to_idx = {all_nodes[i].id: i for i in range(len(all_nodes))}
        seeded_routes = []
        for r in seed_sol.routes:
            seeded_routes.append([id_to_idx[nid] for nid in r.node_ids])
        while len(seeded_routes) < eff_k:
            seeded_routes.append([])
        seeded_routes = seeded_routes[:eff_k]
        initial = routing.ReadAssignmentFromRoutes(seeded_routes, True)
        assign = routing.SolveFromAssignmentWithParameters(initial, sp)
    else:
        assign = routing.SolveWithParameters(sp)
    if not assign:
        return None

    routes_ids = []
    for v in range(eff_k):
        idx = routing.Start(v)
        r = []
        while not routing.IsEnd(idx):
            n_idx = manager.IndexToNode(idx)
            if n_idx != 0:
                r.append(all_nodes[n_idx].id)
            idx = assign.Value(routing.NextVar(idx))
        if r:
            routes_ids.append(r)
    return solution_from_routes(routes_ids, depot.id, dist_map, node_map)


def algo_ortools_gls(nodes, k, depot, dist_map, node_map):
    from ortools.constraint_solver import routing_enums_pb2
    return _run_ortools(
        nodes, k, depot, dist_map, node_map,
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    )


def algo_ortools_sa(nodes, k, depot, dist_map, node_map):
    from ortools.constraint_solver import routing_enums_pb2
    return _run_ortools(
        nodes, k, depot, dist_map, node_map,
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
    )


def algo_ortools_tabu(nodes, k, depot, dist_map, node_map):
    from ortools.constraint_solver import routing_enums_pb2
    return _run_ortools(
        nodes, k, depot, dist_map, node_map,
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    )


def algo_ortools_pca(nodes, k, depot, dist_map, node_map):
    from ortools.constraint_solver import routing_enums_pb2
    return _run_ortools(
        nodes, k, depot, dist_map, node_map,
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        time_limit=3,
    )


def algo_ortools_savings(nodes, k, depot, dist_map, node_map):
    # SAVINGS first-solution can't honor hard-k; seed with Clarke-Wright
    # (same family of heuristic) then polish with GLS.
    from ortools.constraint_solver import routing_enums_pb2
    return _run_ortools(
        nodes, k, depot, dist_map, node_map,
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        time_limit=5, hard_k=True, seed_from_savings=True,
    )


def algo_ortools_christofides(nodes, k, depot, dist_map, node_map):
    from ortools.constraint_solver import routing_enums_pb2
    return _run_ortools(
        nodes, k, depot, dist_map, node_map,
        routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        time_limit=5, hard_k=True, seed_from_savings=True,
    )


def algo_ortools_parallel_cheapest(nodes, k, depot, dist_map, node_map):
    from ortools.constraint_solver import routing_enums_pb2
    return _run_ortools(
        nodes, k, depot, dist_map, node_map,
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        time_limit=5, hard_k=True, seed_from_savings=True,
    )


# ───────────────── recursive cluster-and-solve (notebook-faithful) ──────
# These functions are byte-identical copies of the notebook solver in
# QAOA.py (see lines cluster_nodes (258-311), _classical_optimal_cost (466-502), solve_brute_force (503-1034), allocate_vehicles (1045-1065), create_super_nodes (1066-1073), build_super_dist_map (1074-1085), _orient_segment (1086-1097), merge_super_solution (1098-1217), vrp_solver_plain (1535-1611)).  Any change here must be mirrored there
# so the benchmark and the notebook produce identical results.
LEAF_SIZE = 4
GLOBAL_MAX_DIST: float = 1.0
QAOA_STATS: Dict[str, int] = {"success": 0, "fallback": 0}
QAOA_LOG: List[dict] = []

def cluster_nodes(nodes: List[Node], depot: Node,
                    num_clusters: int) -> List[List[Node]]:
    """Partition *nodes* into *num_clusters* geographic groups using angular
    sweep initialisation followed by k-means refinement (up to 10 iterations).
    Clusters that collapse are re-split along the principal axis."""
    if num_clusters <= 1: return [nodes]
    # Guard: need at least 2 nodes to form 2 clusters.
    # The old guard was "len(nodes) <= 3" which silently returned a single
    # cluster for n=3,k=2 — the solver then re-entered cluster_nodes with
    # the same arguments, causing infinite recursion whenever LEAF_SIZE <= 3.
    # Correct threshold is < 2: fewer than 2 nodes cannot be split at all.
    if len(nodes) < 2: return [nodes]
    num_clusters = min(num_clusters, len(nodes))
    def angle(n): return math.atan2(n.y - depot.y, n.x - depot.x)
    sorted_nodes = sorted(nodes, key=angle)
    clusters = [[] for _ in range(num_clusters)]
    for i, n in enumerate(sorted_nodes): clusters[i % num_clusters].append(n)
    # Pre-build node coordinate array once for vectorised distance computation.
    node_xy = np.array([(n.x, n.y) for n in nodes])          # (N, 2)
    for _ in range(10):
        centroids = np.array([
            (np.mean([n.x for n in cl]), np.mean([n.y for n in cl]))
            if cl else (depot.x, depot.y)
            for cl in clusters
        ])                                                      # (K, 2)
        # Compute all node-centroid distances in one vectorised call: (N, K)
        dists       = np.linalg.norm(node_xy[:, None, :] - centroids[None, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)                 # (N,)
        new_clusters = [[] for _ in range(num_clusters)]
        for i, n in enumerate(nodes):
            new_clusters[assignments[i]].append(n)
        new_clusters = [c for c in new_clusters if c]
        # ISSUE 8 FIX: split along principal axis, not by list index
        while len(new_clusters) < num_clusters:
            largest = max(new_clusters, key=len)
            if len(largest) < 2: break
            lx = np.array([n.x for n in largest])
            ly = np.array([n.y for n in largest])
            if np.var(lx) >= np.var(ly):
                order = np.argsort(lx)
            else:
                order = np.argsort(ly)
            sorted_cluster = [largest[int(idx)] for idx in order]
            mid = len(sorted_cluster) // 2
            new_clusters.remove(largest)
            new_clusters += [sorted_cluster[:mid], sorted_cluster[mid:]]
        # Use frozenset comparison so order-scrambling from .remove()+= does not
        # prevent early exit when the partition has genuinely converged.
        if ({frozenset(n.id for n in cl) for cl in new_clusters} ==
                {frozenset(n.id for n in cl) for cl in clusters}):
            break
        clusters = new_clusters
    return [c for c in clusters if c]


def _classical_optimal_cost(node_ids, eff_k, depot_id, dist_map):
    """Brute-force optimal cost for a small leaf (used for QAOA classification)."""
    if eff_k == 1:
        best = float("inf")
        for perm in itertools.permutations(node_ids):
            d = route_distance(list(perm), depot_id, dist_map)
            if d < best: best = d
        return best
    def _parts(ids, k):
        """Generate all partitions of ids into exactly k non-empty groups."""
        ids_l = list(ids)
        n = len(ids_l)
        if k == 1:
            yield (tuple(ids_l),)
            return
        if n == k:
            yield tuple((x,) for x in ids_l)
            return
        if n < k or k <= 0:
            return
        first = ids_l[0]; rest = ids_l[1:]
        # Case 1: first is a singleton group; partition rest into k-1
        for sub in _parts(tuple(rest), k - 1):
            yield ((first,),) + sub
        # Case 2: first joins an existing group; partition rest into k
        for sub in _parts(tuple(rest), k):
            for i in range(len(sub)):
                yield sub[:i] + ((first,) + sub[i],) + sub[i+1:]
    best = float("inf")
    for part in _parts(tuple(node_ids), eff_k):
        if len(part) != eff_k: continue
        for combo in itertools.product(*[itertools.permutations(b) for b in part]):
            total = sum(route_distance(list(p), depot_id, dist_map) for p in combo)
            if total < best: best = total
    return best


def solve_brute_force(nodes, k, depot, dist_map):
    """QAOA-based solver for VRP leaves (len(nodes) <= LEAF_SIZE).
    Closed-tour only: depot is always included.
    Logs every call into QAOA_LOG with outcome classification.
    """
    if not nodes:
        return VRPSolution(routes=[])

    if GLOBAL_MAX_DIST <= 1.0 + 1e-9:
        raise RuntimeError(
            "GLOBAL_MAX_DIST not initialised (still placeholder 1.0). "
            "Run the data-loading cell before solving."
        )

    node_ids = [n.id for n in nodes]
    node_map = {n.id: n for n in nodes}
    eff_k    = min(k, len(node_ids))

    _PREFIX = f"  [QAOA leaf n={len(node_ids)} k={eff_k}]"

    # classical fallback
    def _classical_fallback(reason=""):
        QAOA_STATS["fallback"] += 1
        print(f"{_PREFIX} *** FALLBACK #{QAOA_STATS['fallback']} to classical "
              f"brute-force{(' -- ' + reason) if reason else ''}")
        if eff_k == 1:
            best_dist = float("inf"); best_perm = node_ids
            for perm in itertools.permutations(node_ids):
                d = route_distance(list(perm), depot.id, dist_map)
                if d < best_dist: best_dist = d; best_perm = list(perm)
            load = sum(n.demand for n in nodes)
            _sol = VRPSolution(routes=[Route(node_ids=list(best_perm),
                                             distance=best_dist, load=load)])
            _sol.solver_used = "classical_fallback"
            return _sol
        def _parts(ids, k):
            """Generate all partitions of ids into exactly k non-empty groups."""
            ids_l = list(ids)
            n = len(ids_l)
            if k == 1:
                yield (tuple(ids_l),)
                return
            if n == k:
                yield tuple((x,) for x in ids_l)
                return
            if n < k or k <= 0:
                return
            first = ids_l[0]; rest = ids_l[1:]
            # Case 1: first is a singleton group; partition rest into k-1
            for sub in _parts(tuple(rest), k - 1):
                yield ((first,),) + sub
            # Case 2: first joins an existing group; partition rest into k
            for sub in _parts(tuple(rest), k):
                for i in range(len(sub)):
                    yield sub[:i] + ((first,) + sub[i],) + sub[i+1:]
        best_dist = float("inf"); best_routes = None
        for partition in _parts(tuple(node_ids), eff_k):
            if len(partition) != eff_k: continue
            for combo in itertools.product(*[itertools.permutations(b) for b in partition]):
                total = sum(route_distance(list(p), depot.id, dist_map) for p in combo)
                if total < best_dist:
                    best_dist = total; best_routes = [list(p) for p in combo]
        if best_routes is None:
            best_routes = [list(node_ids[i::eff_k]) for i in range(eff_k)]
            best_routes = [r for r in best_routes if r]
        _sol = VRPSolution(routes=[
            Route(node_ids=r, distance=route_distance(r, depot.id, dist_map),
                  load=sum(node_map[nid].demand for nid in r))
            for r in best_routes
        ])
        _sol.solver_used = "classical_fallback"
        return _sol

    try:
        import numpy as _np
        from qiskit import transpile as _transpile
        from qiskit.circuit.library import QAOAAnsatz as _QAOAAnsatz
        from qiskit_aer import AerSimulator as _AerSim
        from qiskit_aer.primitives import Estimator as _AerEst
        from qiskit_algorithms.optimizers import ADAM as _ADAM, COBYLA as _COBYLA
        from qiskit_algorithms.utils import algorithm_globals as _ag
        from qiskit_optimization import QuadraticProgram as _QP
        from qiskit_optimization.converters import QuadraticProgramToQubo as _QP2Q
        print(f"{_PREFIX} qiskit imports OK -- entering QAOA path")
    except ImportError as e:
        _sol = _classical_fallback(f"qiskit not installed: {e}")
        QAOA_LOG.append({
            "node_ids": node_ids, "k": eff_k, "n_qubits": 0,
            "outcome": "non-valid", "qaoa_cost": float("inf"),
            "optimal_cost": None, "gap_pct": None,
            "valid_frac": 0.0, "n_valid_unique": 0,
            "best_prob_rank": -1, "solver_used": "classical_fallback",
            "reason": f"qiskit not installed: {e}",
        })
        return _sol

    _CONFIGS = [
    
        {"reps": 2, "optimizer": "COBYLA", "maxiter": 50, "restarts": 3},
        
    ]
    _SHOTS    = 50_000
    _DECODE_K = 100
    penalty_mul=2
    import hashlib as _hl
    # Generate a random seed each time instead of deterministic seeding
    _leaf_seed = random.randint(0, 2**31 - 1)
    _ag.random_seed = _leaf_seed
    _np.random.seed(_leaf_seed)

    # ── QUBO construction ──────────────────────────────────────────────────────
    # Two formulations depending on eff_k:
    #   k=1 → Position-indexed QUBO (subtour-free by construction, fewer qubits)
    #   k>1 → Edge-based QUBO with post-hoc subtour filtering
    # ──────────────────────────────────────────────────────────────────────────

    all_ids  = [depot.id] + node_ids
    m1       = len(all_ids)
    dist_mat = _np.zeros((m1, m1), dtype=float)
    for qi, a in enumerate(all_ids):
        for qj, b in enumerate(all_ids):
            if a != b and (a, b) in dist_map:
                dist_mat[qi, qj] = dist_map[(a, b)]

    _gmax = float(GLOBAL_MAX_DIST) if GLOBAL_MAX_DIST > 0 else (
        float(_np.max(dist_mat)) if _np.max(dist_mat) > 0 else 1.0)
    dist_norm = dist_mat / _gmax

    _N = len(node_ids)

    if eff_k == 1:
        # ═══════════════════════════════════════════════════════════════════
        # POSITION-INDEXED QUBO (k=1 only)
        # Variables: p_{i}_{t}  = 1 if customer i (0-indexed) at position t
        # Subtours are impossible: the assignment structure guarantees a
        # single Hamiltonian path through all customers.
        # ═══════════════════════════════════════════════════════════════════
        penalty = _N * penalty_mul

        qp = _QP()
        for i in range(_N):
            for t in range(_N):
                qp.binary_var(f"p_{i}_{t}")

        _lin = {}
        _quad = {}
        for i in range(_N):
            ci = i + 1   # index into dist_norm (0 = depot)
            # depot → first stop
            _lin[f"p_{i}_0"] = _lin.get(f"p_{i}_0", 0) + float(dist_norm[0, ci])
            # last stop → depot
            _lin[f"p_{i}_{_N-1}"] = _lin.get(f"p_{i}_{_N-1}", 0) + float(dist_norm[ci, 0])

        # consecutive-stop costs (quadratic)
        for t in range(_N - 1):
            for i in range(_N):
                ci = i + 1
                for j in range(_N):
                    cj = j + 1
                    if i != j:
                        key = (f"p_{i}_{t}", f"p_{j}_{t+1}")
                        _quad[key] = _quad.get(key, 0) + float(dist_norm[ci, cj])

        qp.minimize(linear=_lin, quadratic=_quad)

        for i in range(_N):
            qp.linear_constraint(
                {f"p_{i}_{t}": 1 for t in range(_N)},
                sense="==", rhs=1, name=f"cust_{i}")
        for t in range(_N):
            qp.linear_constraint(
                {f"p_{i}_{t}": 1 for i in range(_N)},
                sense="==", rhs=1, name=f"pos_{t}")

        qubo = _QP2Q(penalty=penalty).convert(qp)
        n_qubits = len(qubo.variables)
        print(f"{_PREFIX} QUBO (position-indexed, subtour-free) -- {n_qubits} qubits, penalty={penalty:.4f}")

        var_names = [v.name for v in qubo.variables]
        # Build position-variable index map
        _pos_var_map = {}  # var_index -> (customer_idx, position)
        for vi, vn in enumerate(var_names):
            if vn.startswith("p_"):
                parts = vn.split("_")
                _pos_var_map[vi] = (int(parts[1]), int(parts[2]))

        ising_op, offset = qubo.to_ising()
        raw      = _np.array([abs(c) for _, c in ising_op.to_list()], dtype=float)
        op_scale = float(_np.max(raw)) if len(raw) > 0 and _np.max(raw) > 0 else 1.0
        ising_n  = ising_op / op_scale

        # ── decode helpers (position-indexed) ──────────────────────────────
        def _decode_counts(counts):
            total = int(sum(counts.values()))
            total_q = max(len(s.replace(" ", "")) for s in counts)
            valid_costs = {}; valid_routes_ = {}

            for bitstr, cnt in counts.items():
                s  = bitstr.replace(" ", "")
                bf = _np.array([1 if c == "1" else 0 for c in s], dtype=_np.int8)
                if len(bf) < total_q:
                    bf = _np.concatenate([_np.zeros(total_q - len(bf), dtype=_np.int8), bf])
                bf = bf[::-1]

                # Extract assignment matrix
                p = _np.zeros((_N, _N), dtype=int)
                for vi, (ci, ti) in _pos_var_map.items():
                    if vi < len(bf):
                        p[ci, ti] = bf[vi]

                # Valid iff doubly-stochastic (each row & col sums to 1)
                if not (_np.all(p.sum(axis=1) == 1) and _np.all(p.sum(axis=0) == 1)):
                    continue

                # Extract route
                perm = tuple(int(_np.argmax(p[:, t])) for t in range(_N))
                route_ids = [all_ids[pi + 1] for pi in perm]  # customer indices to node IDs
                cost = float(route_distance(route_ids, depot.id, dist_map))

                lbl = str(perm)
                valid_costs[lbl] = valid_costs.get(lbl, 0) + int(cnt)
                valid_routes_[lbl] = route_ids

            valid_frac = int(sum(valid_costs.values())) / total if total > 0 else 0.0
            n_valid_unique = len(valid_costs)

            if not valid_costs:
                return float("inf"), float("inf"), float("inf"), None, valid_frac, 0, -1

            sorted_by_prob = sorted(valid_costs, key=valid_costs.__getitem__, reverse=True)

            mp_lbl = sorted_by_prob[0]
            mp_cost = float(route_distance(valid_routes_[mp_lbl], depot.id, dist_map))

            topk_lbls = sorted_by_prob[:_DECODE_K]
            topk_lbl = min(topk_lbls,
                           key=lambda l: float(route_distance(valid_routes_[l], depot.id, dist_map)))
            topk_cost = float(route_distance(valid_routes_[topk_lbl], depot.id, dist_map))

            global_lbl = min(valid_costs.keys(),
                             key=lambda l: float(route_distance(valid_routes_[l], depot.id, dist_map)))
            global_cost = float(route_distance(valid_routes_[global_lbl], depot.id, dist_map))

            best_prob_rank = sorted_by_prob.index(global_lbl) + 1

            return topk_cost, mp_cost, global_cost, valid_routes_[topk_lbl], valid_frac, n_valid_unique, best_prob_rank

    else:
        # ═══════════════════════════════════════════════════════════════════
        # EDGE-BASED QUBO (k > 1)
        # Variables: x_{i}_{j} = 1 if arc (i→j) is used
        # Post-hoc subtour filtering on measured bitstrings.
        # ═══════════════════════════════════════════════════════════════════
        penalty = _N * 2.0

        qp = _QP()
        for i in range(m1):
            for j in range(m1):
                if i != j: qp.binary_var(f"x_{i}_{j}")
        qp.minimize(linear={f"x_{i}_{j}": float(dist_norm[i, j])
                             for i in range(m1) for j in range(m1) if i != j})
        for i in range(1, m1):
            qp.linear_constraint({f"x_{i}_{j}": 1 for j in range(m1) if j != i},
                                  sense="==", rhs=1, name=f"out_{i}")
            qp.linear_constraint({f"x_{j}_{i}": 1 for j in range(m1) if j != i},
                                  sense="==", rhs=1, name=f"in_{i}")
        qp.linear_constraint({f"x_{0}_{j}": 1 for j in range(1, m1)},
                              sense="==", rhs=eff_k, name="out_0")
        qp.linear_constraint({f"x_{j}_{0}": 1 for j in range(1, m1)},
                              sense="==", rhs=eff_k, name="in_0")

        qubo = _QP2Q(penalty=penalty).convert(qp)
        n_qubits = len(qubo.variables)
        print(f"{_PREFIX} QUBO (edge-based, k={eff_k}) -- {n_qubits} qubits, penalty={penalty:.4f}")

        var_names = [v.name for v in qubo.variables]
        ak, ai_arr, aj_arr = [], [], []
        for ki, name in enumerate(var_names):
            if name.startswith("x_"):
                _, si, sj = name.split("_")
                ak.append(ki); ai_arr.append(int(si)); aj_arr.append(int(sj))
        arc_k = _np.array(ak, dtype=int)
        arc_i = _np.array(ai_arr, dtype=int)
        arc_j = _np.array(aj_arr, dtype=int)

        ising_op, offset = qubo.to_ising()
        raw      = _np.array([abs(c) for _, c in ising_op.to_list()], dtype=float)
        op_scale = float(_np.max(raw)) if len(raw) > 0 and _np.max(raw) > 0 else 1.0
        ising_n  = ising_op / op_scale

        # ── decode helpers (edge-based) ────────────────────────────────────
        def _degree_ok(bits):
            idx = _np.flatnonzero(bits)
            od  = _np.bincount(arc_i[idx], minlength=m1)
            id_ = _np.bincount(arc_j[idx], minlength=m1)
            return (od[0] == eff_k and id_[0] == eff_k and
                    bool(_np.all(od[1:] == 1) and _np.all(id_[1:] == 1)))

        def _has_subtour(edges):
            from collections import defaultdict as _dd
            succ = _dd(list)
            for a, b in edges:
                succ[a].append(b)
            depot_exits = succ.get(0, [])
            depot_entries = [a for a, b in edges if b == 0]
            if len(depot_exits) != eff_k or len(depot_entries) != eff_k:
                return True
            non_depot_ids = set(range(1, m1))
            for nd in non_depot_ids:
                if len(succ.get(nd, [])) != 1:
                    return True
            pred_count = _dd(int)
            for a, b in edges:
                if b != 0:
                    pred_count[b] += 1
            for nd in non_depot_ids:
                if pred_count.get(nd, 0) != 1:
                    return True
            visited_global = set()
            completed = 0
            for s in depot_exits:
                cur, path = s, set()
                while True:
                    if cur == 0:
                        completed += 1
                        break
                    if cur in path or cur in visited_global:
                        return True
                    path.add(cur)
                    nxt_list = succ.get(cur, [])
                    if len(nxt_list) != 1:
                        return True
                    cur = nxt_list[0]
                visited_global |= path
            if visited_global != non_depot_ids:
                return True
            return completed != eff_k

        def _bits_to_edges(bits):
            idx = _np.flatnonzero(_np.asarray(bits) >= 0.5)
            return list(zip(arc_i[idx].tolist(), arc_j[idx].tolist()))

        def _edges_cost(edges):
            return float(sum(dist_mat[a, b] for a, b in edges))

        def _decode_counts(counts):
            total   = int(sum(counts.values()))
            total_q = max(len(s.replace(" ", "")) for s in counts)
            valid_costs = {}; valid_edges_ = {}
            for bitstr, cnt in counts.items():
                s  = bitstr.replace(" ", "")
                bf = _np.array([1 if c == "1" else 0 for c in s], dtype=_np.int8)
                if len(bf) < total_q:
                    bf = _np.concatenate([_np.zeros(total_q - len(bf), dtype=_np.int8), bf])
                bf = bf[::-1]
                b  = bf[arc_k]
                if _degree_ok(b):
                    edges = _bits_to_edges(b)
                    if not _has_subtour(edges):
                        lbl = "".join("1" if x else "0" for x in b)
                        valid_costs[lbl]  = valid_costs.get(lbl, 0) + int(cnt)
                        valid_edges_[lbl] = edges
            valid_frac     = int(sum(valid_costs.values())) / total if total > 0 else 0.0
            n_valid_unique = len(valid_costs)

            if not valid_costs:
                return float("inf"), float("inf"), float("inf"), None, valid_frac, 0, -1

            sorted_by_prob = sorted(valid_costs, key=valid_costs.__getitem__, reverse=True)

            mp_lbl  = sorted_by_prob[0]
            mp_cost = _edges_cost(valid_edges_[mp_lbl])

            topk_lbls = sorted_by_prob[:_DECODE_K]
            topk_lbl  = min(topk_lbls, key=lambda l: _edges_cost(valid_edges_[l]))
            topk_cost = _edges_cost(valid_edges_[topk_lbl])

            global_lbl  = min(valid_costs.keys(), key=lambda l: _edges_cost(valid_edges_[l]))
            global_cost = _edges_cost(valid_edges_[global_lbl])

            best_prob_rank = sorted_by_prob.index(global_lbl) + 1

            return topk_cost, mp_cost, global_cost, valid_edges_[topk_lbl], valid_frac, n_valid_unique, best_prob_rank

    # ═══════════════════════════════════════════════════════════════════════
    # QAOA EXECUTION (shared by both formulations)
    # ═══════════════════════════════════════════════════════════════════════

    # ── run all configs ────────────────────────────────────────────────────
    backend_sv = _AerSim(method="statevector", seed_simulator=_leaf_seed)
    est        = _AerEst(run_options={"seed_simulator": _leaf_seed})

    all_results = []

    for ci, cfg in enumerate(_CONFIGS, 1):
        reps       = cfg["reps"]
        opt_name   = cfg["optimizer"].upper()
        maxiter    = cfg["maxiter"]
        n_restarts = cfg["restarts"]

        print(f"{_PREFIX} config {ci}/{len(_CONFIGS)} -- reps={reps} {opt_name} maxiter={maxiter} restarts={n_restarts}")

        ansatz = _QAOAAnsatz(ising_n, reps=reps)
        tqc    = _transpile(ansatz, backend=backend_sv, optimization_level=3)

        def _energy(theta, _tqc=tqc):
            theta = _np.asarray(theta, dtype=float).ravel()
            job   = est.run([_tqc], [ising_n], parameter_values=[theta])
            return float(job.result().values[0]) * op_scale + offset

        opt = (_ADAM(maxiter=maxiter, amsgrad=False)
               if opt_name == "ADAM" else _COBYLA(maxiter=maxiter))

        best_res = None
        for ri in range(n_restarts):
            x0  = 2 * _np.pi * _np.random.rand(ansatz.num_parameters)
            res = opt.minimize(fun=_energy, x0=x0)
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        print(f"{_PREFIX}   energy={best_res.fun:.6f}")

        circ  = tqc.copy()
        if not circ.cregs: circ.measure_all()
        bound  = circ.assign_parameters(best_res.x, inplace=False)
        counts = backend_sv.run(bound, shots=_SHOTS).result().get_counts()

        topk_cost, mp_cost, global_cost, topk_result, vf, n_uniq, prob_rank = _decode_counts(counts)
        status = (f"topk={topk_cost:.1f} mp={mp_cost:.1f} global={global_cost:.1f}"
                  if topk_cost < float("inf") else "NO feasible solution")
        print(f"{_PREFIX}   valid%={vf:.4%}  n_valid={n_uniq}  "
              f"prob_rank={prob_rank}  {status}")
        all_results.append((topk_cost, mp_cost, global_cost, topk_result, vf, n_uniq, prob_rank))

    # ── pick best across configs ───────────────────────────────────────────
    best_topk, best_mp, best_global, best_result, best_vf, best_n_uniq, best_prob_rank = \
        min(all_results, key=lambda t: t[0])

    # ── compute classical optimal for classification ───────────────────────
    opt_cost = _classical_optimal_cost(node_ids, eff_k, depot.id, dist_map)

    if best_result is None:
        _sol = _classical_fallback("no config found a feasible solution")
        QAOA_LOG.append({
            "node_ids": node_ids, "k": eff_k, "n_qubits": n_qubits,
            "outcome": "non-valid",
            "qaoa_cost": float("inf"), "optimal_cost": opt_cost,
            "gap_pct": None,
            "valid_frac": best_vf, "n_valid_unique": 0,
            "best_prob_rank": -1, "solver_used": "classical_fallback",
            "reason": "no feasible solution found",
        })
        return _sol

    # ── Convert result to route IDs ────────────────────────────────────────
    if eff_k == 1:
        # Position-indexed: best_result is already a list of node IDs
        routes_ids = [best_result]
    else:
        # Edge-based: best_result is a list of edges, need to convert
        non_depot_edges = [(a, b) for a, b in best_result if a != 0]
        succ_nd = {a: b for a, b in non_depot_edges}
        starts  = [b for a, b in best_result if a == 0]
        routes_ids = []
        for s in starts:
            route_qi = []; cur = s; steps = 0
            while cur != 0:
                if steps > m1 + 1:
                    routes_ids = []
                    break
                route_qi.append(cur)
                nxt = succ_nd.get(cur)
                if nxt is None:
                    routes_ids = []
                    break
                cur = nxt
                steps += 1
            else:
                routes_ids.append([all_ids[qi] for qi in route_qi])
            if not routes_ids:
                break
        if len(routes_ids) != eff_k or any(len(r) == 0 for r in routes_ids):
            routes_ids = []

    if not routes_ids:
        _sol = _classical_fallback("result-to-route conversion failed")
        QAOA_LOG.append({
            "node_ids": node_ids, "k": eff_k, "n_qubits": n_qubits,
            "outcome": "non-valid",
            "qaoa_cost": float("inf"), "optimal_cost": opt_cost,
            "gap_pct": None,
            "valid_frac": best_vf, "n_valid_unique": best_n_uniq,
            "best_prob_rank": best_prob_rank, "solver_used": "classical_fallback",
            "reason": "result-to-route conversion failed",
        })
        return _sol

    QAOA_STATS["success"] += 1

    # ── classify outcome ───────────────────────────────────────────────────
    gap_pct = (best_topk - opt_cost) / opt_cost * 100 if opt_cost > 0 else 0.0
    if abs(best_topk - opt_cost) < 1e-4:
        outcome = "optimal"
    else:
        outcome = "feasible"

    formulation = "pos-idx" if eff_k == 1 else "edge"
    print(f"{_PREFIX} QAOA {outcome.upper()} #{QAOA_STATS['success']} ({formulation}) -- "
          f"topk={best_topk:.4f}  mp={best_mp:.4f}  global={best_global:.4f}  "
          f"optimal={opt_cost:.4f}  gap={gap_pct:.2f}%  "
          f"rank={best_prob_rank}/{best_n_uniq}  routes={len(routes_ids)}")

    QAOA_LOG.append({
        "node_ids": node_ids, "k": eff_k, "n_qubits": n_qubits,
        "outcome": outcome,
        "qaoa_cost": best_topk, "optimal_cost": opt_cost,
        "gap_pct": round(gap_pct, 4),
        "valid_frac": best_vf, "n_valid_unique": best_n_uniq,
        "best_prob_rank": best_prob_rank, "solver_used": "QAOA",
        "reason": "",
        "formulation": formulation,
    })

    _sol = VRPSolution(routes=[
        Route(node_ids=r,
              distance=route_distance(r, depot.id, dist_map),
              load=sum(node_map[nid].demand for nid in r))
        for r in routes_ids
    ])
    _sol.solver_used = "QAOA"
    return _sol


def allocate_vehicles(nodes: List[Node], clusters: List[List[Node]],
                        k: int) -> List[int]:
    """Distribute *k* vehicles across *clusters* proportional to demand,
    guaranteeing at least 1 vehicle per cluster.  Only called when
    len(clusters) <= k."""
    if not clusters: return []
    total_demand = sum(n.demand for n in nodes)
    if total_demand == 0:
        tc = len(nodes); allocs = [max(1, round(len(cl)/tc*k)) for cl in clusters]
    else:
        cds = [sum(n.demand for n in cl) for cl in clusters]
        allocs = [max(1, round(cd/total_demand*k)) for cd in cds]
    while sum(allocs) > k:
        mi = int(np.argmax(allocs))
        if allocs[mi] > 1: allocs[mi] -= 1
        else: break
    while sum(allocs) < k:
        ratios = [len(clusters[i])/allocs[i] for i in range(len(clusters))]
        allocs[int(np.argmax(ratios))] += 1
    return allocs


def create_super_nodes(clusters: List[List[Node]],
                         csols: List['VRPSolution']) -> List[Node]:
    """Create one synthetic Node per cluster, positioned at the cluster centroid.
    Super-node IDs are negative (−1, −2, …) to avoid collision with real IDs."""
    return [Node(id=-(i+1), x=np.mean([n.x for n in cl]),
                 y=np.mean([n.y for n in cl]), demand=0)
            for i, cl in enumerate(clusters)]


def build_super_dist_map(super_nodes, depot):
    """Build the macro distance map for QAOA using centroid-to-centroid Euclidean distance.

    super_nodes are already positioned at cluster centroids by create_super_nodes().
    Using centroid distance gives QAOA an honest abstraction of the macro-jump cost
    between clusters without any optimistic bias from interior nodes or endpoint luck.
    The greedy stitcher (merge_super_solution) handles micro endpoint alignment
    independently, so the two concerns are cleanly separated.
    """
    all_sn = [depot] + super_nodes
    return {(a.id, b.id): euclidean(a, b) for a in all_sn for b in all_sn}


def _orient_segment(seg, anchor_id, dist_map):
    """Return seg (possibly reversed) so its head is the endpoint closest to anchor_id.

    4-way evaluation: we compare dist(anchor→seg[0]) vs dist(anchor→seg[-1]).
    If the tail is closer, reverse the segment so the physical endpoints align.
    """
    if not seg or len(seg) < 2 or dist_map is None:
        return seg
    d_fwd = dist_map.get((anchor_id, seg[0]),  float('inf'))
    d_rev = dist_map.get((anchor_id, seg[-1]), float('inf'))
    return list(reversed(seg)) if d_rev < d_fwd else seg


def merge_super_solution(ssol, csols, clusters, depot=None, dist_map=None):
    """Merge cluster sub-solutions according to the super-solution route order.

    For each super-route (a group of clusters assigned to one vehicle by QAOA),
    collect all real nodes from those clusters and use QAOA (solve_brute_force)
    to find the optimal visiting sequence, rather than greedy stitching.

    Single-cluster super-routes keep their existing sub-solution ordering
    (already optimised by the leaf-level QAOA call).
    """
    merged = []
    depot_id = depot.id if depot is not None else None

    for sr in ssol.routes:
        valid_cis = [-(sn_id)-1 for sn_id in sr.node_ids
                     if 0 <= -(sn_id)-1 < len(csols)]
        if len(valid_cis) < len(sr.node_ids):
            print(f"WARNING: merge_super_solution dropped "
                  f"{len(sr.node_ids) - len(valid_cis)} super-node(s) with "
                  f"out-of-range IDs in route {sr.node_ids}")
        if not valid_cis:
            # Super-node IDs are out of range — this super-vehicle slot
            # must not be silently dropped or we lose a route from the count.
            # Find any cluster not yet referenced by merged routes and assign it.
            used_cis = set()
            for prev_r in merged:
                pass  # we don't track per-route cluster ownership here
            print(f"WARNING merge_super_solution: super-route {sr.node_ids} has "
                  f"no valid cluster indices -- skipping slot (upstream bug)")
            continue

        if len(valid_cis) == 1:
            # Single cluster in this super-route — keep existing ordering
            ci = valid_cis[0]
            for r in csols[ci].routes:
                if r.node_ids:
                    merged.append(Route(node_ids=list(r.node_ids),
                                        distance=r.distance, load=r.load))
        else:
            # Multiple clusters assigned to one vehicle by the super-QAOA.
            # Collect ALL real nodes from these clusters and re-sequence
            # them with QAOA (solve_brute_force, k=1) for optimal ordering.
            all_real_nodes = []
            for ci in valid_cis:
                all_real_nodes.extend(clusters[ci])

            if not all_real_nodes:
                continue

            if len(all_real_nodes) <= LEAF_SIZE and dist_map is not None and depot is not None:
                # Small enough for direct QAOA sequencing
                re_sol = solve_brute_force(all_real_nodes, 1, depot, dist_map)
                for r in re_sol.routes:
                    if r.node_ids:
                        merged.append(Route(node_ids=list(r.node_ids),
                                            distance=r.distance, load=r.load))
            else:
                # Too many nodes for a single QAOA call — use endpoint-aware
                # stitching as before (each cluster was already QAOA-optimised
                # internally, so we just need to chain them well).
                combined_ids = []
                combined_load = 0.0
                for ci in valid_cis:
                    non_empty = [r for r in csols[ci].routes if r.node_ids]
                    load = sum(r.load for r in csols[ci].routes)
                    for sub_r in non_empty:
                        seg = list(sub_r.node_ids)
                        anchor = combined_ids[-1] if combined_ids else depot_id
                        seg = _orient_segment(seg, anchor, dist_map)
                        combined_ids.extend(seg)
                        combined_load += load

                if combined_ids:
                    d = (route_distance(combined_ids, depot.id, dist_map)
                         if depot is not None and dist_map is not None else 0.0)
                    merged.append(Route(node_ids=combined_ids, distance=d,
                                        load=combined_load))

    if not merged:
        for csol in csols:
            for r in csol.routes:
                merged.append(r)

    expected_k = len(ssol.routes)
    if len(merged) != expected_k:
        print(f"WARNING merge_super_solution: produced {len(merged)} routes, "
              f"expected {expected_k} -- re-grouping to match")
        # Too many: merge smallest pairs until we reach expected_k
        while len(merged) > expected_k:
            merged.sort(key=lambda r: len(r.node_ids))
            a, b = merged[0], merged[1]
            combined_ids = list(a.node_ids) + list(b.node_ids)
            d = (route_distance(combined_ids, depot.id, dist_map)
                 if depot is not None and dist_map is not None else 0.0)
            load = a.load + b.load
            merged = merged[2:] + [Route(node_ids=combined_ids, distance=d, load=load)]
        # Too few: split the largest route at its longest interior edge
        while len(merged) < expected_k:
            merged.sort(key=lambda r: len(r.node_ids), reverse=True)
            biggest = merged[0].node_ids
            if len(biggest) < 2:
                break  # cannot split a single-node route
            mid = len(biggest) // 2
            left, right = list(biggest[:mid]), list(biggest[mid:])
            dl = (route_distance(left,  depot.id, dist_map)
                  if depot is not None and dist_map is not None else 0.0)
            dr = (route_distance(right, depot.id, dist_map)
                  if depot is not None and dist_map is not None else 0.0)
            nm_msol = {}
            for cl in clusters:
                for nd in cl:
                    nm_msol[nd.id] = nd
            merged = merged[1:] + [
                Route(node_ids=left,  distance=dl,
                      load=sum(nm_msol[x].demand for x in left  if x in nm_msol)),
                Route(node_ids=right, distance=dr,
                      load=sum(nm_msol[x].demand for x in right if x in nm_msol)),
            ]
    return VRPSolution(routes=merged)


def vrp_solver_plain(k, nodes, depot, dist_map=None, dist_matrix=None, _depth=0):
    if dist_map is None:
        dist_map = build_dist_map(nodes, depot, dist_matrix)
    if not nodes: return VRPSolution(routes=[])
    if len(nodes) == 1:
        nid = nodes[0].id; d = dist_map[(depot.id, nid)] * 2
        return VRPSolution(routes=[Route(node_ids=[nid], distance=d, load=nodes[0].demand)])

    if len(nodes) <= LEAF_SIZE:
        return solve_brute_force(nodes, k, depot, dist_map)

    clusters = cluster_nodes(nodes, depot, max(2, math.ceil(math.sqrt(len(nodes)))))

    if k == 1:
        cluster_solutions = [vrp_solver_plain(1, cl, depot, dist_map, _depth=_depth+1)
                             for cl in clusters]
        # Use QAOA (solve_brute_force) to find the best super-node ordering,
        # then merge with endpoint-aware stitching instead of blind concatenation.
        super_nodes = create_super_nodes(clusters, cluster_solutions)
        sdm         = build_super_dist_map(super_nodes, depot)
        if len(super_nodes) > LEAF_SIZE:
            super_solution = vrp_solver_plain(1, super_nodes, depot, dist_map=sdm, _depth=_depth+1)
        else:
            super_solution = solve_brute_force(super_nodes, 1, depot, sdm)
        _sol = merge_super_solution(super_solution, cluster_solutions, clusters, depot, dist_map)
        try:
            assert len({nid for r in _sol.routes for nid in r.node_ids}) == len(nodes)
        except AssertionError:
            print(f"WARNING (k=1 plain): merge validation failed -- concatenating all nodes into 1 route")
            all_ids_k1p = [nid for cs in cluster_solutions
                           for r in cs.routes for nid in r.node_ids]
            _sol = VRPSolution(routes=[Route(
                node_ids=all_ids_k1p,
                distance=route_distance(all_ids_k1p, depot.id, dist_map),
                load=sum(n.demand for n in nodes)
            )])
        return _sol

    C = len(clusters)

    if C > k:
        cluster_solutions = [vrp_solver_plain(1, cl, depot, dist_map, _depth=_depth+1)
                             for cl in clusters]
        # ISSUE 1 FIX: use real inter-cluster distances for super-VRP
        super_nodes = create_super_nodes(clusters, cluster_solutions)
        sdm         = build_super_dist_map(super_nodes, depot)
        if len(super_nodes) > LEAF_SIZE:
            super_solution = vrp_solver_plain(k, super_nodes, depot, dist_map=sdm, _depth=_depth+1)
        else:
            super_solution = solve_brute_force(super_nodes, k, depot, sdm)
        _sol = merge_super_solution(super_solution, cluster_solutions, clusters, depot, dist_map)
        try:
            assert len({nid for r in _sol.routes for nid in r.node_ids}) == len(nodes)
        except AssertionError:
            print(f"WARNING (C>k plain): merge validation failed -- re-grouping {len(cluster_solutions)} cluster routes into {k}")
            flat_routes_p = [r for cs in cluster_solutions for r in cs.routes]
            groups_p = [[] for _ in range(k)]
            for ci, cr in enumerate(flat_routes_p):
                groups_p[ci % k].extend(cr.node_ids)
            _sol = VRPSolution(routes=[
                Route(node_ids=g,
                      distance=route_distance(g, depot.id, dist_map),
                      load=sum({n.id: n for n in nodes}[nid].demand for nid in g))
                for g in groups_p if g
            ])
        return _sol

    else:
        vehicle_alloc     = allocate_vehicles(nodes, clusters, k)
        cluster_solutions = [vrp_solver_plain(vehicle_alloc[i], cl, depot, dist_map, _depth=_depth+1)
                             for i, cl in enumerate(clusters)]
        all_routes = [r for csol in cluster_solutions for r in csol.routes]
        return VRPSolution(routes=all_routes)



def algo_recursive_qaoa(nodes, k, depot, dist_map, node_map):
    """Recursive QAOA solver: uses Qiskit QAOA at leaves (notebook-faithful)."""
    # Ensure GLOBAL_MAX_DIST is set from the distance map before the leaf solver
    # checks it (guards the "GLOBAL_MAX_DIST not initialised" error path).
    global GLOBAL_MAX_DIST
    if dist_map:
        GLOBAL_MAX_DIST = max(GLOBAL_MAX_DIST, max(dist_map.values()))
    return vrp_solver_plain(k, nodes, depot, dist_map)



# ───────────────────── combined "X + 2-opt" wrappers ─────────────────────
def with_2opt(algo_fn):
    def wrapped(nodes, k, depot, dist_map, node_map):
        sol = algo_fn(nodes, k, depot, dist_map, node_map)
        if sol is None:
            return None
        return apply_local_search(sol, depot.id, dist_map, node_map, two_opt)
    return wrapped


def with_oropt(algo_fn):
    def wrapped(nodes, k, depot, dist_map, node_map):
        sol = algo_fn(nodes, k, depot, dist_map, node_map)
        if sol is None:
            return None
        return apply_local_search(sol, depot.id, dist_map, node_map, or_opt)
    return wrapped


# ───────────────────── instance loading ─────────────────────────────────
def list_array_instances() -> List[str]:
    names = []
    for p in sorted(ARRAYS_DIR.glob("*.py")):
        if p.stem.startswith("__"):
            continue
        names.append(p.stem)
    return names


def load_instance(name: str) -> Tuple[List[Node], Node, list]:
    mod = importlib.import_module(f"arrays.{name}")
    coords = mod.node_coords
    dist_matrix = mod.dist_matrix
    depot = Node(id=0, x=coords[0][0], y=coords[0][1], demand=0.0)
    nodes = [Node(id=nid, x=xy[0], y=xy[1], demand=1.0)
             for nid, xy in coords.items() if nid != 0]
    return nodes, depot, dist_matrix


def read_vrp_k(name: str) -> int | None:
    """Return MAX_VEHICLES from the matching .vrp file, or None."""
    m = re.match(r"RioClaroPostToy_(\d+)_(\d+)", name)
    if not m:
        return None
    n_nodes = int(m.group(1))
    vrp_path = INSTANCES_DIR / f"{n_nodes}-Nodes" / f"{name}.vrp"
    if not vrp_path.exists():
        return None
    try:
        for line in vrp_path.read_text().splitlines():
            if line.startswith("MAX_VEHICLES"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return None


# ───────────────────── validation ───────────────────────────────────────
def validate(sol: VRPSolution, nodes: List[Node]) -> bool:
    expected = {n.id for n in nodes}
    seen: set[int] = set()
    for r in sol.routes:
        for nid in r.node_ids:
            if nid in seen:
                return False
            seen.add(nid)
    return seen == expected


# ───────────────────── registry + runner ────────────────────────────────
ALGORITHMS: List[Tuple[str, Callable]] = [
    ("Recursive QAOA (NO 2-opt) [Qiskit at leaves]", algo_recursive_qaoa),
    ("Recursive QAOA + 2-opt [Qiskit at leaves]", with_2opt(algo_recursive_qaoa)),
    ("Nearest-Neighbour", algo_nearest_neighbour),
    ("Nearest-Neighbour + 2-opt", with_2opt(algo_nearest_neighbour)),
    ("Sweep", algo_sweep),
    ("Sweep + 2-opt", with_2opt(algo_sweep)),
    ("Clarke-Wright (Parallel)", algo_savings_parallel),
    ("Clarke-Wright (Parallel) + 2-opt", with_2opt(algo_savings_parallel)),
    ("Clarke-Wright (Parallel) + Or-opt", with_oropt(algo_savings_parallel)),
    ("Clarke-Wright (Sequential)", algo_savings_sequential),
    ("Clarke-Wright (Sequential) + 2-opt", with_2opt(algo_savings_sequential)),
    ("Nearest Insertion", algo_nearest_insertion),
    ("Nearest Insertion + 2-opt", with_2opt(algo_nearest_insertion)),
    ("Farthest Insertion", algo_farthest_insertion),
    ("Farthest Insertion + 2-opt", with_2opt(algo_farthest_insertion)),
    ("Cheapest Insertion", algo_cheapest_insertion),
    ("Cheapest Insertion + 2-opt", with_2opt(algo_cheapest_insertion)),
    ("Simulated Annealing", algo_simulated_annealing),
    ("Tabu Search", algo_tabu_search),
    ("Iterated Local Search", algo_iterated_local_search),
    ("Genetic Algorithm", algo_genetic),
    ("OR-Tools (PATH_CHEAPEST_ARC + GLS)", algo_ortools_pca),
    ("OR-Tools (SAVINGS + GLS)", algo_ortools_savings),
    ("OR-Tools (Christofides + GLS)", algo_ortools_christofides),
    ("OR-Tools (Parallel Cheapest Ins. + GLS)", algo_ortools_parallel_cheapest),
    ("OR-Tools (SA)", algo_ortools_sa),
    ("OR-Tools (Tabu)", algo_ortools_tabu),
    ("OR-Tools (GLS, 10s)", algo_ortools_gls),
]


# ───────────────────── plotting ─────────────────────────────────────────
CLUSTER_COLORS = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
    "#800000", "#aaffc3", "#808000", "#000075", "#a9a9a9", "#ffe119",
]


def plot_solution(sol: VRPSolution, depot: Node, coords: Dict[int, Tuple[float, float]],
                  title: str, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    # All nodes
    xs = [xy[0] for xy in coords.values()]
    ys = [xy[1] for xy in coords.values()]
    ax.scatter(xs, ys, s=20, c="#cccccc", zorder=1)
    # Depot
    ax.scatter([depot.x], [depot.y], s=200, c="black", marker="*",
               zorder=5, label="Depot")

    for i, r in enumerate(sol.routes):
        if not r.node_ids:
            continue
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        path = [depot.id] + r.node_ids + [depot.id]
        px = [coords[n][0] for n in path]
        py = [coords[n][1] for n in path]
        ax.plot(px, py, "-", color=color, linewidth=1.5, zorder=2,
                label=f"Route {i+1} (d={r.distance:.0f})")
        rx = [coords[n][0] for n in r.node_ids]
        ry = [coords[n][1] for n in r.node_ids]
        ax.scatter(rx, ry, s=35, color=color, zorder=4, edgecolor="black",
                   linewidth=0.3)

    ax.set_title(f"{title}\nTotal: {sol.total_distance:.1f}   "
                 f"Routes: {sol.num_vehicles_used}   "
                 f"\u03c3: {sol.distance_std:.1f}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_all_algorithms_grid(sols: Dict[str, VRPSolution], depot: Node,
                             coords: Dict[int, Tuple[float, float]],
                             instance_name: str, k: int, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(n, s) for n, s in sols.items() if s is not None]
    if not items:
        return
    ncols = 3
    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    xs = [xy[0] for xy in coords.values()]
    ys = [xy[1] for xy in coords.values()]

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, (aname, sol) in enumerate(items):
        ax = axes[idx // ncols, idx % ncols]
        ax.set_visible(True)
        ax.scatter(xs, ys, s=10, c="#cccccc", zorder=1)
        ax.scatter([depot.x], [depot.y], s=120, c="black", marker="*", zorder=5)
        for i, r in enumerate(sol.routes):
            if not r.node_ids:
                continue
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            path = [depot.id] + r.node_ids + [depot.id]
            px = [coords[n][0] for n in path]
            py = [coords[n][1] for n in path]
            ax.plot(px, py, "-", color=color, linewidth=1.2, zorder=2)
            rx = [coords[n][0] for n in r.node_ids]
            ry = [coords[n][1] for n in r.node_ids]
            ax.scatter(rx, ry, s=18, color=color, zorder=4,
                       edgecolor="black", linewidth=0.2)
        ax.set_title(f"{aname}\nd={sol.total_distance:.0f}  "
                     f"r={sol.num_vehicles_used}  \u03c3={sol.distance_std:.0f}",
                     fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"{instance_name} — {k} vehicles — classical VRP comparison",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ───────────────────── PDF BI report ────────────────────────────────────
def generate_pdf_report(results: List[dict], instance_name: str, k: int,
                        n_nodes: int, stamp: str,
                        data_dir: Path, plots_dir: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.backends.backend_pdf import PdfPages

        sorted_res = sorted(results, key=lambda r: r.get("gap", 0))
        valid = [r for r in sorted_res if r["valid"]]
        best_total = min(r["total"] for r in valid) if valid else None
        worst_total = max(r["total"] for r in valid) if valid else None

        prefix = f"{instance_name}_{k}veh_{n_nodes}nodes"
        pdf_path = data_dir / f"{prefix}_benchmark_report.pdf"

        COL_W = [0.38, 0.06, 0.10, 0.09, 0.10, 0.08, 0.07, 0.08]  # relative col widths

        GREEN_BEST = "#c6efce"
        RED_WORST = "#ffc7ce"
        GREY_INV = "#e0e0e0"
        BLUE_BAR = "#4472c4"
        GREEN_BAR = "#00b050"

        with PdfPages(pdf_path) as pdf:
            # ── PAGE 1: Cover ─────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.set_axis_off()
            ax.axhline(0.82, color="#333333", linewidth=3, xmin=0.05, xmax=0.95)
            ax.axhline(0.16, color="#333333", linewidth=1, xmin=0.05, xmax=0.95)
            ax.text(0.5, 0.91, "VRP Benchmark Report", ha="center", va="center",
                    fontsize=28, fontweight="bold", transform=ax.transAxes)
            ax.text(0.5, 0.74, instance_name, ha="center", va="center",
                    fontsize=20, color="#1a1a8c", transform=ax.transAxes)
            meta = [
                ("Vehicles (k)", str(k)),
                ("Nodes (n)", str(n_nodes)),
                ("Algorithms tested", str(len(results))),
                ("Valid solutions", str(len(valid))),
                ("Run timestamp", stamp),
            ]
            for row_i, (label, val) in enumerate(meta):
                y = 0.60 - row_i * 0.075
                ax.text(0.32, y, label + ":", ha="right", fontsize=13,
                        color="#555555", transform=ax.transAxes)
                ax.text(0.35, y, val, ha="left", fontsize=13, fontweight="bold",
                        transform=ax.transAxes)
            if best_total is not None:
                ax.text(0.5, 0.10, f"Best total distance: {best_total:.2f}",
                        ha="center", fontsize=12, color="#006400", transform=ax.transAxes)
            ax.text(0.5, 0.04, "Generated by classical_vrp_benchmark.py",
                    ha="center", fontsize=9, color="#888888", transform=ax.transAxes)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # ── PAGE 2: Results Table ─────────────────────────────────────
            ncols = 8
            col_labels = ["Algorithm", "k", "Total Dist", "Std Dev",
                          "W_Fairness", "Time(s)", "Valid", "Gap%"]
            rows = []
            for r in sorted_res:
                rows.append([
                    r["algo"],
                    str(r["k_used"]),
                    f"{r['total']:.2f}",
                    f"{r['std']:.2f}",
                    f"{r.get('weighted_fairness', 0):.2f}",
                    f"{r['time']:.4f}",
                    "OK" if r["valid"] else "INVALID",
                    f"{r.get('gap', 0):.1f}",
                ])
            nrows = len(rows)
            fig_h = max(6, 1.0 + nrows * 0.32)
            fig2, ax2 = plt.subplots(figsize=(14, fig_h))
            ax2.set_axis_off()
            ax2.set_title(f"{instance_name} — Algorithm Results (sorted by Gap%)",
                          fontsize=12, fontweight="bold", pad=8)
            tbl = ax2.table(
                cellText=rows,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7.5)
            tbl.scale(1.0, 1.6)
            # colour header
            for j in range(ncols):
                tbl[0, j].set_facecolor("#2e4057")
                tbl[0, j].set_text_props(color="white", fontweight="bold")
            # colour data rows
            for row_i, r in enumerate(sorted_res):
                fc = GREY_INV if not r["valid"] else (
                    GREEN_BEST if r["total"] == best_total else (
                        RED_WORST if r["total"] == worst_total else "#ffffff"))
                for j in range(ncols):
                    tbl[row_i + 1, j].set_facecolor(fc)
            fig2.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

            # ── PAGE 3: Distance Bar Chart ────────────────────────────────
            fig_h3 = max(6, len(sorted_res) * 0.38 + 1.5)
            fig3, ax3 = plt.subplots(figsize=(12, fig_h3))
            labels3 = [r["algo"] for r in sorted_res]
            vals3 = [r["total"] for r in sorted_res]
            colors3 = []
            for r in sorted_res:
                if not r["valid"]:
                    colors3.append("#a9a9a9")
                elif best_total is not None and abs(r["total"] - best_total) < 1e-6:
                    colors3.append(GREEN_BAR)
                else:
                    colors3.append(BLUE_BAR)
            bars = ax3.barh(range(len(labels3)), vals3, color=colors3, edgecolor="white",
                            linewidth=0.5)
            ax3.set_yticks(range(len(labels3)))
            ax3.set_yticklabels(labels3, fontsize=7)
            ax3.invert_yaxis()
            if best_total is not None:
                ax3.axvline(best_total, color=GREEN_BAR, linestyle="--",
                            linewidth=1.5, label=f"Best: {best_total:.1f}")
            ax3.set_xlabel("Total Distance", fontsize=10)
            ax3.set_title(f"{instance_name} — Total Distance by Algorithm", fontsize=12,
                          fontweight="bold")
            ax3.legend(fontsize=9)
            for bar, val in zip(bars, vals3):
                ax3.text(val + max(vals3) * 0.005, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}", va="center", fontsize=6)
            ax3.grid(axis="x", alpha=0.3)
            fig3.tight_layout()
            pdf.savefig(fig3)
            plt.close(fig3)

            # ── PAGE 4: Fairness Analysis ─────────────────────────────────
            fig_h4 = max(6, len(sorted_res) * 0.38 + 2)
            fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, fig_h4))

            def _bar_fairness(ax, metric_key, title, best_is_min=True):
                vals = [r.get(metric_key, 0) for r in sorted_res]
                best_val = min(v for v, r in zip(vals, sorted_res) if r["valid"]) if valid else None
                cols = []
                for v, r in zip(vals, sorted_res):
                    if not r["valid"]:
                        cols.append("#a9a9a9")
                    elif best_val is not None and abs(v - best_val) < 1e-6:
                        cols.append(GREEN_BAR)
                    else:
                        cols.append(BLUE_BAR)
                ax.barh(range(len(sorted_res)), vals, color=cols, edgecolor="white",
                        linewidth=0.5)
                ax.set_yticks(range(len(sorted_res)))
                ax.set_yticklabels([r["algo"] for r in sorted_res], fontsize=7)
                ax.invert_yaxis()
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.grid(axis="x", alpha=0.3)

            _bar_fairness(ax4a, "std", "Std Dev of Route Distances (lower = fairer)")
            _bar_fairness(ax4b, "weighted_fairness", "Weighted Fairness (lower = better)")

            # Top-3 fairest text box
            top3_fair = sorted(valid, key=lambda r: r["std"])[:3]
            box_lines = ["Top-3 Fairest (lowest Std Dev):"]
            for rank_i, r in enumerate(top3_fair, 1):
                box_lines.append(f"  {rank_i}. {r['algo']}  (std={r['std']:.2f})")
            fig4.text(0.5, 0.01, "\n".join(box_lines), ha="center", fontsize=8,
                      va="bottom", bbox=dict(boxstyle="round", facecolor="#fffbe6",
                                             edgecolor="#ccaa00", alpha=0.9))
            fig4.suptitle(f"{instance_name} — Fairness Metrics", fontsize=12,
                          fontweight="bold")
            fig4.tight_layout(rect=(0, 0.1, 1, 0.96))
            pdf.savefig(fig4)
            plt.close(fig4)

            # ── PAGE 5: Summary Stats + Winners ──────────────────────────
            fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(12, 10),
                                               gridspec_kw={"height_ratios": [1, 1]})
            fig5.suptitle(f"{instance_name} — Summary Statistics", fontsize=13,
                          fontweight="bold")

            # aggregate stats table
            ax5a.set_axis_off()
            stat_metrics = [
                ("Total Distance", "total"),
                ("Std Dev", "std"),
                ("Weighted Fairness", "weighted_fairness"),
                ("Time (s)", "time"),
            ]
            stat_rows = []
            for label, key in stat_metrics:
                vals = [r[key] for r in valid] if valid else [0]
                stat_rows.append([
                    label,
                    f"{min(vals):.4f}",
                    f"{max(vals):.4f}",
                    f"{float(np.mean(vals)):.4f}",
                    f"{float(np.std(vals)):.4f}",
                ])
            tbl5 = ax5a.table(
                cellText=stat_rows,
                colLabels=["Metric", "Min", "Max", "Avg", "Std"],
                loc="center",
                cellLoc="center",
            )
            tbl5.auto_set_font_size(False)
            tbl5.set_fontsize(10)
            tbl5.scale(1.2, 2.0)
            for j in range(5):
                tbl5[0, j].set_facecolor("#2e4057")
                tbl5[0, j].set_text_props(color="white", fontweight="bold")
            ax5a.set_title("Aggregate Statistics (valid solutions only)",
                           fontsize=11, pad=6)

            # winners panel
            ax5b.set_axis_off()
            if valid:
                best_dist_r = min(valid, key=lambda r: r["total"])
                best_fair_r = min(valid, key=lambda r: r["std"])
                best_wf_r = min(valid, key=lambda r: r.get("weighted_fairness", float("inf")))
                winners = [
                    ("Best Total Distance", best_dist_r["algo"],
                     f"{best_dist_r['total']:.2f}", GREEN_BEST),
                    ("Best Fairness (min Std Dev)", best_fair_r["algo"],
                     f"std={best_fair_r['std']:.2f}", "#cce5ff"),
                    ("Best Weighted Fairness", best_wf_r["algo"],
                     f"{best_wf_r.get('weighted_fairness', 0):.2f}", "#e8d5f5"),
                ]
                ax5b.set_title("Winners", fontsize=11, pad=6)
                for wi, (cat, algo, val, fc) in enumerate(winners):
                    y = 0.75 - wi * 0.28
                    rect = mpatches.FancyBboxPatch((0.05, y - 0.10), 0.90, 0.22,
                                                   boxstyle="round,pad=0.02",
                                                   facecolor=fc, edgecolor="#888888",
                                                   transform=ax5b.transAxes, zorder=2)
                    ax5b.add_patch(rect)
                    ax5b.text(0.50, y + 0.06, cat, ha="center", fontsize=11,
                              fontweight="bold", transform=ax5b.transAxes, zorder=3)
                    ax5b.text(0.50, y - 0.02, algo, ha="center", fontsize=9,
                              color="#1a1a8c", transform=ax5b.transAxes, zorder=3)
                    ax5b.text(0.50, y - 0.08, val, ha="center", fontsize=10,
                              color="#444444", transform=ax5b.transAxes, zorder=3)

            fig5.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig5)
            plt.close(fig5)

        print(f"  ✓ PDF report: {pdf_path.relative_to(pdf_path.parents[3])}")
    except Exception as e:
        print(f"  PDF report failed: {e}")


# ───────────────────── results folder scaffolding ───────────────────────
_GLOBAL_RUN_STAMP: str | None = None  # Set once at start of main()

def set_global_run_stamp():
    """Set the global run timestamp (called once at start of main)."""
    global _GLOBAL_RUN_STAMP
    _GLOBAL_RUN_STAMP = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return _GLOBAL_RUN_STAMP

def make_result_dirs(instance_name: str, k: int) -> Tuple[Path, Path, Path, str]:
    """Create result directories under a timestamped run folder.
    
    Structure:
        results/
            <YYYY-MM-DD_HHMMSS>/
                <instance>_k<k>/
                    data/
                    plots/
                    logs/
    """
    if _GLOBAL_RUN_STAMP is None:
        raise RuntimeError("Global run stamp not set. Call set_global_run_stamp() first.")
    
    run_dir = ROOT / "results" / _GLOBAL_RUN_STAMP
    instance_dir = run_dir / f"{instance_name}_k{k}"
    data_dir = instance_dir / "data"
    plots_dir = instance_dir / "plots"
    logs_dir = instance_dir / "logs"
    for d in (data_dir, plots_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return data_dir, plots_dir, logs_dir, _GLOBAL_RUN_STAMP


def run_one_instance(name: str, k: int, save_outputs: bool = True,
                     active_algorithms=None, run_timestamp: str | None = None) -> List[dict]:
    if active_algorithms is None:
        active_algorithms = ALGORITHMS
    nodes, depot, dist_matrix = load_instance(name)
    dist_map = build_dist_map(nodes, depot, dist_matrix)
    node_map = {n.id: n for n in nodes}
    coords = {depot.id: (depot.x, depot.y)}
    for n in nodes:
        coords[n.id] = (n.x, n.y)

    print(f"\n=== {name}  (n={len(nodes)}, k={k}) ===")
    header = f"{'Algorithm':<42}{'k':>4}{'Total':>15}{'Std':>12}{'W_Fair':>10}{'Time_s':>10}  Valid   Gap%"
    print(header)
    print("-" * len(header))

    results: List[dict] = []
    sol_by_algo: Dict[str, VRPSolution] = {}
    for aname, fn in active_algorithms:
        t0 = time.perf_counter()
        try:
            sol = fn(nodes, k, depot, dist_map, node_map)
        except Exception as e:
            print(f"{aname:<42}  ERROR: {e}")
            continue
        elapsed = time.perf_counter() - t0
        if sol is None:
            print(f"{aname:<42}  SKIPPED")
            continue
        ok = validate(sol, nodes)
        sol_by_algo[aname] = sol
        results.append({
            "instance": name,
            "algo": aname,
            "k_used": sol.num_vehicles_used,
            "total": sol.total_distance,
            "std": sol.distance_std,
            "weighted_fairness": sol.weighted_fairness,
            "time": elapsed,
            "valid": ok,
        })

    valid = [r for r in results if r["valid"]]
    if valid:
        best = min(r["total"] for r in valid)
        for r in results:
            r["gap"] = 100 * (r["total"] - best) / best if best > 0 else 0.0

    for r in results:
        print(f"{r['algo']:<42}{r['k_used']:>4}{r['total']:>15.2f}"
              f"{r['std']:>12.2f}{r.get('weighted_fairness', 0):>10.2f}"
              f"{r['time']:>10.4f}  "
              f"{'OK ' if r['valid'] else 'BAD'}  {r.get('gap', 0):>6.2f}")

    if save_outputs:
        data_dir, plots_dir, logs_dir, stamp = make_result_dirs(name, k)
        n = len(nodes)
        prefix = f"{name}_{k}veh_{n}nodes"

        # CSV matching the notebook's schema
        csv_path = data_dir / f"{prefix}_benchmark_algorithms.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Algorithm", "k_used", "Total_Distance", "Distance_Std",
                        "Weighted_Fairness", "Time_sec", "Valid", "Gap_vs_Best_Pct"])
            for r in results:
                w.writerow([r["algo"], r["k_used"], f"{r['total']:.2f}",
                            f"{r['std']:.2f}", f"{r.get('weighted_fairness', 0):.2f}",
                            f"{r['time']:.4f}",
                            "OK" if r["valid"] else "INVALID",
                            f"{r.get('gap', 0):.1f}"])

        # JSON report
        json_path = data_dir / f"{prefix}_benchmark_report.json"
        with json_path.open("w") as f:
            json.dump({
                "instance": name,
                "n_nodes": n,
                "k_vehicles": k,
                "timestamp": stamp,
                "results": results,
            }, f, indent=2)

        # Markdown summary
        md_path = data_dir / f"{prefix}_benchmark_summary.md"
        lines = [f"# Benchmark Results: {name} (k={k}, n={n})\n",
                 f"**Date:** {stamp}\n\n",
                 "| Algorithm | k | Total | Std | W_Fair | Time(s) | Valid | Gap% |",
                 "|---|---|---|---|---|---|---|---|"]
        for r in sorted(results, key=lambda r: r.get("gap", 0)):
            lines.append(
                f"| {r['algo']} | {r['k_used']} | {r['total']:.2f} | "
                f"{r['std']:.2f} | {r.get('weighted_fairness', 0):.2f} | {r['time']:.4f} | "
                f"{'OK' if r['valid'] else 'INVALID'} | {r.get('gap', 0):.1f} |"
            )
        md_path.write_text("\n".join(lines) + "\n")

        # PDF BI report
        try:
            generate_pdf_report(results, name, k, n, stamp, data_dir, plots_dir)
        except Exception as e:
            print(f"  PDF report failed: {e}")

        # Plots: individual + all-in-one grid
        for aname, sol in sol_by_algo.items():
            slug = re.sub(r"[^A-Za-z0-9_.+-]+", "_", aname).strip("_")
            p = plots_dir / f"{prefix}_{slug}.png"
            try:
                plot_solution(sol, depot, coords, f"{name} — {aname}", p)
            except Exception as e:
                print(f"  plot failed for {aname}: {e}")
        grid_path = plots_dir / f"{prefix}_all_algorithms_grid.png"
        try:
            plot_all_algorithms_grid(sol_by_algo, depot, coords, name, k, grid_path)
        except Exception as e:
            print(f"  grid plot failed: {e}")

        rel_path = data_dir.relative_to(ROOT)
        print(f"  ✓ Saved to: {rel_path}")

    return results


def write_comparison(all_results: List[dict], out_path: Path):
    import csv
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Instance", "Algorithm", "k_used", "Total_Distance",
                    "Distance_Std", "Weighted_Fairness", "Time_sec", "Valid", "Gap_vs_Best_Pct"])
        for r in all_results:
            w.writerow([r["instance"], r["algo"], r["k_used"],
                        f"{r['total']:.2f}", f"{r['std']:.2f}",
                        f"{r.get('weighted_fairness', 0):.2f}",
                        f"{r['time']:.4f}", "OK" if r["valid"] else "INVALID",
                        f"{r.get('gap', 0):.2f}"])



def update_runs_index(run_stamp: str, instance_k_pairs: List[Tuple[str, int]], all_results: List[dict]):
    """Update the global RUNS_INDEX.md file with this run's metadata."""
    index_path = ROOT / "results" / "RUNS_INDEX.md"
    
    # Build this run's entry
    valid_results = [r for r in all_results if r["valid"]]
    if valid_results:
        best = min(valid_results, key=lambda r: r["total"])
        summary = f"best: {best['algo']} ({best['total']:.1f})"
    else:
        summary = "no valid solutions"
    
    instances_str = ", ".join([f"{name} (k={k})" for name, k in instance_k_pairs])
    entry = f"\n## {run_stamp}\n**Instances:** {instances_str}\n**Summary:** {summary}\n**Folder:** `results/{run_stamp}/`\n"
    
    # Read existing index or create header
    if index_path.exists():
        content = index_path.read_text()
    else:
        content = """# Benchmark Runs Index

All VRP benchmark runs are timestamped and organized in dated folders.

---\n"""
    
    # Append new entry
    content += entry
    index_path.write_text(content)
    print(f"✓ Updated {index_path.relative_to(ROOT)}")

def create_run_manifest(run_stamp: str, instance_k_pairs: List[Tuple[str, int]]):
    """Create a manifest.md file for this run."""
    run_dir = ROOT / "results" / run_stamp
    manifest_path = run_dir / "manifest.md"
    
    lines = [
        f"# Benchmark Run Manifest",
        f"",
        f"**Started:** {run_stamp}",
        f"**Completed:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Instances Benchmarked",
        f""
    ]
    
    for name, k in instance_k_pairs:
        lines.append(f"- `{name}` with k={k}")
        inst_dir = run_dir / f"{name}_k{k}"
        if inst_dir.exists():
            data_files = list((inst_dir / "data").glob("*"))
            lines.append(f"  - Results: {len(data_files)} files")
    
    lines.extend([
        f"",
        f"## File Organization",
        f""",
        f"Each instance has its own folder:",
        f""",
        f"```",
        f"{run_stamp}/",
        f"├── manifest.md (this file)",
        f"├── instance_1_k7/",
        f"│   ├── data/",
        f"│   │   ├── benchmark_algorithms.csv",
        f"│   │   ├── benchmark_report.json",
        f"│   │   └── benchmark_summary.md",
        f"│   ├── plots/",
        f"│   └── logs/",
        f"├── instance_2_k15/",
        f"│   └── ...",
        f"```",
    ])
    
    manifest_path.write_text("\n".join(lines) + "\n")
    print(f"✓ Created manifest: {manifest_path.relative_to(ROOT)}")

def print_winners(all_results: List[dict]):
    by_inst: Dict[str, List[dict]] = {}
    for r in all_results:
        by_inst.setdefault(r["instance"], []).append(r)
    print("\n" + "="*80)
    print("WINNERS BY INSTANCE (Best Distance vs Best Fairness)")
    print("="*80)
    for inst, rs in by_inst.items():
        valid = [r for r in rs if r["valid"]]
        if not valid:
            print(f"  {inst}: no valid solutions")
            continue
        best_dist = min(valid, key=lambda r: r["total"])
        best_fair = min(valid, key=lambda r: r["std"])
        print(f"  {inst:<28}  best distance: {best_dist['algo']:<35} "
              f"({best_dist['total']:.1f})  |  best fairness: "
              f"{best_fair['algo']:<35} (std={best_fair['std']:.1f})")
    print("="*80)


# ───────────────────── entry point ──────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Classical VRP benchmark")
    ap.add_argument("--instance", help="e.g. 50_0 (runs RioClaroPostToy_50_0). "
                                       "Omit to run all.")
    ap.add_argument("--k", type=int, default=None,
                    help="Number of vehicles. Default: MAX_VEHICLES from .vrp, "
                         "or 7 if not found.")
    ap.add_argument("--out", default="classical_benchmark_results.csv",
                    help="Output CSV path.")
    ap.add_argument("--algos", nargs="+", metavar="ALGO",
                    help="Run only algorithms whose names contain any of these "
                         "substrings (case-insensitive). "
                         "E.g. --algos recursive qaoa")
    args = ap.parse_args()

    # Set global run timestamp
    run_stamp = set_global_run_stamp()
    
    print("\n" + "="*80)
    print("VRP BENCHMARK")
    print("="*80)
    print(f"Run ID: {run_stamp}")
    print("✓ Weighted Fairness metric: ((total_distance/k) + distance_std) / 2")
    print("✓ Recursive QAOA solver: uses Qiskit for small leaves (≤5 nodes)")
    print(f"✓ Results: results/{run_stamp}/")
    print("="*80 + "\n")

    all_names = list_array_instances()
    if args.instance:
        target = f"RioClaroPostToy_{args.instance}"
        if target not in all_names:
            sys.exit(f"Instance '{target}' not in arrays/. Available: {all_names}")
        names = [target]
    else:
        names = all_names

    algo_filter = [s.lower() for s in args.algos] if args.algos else None
    if algo_filter:
        active_algos = [(n, f) for n, f in ALGORITHMS
                        if any(s in n.lower() for s in algo_filter)]
        print(f"Running {len(active_algos)} algorithm(s) matching {args.algos}:")
        for n, _ in active_algos:
            print(f"  • {n}")
    else:
        active_algos = ALGORITHMS

    all_results: List[dict] = []
    instance_k_pairs: List[Tuple[str, int]] = []
    for name in names:
        k = args.k or read_vrp_k(name) or 7
        instance_k_pairs.append((name, k))
        all_results.extend(run_one_instance(name, k, active_algorithms=active_algos))

    # Create run manifest and update global index
    create_run_manifest(run_stamp, instance_k_pairs)
    update_runs_index(run_stamp, instance_k_pairs, all_results)
    
    out_path = ROOT / args.out
    write_comparison(all_results, out_path)
    print_winners(all_results)
    
    print(f"\n{'='*80}")
    print(f"✓ Run complete! Results at: results/{run_stamp}/")
    print(f"✓ See index: results/RUNS_INDEX.md")
    print(f"✓ Global CSV: {out_path}")
    print(f"{'='*80 + chr(10)}")


if __name__ == "__main__":
    main()
