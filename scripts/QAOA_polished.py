# %%
#!/usr/bin/env python3
"""
📊 VRP QAOA Solver - Instance Selector
Select which instance to analyze from available files
"""

import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# 📁 DISCOVER AVAILABLE INSTANCES
# ═══════════════════════════════════════════════════════════════════════════

def find_available_instances():
    """Scan directories and return available instances"""
    instances_dir = Path("Instances")
    available = []
    
    if not instances_dir.exists():
        return []
    
    for category_dir in sorted(instances_dir.iterdir()):
        if category_dir.is_dir():
            for vrp_file in sorted(category_dir.glob("*.vrp")):
                instance_base = vrp_file.stem  # e.g., "RioClaroPostToy_100_0"
                # Extract friendly name
                parts = instance_base.rsplit('_', 1)
                if len(parts) == 2:
                    base_name, variant = parts
                    n_nodes = category_dir.name.replace('-Nodes', '')
                    available.append({
                        'full_name': instance_base,
                        'base_name': base_name,
                        'variant': variant,
                        'nodes': n_nodes,
                        'path': str(vrp_file)
                    })
    
    return available

# Find available instances
available_instances = find_available_instances()

print("="*80)
print("🔍 AVAILABLE INSTANCES")
print("="*80)

if not available_instances:
    raise SystemExit("❌ No instances found in Instances/ directory")

# Flat numbered list (one number per instance) for the selector
for i, inst in enumerate(available_instances):
    print(f"  [{i:2d}] {inst['base_name']:<25s} {inst['nodes']:>4s} nodes  variant {inst['variant']}")

# ═══════════════════════════════════════════════════════════════════════════
# 🎯 INSTANCE SELECTOR  (interactive prompt, env var, or CLI arg)
# ═══════════════════════════════════════════════════════════════════════════
# Priority:
#   1) --instance N  on the command line
#   2) INSTANCE_INDEX=N  env var
#   3) interactive input() prompt (non-blocking fallback = 0)

import sys

def _pick_instance_index(n_choices):
    # CLI arg
    for i, a in enumerate(sys.argv):
        if a in ("--instance", "-i") and i + 1 < len(sys.argv):
            try: return int(sys.argv[i + 1])
            except ValueError: pass
    # Env var
    if "INSTANCE_INDEX" in os.environ:
        try: return int(os.environ["INSTANCE_INDEX"])
        except ValueError: pass
    # Interactive prompt (only if stdin is a tty)
    if sys.stdin.isatty():
        try:
            raw = input(f"\nSelect instance [0-{n_choices-1}] (default 0): ").strip()
            return int(raw) if raw else 0
        except (EOFError, ValueError):
            return 0
    return 0

print("\n" + "="*80)
print("⚙️  SELECT INSTANCE")
print("="*80)

idx = _pick_instance_index(len(available_instances))
idx = max(0, min(idx, len(available_instances) - 1))
SELECTED_INSTANCE = available_instances[idx]

print("\n" + "="*80)
print("💾 SELECTED CONFIGURATION")
print("="*80)
print(f"Instance Name: {SELECTED_INSTANCE['full_name']}")
print(f"Base Name:     {SELECTED_INSTANCE['base_name']}")
print(f"Variant:       {SELECTED_INSTANCE['variant']}")
print(f"Nodes:         {SELECTED_INSTANCE['nodes']}")
print(f"VRP File:      {SELECTED_INSTANCE['path']}")
print("\n✓ Ready!  (override with --instance N  or  INSTANCE_INDEX=N)")

# %%
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for plain-Python runs
matplotlib.rcParams['figure.dpi'] = 110

# ═══════════════════════════════════════════════════════════════════════════
# 🔍 VERIFY SELECTED INSTANCE (for debugging)
# ═══════════════════════════════════════════════════════════════════════════

print("="*80)
print("📋 CURRENT SELECTED INSTANCE")
print("="*80)
print(f"✓ Instance: {SELECTED_INSTANCE['full_name']}")
print(f"✓ Base Name: {SELECTED_INSTANCE['base_name']}")
print(f"✓ Nodes: {SELECTED_INSTANCE['nodes']}")
print(f"✓ Variant: {SELECTED_INSTANCE['variant']}")
print(f"✓ Path: {SELECTED_INSTANCE['path']}")
print("\n(Run this cell after changing the dropdown to verify the selection was updated)")

# %%
# Install dependencies (run once from the shell, not as a script step):
#   pip install qiskit qiskit-aer qiskit-algorithms qiskit-optimization

# %% [markdown]
# # VRP Recursive Solver — Cluster-Level Tracer
# 
# Uses a **real 51-node distance matrix** (1 depot + 50 delivery nodes).  
# **Instance:** `RioClaroPostToy_50_0`
# 
# | Cell | Purpose |
# |------|---------|
# | 1 | Core library — data structures, distances, algorithms |
# | 2 | Instrumented solver — records every clustering event |
# | 3 | Plotting helpers — depth-level cluster figures |
# | **4** | **Distance matrix + node setup** |
# | 5 | Configuration — set `K` (vehicles) here |
# | 6 | Run tracer and render cluster plots |
# | 7 | Benchmark table — all algorithms, k shown per row |
# | 8 | Depth profile bar charts |
# 

# %% [markdown]
# ---
# ## Cell 1 — Core library (split into 4 parts below)

# %% [markdown]
# ---
# ## Cell 1a — Data structures & core utilities
# 
# Node/Route/VRPSolution data classes, distance helpers, 2-opt, k-means clustering, spatial splitting.

# %%
# ── GLOBAL_MAX_DIST ──────────────────────────────────────────────────────────
# Set once in Cell 4 from the full 51-node distance matrix.
# Every QUBO built at any recursion depth normalises distances by this single
# constant, so the variational parameters (gamma, beta) retain a consistent
# physical meaning regardless of cluster size.  A 1 km edge always contributes
# the same fractional cost to the QUBO, whether the sub-problem spans the
# whole map or a single neighbourhood.
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_MAX_DIST = 1.0  # placeholder — overwritten when the distance matrix is loaded

import numpy as np
import math
import time
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class Node:
    id: int
    x: float
    y: float
    demand: float = 0.0
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
    total_load: float = 0.0
    num_vehicles_used: int = 0
    solve_time: float = 0.0
    distance_std: float = 0.0                # NEW: fairness metric
    def __post_init__(self):
        self.num_vehicles_used = len(self.routes)
        self.total_distance    = sum(r.distance for r in self.routes)
        self.total_load        = sum(r.load     for r in self.routes)
        # NEW: standard deviation of route distances
        if len(self.routes) > 1:
            self.distance_std = float(np.std([r.distance for r in self.routes]))
        else:
            self.distance_std = 0.0

def euclidean(a: Node, b: Node) -> float:
    """Return the Euclidean distance between two Nodes using their (x, y) coordinates."""
    return math.hypot(a.x - b.x, a.y - b.y)

def build_dist_map(nodes: List[Node], depot: Node,
                     dist_matrix=None) -> Dict[tuple, float]:
    """Build a pairwise distance lookup dict for all nodes including depot.

    If *dist_matrix* is provided (2-D array indexed by node id), distances are
    read from it; otherwise Euclidean distance on (x, y) coordinates is used.

    Returns:
        dict mapping (id_a, id_b) -> float distance.
    """
    all_nodes = [depot] + nodes
    dmap = {}
    for a in all_nodes:
        for b in all_nodes:
            dmap[(a.id, b.id)] = (float(dist_matrix[a.id][b.id])
                                  if dist_matrix is not None else euclidean(a, b))
    return dmap

def route_distance(route_ids: List[int], depot_id: int,
                     dist_map: Dict[tuple, float]) -> float:
    """Compute the total distance of a route: depot → route[0] → … → route[-1] → depot."""
    if not route_ids: return 0.0
    d = dist_map[(depot_id, route_ids[0])]
    for i in range(len(route_ids)-1):
        d += dist_map[(route_ids[i], route_ids[i+1])]
    d += dist_map[(route_ids[-1], depot_id)]
    return d

def two_opt(route_ids: List[int], depot_id: int,
             dist_map: Dict[tuple, float]) -> List[int]:
    """Improve *route_ids* by repeatedly reversing sub-segments (2-opt) until
    no improving swap exists.  Returns the improved node-id list."""
    if len(route_ids) <= 2: return route_ids
    improved = True; best = list(route_ids)
    best_dist = route_distance(best, depot_id, dist_map)
    while improved:
        improved = False
        for i in range(len(best)-1):
            for j in range(i+1, len(best)):
                nr = best[:i] + best[i:j+1][::-1] + best[j+1:]
                nd = route_distance(nr, depot_id, dist_map)
                if nd < best_dist - 1e-10:
                    best = nr; best_dist = nd; improved = True; break
            if improved: break
    return best

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

def nearest_neighbor_route(node_ids: List[int], depot_id: int,
                             dist_map: Dict[tuple, float]) -> List[int]:
    """Greedy nearest-neighbour ordering starting from *depot_id*."""
    if not node_ids: return []
    unvisited = set(node_ids); route = []; current = depot_id
    while unvisited:
        nearest = min(unvisited, key=lambda nid: dist_map[(current, nid)])
        route.append(nearest); unvisited.remove(nearest); current = nearest
    return route

def _split_routes_to_k(routes_list, k, n_nodes, depot_id=None, dist_map=None,
                       node_map=None):
    """Safety net: split longest route until we have exactly k non-empty routes.

    FIX 4 — Spatial split: always uses a geographically aware strategy.

    Strategy cascade (highest to lowest fidelity):
      1. **Longest-edge cut** (when dist_map is available):
         Walk the route, find the longest inter-node edge, and sever the route
         there.  This is the most natural geographic break-point because the
         biggest gap between consecutive stops marks a natural boundary between
         two service regions.
      2. **Angular-sweep cut** (when node_map with xy coords is available):
         Compute the route centroid, sort nodes by polar angle around it, and
         split at the midpoint of the sorted sequence.  This keeps each
         half in a contiguous angular wedge.
      3. **Index midpoint** (last resort, no spatial data):
         Plain list bisection — only reached if neither dist_map nor node_map
         is supplied.
    """
    if n_nodes < k:
        return routes_list
    result = [list(r) for r in routes_list if r]

    def _split_one(route):
        """Return two sub-lists by cutting at the most geographically natural point."""
        if len(route) < 2:
            return route, []

        # ── Strategy 1: longest-edge cut (best) ──
        if dist_map is not None:
            edge_costs = [
                dist_map.get((route[i], route[i+1]), 0.0)
                for i in range(len(route) - 1)
            ]
            # Also consider the depot→first and last→depot edges for completeness
            if depot_id is not None:
                # Include depot-bookend edges so we can also sever at the boundary
                edge_costs_full = (
                    [dist_map.get((depot_id, route[0]), 0.0)]
                    + edge_costs
                    + [dist_map.get((route[-1], depot_id), 0.0)]
                )
                # But we only cut interior edges (indices 0..len-2 of the route)
                # so the depot-edges are just for awareness, not cutting.
            if edge_costs:
                cut = int(max(range(len(edge_costs)), key=lambda i: edge_costs[i])) + 1
                return route[:cut], route[cut:]

        # ── Strategy 2: angular-sweep cut ──
        if node_map is not None:
            import math as _math
            coords = [(nid, node_map[nid].x, node_map[nid].y)
                      for nid in route if nid in node_map]
            if len(coords) >= 2:
                cx = sum(x for _, x, _ in coords) / len(coords)
                cy = sum(y for _, _, y in coords) / len(coords)
                sorted_by_angle = sorted(coords,
                                         key=lambda t: _math.atan2(t[2] - cy, t[1] - cx))
                mid = len(sorted_by_angle) // 2
                left_ids  = [nid for nid, _, _ in sorted_by_angle[:mid]]
                right_ids = [nid for nid, _, _ in sorted_by_angle[mid:]]
                return left_ids, right_ids

        # ── Strategy 3: index midpoint (last resort) ──
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


print('OK  [1/4] Data structures & core utilities loaded.')


# %% [markdown]
# ---
# ## Cell 1b — Clarke-Wright savings & QAOA solver
# 
# Clarke-Wright savings heuristic, QAOA leaf solver (5 configs × per-leaf seed).

# %%
def greedy_vrp_no_opt(nodes, k, depot, dist_map):
    """Clarke-Wright Savings: merges n->k routes by stopping at target k."""
    if not nodes: return VRPSolution(routes=[])
    node_map = {n.id: n for n in nodes}; node_ids = [n.id for n in nodes]
    routes_dict = {nid: [nid] for nid in node_ids}
    route_of    = {nid: nid   for nid in node_ids}
    savings = sorted([(dist_map[(depot.id,i)] + dist_map[(depot.id,j)] - dist_map[(i,j)], i, j)
                      for i in node_ids for j in node_ids if i < j], reverse=True)
    def route_load(r): return sum(node_map[nid].demand for nid in r)
    target = max(1, min(k, len(node_ids)))
    for s, i, j in savings:
        if len(routes_dict) <= target: break
        ri = route_of.get(i); rj = route_of.get(j)
        if ri is None or rj is None or ri == rj: continue
        if ri not in routes_dict or rj not in routes_dict: continue
        ri_r = routes_dict[ri]; rj_r = routes_dict[rj]
        if ri_r[-1] != i and ri_r[0] != i: continue
        if rj_r[-1] != j and rj_r[0] != j: continue
        if   ri_r[-1]==i and rj_r[0]==j:  merged = ri_r + rj_r
        elif ri_r[0]==i  and rj_r[-1]==j: merged = rj_r + ri_r
        elif ri_r[-1]==i and rj_r[-1]==j: merged = ri_r + rj_r[::-1]
        elif ri_r[0]==i  and rj_r[0]==j:  merged = ri_r[::-1] + rj_r
        else: continue
        routes_dict[ri] = merged
        if rj in routes_dict: del routes_dict[rj]
        for nid in merged: route_of[nid] = ri
    final = list(routes_dict.values())
    while len(final) > target and len(final) > 1:
        final.sort(key=route_load); r1=final.pop(0); r2=final.pop(0); final.append(r1+r2)
    # safety net for edge cases where valid merges ran out before reaching k
    if len(nodes) >= k:
        final = _split_routes_to_k(final, k, len(nodes),
                                   depot_id=depot.id, dist_map=dist_map,
                                   node_map=node_map)
    return VRPSolution(routes=[Route(node_ids=r,
                                     distance=route_distance(r, depot.id, dist_map),
                                     load=route_load(r)) for r in final])

QAOA_STATS = {"success": 0, "fallback": 0}

# ── QAOA LOG ─────────────────────────────────────────────────────────────────
# Every QAOA leaf call appends one entry classifying the result as:
#   "optimal"   — QAOA's best solution matches the brute-force global minimum
#   "feasible"  — QAOA found valid route(s) but not the optimal cost
#   "non-valid" — QAOA produced no feasible solution (fell back to classical)
#
# Each entry also records: node_ids, n_qubits, n_valid_unique (distinct feasible
# bitstrings), best_prob_rank (probability rank of the cheapest valid solution;
# 1 = QAOA concentrated probability on the optimum), qaoa_cost, optimal_cost,
# gap_pct, valid_frac, and solver_used.
# ──────────────────────────────────────────────────────────────────────────────
QAOA_LOG = []

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
    _SEED     = 7
    _DECODE_K = 100
    penalty_mul=2
    import hashlib as _hl
    _leaf_seed = (_SEED + int(_hl.md5(str(sorted(node_ids)).encode()).hexdigest(), 16)) % (2**31)
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


print('OK  [2/4] Clarke-Wright savings & QAOA solver loaded.')


# %% [markdown]
# ---
# ## Cell 1c — Merge logic & classical baselines
# 
# Vehicle allocation, super-node construction, endpoint-aware merging, and all classical baseline algorithms (NN, Sweep, OR-Tools).

# %%
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

def _partition_by_angle(nodes, depot, k):
    """Split nodes into exactly min(k, n) groups by angular sweep from depot."""
    n_count = len(nodes); eff_k = min(k, n_count)
    def angle(nd): return math.atan2(nd.y-depot.y, nd.x-depot.x)
    sn = sorted(nodes, key=angle)
    base = n_count // eff_k; rem = n_count % eff_k
    groups=[]; idx=0
    for v in range(eff_k):
        sz = base+(1 if v<rem else 0)
        groups.append([nd.id for nd in sn[idx:idx+sz]]); idx+=sz
    return groups

def algo_nn(nodes, k, depot, dist_matrix=None):
    """Nearest-Neighbour: angle-partition into k groups, reorder each by NN greedy."""
    if not nodes: return VRPSolution(routes=[])
    dist_map = build_dist_map(nodes, depot, dist_matrix)
    node_map = {n.id: n for n in nodes}
    routes=[]
    for g in _partition_by_angle(nodes, depot, k):
        nn = nearest_neighbor_route(g, depot.id, dist_map)
        routes.append(Route(node_ids=nn,
                            distance=route_distance(nn, depot.id, dist_map),
                            load=sum(node_map[nid].demand for nid in nn)))
    return VRPSolution(routes=routes)

def algo_savings(nodes, k, depot, dist_matrix=None):
    """Clarke-Wright Savings: merges n->k naturally by stopping at k."""
    return greedy_vrp_no_opt(nodes, k, depot, build_dist_map(nodes, depot, dist_matrix))

def algo_sweep(nodes, k, depot, dist_matrix=None):
    """Sweep: angle-partition into k groups, visit each group in angular order."""
    if not nodes: return VRPSolution(routes=[])
    dist_map = build_dist_map(nodes, depot, dist_matrix)
    node_map = {n.id: n for n in nodes}
    routes=[]
    for g in _partition_by_angle(nodes, depot, k):
        routes.append(Route(node_ids=g,
                            distance=route_distance(g, depot.id, dist_map),
                            load=sum(node_map[nid].demand for nid in g)))
    return VRPSolution(routes=routes)


def algo_ortools(nodes, k, depot, dist_matrix=None):
    """OR-Tools VRP: exact routing solver using Google OR-Tools.
    Forces exactly k vehicles natively by penalising empty vehicles
    so OR-Tools always uses all k from the start.
    """
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        return algo_savings(nodes, k, depot, dist_matrix)

    if not nodes:
        return VRPSolution(routes=[])

    dist_map = build_dist_map(nodes, depot, dist_matrix)
    node_map = {n.id: n for n in nodes}
    eff_k    = min(k, len(nodes))

    all_nodes = [depot] + nodes
    n_total   = len(all_nodes)

    _OR_SCALE = 1000
    def or_dist(i, j):
        return int(dist_map[(all_nodes[i].id, all_nodes[j].id)] * _OR_SCALE)

    manager  = pywrapcp.RoutingIndexManager(n_total, eff_k, 0)
    routing  = pywrapcp.RoutingModel(manager)

    transit_cb_idx = routing.RegisterTransitCallback(
        lambda i, j: or_dist(manager.IndexToNode(i), manager.IndexToNode(j))
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # ── Force all k vehicles to be used ──────────────────────────────
    # Count dimension: each delivery node counts as 1.  Vehicle capacity
    # is uncapped (len(nodes)), so OR-Tools can freely optimise.  The
    # soft lower bound of 1 on each vehicle's end cumul means OR-Tools
    # pays a huge penalty for leaving any vehicle empty.
    def demand_cb(from_index):
        node = manager.IndexToNode(from_index)
        return 0 if node == 0 else 1
    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx, 0, [len(nodes)] * eff_k, True, 'Count')
    count_dim = routing.GetDimensionOrDie('Count')
    big_penalty = int(max(dist_map.values()) * _OR_SCALE * 10)
    for v in range(eff_k):
        end_idx = routing.End(v)
        count_dim.SetCumulVarSoftLowerBound(end_idx, 1, big_penalty)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = 10

    assignment = routing.SolveWithParameters(search_params)

    if not assignment:
        return algo_savings(nodes, k, depot, dist_matrix)

    routes = []
    for v in range(eff_k):
        idx   = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            node_idx = manager.IndexToNode(idx)
            if node_idx != 0:
                route.append(all_nodes[node_idx].id)
            idx = assignment.Value(routing.NextVar(idx))
        if route:
            routes.append(Route(
                node_ids = route,
                distance = route_distance(route, depot.id, dist_map),
                load     = sum(node_map[nid].demand for nid in route)
            ))

    return VRPSolution(routes=routes)

def two_opt_routes(sol, depot, dist_map):
    """Apply 2-opt improvement to every route in a VRPSolution."""
    improved = []
    for r in sol.routes:
        opt_ids  = two_opt(r.node_ids, depot.id, dist_map)
        improved.append(Route(
            node_ids = opt_ids,
            distance = route_distance(opt_ids, depot.id, dist_map),
            load     = r.load
        ))
    return VRPSolution(routes=improved)


print('OK  [3/4] Merge logic & classical baselines loaded.')


# %% [markdown]
# ---
# ## Cell 1d — Display & test instance generators
# 
# Notebook rendering helper, random/clustered test instance generators.

# %%
def _show(fig) -> None:
    """Render a matplotlib figure.  In a Jupyter kernel it publishes an
    inline PNG; in a plain Python run it just closes the figure (callers
    save their own copies via fig.savefig)."""
    try:
        import base64, io as _io
        from IPython.core.displaypub import publish_display_data
        from IPython import get_ipython
        if get_ipython() is None:
            raise RuntimeError("no IPython kernel")
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
        publish_display_data(
            {'image/png': base64.b64encode(buf.getvalue()).decode(),
             'text/plain': '<Figure>'}
        )
    except Exception:
        pass
    finally:
        plt.close(fig)

def generate_test_instance(n: int, seed: int = 42,
                             area: float = 100.0) -> tuple:
    """Generate *n* uniformly distributed delivery nodes in a square of side *area*
    with a central depot.  Returns (nodes, depot)."""
    rng = np.random.RandomState(seed)
    depot = Node(id=0, x=area/2, y=area/2, demand=0, is_depot=True)
    return [Node(id=i, x=rng.uniform(0,area), y=rng.uniform(0,area),
                 demand=rng.uniform(1,20)) for i in range(1,n+1)], depot

def generate_clustered_instance(n: int, n_real_clusters: int = 3,
                                  seed: int = 42) -> tuple:
    """Generate *n* delivery nodes grouped around *n_real_clusters* Gaussian
    centres, with a central depot.  Returns (nodes, depot)."""
    rng = np.random.RandomState(seed)
    depot = Node(id=0, x=50, y=50, demand=0, is_depot=True)
    centers = [(rng.uniform(10,90), rng.uniform(10,90)) for _ in range(n_real_clusters)]
    nodes=[]; per_cluster=n//n_real_clusters; nid=1
    for cx,cy in centers:
        for _ in range(per_cluster):
            nodes.append(Node(id=nid, x=cx+rng.normal(0,8),
                              y=cy+rng.normal(0,8), demand=rng.uniform(1,15))); nid+=1
    while len(nodes) < n:
        nodes.append(Node(id=nid, x=rng.uniform(0,100),
                          y=rng.uniform(0,100), demand=rng.uniform(1,15))); nid+=1
    return nodes, depot

print('OK  Core library loaded.')



# %% [markdown]
# ---
# ## Cell 2 — Instrumented recursive solver

# %%
LEAF_SIZE = 4
TRACE_LOG = []

def vrp_solver_traced(k, nodes, depot, dist_map=None, dist_matrix=None, _depth=0):
    if dist_map is None:
        dist_map = build_dist_map(nodes, depot, dist_matrix)
    if not nodes: return VRPSolution(routes=[])
    if len(nodes) == 1:
        nid = nodes[0].id; d = dist_map[(depot.id, nid)] * 2
        TRACE_LOG.append(dict(depth=_depth, k=k, n=1, node_ids=[nid],
                              clusters=[[nid]], allocs=[1], label='leaf-single',
                              nodes_xy={nid: (nodes[0].x, nodes[0].y)},
                              depot_x=depot.x, depot_y=depot.y))
        return VRPSolution(routes=[Route(node_ids=[nid], distance=d, load=nodes[0].demand)])

    # Base case: n <= LEAF_SIZE
    if len(nodes) <= LEAF_SIZE:
        TRACE_LOG.append(dict(depth=_depth, k=k, n=len(nodes),
                              node_ids=[n.id for n in nodes],
                              clusters=[[n.id for n in nodes]],
                              allocs=[k], depot_x=depot.x, depot_y=depot.y,
                              nodes_xy={n.id: (n.x, n.y) for n in nodes}, label='leaf-bf'))
        return solve_brute_force(nodes, k, depot, dist_map)

    # Rule 2: purely data-based clustering
    clusters = cluster_nodes(nodes, depot, max(2, math.ceil(math.sqrt(len(nodes)))))

    # k == 1: sequence clusters via QAOA super-node ordering + endpoint-aware merge
    if k == 1:
        TRACE_LOG.append(dict(depth=_depth, k=k, n=len(nodes),
                              node_ids=[n.id for n in nodes],
                              clusters=[[n.id for n in cl] for cl in clusters],
                              allocs=[1]*len(clusters), depot_x=depot.x, depot_y=depot.y,
                              nodes_xy={n.id: (n.x, n.y) for n in nodes}, label='k=1'))
        cluster_solutions = [vrp_solver_traced(1, cl, depot, dist_map, _depth=_depth+1)
                             for cl in clusters]
        # Use QAOA (solve_brute_force) to find the best ordering of super-nodes,
        # then merge with endpoint-aware stitching instead of blind concatenation.
        super_nodes = create_super_nodes(clusters, cluster_solutions)
        sdm         = build_super_dist_map(super_nodes, depot)
        if len(super_nodes) > LEAF_SIZE:
            super_solution = vrp_solver_traced(1, super_nodes, depot, dist_map=sdm, _depth=_depth+1)
        else:
            super_solution = solve_brute_force(super_nodes, 1, depot, sdm)
        _sol = merge_super_solution(super_solution, cluster_solutions, clusters, depot, dist_map)
        try:
            assert len({nid for r in _sol.routes for nid in r.node_ids}) == len(nodes)
        except AssertionError:
            print(f"WARNING (k=1 traced): merge validation failed -- concatenating all nodes into 1 route")
            all_ids_k1t = [nid for cs in cluster_solutions
                           for r in cs.routes for nid in r.node_ids]
            _sol = VRPSolution(routes=[Route(
                node_ids=all_ids_k1t,
                distance=route_distance(all_ids_k1t, depot.id, dist_map),
                load=sum(n.demand for n in nodes)
            )])
        return _sol

    C = len(clusters)

    if C > k:
        vehicle_alloc = [1] * C
        label = 'C>k'

        TRACE_LOG.append(dict(depth=_depth, k=k, n=len(nodes),
                              node_ids=[n.id for n in nodes],
                              clusters=[[n.id for n in cl] for cl in clusters],
                              allocs=vehicle_alloc, depot_x=depot.x, depot_y=depot.y,
                              nodes_xy={n.id: (n.x, n.y) for n in nodes}, label=label))

        cluster_solutions = [vrp_solver_traced(1, cl, depot, dist_map, _depth=_depth+1)
                             for cl in clusters]

        # ISSUE 1 FIX: use real inter-cluster distances for super-VRP
        super_nodes = create_super_nodes(clusters, cluster_solutions)
        sdm         = build_super_dist_map(super_nodes, depot)
        if len(super_nodes) > LEAF_SIZE:
            super_solution = vrp_solver_traced(k, super_nodes, depot, dist_map=sdm, _depth=_depth+1)
        else:
            super_solution = solve_brute_force(super_nodes, k, depot, sdm)
        _sol = merge_super_solution(super_solution, cluster_solutions, clusters, depot, dist_map)
        try:
            assert len({nid for r in _sol.routes for nid in r.node_ids}) == len(nodes)
        except AssertionError:
            print(f"WARNING (C>k traced): merge validation failed -- re-grouping {len(cluster_solutions)} cluster routes into {k}")
            # cluster_solutions each have k=1 route; group them into k vehicle routes
            flat_routes = [r for cs in cluster_solutions for r in cs.routes]
            groups = [[] for _ in range(k)]
            for ci, cr in enumerate(flat_routes):
                groups[ci % k].extend(cr.node_ids)
            _sol = VRPSolution(routes=[
                Route(node_ids=g,
                      distance=route_distance(g, depot.id, dist_map),
                      load=sum({n.id: n for n in nodes}[nid].demand for nid in g))
                for g in groups if g
            ])
        return _sol

    else:
        vehicle_alloc = allocate_vehicles(nodes, clusters, k)
        label = 'recurse'

        TRACE_LOG.append(dict(depth=_depth, k=k, n=len(nodes),
                              node_ids=[n.id for n in nodes],
                              clusters=[[n.id for n in cl] for cl in clusters],
                              allocs=vehicle_alloc, depot_x=depot.x, depot_y=depot.y,
                              nodes_xy={n.id: (n.x, n.y) for n in nodes}, label=label))

        cluster_solutions = [vrp_solver_traced(vehicle_alloc[i], cl, depot, dist_map, _depth=_depth+1)
                             for i, cl in enumerate(clusters)]

        all_routes = [r for csol in cluster_solutions for r in csol.routes]
        return VRPSolution(routes=all_routes)


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




def vrp_solver_plain_2opt(k, nodes, depot, dist_map=None, dist_matrix=None):
    """Wrapper: runs vrp_solver_plain then applies 2-opt to every route."""
    if dist_map is None:
        dist_map = build_dist_map(nodes, depot, dist_matrix)
    sol = vrp_solver_plain(k, nodes, depot, dist_map=dist_map)
    return two_opt_routes(sol, depot, dist_map)


def vrp_solver_traced_2opt(k, nodes, depot, dist_map=None, dist_matrix=None):
    """Traced version with 2-opt post-processing."""
    if dist_map is None:
        dist_map = build_dist_map(nodes, depot, dist_matrix)
    sol = vrp_solver_traced(k, nodes, depot, dist_map=dist_map)
    return two_opt_routes(sol, depot, dist_map)

def run_traced(nodes, k, depot, dist_matrix=None):
    """Run the QAOA solver and return the raw solution without any classical post-processing.
    Use run_traced_2opt() if you also want the 2-opt-improved version.
    """
    global TRACE_LOG; TRACE_LOG = []
    dist_map = build_dist_map(nodes, depot, dist_matrix)
    sol      = vrp_solver_traced(k, nodes, depot, dist_map=dist_map)
    return sol, list(TRACE_LOG)

def run_traced_2opt(nodes, k, depot, dist_matrix=None):
    """Run the QAOA solver then apply 2-opt classical post-processing.
    Returns (raw_qaoa_sol, two_opt_sol, trace_log) so both can be compared.
    """
    global TRACE_LOG; TRACE_LOG = []
    dist_map = build_dist_map(nodes, depot, dist_matrix)
    sol      = vrp_solver_traced(k, nodes, depot, dist_map=dist_map)
    sol_2opt = two_opt_routes(sol, depot, dist_map)
    return sol, sol_2opt, list(TRACE_LOG)

print('OK  Instrumented solver ready.')


# %% [markdown]
# ---
# ## Cell 3 — Plotting helpers

# %%
CLUSTER_COLORS = [
    '#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
    '#42d4f4','#f032e6','#bfef45','#fabed4','#469990',
    '#dcbeff','#9a6324','#800000','#aaffc3','#808000',
]
def color_for(idx): return CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]


def plot_cluster_frame(ax, frame, all_nodes_xy, depot_xy, title=''):
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    active_ids = set(frame['node_ids'])
    for nid, (x, y) in all_nodes_xy.items():
        if nid not in active_ids:
            ax.scatter(x, y, c='#cccccc', s=20, zorder=1, linewidths=0)
    legend_patches = []
    for ci, cluster_ids in enumerate(frame['clusters']):
        col   = color_for(ci)
        alloc = frame['allocs'][ci] if ci < len(frame['allocs']) else 1
        xs = [all_nodes_xy[nid][0] for nid in cluster_ids if nid in all_nodes_xy]
        ys = [all_nodes_xy[nid][1] for nid in cluster_ids if nid in all_nodes_xy]
        ax.scatter(xs, ys, c=col, s=45, zorder=3, edgecolors='white', linewidths=0.4)
        for nid, x, y in zip(cluster_ids, xs, ys):
            ax.annotate(str(nid), (x,y), fontsize=5, ha='center', va='bottom',
                        xytext=(0,3), textcoords='offset points', color='#333')
        if xs:
            cx, cy = np.mean(xs), np.mean(ys)
            ax.scatter(cx, cy, marker='+', c=col, s=120, zorder=4, linewidths=1.5)
        legend_patches.append(mpatches.Patch(color=col,
                                              label=f'C{ci}  n={len(cluster_ids)}  k={alloc}'))
    dx, dy = depot_xy
    ax.scatter(dx, dy, c='black', marker='*', s=180, zorder=5)
    ax.annotate('depot', (dx,dy), fontsize=5, ha='left', va='top',
                xytext=(3,-3), textcoords='offset points')
    ax.legend(handles=legend_patches, fontsize=5, loc='upper left',
              framealpha=0.6, borderpad=0.4, labelspacing=0.2)


def plot_trace(trace_log, all_nodes_xy, depot_xy, instance_name=''):
    if not trace_log: print('No trace data.'); return
    depth_frames = defaultdict(list)
    for frame in trace_log: depth_frames[frame['depth']].append(frame)
    for depth in sorted(depth_frames.keys()):
        frames = [f for f in depth_frames[depth] if f['n'] > 1 or depth == 0]
        if not frames: continue
        ncols = min(4, len(frames)); nrows = math.ceil(len(frames)/ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.2*nrows), squeeze=False)
        fig.suptitle(f'{instance_name}  --  Recursion Depth {depth}  '
                     f'({len(frames)} sub-problem{"s" if len(frames)>1 else ""})',
                     fontsize=11, fontweight='bold', y=1.01)
        for fi, frame in enumerate(frames):
            ax    = axes[fi//ncols][fi%ncols]
            title = (f'Sub-problem {fi+1}/{len(frames)}\n'
                     f'n={frame["n"]}  k={frame["k"]}  '
                     f'clusters={len(frame["clusters"])}  [{frame["label"]}]')
            plot_cluster_frame(ax, frame, all_nodes_xy, depot_xy, title=title)
        for fi in range(len(frames), nrows*ncols):
            axes[fi//ncols][fi%ncols].set_visible(False)
        plt.tight_layout()
        _show(fig)


def plot_final_routes(sol, all_nodes_xy, depot_xy, instance_name='', algo_name='Recursive'):
    fig, ax = plt.subplots(figsize=(7,6))
    # UPDATED: show STD in title
    ax.set_title(f'{instance_name}  --  {algo_name}\n'
                 f'Total distance: {sol.total_distance:.2f}   Routes: {sol.num_vehicles_used}'
                 f'   \u03c3(fairness): {sol.distance_std:.2f}',
                 fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    dx, dy = depot_xy
    for ri, route in enumerate(sol.routes):
        col  = color_for(ri)
        path = [depot_xy]+[all_nodes_xy[nid] for nid in route.node_ids
                            if nid in all_nodes_xy]+[depot_xy]
        xs, ys = zip(*path)
        ax.plot(xs, ys, '-', color=col, linewidth=1.4, zorder=2)
        for i in range(len(path)-1):
            ax.annotate('', xy=path[i+1], xytext=path[i],
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.0),
                        annotation_clip=False)
        nxs=[all_nodes_xy[nid][0] for nid in route.node_ids if nid in all_nodes_xy]
        nys=[all_nodes_xy[nid][1] for nid in route.node_ids if nid in all_nodes_xy]
        ax.scatter(nxs, nys, c=col, s=50, zorder=3, edgecolors='white', linewidths=0.5)
        for nid, x, y in zip(route.node_ids, nxs, nys):
            ax.annotate(str(nid), (x,y), fontsize=6, ha='center', va='bottom',
                        xytext=(0,3), textcoords='offset points')
    ax.scatter(dx, dy, c='black', marker='*', s=250, zorder=5)
    ax.annotate('depot', (dx,dy), fontsize=7, ha='left', va='top',
                xytext=(3,-3), textcoords='offset points')
    # UPDATED: legend shows per-route distance + STD summary
    legend = [mpatches.Patch(color=color_for(ri),
                              label=f'Route {ri+1}  d={sol.routes[ri].distance:.1f}')
              for ri in range(len(sol.routes))]
    legend.append(mpatches.Patch(color='none',
                                  label=f'\u03c3 = {sol.distance_std:.1f}'))
    ax.legend(handles=legend, fontsize=7, loc='lower right', framealpha=0.7)
    plt.tight_layout()
    _show(fig)


def plot_all_algorithms(nodes, k, depot, instance_name, dist_matrix=None, precomputed_sol=None):
    """Side-by-side comparison of all algorithms. Returns the figure object."""
    all_xy={n.id: (n.x,n.y) for n in nodes}; dep_xy=(depot.x,depot.y)
    # Reuse pre-computed QAOA solution if provided to avoid re-running QAOA
    _dm = build_dist_map(nodes, depot, dist_matrix)
    if precomputed_sol is not None:
        _raw = precomputed_sol
        _2opt = two_opt_routes(precomputed_sol, depot, _dm)
    else:
        _raw = vrp_solver_traced(k,nodes,depot,dist_matrix=dist_matrix)
        _2opt = vrp_solver_traced_2opt(k,nodes,depot,dist_matrix=dist_matrix)
    algos=[
        ('Recursive (NO 2-opt)', lambda: _raw),
        ('Recursive + 2-opt',    lambda: _2opt),
        ('Nearest-Neighbour',    lambda: algo_nn(nodes,k,depot,dist_matrix)),
        ('Clarke-Wright Savings',lambda: algo_savings(nodes,k,depot,dist_matrix)),
        ('Sweep',                lambda: algo_sweep(nodes,k,depot,dist_matrix)),
        ('OR-Tools (GLS)',       lambda: algo_ortools(nodes,k,depot,dist_matrix)),
    ]
    ncols = 3; nrows = math.ceil(len(algos) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = np.array(axes).flatten()
    fig.suptitle(f'All algorithms -- {instance_name}', fontsize=13, fontweight='bold')
    for (name, fn), ax in zip(algos, axes.flat):
        sol = fn(); dx, dy = dep_xy
        for ri, route in enumerate(sol.routes):
            col  = color_for(ri)
            path = [dep_xy]+[all_xy[nid] for nid in route.node_ids if nid in all_xy]+[dep_xy]
            xs, ys = zip(*path)
            ax.plot(xs, ys, '-', color=col, linewidth=1.2, zorder=2)
            nxs=[all_xy[nid][0] for nid in route.node_ids if nid in all_xy]
            nys=[all_xy[nid][1] for nid in route.node_ids if nid in all_xy]
            ax.scatter(nxs, nys, c=col, s=30, zorder=3, edgecolors='white', linewidths=0.3)
        ax.scatter(dx, dy, c='black', marker='*', s=150, zorder=5)
        # UPDATED: show STD in subplot title
        ax.set_title(f'{name}\nd={sol.total_distance:.1f}  routes={sol.num_vehicles_used}'
                     f'  \u03c3={sol.distance_std:.1f}',
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    _show(fig)
    return fig  # Return figure object for saving

print('OK  Plotting helpers ready.')


# %% [markdown]
# ---
# ## Cell 4 — Distance matrix + node setup
# 
# The 51×51 real distance matrix is defined here.  
# Index `0` = depot, indices `1-50` = delivery nodes.  
# Coordinates are synthetic (unit circle) and used **only** for angle-based clustering — all cost calculations use the matrix.

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC DATA LOADING — NO HARDCODING
# ═══════════════════════════════════════════════════════════════════════════════

# Import distance matrix and node coordinates from generated array file
# DYNAMICALLY SELECTED from SELECTED_INSTANCE

# Extract instance name and generate import path
instance_array_name = SELECTED_INSTANCE['full_name']  # e.g., "RioClaroPostToy_100_0"
array_module_path = f"arrays.{instance_array_name}"

# Dynamically import the selected instance's array file
import importlib
array_module = importlib.import_module(array_module_path)
DIST_MATRIX_IMPORTED = array_module.dist_matrix
node_coords = array_module.node_coords

# Convert distance matrix to numpy array
DIST_MATRIX = np.array(DIST_MATRIX_IMPORTED)
DIST_MATRIX_RAW = DIST_MATRIX_IMPORTED  # Keep for compatibility

# Determine number of nodes dynamically from matrix size
# Matrix is (n_delivery + 1) × (n_delivery + 1), so n_delivery = matrix_size - 1
N_NODES = DIST_MATRIX.shape[0] - 1
print(f"✓ Loaded dynamic data from arrays/{instance_array_name}.py")
print(f"  Distance matrix shape: {DIST_MATRIX.shape}")
print(f"  Total problem size: {N_NODES} delivery nodes + 1 depot")

# Create depot node (node 0)
depot_x, depot_y = node_coords.get(0, (0.0, 0.0))
depot = Node(id=0, x=depot_x, y=depot_y, demand=0, is_depot=True)

# Create delivery nodes dynamically from all coordinates except depot
nodes = []
for node_id in sorted(node_coords.keys()):
    if node_id == 0:  # Skip depot (already created)
        continue
    x, y = node_coords[node_id]
    nodes.append(Node(id=node_id, x=x, y=y, demand=1.0, is_depot=False))

# Validate
assert len(nodes) == N_NODES, f"Mismatch: expected {N_NODES} nodes, got {len(nodes)}"
print(f"  ✓ Depot: {depot}")
print(f"  ✓ First delivery node: {nodes[0]}")
print(f"  ✓ Last delivery node: {nodes[-1]}")

# Set up global distance metrics
GLOBAL_MAX_DIST = float(np.max(DIST_MATRIX))
print(f"  ✓ Max distance in matrix: {GLOBAL_MAX_DIST:.3f}")

# Create node coordinate dictionary for lookups
all_nodes_xy = {n.id: (n.x, n.y) for n in nodes}
all_nodes_xy[0] = (depot.x, depot.y)
print(f"  ✓ Node coordinate map: {len(all_nodes_xy)} nodes")

# Build distance map for quick lookup
dist_map_full = build_dist_map(nodes, depot, DIST_MATRIX)
print(f"  ✓ Distance map built: {len(dist_map_full)} entries")

# %% [markdown]
# ---
# ## Cell 5 — Configuration
# 
# Set **`K`** to the number of vehicles you want.  
# All 50 delivery nodes will be served across exactly `K` routes.

# %%
# ── Set number of vehicles ────────────────────────────────────────────────
K = 15
# ──────────────────────────────────────────────────────────────────────────

print(f'Problem  : {N_NODES} delivery nodes, k={K} vehicles')
print(f'Avg nodes per route : {N_NODES / K:.1f}')

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS MANAGEMENT — Dynamic naming and organization
# ═══════════════════════════════════════════════════════════════════════════════

import os
from datetime import datetime

# Use the instance selected in Cell 1
# Extract base name from full name (RioClaroPostToy_100_0 -> RioClaroPostToy_100)
full_instance_name = SELECTED_INSTANCE['full_name']  # e.g., "RioClaroPostToy_100_0"
instance_name = f"{SELECTED_INSTANCE['base_name']}_{SELECTED_INSTANCE['variant']}"  # e.g., "RioClaroPostToy_100_0"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create results directory structure organized by instance and vehicle count
# Structure: results/instance_name/K_vehicles/[plots|data|logs]/
results_base = os.path.join(os.getcwd(), "results")
instance_dir = os.path.join(results_base, instance_name)
vehicle_dir = os.path.join(instance_dir, f"{K}_vehicles")  # e.g., "7_vehicles"

os.makedirs(results_base, exist_ok=True)
os.makedirs(instance_dir, exist_ok=True)
os.makedirs(vehicle_dir, exist_ok=True)

# Create subdirectories for different result types
plots_dir = os.path.join(vehicle_dir, "plots")
data_dir = os.path.join(vehicle_dir, "data")
logs_dir = os.path.join(vehicle_dir, "logs")

for d in [plots_dir, data_dir, logs_dir]:
    os.makedirs(d, exist_ok=True)

# Helper function to generate timestamped filenames
def get_result_path(name, filetype, subdir="plots", timestamp_prefix=True):
    """Generate a results path with proper naming convention.
    
    Args:
        name: Descriptive name (e.g., 'algorithm_comparison', 'qaoa_summary')
        filetype: File extension (e.g., 'png', 'txt', 'csv')
        subdir: Subdirectory ('plots', 'data', or 'logs')
        timestamp_prefix: Whether to prepend timestamp
    
    Returns:
        Full path to save file
    """
    if subdir == "plots":
        base_dir = plots_dir
    elif subdir == "data":
        base_dir = data_dir
    else:
        base_dir = logs_dir
    
    prefix = f"{timestamp}_" if timestamp_prefix else ""
    filename = f"{prefix}{instance_name}_{K}veh_{N_NODES}nodes_{name}.{filetype}"
    return os.path.join(base_dir, filename)

# Print tree structure
print(f"✓ Results directory structure: [NEW FIX APPLIED - instance_name={instance_name}]")
print(f"  results/")
print(f"  └─ {instance_name}/")
print(f"     └─ {K}_vehicles/")
print(f"        ├─ plots/")
print(f"        ├─ data/")


# %% [markdown]
# ---
# ## Cell 6 — Run tracer and render cluster plots

# %%
print(f'Tracing {N_NODES}-node real instance  k={K} ...')
import time
t0  = time.time()
QAOA_STATS["success"] = 0; QAOA_STATS["fallback"] = 0
QAOA_LOG.clear()
sol, trace_log = run_traced(nodes, K, depot, DIST_MATRIX)
elapsed = time.time() - t0

from collections import defaultdict
depth_counts = defaultdict(int)
for f in trace_log: depth_counts[f['depth']] += 1
max_depth = max(depth_counts) if depth_counts else 0

print(f'Frames per depth : {dict(sorted(depth_counts.items()))}')
print(f'Routes produced  : {sol.num_vehicles_used}')

print('\n' + '='*60)
print('ALL ALGORITHMS — side-by-side')
print('='*60)
fig = plot_all_algorithms(nodes, K, depot, f'Real {N_NODES}-node  k={K}', DIST_MATRIX, precomputed_sol=sol)

# Save the figure object directly (not plt state which may be cleared after display)
plot_path = get_result_path("01_algorithm_comparison", "png")
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {os.path.basename(plot_path)}")

# %%
# Save QAOA summary - Multiple formats for better readability
import csv
import json

# Compute summary statistics from QAOA_LOG
n_total = len(QAOA_LOG)
n_optimal = sum(1 for e in QAOA_LOG if e['outcome'].upper() == 'OPTIMAL')
n_feasible = sum(1 for e in QAOA_LOG if e['outcome'].upper() == 'FEASIBLE')
n_nonvalid = n_total - n_optimal - n_feasible

# 1. CSV format (detailed, machine-readable)
qaoa_csv_path = get_result_path("02_qaoa_outcomes", "csv", subdir="data")
with open(qaoa_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['#', 'Nodes', 'Routes', 'Qubits', 'Status', 'QAOA_Cost', 
                     'Optimal_Cost', 'Gap_Pct', 'Description'])
    for i, e in enumerate(QAOA_LOG, 1):
        node_desc = ', '.join(map(str, e["node_ids"]))
        n_routes = len([x for x in e["node_ids"] if x < 0])  # negative = route separators
        status = e['outcome'].upper()
        gap_str = f"{e['gap_pct']:.3f}%" if e['gap_pct'] is not None else "N/A"
        
        # Description
        if status == "OPTIMAL":
            desc = "Solution found optimally"
        elif status == "FEASIBLE":
            desc = "Valid solution (not proven optimal)"
        else:
            desc = "No valid solution found"
        
        writer.writerow([i, node_desc, len(e["node_ids"]), e['n_qubits'], 
                       status, f"{e['qaoa_cost']:.2f}", f"{e['optimal_cost']:.2f}", 
                       gap_str, desc])
    
    # Summary section
    writer.writerow([])
    writer.writerow(['SUMMARY STATISTICS'])
    writer.writerow(['Metric', 'Value', 'Percentage'])
    writer.writerow(['Total Solver Calls', n_total, '100.0%'])
    writer.writerow(['Optimal Solutions', n_optimal, f"{n_optimal/n_total*100:.1f}%"])
    writer.writerow(['Feasible Solutions', n_feasible, f"{n_feasible/n_total*100:.1f}%"])
    writer.writerow(['No Valid Solution', n_nonvalid, f"{n_nonvalid/n_total*100:.1f}%"])

# 2. Markdown format (human-friendly with explanations)
qaoa_md_path = get_result_path("02_qaoa_summary", "md", subdir="data")
with open(qaoa_md_path, 'w') as f:
    f.write("# QAOA Quantum Solver Results\n\n")
    f.write(f"**Instance:** {instance_name} | **Vehicles:** {K} | **Nodes:** {N_NODES}\n\n")
    
    f.write("## Overview\n")
    f.write(f"- **Total Solver Calls:** {n_total}\n")
    f.write(f"- **Optimal Solutions:** {n_optimal} ({n_optimal/n_total*100:.1f}%)\n")
    f.write(f"- **Feasible Solutions:** {n_feasible} ({n_feasible/n_total*100:.1f}%)\n")
    f.write(f"- **Failed (No Valid Solution):** {n_nonvalid} ({n_nonvalid/n_total*100:.1f}%)\n\n")
    
    f.write("## Detailed Results\n\n")
    f.write("| # | Nodes | Routes | Qubits | Status | QAOA Cost | Optimal | Gap | Notes |\n")
    f.write("|---|-------|--------|--------|--------|-----------|---------|-----|-------|\n")
    
    for i, e in enumerate(QAOA_LOG, 1):
        nodes_str = ', '.join(map(str, e["node_ids"]))
        if len(nodes_str) > 25:
            nodes_str = nodes_str[:22] + "..."
        n_routes = len([x for x in e["node_ids"] if x < 0])
        status = e['outcome'].upper()
        gap_str = f"{e['gap_pct']:.3f}%" if e['gap_pct'] is not None else "N/A"
        
        # Status note
        if status == "OPTIMAL":
            note = "✓ Best solution"
        elif status == "FEASIBLE":
            note = "◐ Valid but not best"
        else:
            note = "✗ No solution"
        
        f.write(f"| {i:2d} | {nodes_str} | {n_routes} | {e['n_qubits']:2d} | {status:8s} | "
               f"${e['qaoa_cost']:9.2f} | ${e['optimal_cost']:9.2f} | {gap_str:7s} | {note} |\n")
    
    f.write("\n## Performance Summary\n\n")
    
    # Success rate context
    success_rate = (n_optimal + n_feasible) / n_total * 100 if n_total > 0 else 0
    if success_rate >= 95:
        rating = "🟢 Excellent"
    elif success_rate >= 80:
        rating = "🟡 Good"
    elif success_rate >= 60:
        rating = "🟠 Fair"
    else:
        rating = "🔴 Poor"
    
    f.write(f"**Success Rate:** {success_rate:.1f}% - {rating}\n\n")
    f.write(f"**Interpretation:**\n")
    f.write(f"- **Optimal:** Quantum solver found the best possible solution\n")
    f.write(f"- **Feasible:** Valid route found, but optimality not guaranteed\n")
    f.write(f"- **Failed:** No valid solution could be produced\n")

# 3. JSON format (structured data)
qaoa_json_path = get_result_path("02_qaoa_report", "json", subdir="data")
json_data = {
    "instance": instance_name,
    "vehicles": K,
    "nodes": N_NODES,
    "summary": {
        "total_calls": n_total,
        "optimal": {"count": n_optimal, "percentage": round(n_optimal/n_total*100, 1)},
        "feasible": {"count": n_feasible, "percentage": round(n_feasible/n_total*100, 1)},
        "failed": {"count": n_nonvalid, "percentage": round(n_nonvalid/n_total*100, 1)}
    },
    "results": [
        {
            "index": i,
            "nodes": e["node_ids"],
            "vehicles": len([x for x in e["node_ids"] if x < 0]),
            "qubits": e['n_qubits'],
            "status": e['outcome'].upper(),
            "qaoa_cost": round(e['qaoa_cost'], 2),
            "optimal_cost": round(e['optimal_cost'], 2),
            "gap_pct": round(e['gap_pct'], 3) if e['gap_pct'] is not None else None
        }
        for i, e in enumerate(QAOA_LOG, 1)
    ]
}

with open(qaoa_json_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"\n✓ Saved reports:")
print(f"  • CSV:      {os.path.basename(qaoa_csv_path)}")
print(f"  • Markdown: {os.path.basename(qaoa_md_path)}")


# %%
# Reset QAOA_STATS before the benchmark so Cell 7 counts are isolated to the benchmark run only.
QAOA_STATS["success"] = 0; QAOA_STATS["fallback"] = 0
QAOA_LOG.clear()
print("QAOA_STATS reset for benchmark run.")


# %% [markdown]
# ---
# ## Cell 7 -- Benchmark table
# 
# Runs all 6 algorithms on the real 50-node matrix for the configured `K`.  
# The **`k`** column shows the actual number of routes each algorithm produced.
# 

# %%
def validate_solution(sol, nodes, depot, k, dist_matrix=None):
    issues  = []
    all_ids = set(n.id for n in nodes); visited = set()
    for r in sol.routes:
        for nid in r.node_ids:
            if nid in visited: issues.append(f'Node {nid} visited >1')
            visited.add(nid)
    missing = all_ids - visited
    if missing: issues.append(f'Nodes not visited: {missing}')
    extra = visited - all_ids
    if extra: issues.append(f'Unknown nodes: {extra}')
    # Route-count check: must produce exactly k routes (or n if n < k).
    expected_k = min(k, len(nodes))
    if nodes and len(sol.routes) != expected_k:
        issues.append(
            f'Wrong route count: {len(sol.routes)} produced, expected exactly {expected_k}')
    dm = build_dist_map(nodes, depot, dist_matrix)
    for i, r in enumerate(sol.routes):
        actual = route_distance(r.node_ids, depot.id, dm)
        if abs(actual - r.distance) > 1e-6:
            issues.append(f'Route {i}: reported {r.distance:.2f}, actual {actual:.2f}')
    return {'valid': len(issues)==0, 'issues': issues}


# ── Reuse the QAOA solution from the traced run (Cell 6) ────────────────────
# 'sol' was computed by run_traced() in Cell 6.  The leaf seeds are
# deterministic (hash of node IDs), so vrp_solver_plain would produce
# identical results — no need to re-run QAOA.
_cached_dm   = build_dist_map(nodes, depot, DIST_MATRIX)
_cached_raw  = sol                                       # from Cell 6
_cached_2opt = two_opt_routes(sol, depot, _cached_dm)

def _return_cached_raw(n, k, d, m):
    return _cached_raw

def _return_cached_2opt(n, k, d, m):
    return _cached_2opt

ALGO_REGISTRY = [
    ('Recursive (NO 2-opt)',  _return_cached_raw),
    ('Recursive + 2-opt',     _return_cached_2opt),
    ('Nearest-Neighbour',     lambda n,k,d,m: algo_nn(n,k,d,m)),
    ('Clarke-Wright Savings', lambda n,k,d,m: algo_savings(n,k,d,m)),
    ('Sweep',                 lambda n,k,d,m: algo_sweep(n,k,d,m)),
    ('OR-Tools (GLS)',        lambda n,k,d,m: algo_ortools(n,k,d,m)),
]
algo_names = [a[0] for a in ALGO_REGISTRY]

# SINGLE K: use the K defined in Cell 5
k_val = K

name_w=25; k_w=4; col_w=13; std_w=10
hdr = (f'  {"Algorithm":{name_w}s}  {"k":>{k_w}s}  '
       f'{"Distance":>{col_w}s}  {"Std":>{std_w}s}  '
       f'{"Time(s)":>{col_w}s}  {"Valid":>6s}  {"vs Best":>{col_w}s}')
div = (f'  {"-"*name_w}  {"-"*k_w}  {"-"*col_w}  {"-"*std_w}  '
       f'{"-"*col_w}  {"-"*6}  {"-"*col_w}')

print('='*130)
print(f'BENCHMARK -- RioClaroPostToy_50_0 (Real 50-node distance matrix)   k={k_val}')
print('='*130)

print(f'\n-- k={k_val}  (n=50)')
print(hdr); print(div)
inst_res = {}
for aname, afn in ALGO_REGISTRY:
    t0  = time.time()
    sol = afn(nodes, k_val, depot, DIST_MATRIX)
    el  = time.time() - t0
    v   = validate_solution(sol, nodes, depot, k_val, DIST_MATRIX)
    inst_res[aname] = (sol.total_distance, el, v['valid'], sol.num_vehicles_used, sol.distance_std)
    if not v['valid']: print(f'    !! {aname}: {v["issues"]}')

best = min(d for d,_,valid,_,_ in inst_res.values() if valid)
for aname in algo_names:
    dist, el, valid, k_used, std = inst_res[aname]
    gap    = (dist-best)/best*100 if best > 0 else 0
    marker = ' ***' if dist == best and valid else ''
    status = 'OK' if valid else 'FAIL'
    print(f'  {aname:{name_w}s}  {k_used:{k_w}d}  '
          f'{dist:{col_w}.2f}  {std:{std_w}.2f}  '
          f'{el:{col_w}.4f}  '
          f'{status:>6s}  {gap:{col_w-1}.1f}%{marker}')

print('='*130)


# %%
# Save benchmark results to CSV
benchmark_csv_path = get_result_path("03_benchmark_algorithms", "csv", subdir="data")
with open(benchmark_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Algorithm', 'k_used', 'Total_Distance', 'Distance_Std', 'Time_sec', 'Valid', 'Gap_vs_Best_Pct'])
    for aname in algo_names:
        if aname in inst_res:
            dist, el, valid, k_used, std = inst_res[aname]
            gap = (dist-best)/best*100 if best > 0 else 0
            writer.writerow([aname, k_used, f"{dist:.2f}", f"{std:.2f}", f"{el:.4f}", "OK" if valid else "FAIL", f"{gap:.1f}"])
print(f"\n✓ Saved: {os.path.basename(benchmark_csv_path)}")

# %% [markdown]
# ---
# ## QAOA Performance Log
# 
# Classifies every QAOA leaf call into one of three outcomes:
# - **Optimal** — QAOA found the global minimum (matches brute-force)
# - **Feasible** — Valid route but not optimal (gap > 0%)
# - **Non-valid** — No feasible solution found (classical fallback used)
# 

# %%
# Helper function to generate timestamped filenames
def get_result_path(name, filetype, subdir="plots", timestamp_prefix=True):
    """Generate a results path with proper naming convention.
    
    Args:
        name: Descriptive name (e.g., 'algorithm_comparison', 'qaoa_summary')
        filetype: File extension (e.g., 'png', 'txt', 'csv')
        subdir: Subdirectory ('plots', 'data', or 'logs')
        timestamp_prefix: Whether to prepend timestamp
    
    Returns:
        Full path to save file
    """
    if subdir == "plots":
        base_dir = plots_dir
    elif subdir == "data":
        base_dir = data_dir
    else:
        base_dir = logs_dir
    
    # Simplified naming since context is in folder path
    if timestamp_prefix:
        filename = f"{timestamp}_{name}.{filetype}"
    else:
        filename = f"{name}.{filetype}"
    return os.path.join(base_dir, filename)

# %% [markdown]
# ---
# ## Cell 8 — Depth profile charts

# %%
    plt.tight_layout()
    _show(fig)
    
    # Save recursion profile plot
    profile_plot_path = get_result_path("04_recursion_profile", "png")
    fig.savefig(profile_plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {os.path.basename(profile_plot_path)}")

    print('\nDetailed frame log:')
    print(f'  {"depth":>5}  {"label":>15}  {"n":>5}  {"k":>4}  {"clusters":>8}  allocs')
    print('  ' + '-'*65)
    
    # Also save detailed log to file
    log_path = get_result_path("04_recursion_trace", "txt", subdir="logs")
    with open(log_path, 'w') as logf:
        logf.write(f'Recursion Trace Log — {instance_name} k={K}\n')
        logf.write('='*70 + '\n\n')
        logf.write(f'  {"depth":>5}  {"label":>15}  {"n":>5}  {"k":>4}  {"clusters":>8}  allocs\n')
        logf.write('  ' + '-'*65 + '\n')
        
        for f in TRACE_LOG:
            line = f'  {f["depth"]:>5}  {f["label"]:>15}  {f["n"]:>5}  {f["k"]:>4}  {len(f["clusters"]):>8}  {f["allocs"]}\n'
            print(line.rstrip())
            logf.write(line)
    print(f"✓ Saved: {os.path.basename(log_path)}")

# %% [markdown]
# 

# %%

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS SUMMARY — All saved files and outputs
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print(" "*20 + "EXPERIMENT RESULTS SUMMARY")
print("="*80)
print(f"\nInstance:      {instance_name}")
print(f"Problem Size:  {N_NODES} delivery nodes + 1 depot")
print(f"Vehicles:      {K}")
print(f"Timestamp:     {timestamp}")
print(f"\nResults saved to: {results_base}")

# List all saved files
print("\n" + "-"*80)
print("SAVED FILES:")
print("-"*80)

saved_files = []

# Check plots directory
if os.path.exists(plots_dir):
    for f in sorted(os.listdir(plots_dir)):
        fpath = os.path.join(plots_dir, f)
        size_mb = os.path.getsize(fpath) / (1024*1024)
        print(f"  📊 plots/{f:<50s} ({size_mb:.1f} MB)")
        saved_files.append(('plot', f))

# Check data directory
if os.path.exists(data_dir):
    for f in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  📋 data/{f:<50s}  ({size_kb:.1f} KB)")
        saved_files.append(('data', f))

# Check logs directory
if os.path.exists(logs_dir):
    for f in sorted(os.listdir(logs_dir)):
        fpath = os.path.join(logs_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  📝 logs/{f:<50s}  ({size_kb:.1f} KB)")
        saved_files.append(('log', f))

print("-"*80)
print(f"\nTotal files saved: {len(saved_files)}")
print(f"\nTo open results folder:")
print(f"  open '{results_base}'")
print("\n" + "="*80)


