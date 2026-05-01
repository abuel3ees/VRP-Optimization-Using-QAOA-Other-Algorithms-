#!/usr/bin/env python3
"""
Experiment 3: Fix Decomposition — K Clusters Instead of sqrt(n)
================================================================
Root cause: ceil(sqrt(50)) = 8 clusters > K=7 vehicles → triggers C>k branch
where every cluster gets 1 vehicle, destroying vehicle allocation.

Fix: Use K clusters at root so each cluster maps to exactly 1 vehicle.
     Deeper levels (k=1 TSP) still use sqrt(n) for recursive subdivision.

Variants tested:
  A. BASE:    ceil(sqrt(n)) clusters at all levels     (original)
  B. K-MATCH: K clusters at root, sqrt(n) at sub-levels (fix)
  C. K-DIST:  K clusters at root using distance-weighted k-means (enhanced fix)
"""

import sys, os, time, math, itertools, json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

QAOA_VERBOSE = False  # Suppress leaf output for faster comparison

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

GLOBAL_MAX_DIST = 1.0

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
    distance_std: float = 0.0
    def __post_init__(self):
        self.num_vehicles_used = len(self.routes)
        self.total_distance    = sum(r.distance for r in self.routes)
        self.total_load        = sum(r.load     for r in self.routes)
        if len(self.routes) > 1:
            self.distance_std = float(np.std([r.distance for r in self.routes]))
        else:
            self.distance_std = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# DISTANCE & ROUTE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def euclidean(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def build_dist_map(nodes, depot, dist_matrix=None):
    all_nodes = [depot] + nodes
    dmap = {}
    for a in all_nodes:
        for b in all_nodes:
            dmap[(a.id, b.id)] = (float(dist_matrix[a.id][b.id])
                                  if dist_matrix is not None else euclidean(a, b))
    return dmap

def route_distance(route_ids, depot_id, dist_map):
    if not route_ids: return 0.0
    d = dist_map[(depot_id, route_ids[0])]
    for i in range(len(route_ids)-1):
        d += dist_map[(route_ids[i], route_ids[i+1])]
    d += dist_map[(route_ids[-1], depot_id)]
    return d

def two_opt(route_ids, depot_id, dist_map):
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

def two_opt_routes(sol, depot, dist_map):
    improved = []
    for r in sol.routes:
        opt_ids = two_opt(r.node_ids, depot.id, dist_map)
        improved.append(Route(
            node_ids=opt_ids,
            distance=route_distance(opt_ids, depot.id, dist_map),
            load=r.load
        ))
    return VRPSolution(routes=improved)


# ═══════════════════════════════════════════════════════════════════════════════
# CLUSTERING VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_nodes(nodes, depot, num_clusters):
    """Original: angular-sweep + k-means."""
    if num_clusters <= 1: return [nodes]
    if len(nodes) < 2: return [nodes]
    num_clusters = min(num_clusters, len(nodes))
    def angle(n): return math.atan2(n.y - depot.y, n.x - depot.x)
    sorted_nodes = sorted(nodes, key=angle)
    clusters = [[] for _ in range(num_clusters)]
    for i, n in enumerate(sorted_nodes): clusters[i % num_clusters].append(n)
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
        if ({frozenset(n.id for n in cl) for cl in new_clusters} ==
                {frozenset(n.id for n in cl) for cl in clusters}):
            break
        clusters = new_clusters
    return [c for c in clusters if c]


def cluster_nodes_dist_weighted(nodes, depot, num_clusters, dist_map):
    """Distance-weighted k-means: penalises assigning far-from-depot nodes
    to the same cluster as near-depot nodes, encouraging compact routes."""
    if num_clusters <= 1: return [nodes]
    if len(nodes) < 2: return [nodes]
    num_clusters = min(num_clusters, len(nodes))

    # Initialise with angular sweep (same as original)
    def angle(n): return math.atan2(n.y - depot.y, n.x - depot.x)
    sorted_nodes = sorted(nodes, key=angle)
    clusters = [[] for _ in range(num_clusters)]
    for i, n in enumerate(sorted_nodes): clusters[i % num_clusters].append(n)

    node_xy = np.array([(n.x, n.y) for n in nodes])
    depot_dists = np.array([dist_map.get((depot.id, n.id), euclidean(depot, n)) for n in nodes])
    max_depot_dist = depot_dists.max() if depot_dists.max() > 0 else 1.0

    for _ in range(15):
        centroids = np.array([
            (np.mean([n.x for n in cl]), np.mean([n.y for n in cl]))
            if cl else (depot.x, depot.y)
            for cl in clusters
        ])
        # Euclidean distance to centroids
        geo_dists = np.linalg.norm(node_xy[:, None, :] - centroids[None, :, :], axis=2)

        # Compute cluster "depot distance profile" — average depot distance per cluster
        cluster_depot_profile = np.array([
            np.mean([dist_map.get((depot.id, n.id), euclidean(depot, n)) for n in cl])
            if cl else 0.0
            for cl in clusters
        ])

        # Penalty: assigning a node to a cluster whose depot-distance profile is
        # very different from the node's own depot distance
        node_depot_dists = depot_dists[:, None]  # (n_nodes, 1)
        cluster_profiles = cluster_depot_profile[None, :]  # (1, n_clusters)
        profile_penalty = np.abs(node_depot_dists - cluster_profiles) / max_depot_dist

        # Combined metric: 70% geometry + 30% depot-distance compatibility
        combined = 0.7 * geo_dists / (geo_dists.max() if geo_dists.max() > 0 else 1.0) + \
                   0.3 * profile_penalty

        assignments = np.argmin(combined, axis=1)
        new_clusters = [[] for _ in range(num_clusters)]
        for i, n in enumerate(nodes):
            new_clusters[assignments[i]].append(n)
        new_clusters = [c for c in new_clusters if c]
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
        if ({frozenset(n.id for n in cl) for cl in new_clusters} ==
                {frozenset(n.id for n in cl) for cl in clusters}):
            break
        clusters = new_clusters
    return [c for c in clusters if c]


def nearest_neighbor_route(node_ids, depot_id, dist_map):
    if not node_ids: return []
    unvisited = set(node_ids); route = []; current = depot_id
    while unvisited:
        nearest = min(unvisited, key=lambda nid: dist_map[(current, nid)])
        route.append(nearest); unvisited.remove(nearest); current = nearest
    return route


# ═══════════════════════════════════════════════════════════════════════════════
# CLARKE-WRIGHT & SPLIT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _split_routes_to_k(routes_list, k, n_nodes, depot_id=None, dist_map=None, node_map=None):
    if n_nodes < k: return routes_list
    result = [list(r) for r in routes_list if r]
    def _split_one(route):
        if len(route) < 2: return route, []
        if dist_map is not None:
            edge_costs = [dist_map.get((route[i], route[i+1]), 0.0) for i in range(len(route)-1)]
            if edge_costs:
                cut = int(max(range(len(edge_costs)), key=lambda i: edge_costs[i])) + 1
                return route[:cut], route[cut:]
        if node_map is not None:
            coords = [(nid, node_map[nid].x, node_map[nid].y) for nid in route if nid in node_map]
            if len(coords) >= 2:
                cx = sum(x for _, x, _ in coords) / len(coords)
                cy = sum(y for _, _, y in coords) / len(coords)
                sorted_by_angle = sorted(coords, key=lambda t: math.atan2(t[2]-cy, t[1]-cx))
                mid = len(sorted_by_angle) // 2
                return [nid for nid,_,_ in sorted_by_angle[:mid]], [nid for nid,_,_ in sorted_by_angle[mid:]]
        mid = len(route) // 2
        return route[:mid], route[mid:]
    while len(result) < k:
        idx = max(range(len(result)), key=lambda i: len(result[i]))
        r = result[idx]
        if len(r) < 2: break
        left, right = _split_one(r)
        if not right: break
        result[idx] = left; result.append(right)
    return result

def greedy_vrp_no_opt(nodes, k, depot, dist_map):
    if not nodes: return VRPSolution(routes=[])
    node_map = {n.id: n for n in nodes}; node_ids = [n.id for n in nodes]
    routes_dict = {nid: [nid] for nid in node_ids}
    route_of = {nid: nid for nid in node_ids}
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
    if len(nodes) >= k:
        final = _split_routes_to_k(final, k, len(nodes), depot_id=depot.id, dist_map=dist_map, node_map=node_map)
    return VRPSolution(routes=[Route(node_ids=r, distance=route_distance(r, depot.id, dist_map),
                                     load=route_load(r)) for r in final])


# ═══════════════════════════════════════════════════════════════════════════════
# QAOA SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

QAOA_STATS = {"success": 0, "fallback": 0}
QAOA_LOG = []
LEAF_SIZE = 4

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
        {"reps": 2, "optimizer": "ADAM",   "maxiter": 30,  "restarts": 1},
        {"reps": 5, "optimizer": "ADAM",   "maxiter": 30,  "restarts": 1},
        {"reps": 2, "optimizer": "COBYLA", "maxiter": 100, "restarts": 1},
        {"reps": 2, "optimizer": "ADAM",   "maxiter": 80,  "restarts": 2},
        {"reps": 5, "optimizer": "COBYLA", "maxiter": 100, "restarts": 1},
    ]
    _SHOTS    = 50_000
    _SEED     = 7
    _DECODE_K = 10

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
        penalty = _N * 2.0

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



def allocate_vehicles(nodes, clusters, k):
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

def create_super_nodes(clusters, csols):
    return [Node(id=-(i+1), x=np.mean([n.x for n in cl]),
                 y=np.mean([n.y for n in cl]), demand=0)
            for i, cl in enumerate(clusters)]

def build_super_dist_map_real(super_nodes, depot, clusters, dist_map):
    """Minimum real distance between any two nodes across clusters."""
    sdm = {}
    all_sn = [depot] + super_nodes
    cluster_node_ids = {}
    for sn in super_nodes:
        ci = -(sn.id) - 1
        cluster_node_ids[sn.id] = [n.id for n in clusters[ci]]
    cluster_node_ids[depot.id] = [depot.id]
    for a in all_sn:
        for b in all_sn:
            if a.id == b.id:
                sdm[(a.id, b.id)] = 0.0
                continue
            ids_a = cluster_node_ids[a.id]
            ids_b = cluster_node_ids[b.id]
            min_d = float('inf')
            for na_id in ids_a:
                for nb_id in ids_b:
                    d = dist_map.get((na_id, nb_id), float('inf'))
                    if d < min_d:
                        min_d = d
            sdm[(a.id, b.id)] = min_d
    return sdm

def _orient_segment(seg, anchor_id, dist_map):
    if not seg or len(seg) < 2 or dist_map is None: return seg
    d_fwd = dist_map.get((anchor_id, seg[0]), float('inf'))
    d_rev = dist_map.get((anchor_id, seg[-1]), float('inf'))
    return list(reversed(seg)) if d_rev < d_fwd else seg

def merge_super_solution(ssol, csols, clusters, depot=None, dist_map=None):
    merged = []; depot_id = depot.id if depot is not None else None
    for sr in ssol.routes:
        valid_cis = [-(sn_id)-1 for sn_id in sr.node_ids if 0 <= -(sn_id)-1 < len(csols)]
        if not valid_cis: continue
        if len(valid_cis) == 1:
            ci = valid_cis[0]
            for r in csols[ci].routes:
                if r.node_ids:
                    merged.append(Route(node_ids=list(r.node_ids), distance=r.distance, load=r.load))
        else:
            all_real_nodes = []
            for ci in valid_cis: all_real_nodes.extend(clusters[ci])
            if not all_real_nodes: continue
            if len(all_real_nodes) <= LEAF_SIZE and dist_map is not None and depot is not None:
                re_sol = solve_brute_force(all_real_nodes, 1, depot, dist_map)
                for r in re_sol.routes:
                    if r.node_ids:
                        merged.append(Route(node_ids=list(r.node_ids), distance=r.distance, load=r.load))
            else:
                combined_ids = []; combined_load = 0.0
                for ci in valid_cis:
                    non_empty = [r for r in csols[ci].routes if r.node_ids]
                    load = sum(r.load for r in csols[ci].routes)
                    for sub_r in non_empty:
                        seg = list(sub_r.node_ids)
                        anchor = combined_ids[-1] if combined_ids else depot_id
                        seg = _orient_segment(seg, anchor, dist_map)
                        combined_ids.extend(seg); combined_load += load
                if combined_ids:
                    d = route_distance(combined_ids, depot.id, dist_map) if depot and dist_map else 0.0
                    merged.append(Route(node_ids=combined_ids, distance=d, load=combined_load))
    if not merged:
        for csol in csols:
            for r in csol.routes: merged.append(r)
    expected_k = len(ssol.routes)
    while len(merged) > expected_k:
        merged.sort(key=lambda r: len(r.node_ids))
        a, b = merged[0], merged[1]
        ids = list(a.node_ids) + list(b.node_ids)
        d = route_distance(ids, depot.id, dist_map) if depot and dist_map else 0.0
        merged = merged[2:] + [Route(node_ids=ids, distance=d, load=a.load+b.load)]
    while len(merged) < expected_k:
        merged.sort(key=lambda r: len(r.node_ids), reverse=True)
        biggest = merged[0].node_ids
        if len(biggest) < 2: break
        mid = len(biggest) // 2
        left, right = list(biggest[:mid]), list(biggest[mid:])
        dl = route_distance(left, depot.id, dist_map) if depot and dist_map else 0.0
        dr = route_distance(right, depot.id, dist_map) if depot and dist_map else 0.0
        nm = {n.id: n for cl in clusters for n in cl}
        merged = merged[1:] + [
            Route(node_ids=left, distance=dl, load=sum(nm[x].demand for x in left if x in nm)),
            Route(node_ids=right, distance=dr, load=sum(nm[x].demand for x in right if x in nm))]
    return VRPSolution(routes=merged)


# ═══════════════════════════════════════════════════════════════════════════════
# RECURSIVE SOLVERS (3 VARIANTS)
# ═══════════════════════════════════════════════════════════════════════════════

# Variant selector (module-level)
_CLUSTER_MODE = "base"  # "base", "k_match", "k_dist"

def _get_num_clusters(n_nodes, k, depth):
    """Decide number of clusters based on mode."""
    if _CLUSTER_MODE == "base":
        return max(2, math.ceil(math.sqrt(n_nodes)))
    elif _CLUSTER_MODE == "k_match":
        if depth == 0 and k > 1:
            return k  # Match clusters to vehicles at root
        return max(2, math.ceil(math.sqrt(n_nodes)))
    elif _CLUSTER_MODE == "k_dist":
        if depth == 0 and k > 1:
            return k
        return max(2, math.ceil(math.sqrt(n_nodes)))
    return max(2, math.ceil(math.sqrt(n_nodes)))

def vrp_solver(k, nodes, depot, dist_map=None, dist_matrix=None, _depth=0):
    if dist_map is None: dist_map = build_dist_map(nodes, depot, dist_matrix)
    if not nodes: return VRPSolution(routes=[])
    if len(nodes) == 1:
        nid = nodes[0].id; d = dist_map[(depot.id, nid)] * 2
        return VRPSolution(routes=[Route(node_ids=[nid], distance=d, load=nodes[0].demand)])
    if len(nodes) <= LEAF_SIZE:
        return solve_brute_force(nodes, k, depot, dist_map)

    num_cl = _get_num_clusters(len(nodes), k, _depth)

    if _CLUSTER_MODE == "k_dist" and _depth == 0 and k > 1:
        clusters = cluster_nodes_dist_weighted(nodes, depot, num_cl, dist_map)
    else:
        clusters = cluster_nodes(nodes, depot, num_cl)

    def _build_sdm(super_nodes, clusters_local):
        return build_super_dist_map_real(super_nodes, depot, clusters_local, dist_map)

    if k == 1:
        cluster_solutions = [vrp_solver(1, cl, depot, dist_map, _depth=_depth+1) for cl in clusters]
        super_nodes = create_super_nodes(clusters, cluster_solutions)
        sdm = _build_sdm(super_nodes, clusters)
        if len(super_nodes) > LEAF_SIZE:
            super_solution = vrp_solver(1, super_nodes, depot, dist_map=sdm, _depth=_depth+1)
        else:
            super_solution = solve_brute_force(super_nodes, 1, depot, sdm)
        _sol = merge_super_solution(super_solution, cluster_solutions, clusters, depot, dist_map)
        try:
            assert len({nid for r in _sol.routes for nid in r.node_ids}) == len(nodes)
        except AssertionError:
            all_ids = [nid for cs in cluster_solutions for r in cs.routes for nid in r.node_ids]
            _sol = VRPSolution(routes=[Route(node_ids=all_ids,
                distance=route_distance(all_ids, depot.id, dist_map), load=sum(n.demand for n in nodes))])
        return _sol

    C = len(clusters)
    if C > k:
        cluster_solutions = [vrp_solver(1, cl, depot, dist_map, _depth=_depth+1) for cl in clusters]
        super_nodes = create_super_nodes(clusters, cluster_solutions)
        sdm = _build_sdm(super_nodes, clusters)
        if len(super_nodes) > LEAF_SIZE:
            super_solution = vrp_solver(k, super_nodes, depot, dist_map=sdm, _depth=_depth+1)
        else:
            super_solution = solve_brute_force(super_nodes, k, depot, sdm)
        _sol = merge_super_solution(super_solution, cluster_solutions, clusters, depot, dist_map)
        try:
            assert len({nid for r in _sol.routes for nid in r.node_ids}) == len(nodes)
        except AssertionError:
            flat = [r for cs in cluster_solutions for r in cs.routes]
            groups = [[] for _ in range(k)]
            for ci, cr in enumerate(flat): groups[ci % k].extend(cr.node_ids)
            nm = {n.id: n for n in nodes}
            _sol = VRPSolution(routes=[Route(node_ids=g,
                distance=route_distance(g, depot.id, dist_map),
                load=sum(nm[nid].demand for nid in g)) for g in groups if g])
        return _sol
    else:
        vehicle_alloc = allocate_vehicles(nodes, clusters, k)
        cluster_solutions = [vrp_solver(vehicle_alloc[i], cl, depot, dist_map, _depth=_depth+1)
                             for i, cl in enumerate(clusters)]
        all_routes = [r for csol in cluster_solutions for r in csol.routes]
        return VRPSolution(routes=all_routes)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSICAL BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def _partition_by_angle(nodes, depot, k):
    n_count = len(nodes); eff_k = min(k, n_count)
    sn = sorted(nodes, key=lambda nd: math.atan2(nd.y-depot.y, nd.x-depot.x))
    base = n_count // eff_k; rem = n_count % eff_k
    groups=[]; idx=0
    for v in range(eff_k):
        sz = base+(1 if v<rem else 0)
        groups.append([nd.id for nd in sn[idx:idx+sz]]); idx+=sz
    return groups

def algo_nn(nodes, k, depot, dist_matrix=None):
    if not nodes: return VRPSolution(routes=[])
    dist_map = build_dist_map(nodes, depot, dist_matrix)
    node_map = {n.id: n for n in nodes}; routes = []
    for g in _partition_by_angle(nodes, depot, k):
        nn = nearest_neighbor_route(g, depot.id, dist_map)
        routes.append(Route(node_ids=nn, distance=route_distance(nn, depot.id, dist_map),
                            load=sum(node_map[nid].demand for nid in nn)))
    return VRPSolution(routes=routes)

def algo_savings(nodes, k, depot, dist_matrix=None):
    return greedy_vrp_no_opt(nodes, k, depot, build_dist_map(nodes, depot, dist_matrix))

def algo_sweep(nodes, k, depot, dist_matrix=None):
    if not nodes: return VRPSolution(routes=[])
    dist_map = build_dist_map(nodes, depot, dist_matrix)
    node_map = {n.id: n for n in nodes}; routes = []
    for g in _partition_by_angle(nodes, depot, k):
        routes.append(Route(node_ids=g, distance=route_distance(g, depot.id, dist_map),
                            load=sum(node_map[nid].demand for nid in g)))
    return VRPSolution(routes=routes)

def algo_ortools(nodes, k, depot, dist_matrix=None):
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        return algo_savings(nodes, k, depot, dist_matrix)
    if not nodes: return VRPSolution(routes=[])
    dist_map = build_dist_map(nodes, depot, dist_matrix)
    node_map = {n.id: n for n in nodes}
    eff_k = min(k, len(nodes))
    all_nodes = [depot] + nodes; n_total = len(all_nodes)
    _OR_SCALE = 1000
    def or_dist(i, j): return int(dist_map[(all_nodes[i].id, all_nodes[j].id)] * _OR_SCALE)
    manager = pywrapcp.RoutingIndexManager(n_total, eff_k, 0)
    routing = pywrapcp.RoutingModel(manager)
    transit_cb_idx = routing.RegisterTransitCallback(
        lambda i, j: or_dist(manager.IndexToNode(i), manager.IndexToNode(j)))
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)
    def demand_cb(from_index):
        node = manager.IndexToNode(from_index)
        return 0 if node == 0 else 1
    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_cb_idx, 0, [len(nodes)] * eff_k, True, 'Count')
    count_dim = routing.GetDimensionOrDie('Count')
    big_penalty = int(max(dist_map.values()) * _OR_SCALE * 10)
    for v in range(eff_k):
        count_dim.SetCumulVarSoftLowerBound(routing.End(v), 1, big_penalty)
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 10
    assignment = routing.SolveWithParameters(search_params)
    if not assignment: return algo_savings(nodes, k, depot, dist_matrix)
    routes = []
    for v in range(eff_k):
        idx = routing.Start(v); route = []
        while not routing.IsEnd(idx):
            node_idx = manager.IndexToNode(idx)
            if node_idx != 0: route.append(all_nodes[node_idx].id)
            idx = assignment.Value(routing.NextVar(idx))
        if route:
            routes.append(Route(node_ids=route, distance=route_distance(route, depot.id, dist_map),
                                load=sum(node_map[nid].demand for nid in route)))
    return VRPSolution(routes=routes)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_solution(sol, nodes, depot, k, dist_matrix=None):
    issues = []
    all_ids = set(n.id for n in nodes); visited = set()
    for r in sol.routes:
        for nid in r.node_ids:
            if nid in visited: issues.append(f'Node {nid} visited >1')
            visited.add(nid)
    missing = all_ids - visited
    if missing: issues.append(f'Nodes not visited: {missing}')
    extra = visited - all_ids
    if extra: issues.append(f'Unknown nodes: {extra}')
    expected_k = min(k, len(nodes))
    if nodes and len(sol.routes) != expected_k:
        issues.append(f'Wrong route count: {len(sol.routes)}, expected {expected_k}')
    dm = build_dist_map(nodes, depot, dist_matrix)
    for i, r in enumerate(sol.routes):
        actual = route_distance(r.node_ids, depot.id, dm)
        if abs(actual - r.distance) > 1e-6:
            issues.append(f'Route {i}: reported {r.distance:.2f}, actual {actual:.2f}')
    return {'valid': len(issues)==0, 'issues': issues}


# ═══════════════════════════════════════════════════════════════════════════════
# INSTANCE LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_instance():
    import json
    nb_path = os.path.join(os.path.dirname(__file__), '..', 'instance_12_final_with results.ipynb')
    with open(nb_path) as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] != 'code': continue
        src = ''.join(cell['source'])
        if 'DIST_MATRIX_RAW' in src and 'RioClaroPostToy_50_0' in src:
            local_ns = {}
            exec(src, {"__builtins__": __builtins__, "np": np, "Node": Node,
                        "build_dist_map": build_dist_map, "print": lambda *a, **kw: None}, local_ns)
            return (local_ns['DIST_MATRIX'], local_ns['nodes'], local_ns['depot'],
                    local_ns.get('N_NODES', 50))
    raise RuntimeError("Could not find distance matrix in notebook")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print("=" * 100)
    print("EXPERIMENT 3: Fix Decomposition — K Clusters vs sqrt(n)")
    print("=" * 100)

    DIST_MATRIX, nodes, depot, N_NODES = load_instance()
    GLOBAL_MAX_DIST = float(np.max(DIST_MATRIX))
    K = 7
    dist_map_full = build_dist_map(nodes, depot, DIST_MATRIX)

    print(f"  Nodes: {N_NODES}  |  Max dist: {GLOBAL_MAX_DIST:.3f}  |  K={K}  |  LEAF_SIZE={LEAF_SIZE}")
    print(f"  ceil(sqrt({N_NODES})) = {math.ceil(math.sqrt(N_NODES))} clusters (base)")
    print(f"  K = {K} clusters (k-match / k-dist)")

    # ── Show cluster assignments for each mode ────────────────────────────────
    print("\n" + "─" * 80)
    print("CLUSTER ASSIGNMENTS AT ROOT LEVEL")
    print("─" * 80)

    cl_base = cluster_nodes(nodes, depot, max(2, math.ceil(math.sqrt(N_NODES))))
    cl_kmatch = cluster_nodes(nodes, depot, K)
    cl_kdist = cluster_nodes_dist_weighted(nodes, depot, K, dist_map_full)

    for label, cls in [("BASE (sqrt)", cl_base), ("K-MATCH", cl_kmatch), ("K-DIST", cl_kdist)]:
        print(f"\n  [{label}] {len(cls)} clusters:")
        for ci, cl in enumerate(cls):
            centroid = (np.mean([n.x for n in cl]), np.mean([n.y for n in cl]))
            avg_depot = np.mean([dist_map_full[(depot.id, n.id)] for n in cl])
            print(f"    C{ci}: {len(cl):>2d} nodes, "
                  f"centroid=({centroid[0]:.0f},{centroid[1]:.0f}), "
                  f"avg_depot_dist={avg_depot:.0f}, "
                  f"IDs={sorted([n.id for n in cl])}")

    # ── Run all three variants ────────────────────────────────────────────────
    results = {}

    for mode_label, mode in [("BASE (sqrt(n))", "base"),
                              ("K-MATCH", "k_match"),
                              ("K-DIST", "k_dist")]:
        print(f"\n{'─' * 80}")
        print(f"Running QAOA solver: {mode_label}...")
        print("─" * 80)

        _CLUSTER_MODE = mode
        QAOA_STATS["success"] = 0; QAOA_STATS["fallback"] = 0; QAOA_LOG.clear()
        QAOA_VERBOSE = False

        t0 = time.time()
        sol = vrp_solver(K, nodes, depot, dist_matrix=DIST_MATRIX)
        elapsed = time.time() - t0

        log_copy = list(QAOA_LOG)
        stats_copy = dict(QAOA_STATS)

        v = validate_solution(sol, nodes, depot, K, DIST_MATRIX)
        print(f"  Distance: {sol.total_distance:.2f}  |  Valid: {v['valid']}  |  "
              f"Time: {elapsed:.1f}s  |  QAOA calls: {len(log_copy)}  "
              f"(ok={stats_copy['success']}, fb={stats_copy['fallback']})")
        if not v['valid']:
            print(f"  ISSUES: {v['issues']}")

        results[mode_label] = {
            "sol": sol, "time": elapsed, "valid": v,
            "qaoa_log": log_copy, "qaoa_stats": stats_copy
        }

    # ── Classical baselines ───────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("Running classical baselines...")
    print("─" * 80)

    t0 = time.time(); sol_nn = algo_nn(nodes, K, depot, DIST_MATRIX)
    results['Nearest-Neighbour'] = {"sol": sol_nn, "time": time.time()-t0,
        "valid": validate_solution(sol_nn, nodes, depot, K, DIST_MATRIX)}

    t0 = time.time(); sol_cw = algo_savings(nodes, K, depot, DIST_MATRIX)
    results['Clarke-Wright'] = {"sol": sol_cw, "time": time.time()-t0,
        "valid": validate_solution(sol_cw, nodes, depot, K, DIST_MATRIX)}

    t0 = time.time(); sol_sweep = algo_sweep(nodes, K, depot, DIST_MATRIX)
    results['Sweep'] = {"sol": sol_sweep, "time": time.time()-t0,
        "valid": validate_solution(sol_sweep, nodes, depot, K, DIST_MATRIX)}

    t0 = time.time(); sol_ort = algo_ortools(nodes, K, depot, DIST_MATRIX)
    results['OR-Tools (GLS)'] = {"sol": sol_ort, "time": time.time()-t0,
        "valid": validate_solution(sol_ort, nodes, depot, K, DIST_MATRIX)}


    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS TABLES
    # ═══════════════════════════════════════════════════════════════════════════

    # ── TABLE 1: Per-route detail for each QAOA variant ──────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 1: PER-ROUTE DETAIL FOR EACH QAOA VARIANT")
    print("=" * 100)

    for label in ["BASE (sqrt(n))", "K-MATCH", "K-DIST"]:
        sol = results[label]["sol"]
        print(f"\n  [{label}]  Total: {sol.total_distance:.2f}  |  Routes: {sol.num_vehicles_used}  |  σ={sol.distance_std:.2f}")
        print(f"  {'Route':>6s}  {'Nodes':>6s}  {'Distance':>12s}  {'Node IDs'}")
        print("  " + "-" * 80)
        for ri, r in enumerate(sol.routes):
            ids_str = str(sorted(r.node_ids))
            if len(ids_str) > 50: ids_str = ids_str[:48] + ".."
            print(f"  {ri+1:>6d}  {len(r.node_ids):>6d}  {r.distance:>12.2f}  {ids_str}")

    # CW for comparison
    sol_cw = results['Clarke-Wright']['sol']
    print(f"\n  [Clarke-Wright]  Total: {sol_cw.total_distance:.2f}  |  Routes: {sol_cw.num_vehicles_used}  |  σ={sol_cw.distance_std:.2f}")
    print(f"  {'Route':>6s}  {'Nodes':>6s}  {'Distance':>12s}  {'Node IDs'}")
    print("  " + "-" * 80)
    for ri, r in enumerate(sol_cw.routes):
        ids_str = str(sorted(r.node_ids))
        if len(ids_str) > 50: ids_str = ids_str[:48] + ".."
        print(f"  {ri+1:>6d}  {len(r.node_ids):>6d}  {r.distance:>12.2f}  {ids_str}")


    # ── TABLE 2: QAOA leaf performance comparison ────────────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 2: QAOA LEAF PERFORMANCE BY VARIANT")
    print("=" * 100)

    for label in ["BASE (sqrt(n))", "K-MATCH", "K-DIST"]:
        log = results[label]["qaoa_log"]
        n_total = len(log)
        n_opt = sum(1 for e in log if e["outcome"] == "optimal")
        n_feas = sum(1 for e in log if e["outcome"] == "feasible")
        n_nv = sum(1 for e in log if e["outcome"] == "non-valid")
        avg_qubits = np.mean([e["n_qubits"] for e in log]) if log else 0
        max_qubits = max([e["n_qubits"] for e in log]) if log else 0
        print(f"\n  [{label}]")
        print(f"    Total calls: {n_total}  |  Optimal: {n_opt}  |  "
              f"Feasible: {n_feas}  |  Non-valid: {n_nv}")
        print(f"    Avg qubits: {avg_qubits:.1f}  |  Max qubits: {max_qubits}")


    # ── TABLE 3: Full benchmark ──────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 3: FULL BENCHMARK — ALL ALGORITHMS")
    print(f"Instance: RioClaroPostToy_50_0  |  n=50  |  k={K}")
    print("=" * 100)

    valid_dists = [r["sol"].total_distance for r in results.values()
                   if r["valid"]["valid"]]
    best_dist = min(valid_dists) if valid_dists else float('inf')

    print(f"\n  {'Algorithm':<26s}  {'k':>4s}  {'Distance':>12s}  {'Std (σ)':>10s}  {'Time(s)':>10s}  {'Valid':>6s}  {'vs Best':>10s}")
    print("  " + "-" * 88)

    order = ["BASE (sqrt(n))", "K-MATCH", "K-DIST",
             "Nearest-Neighbour", "Clarke-Wright", "Sweep", "OR-Tools (GLS)"]
    for name in order:
        r = results[name]
        sol = r["sol"]; elapsed = r["time"]; v = r["valid"]
        gap = (sol.total_distance - best_dist) / best_dist * 100 if best_dist > 0 else 0
        marker = ' ***' if abs(sol.total_distance - best_dist) < 1e-2 and v['valid'] else ''
        status = 'OK' if v['valid'] else 'FAIL'
        print(f"  {name:<26s}  {sol.num_vehicles_used:>4d}  {sol.total_distance:>12.2f}  "
              f"{sol.distance_std:>10.2f}  {elapsed:>10.2f}  {status:>6s}  {gap:>9.1f}%{marker}")


    # ── TABLE 4: Improvement summary ─────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 4: DECOMPOSITION FIX SUMMARY")
    print("=" * 100)

    d_base = results["BASE (sqrt(n))"]["sol"].total_distance
    d_km = results["K-MATCH"]["sol"].total_distance
    d_kd = results["K-DIST"]["sol"].total_distance
    d_cw = results["Clarke-Wright"]["sol"].total_distance

    print(f"\n  {'Variant':<30s}  {'Distance':>12s}  {'vs Base':>10s}  {'vs CW':>10s}")
    print("  " + "-" * 68)
    for label, d in [("BASE (sqrt(n))", d_base), ("K-MATCH", d_km), ("K-DIST", d_kd), ("Clarke-Wright", d_cw)]:
        vs_base = (d - d_base) / d_base * 100
        vs_cw = (d - d_cw) / d_cw * 100
        print(f"  {label:<30s}  {d:>12.2f}  {vs_base:>9.1f}%  {vs_cw:>9.1f}%")

    print(f"\n  Improvement K-MATCH vs BASE:  {d_base - d_km:>10.2f}  ({(d_base-d_km)/d_base*100:.1f}%)")
    print(f"  Improvement K-DIST vs BASE:   {d_base - d_kd:>10.2f}  ({(d_base-d_kd)/d_base*100:.1f}%)")
    print(f"  Remaining gap K-MATCH vs CW:  {d_km - d_cw:>10.2f}  ({(d_km-d_cw)/d_cw*100:.1f}%)")
    print(f"  Remaining gap K-DIST vs CW:   {d_kd - d_cw:>10.2f}  ({(d_kd-d_cw)/d_cw*100:.1f}%)")


    # Save results
    results_json = {
        "experiment": "Decomposition fix: K clusters vs sqrt(n)",
        "instance": "RioClaroPostToy_50_0",
        "n_nodes": N_NODES, "k": K, "leaf_size": LEAF_SIZE,
        "algorithms": {}
    }
    for name in order:
        r = results[name]
        sol = r["sol"]
        results_json["algorithms"][name] = {
            "total_distance": sol.total_distance,
            "distance_std": sol.distance_std,
            "num_routes": sol.num_vehicles_used,
            "time_seconds": r["time"],
            "valid": r["valid"]["valid"],
            "routes": [{"node_ids": list(rt.node_ids), "distance": rt.distance} for rt in sol.routes]
        }

    out_path = os.path.join(os.path.dirname(__file__), 'exp3_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    print("\n" + "=" * 100)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 100)
