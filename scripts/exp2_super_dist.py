#!/usr/bin/env python3
"""
Experiment 2: Impact of Real Inter-Cluster Distances on Super-Node QAOA
========================================================================
The base code uses Euclidean centroid-to-centroid distances for the super-node
distance map. This means the QAOA that orders/groups clusters at macro level
receives approximate distances that may differ significantly from the actual
travel cost between clusters.

Fix: Replace centroid Euclidean distances with the minimum real distance
between any two nodes across two clusters, using the actual distance matrix.

Produces:
  - Table 1: Super-node distance comparison (centroid vs real)
  - Table 2: Per-route comparison (base vs improved)
  - Table 3: Full benchmark (all algorithms)
  - Table 4: Improvement summary
"""

import sys, os, time, math, itertools, json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
QAOA_VERBOSE = True
# ─────────────────────────────────────────────────────────────────────────────
# Module-level flag: which super-dist map to use
_USE_REAL_SUPER_DIST = False
_SUPER_DIST_LOG = []  # records (level, method, sdm_centroid, sdm_real) for analysis


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
# CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_nodes(nodes, depot, num_clusters):
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

def build_super_dist_map_centroid(super_nodes, depot):
    """BASELINE: Euclidean centroid-to-centroid distances."""
    all_sn = [depot] + super_nodes
    return {(a.id, b.id): euclidean(a, b) for a in all_sn for b in all_sn}

def build_super_dist_map_real(super_nodes, depot, clusters, dist_map):
    """IMPROVED: Minimum real distance between any two nodes across clusters.

    For each pair of super-nodes (representing clusters), compute the minimum
    distance between any node in cluster A and any node in cluster B, using the
    actual distance matrix instead of Euclidean centroid approximation.

    For depot-to-cluster: minimum distance from depot to any node in the cluster.
    """
    sdm = {}
    all_sn = [depot] + super_nodes

    # Pre-compute node ID lists for each super-node
    cluster_node_ids = {}
    for sn in super_nodes:
        ci = -(sn.id) - 1  # cluster index from super-node ID
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
# RECURSIVE SOLVER (parameterised by super-dist method)
# ═══════════════════════════════════════════════════════════════════════════════

def vrp_solver(k, nodes, depot, dist_map=None, dist_matrix=None, _depth=0):
    """Recursive QAOA-VRP solver. Respects _USE_REAL_SUPER_DIST flag."""
    if dist_map is None: dist_map = build_dist_map(nodes, depot, dist_matrix)
    if not nodes: return VRPSolution(routes=[])
    if len(nodes) == 1:
        nid = nodes[0].id; d = dist_map[(depot.id, nid)] * 2
        return VRPSolution(routes=[Route(node_ids=[nid], distance=d, load=nodes[0].demand)])
    if len(nodes) <= LEAF_SIZE:
        return solve_brute_force(nodes, k, depot, dist_map)

    clusters = cluster_nodes(nodes, depot, max(2, math.ceil(math.sqrt(len(nodes)))))

    def _build_sdm(super_nodes, clusters_local):
        """Build super-dist map according to the current experiment mode."""
        if _USE_REAL_SUPER_DIST:
            sdm_real = build_super_dist_map_real(super_nodes, depot, clusters_local, dist_map)
            # Also compute centroid for comparison logging
            sdm_cent = build_super_dist_map_centroid(super_nodes, depot)
            _SUPER_DIST_LOG.append({
                "depth": _depth,
                "n_super": len(super_nodes),
                "n_nodes": len(nodes),
                "centroid": dict(sdm_cent),
                "real": dict(sdm_real),
            })
            return sdm_real
        else:
            return build_super_dist_map_centroid(super_nodes, depot)

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
# DISTANCE MATRIX & NODE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_instance():
    """Load the 51x51 distance matrix and nodes from the notebook."""
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
    print("EXPERIMENT 2: Impact of Real Inter-Cluster Distances on Super-Node QAOA")
    print("=" * 100)

    # Load instance
    print("\nLoading RioClaroPostToy_50_0 instance...")
    DIST_MATRIX, nodes, depot, N_NODES = load_instance()
    GLOBAL_MAX_DIST = float(np.max(DIST_MATRIX))
    print(f"  Nodes: {N_NODES}  |  Distance matrix: {DIST_MATRIX.shape}  |  Max dist: {GLOBAL_MAX_DIST:.3f}")

    K = 7
    print(f"  Vehicles (k): {K}")
    print(f"  LEAF_SIZE: {LEAF_SIZE}")

    dist_map_full = build_dist_map(nodes, depot, DIST_MATRIX)

    # ── Phase 1: Run BASE solver (centroid super-distances) ───────────────────
    print("\n" + "─" * 80)
    print("Phase 1: Running QAOA Recursive solver — CENTROID super-distances (base)...")
    print("─" * 80)
    _USE_REAL_SUPER_DIST = False
    _SUPER_DIST_LOG.clear()
    QAOA_STATS["success"] = 0; QAOA_STATS["fallback"] = 0; QAOA_LOG.clear()
    QAOA_VERBOSE = True

    t0 = time.time()
    sol_base = vrp_solver(K, nodes, depot, dist_matrix=DIST_MATRIX)
    time_base = time.time() - t0

    qaoa_log_base = list(QAOA_LOG)
    qaoa_stats_base = dict(QAOA_STATS)
    print(f"\n  Base solver completed in {time_base:.1f}s")
    print(f"  Total distance: {sol_base.total_distance:.2f}")

    # ── Phase 2: Run IMPROVED solver (real super-distances) ───────────────────
    print("\n" + "─" * 80)
    print("Phase 2: Running QAOA Recursive solver — REAL inter-cluster super-distances...")
    print("─" * 80)
    _USE_REAL_SUPER_DIST = True
    _SUPER_DIST_LOG.clear()
    QAOA_STATS["success"] = 0; QAOA_STATS["fallback"] = 0; QAOA_LOG.clear()
    QAOA_VERBOSE = True

    t0 = time.time()
    sol_improved = vrp_solver(K, nodes, depot, dist_matrix=DIST_MATRIX)
    time_improved = time.time() - t0

    qaoa_log_improved = list(QAOA_LOG)
    qaoa_stats_improved = dict(QAOA_STATS)
    super_dist_log = list(_SUPER_DIST_LOG)
    print(f"\n  Improved solver completed in {time_improved:.1f}s")
    print(f"  Total distance: {sol_improved.total_distance:.2f}")

    # ── Phase 3: Run classical baselines ──────────────────────────────────────
    print("\n" + "─" * 80)
    print("Phase 3: Running classical baselines...")
    print("─" * 80)
    QAOA_VERBOSE = False

    baseline_results = {}

    print("  Running Nearest-Neighbour...")
    t0 = time.time()
    sol_nn = algo_nn(nodes, K, depot, DIST_MATRIX)
    baseline_results['Nearest-Neighbour'] = (sol_nn, time.time() - t0)

    print("  Running Clarke-Wright Savings...")
    t0 = time.time()
    sol_cw = algo_savings(nodes, K, depot, DIST_MATRIX)
    baseline_results['Clarke-Wright'] = (sol_cw, time.time() - t0)

    print("  Running Sweep...")
    t0 = time.time()
    sol_sweep = algo_sweep(nodes, K, depot, DIST_MATRIX)
    baseline_results['Sweep'] = (sol_sweep, time.time() - t0)

    print("  Running OR-Tools (GLS, 10s limit)...")
    t0 = time.time()
    sol_ort = algo_ortools(nodes, K, depot, DIST_MATRIX)
    baseline_results['OR-Tools (GLS)'] = (sol_ort, time.time() - t0)


    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════════════

    # ── TABLE 1: Super-node distance comparison ──────────────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 1: SUPER-NODE DISTANCE COMPARISON (Centroid vs Real)")
    print("=" * 100)

    if super_dist_log:
        for entry_idx, entry in enumerate(super_dist_log):
            print(f"\n  Recursion depth {entry['depth']}, {entry['n_super']} super-nodes, {entry['n_nodes']} nodes")
            cent = entry['centroid']
            real = entry['real']

            # Collect all non-self pairs
            pairs = [(k, cent[k], real[k]) for k in cent if k[0] != k[1]]
            if not pairs:
                print("    (no non-self pairs)")
                continue

            print(f"  {'Pair (A → B)':>20s}  {'Centroid':>12s}  {'Real (min)':>12s}  {'Diff':>10s}  {'Diff%':>8s}")
            print("  " + "-" * 68)

            total_cent = 0; total_real = 0; n_different = 0
            for (a_id, b_id), c_dist, r_dist in sorted(pairs, key=lambda x: abs(x[1]-x[2]), reverse=True):
                diff = c_dist - r_dist
                diff_pct = diff / c_dist * 100 if c_dist > 0 else 0
                total_cent += c_dist; total_real += r_dist
                if abs(diff) > 1e-6: n_different += 1
                print(f"  {f'({a_id} → {b_id})':>20s}  {c_dist:>12.2f}  {r_dist:>12.2f}  {diff:>10.2f}  {diff_pct:>7.1f}%")

            print("  " + "-" * 68)
            overall_diff = total_cent - total_real
            overall_pct = overall_diff / total_cent * 100 if total_cent > 0 else 0
            print(f"  {'SUM':>20s}  {total_cent:>12.2f}  {total_real:>12.2f}  {overall_diff:>10.2f}  {overall_pct:>7.1f}%")
            print(f"  Pairs where centroid ≠ real: {n_different}/{len(pairs)}")
    else:
        print("  No super-dist log entries recorded.")


    # ── TABLE 2: QAOA Leaf Performance Summary ───────────────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 2: QAOA LEAF PERFORMANCE — BASE vs IMPROVED")
    print("=" * 100)

    for label, log in [("BASE (centroid)", qaoa_log_base), ("IMPROVED (real dist)", qaoa_log_improved)]:
        n_total = len(log)
        n_optimal = sum(1 for e in log if e["outcome"] == "optimal")
        n_feasible = sum(1 for e in log if e["outcome"] == "feasible")
        n_nonvalid = sum(1 for e in log if e["outcome"] == "non-valid")
        print(f"\n  [{label}]  Total: {n_total}  Optimal: {n_optimal}  "
              f"Feasible: {n_feasible}  Non-valid: {n_nonvalid}")

    # ── TABLE 3: Per-route comparison ────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 3: PER-ROUTE COMPARISON — BASE (centroid) vs IMPROVED (real dist)")
    print("=" * 100)
    print(f"\n  {'Route':>6s}  {'Base Dist':>12s}  {'Improved Dist':>14s}  {'Change':>10s}  {'Change%':>8s}  {'Nodes(B)':>9s}  {'Nodes(I)':>9s}")
    print("  " + "-" * 75)

    # Align by route index
    n_routes = max(len(sol_base.routes), len(sol_improved.routes))
    total_base = 0; total_improv = 0
    for ri in range(n_routes):
        d_b = sol_base.routes[ri].distance if ri < len(sol_base.routes) else 0
        d_i = sol_improved.routes[ri].distance if ri < len(sol_improved.routes) else 0
        n_b = len(sol_base.routes[ri].node_ids) if ri < len(sol_base.routes) else 0
        n_i = len(sol_improved.routes[ri].node_ids) if ri < len(sol_improved.routes) else 0
        change = d_b - d_i
        change_pct = change / d_b * 100 if d_b > 0 else 0
        total_base += d_b; total_improv += d_i
        print(f"  {ri+1:>6d}  {d_b:>12.2f}  {d_i:>14.2f}  {change:>10.2f}  {change_pct:>7.1f}%  {n_b:>9d}  {n_i:>9d}")
    print("  " + "-" * 75)
    total_change = total_base - total_improv
    total_change_pct = total_change / total_base * 100 if total_base > 0 else 0
    print(f"  {'TOTAL':>6s}  {total_base:>12.2f}  {total_improv:>14.2f}  {total_change:>10.2f}  {total_change_pct:>7.1f}%")


    # ── TABLE 4: Full benchmark ──────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 4: FULL BENCHMARK — ALL ALGORITHMS")
    print(f"Instance: RioClaroPostToy_50_0  |  n=50  |  k={K}")
    print("=" * 100)

    all_results = {
        'QAOA (centroid, base)':  (sol_base,     time_base),
        'QAOA (real dist, impr)': (sol_improved,  time_improved),
    }
    all_results.update(baseline_results)

    for name, (sol, _) in all_results.items():
        v = validate_solution(sol, nodes, depot, K, DIST_MATRIX)
        if not v['valid']:
            print(f"  !! {name}: INVALID -- {v['issues']}")

    best_dist = min(sol.total_distance for sol, _ in all_results.values()
                    if validate_solution(sol, nodes, depot, K, DIST_MATRIX)['valid'])

    print(f"\n  {'Algorithm':<26s}  {'k':>4s}  {'Distance':>12s}  {'Std (σ)':>10s}  {'Time(s)':>10s}  {'Valid':>6s}  {'vs Best':>10s}")
    print("  " + "-" * 88)
    for name in ['QAOA (centroid, base)', 'QAOA (real dist, impr)',
                 'Nearest-Neighbour', 'Clarke-Wright', 'Sweep', 'OR-Tools (GLS)']:
        sol, elapsed = all_results[name]
        v = validate_solution(sol, nodes, depot, K, DIST_MATRIX)
        gap = (sol.total_distance - best_dist) / best_dist * 100 if best_dist > 0 else 0
        marker = ' ***' if sol.total_distance == best_dist and v['valid'] else ''
        status = 'OK' if v['valid'] else 'FAIL'
        print(f"  {name:<26s}  {sol.num_vehicles_used:>4d}  {sol.total_distance:>12.2f}  "
              f"{sol.distance_std:>10.2f}  {elapsed:>10.2f}  {status:>6s}  {gap:>9.1f}%{marker}")


    # ── TABLE 5: Improvement summary ─────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 5: REAL SUPER-DISTANCE IMPROVEMENT SUMMARY")
    print("=" * 100)
    improv_abs = sol_base.total_distance - sol_improved.total_distance
    improv_pct = improv_abs / sol_base.total_distance * 100 if sol_base.total_distance > 0 else 0
    print(f"\n  QAOA base distance (centroid):     {sol_base.total_distance:>12.2f}")
    print(f"  QAOA improved distance (real):     {sol_improved.total_distance:>12.2f}")
    print(f"  Absolute improvement:              {improv_abs:>12.2f}")
    print(f"  Relative improvement:              {improv_pct:>11.1f}%")
    print(f"  Base solve time:                   {time_base:>11.1f}s")
    print(f"  Improved solve time:               {time_improved:>11.1f}s")
    print(f"  Base fairness (σ):                 {sol_base.distance_std:>12.2f}")
    print(f"  Improved fairness (σ):             {sol_improved.distance_std:>12.2f}")
    print(f"  Base QAOA success/fallback:        {qaoa_stats_base['success']}/{qaoa_stats_base['fallback']}")
    print(f"  Improved QAOA success/fallback:    {qaoa_stats_improved['success']}/{qaoa_stats_improved['fallback']}")

    if improv_abs > 0:
        print(f"\n  VERDICT: Real inter-cluster distances IMPROVED the solution by {improv_pct:.1f}%")
    elif improv_abs < 0:
        print(f"\n  VERDICT: Real inter-cluster distances made the solution WORSE by {-improv_pct:.1f}%")
    else:
        print(f"\n  VERDICT: No change (the centroid distances happened to be accurate)")


    # ═══════════════════════════════════════════════════════════════════════════
    # COMPREHENSIVE FINAL SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n\n" + "█" * 100)
    print("█" + " " * 98 + "█")
    print("█" + "   COMPREHENSIVE RESULTS SUMMARY — EXPERIMENT 2".center(98) + "█")
    print("█" + f"   Instance: RioClaroPostToy_50_0  |  n={N_NODES}  |  k={K}  |  Leaf={LEAF_SIZE}".center(98) + "█")
    print("█" + " " * 98 + "█")
    print("█" * 100)

    # ── Section A: Algorithm Ranking ──────────────────────────────────────────
    print("\n┌" + "─" * 98 + "┐")
    print("│" + "  A. ALGORITHM RANKING (sorted by total distance)".ljust(98) + "│")
    print("├" + "─" * 98 + "┤")

    ranked = sorted(all_results.items(), key=lambda x: x[1][0].total_distance)
    best_valid_dist = ranked[0][1][0].total_distance

    hdr = f"│  {'Rank':<5s}  {'Algorithm':<28s}  {'Distance':>12s}  {'σ (fairness)':>13s}  {'Time(s)':>10s}  {'Gap vs Best':>12s}  │"
    print(hdr)
    print("│  " + "─" * 94 + "  │")
    for rank, (name, (sol, elapsed)) in enumerate(ranked, 1):
        gap = (sol.total_distance - best_valid_dist) / best_valid_dist * 100 if best_valid_dist > 0 else 0.0
        marker = " ★" if rank == 1 else ""
        print(f"│  {rank:<5d}  {name:<28s}  {sol.total_distance:>12.2f}  {sol.distance_std:>13.2f}  "
              f"{elapsed:>10.2f}  {gap:>11.1f}%{marker}  │")

    print("├" + "─" * 98 + "┤")

    # ── Section B: QAOA Centroid vs Real — Head-to-Head ──────────────────────
    print("│" + "  B. QAOA HEAD-TO-HEAD: Centroid vs Real Super-Distances".ljust(98) + "│")
    print("├" + "─" * 98 + "┤")

    metrics = [
        ("Total Distance",      f"{sol_base.total_distance:.2f}",     f"{sol_improved.total_distance:.2f}"),
        ("Fairness (σ)",        f"{sol_base.distance_std:.2f}",       f"{sol_improved.distance_std:.2f}"),
        ("Solve Time (s)",      f"{time_base:.1f}",                   f"{time_improved:.1f}"),
        ("QAOA Successes",      str(qaoa_stats_base['success']),      str(qaoa_stats_improved['success'])),
        ("QAOA Fallbacks",      str(qaoa_stats_base['fallback']),     str(qaoa_stats_improved['fallback'])),
        ("Num Routes",          str(sol_base.num_vehicles_used),      str(sol_improved.num_vehicles_used)),
    ]

    # Add per-route min/max
    base_dists = [r.distance for r in sol_base.routes]
    impr_dists = [r.distance for r in sol_improved.routes]
    metrics.append(("Shortest Route",  f"{min(base_dists):.2f}", f"{min(impr_dists):.2f}"))
    metrics.append(("Longest Route",   f"{max(base_dists):.2f}", f"{max(impr_dists):.2f}"))
    metrics.append(("Route Range",     f"{max(base_dists)-min(base_dists):.2f}",
                                        f"{max(impr_dists)-min(impr_dists):.2f}"))

    print(f"│  {'Metric':<28s}  {'Centroid (Base)':>18s}  {'Real Dist (Improved)':>22s}  {'Winner':>18s}  │")
    print("│  " + "─" * 94 + "  │")
    for metric_name, base_val, impr_val in metrics:
        # Determine winner
        try:
            b = float(base_val)
            i = float(impr_val)
            if metric_name == "QAOA Successes":
                winner = "Centroid" if b > i else ("Real" if i > b else "Tie")
            elif metric_name == "QAOA Fallbacks":
                winner = "Centroid" if b < i else ("Real" if i < b else "Tie")
            else:
                winner = "Centroid" if b < i else ("Real" if i < b else "Tie")
        except ValueError:
            winner = "—"
        print(f"│  {metric_name:<28s}  {base_val:>18s}  {impr_val:>22s}  {winner:>18s}  │")

    print("├" + "─" * 98 + "┤")

    # ── Section C: Super-Distance Accuracy ───────────────────────────────────
    print("│" + "  C. SUPER-DISTANCE ACCURACY ANALYSIS".ljust(98) + "│")
    print("├" + "─" * 98 + "┤")

    if super_dist_log:
        print(f"│  {'Level':<10s}  {'#Super':>7s}  {'#Nodes':>7s}  {'Centroid Σ':>14s}  {'Real Σ':>14s}  {'Error':>10s}  {'Error%':>8s}  {'Direction':>12s}  │")
        print("│  " + "─" * 94 + "  │")

        total_cent_all = 0; total_real_all = 0
        for entry in super_dist_log:
            c_sum = sum(v for (a, b), v in entry['centroid'].items() if a != b)
            r_sum = sum(v for (a, b), v in entry['real'].items() if a != b)
            err = c_sum - r_sum
            err_pct = err / c_sum * 100 if c_sum > 0 else 0
            direction = "Overest." if err > 0 else ("Underest." if err < 0 else "Exact")
            total_cent_all += c_sum; total_real_all += r_sum
            label = f"d={entry['depth']}"
            print(f"│  {label:<10s}  {entry['n_super']:>7d}  {entry['n_nodes']:>7d}  "
                  f"{c_sum:>14.1f}  {r_sum:>14.1f}  {err:>10.1f}  {err_pct:>7.1f}%  {direction:>12s}  │")

        print("│  " + "─" * 94 + "  │")
        total_err = total_cent_all - total_real_all
        total_err_pct = total_err / total_cent_all * 100 if total_cent_all > 0 else 0
        n_over = sum(1 for e in super_dist_log
                     if sum(v for (a,b), v in e['centroid'].items() if a!=b)
                      > sum(v for (a,b), v in e['real'].items() if a!=b))
        n_under = sum(1 for e in super_dist_log
                      if sum(v for (a,b), v in e['centroid'].items() if a!=b)
                       < sum(v for (a,b), v in e['real'].items() if a!=b))
        print(f"│  {'TOTAL':<10s}  {'':>7s}  {'':>7s}  "
              f"{total_cent_all:>14.1f}  {total_real_all:>14.1f}  {total_err:>10.1f}  {total_err_pct:>7.1f}%  {'':>12s}  │")
        print(f"│  Overestimates: {n_over}/{len(super_dist_log)}   |   "
              f"Underestimates: {n_under}/{len(super_dist_log)}   |   "
              f"Exact: {len(super_dist_log)-n_over-n_under}/{len(super_dist_log)}".ljust(98) + "│")
    else:
        print("│  No super-distance log entries recorded.".ljust(98) + "│")

    print("├" + "─" * 98 + "┤")

    # ── Section D: QAOA Leaf Performance Comparison ──────────────────────────
    print("│" + "  D. QAOA LEAF PERFORMANCE COMPARISON".ljust(98) + "│")
    print("├" + "─" * 98 + "┤")

    for label, log, stats in [("Centroid (Base)", qaoa_log_base, qaoa_stats_base),
                               ("Real Dist (Improved)", qaoa_log_improved, qaoa_stats_improved)]:
        n_total = len(log)
        n_optimal = sum(1 for e in log if e["outcome"] == "optimal")
        n_feasible = sum(1 for e in log if e["outcome"] == "feasible")
        n_nonvalid = sum(1 for e in log if e["outcome"] == "non-valid")
        opt_rate = n_optimal / n_total * 100 if n_total > 0 else 0
        avg_gap = np.mean([e["gap_pct"] for e in log if e["gap_pct"] is not None and e["outcome"] != "non-valid"]) if log else 0
        avg_vf = np.mean([e["valid_frac"] for e in log if e["outcome"] != "non-valid"]) * 100 if log else 0

        print(f"│  [{label}]".ljust(98) + "│")
        print(f"│    Leaves: {n_total}  |  Optimal: {n_optimal} ({opt_rate:.0f}%)  |  "
              f"Feasible: {n_feasible}  |  Non-valid: {n_nonvalid}  |  "
              f"Avg gap: {avg_gap:.2f}%  |  Avg valid%: {avg_vf:.1f}%".ljust(98) + "│")

    print("├" + "─" * 98 + "┤")

    # ── Section E: QAOA vs Classical ─────────────────────────────────────────
    print("│" + "  E. QAOA vs CLASSICAL — COMPETITIVE ANALYSIS".ljust(98) + "│")
    print("├" + "─" * 98 + "┤")

    qaoa_best_dist = min(sol_base.total_distance, sol_improved.total_distance)
    qaoa_best_label = "Centroid" if sol_base.total_distance <= sol_improved.total_distance else "Real Dist"
    classical_best_name = min(baseline_results.items(), key=lambda x: x[1][0].total_distance)[0]
    classical_best_dist = baseline_results[classical_best_name][0].total_distance

    gap_vs_classical = (qaoa_best_dist - classical_best_dist) / classical_best_dist * 100 if classical_best_dist > 0 else 0

    print(f"│  Best QAOA variant:       {qaoa_best_label:<16s}  distance = {qaoa_best_dist:>12.2f}".ljust(98) + "│")
    print(f"│  Best classical:          {classical_best_name:<16s}  distance = {classical_best_dist:>12.2f}".ljust(98) + "│")
    print(f"│  QAOA gap vs classical:   {gap_vs_classical:>+.1f}%".ljust(98) + "│")

    # Which algorithms QAOA beats?
    qaoa_beats = [name for name, (sol, _) in baseline_results.items()
                  if qaoa_best_dist < sol.total_distance]
    qaoa_loses = [name for name, (sol, _) in baseline_results.items()
                  if qaoa_best_dist >= sol.total_distance]
    print(f"│  QAOA beats:              {', '.join(qaoa_beats) if qaoa_beats else 'None'}".ljust(98) + "│")
    print(f"│  QAOA loses to:           {', '.join(qaoa_loses) if qaoa_loses else 'None'}".ljust(98) + "│")

    print("├" + "─" * 98 + "┤")

    # ── Section F: Conclusion ────────────────────────────────────────────────
    print("│" + "  F. CONCLUSION".ljust(98) + "│")
    print("├" + "─" * 98 + "┤")

    # Determine real vs centroid verdict
    if improv_abs > 1e-3:
        dist_verdict = (f"Real inter-cluster distances IMPROVED total distance by {improv_pct:.1f}% "
                        f"({improv_abs:.2f} units).")
    elif improv_abs < -1e-3:
        dist_verdict = (f"Real inter-cluster distances made the solution WORSE by {-improv_pct:.1f}% "
                        f"({-improv_abs:.2f} units). Centroid approximation was sufficient.")
    else:
        dist_verdict = "No meaningful difference between centroid and real distances."

    # Fairness comparison
    base_fair = sol_base.distance_std
    impr_fair = sol_improved.distance_std
    if base_fair < impr_fair:
        fair_verdict = f"Centroid yields better route fairness (σ={base_fair:.1f} vs {impr_fair:.1f})."
    elif impr_fair < base_fair:
        fair_verdict = f"Real distances yield better route fairness (σ={impr_fair:.1f} vs {base_fair:.1f})."
    else:
        fair_verdict = "Both methods have equivalent fairness."

    conclusions = [
        dist_verdict,
        fair_verdict,
        f"Best overall algorithm: {ranked[0][0]} (distance={ranked[0][1][0].total_distance:.2f}).",
        f"QAOA ({qaoa_best_label}) ranks #{[name for name, _ in ranked].index('QAOA (centroid, base)' if qaoa_best_label == 'Centroid' else 'QAOA (real dist, impr)')+1} of {len(ranked)} algorithms.",
        f"QAOA beats {len(qaoa_beats)}/{len(baseline_results)} classical baselines: {', '.join(qaoa_beats) if qaoa_beats else 'none'}.",
    ]

    # Super-dist insight
    if super_dist_log:
        n_over = sum(1 for e in super_dist_log
                     if sum(v for (a,b), v in e['centroid'].items() if a!=b)
                      > sum(v for (a,b), v in e['real'].items() if a!=b))
        conclusions.append(
            f"Centroid overestimates inter-cluster distances in {n_over}/{len(super_dist_log)} recursion levels "
            f"(net error: {total_err_pct:.1f}%)."
        )

    for i, line in enumerate(conclusions, 1):
        print(f"│  {i}. {line}".ljust(98) + "│")

    print("└" + "─" * 98 + "┘")

    # Save results
    results_json = {
        "experiment": "Real inter-cluster super-distances",
        "instance": "RioClaroPostToy_50_0",
        "n_nodes": N_NODES, "k": K, "leaf_size": LEAF_SIZE,
        "algorithms": {}
    }
    for name, (sol, elapsed) in all_results.items():
        v = validate_solution(sol, nodes, depot, K, DIST_MATRIX)
        results_json["algorithms"][name] = {
            "total_distance": sol.total_distance,
            "distance_std": sol.distance_std,
            "num_routes": sol.num_vehicles_used,
            "time_seconds": elapsed,
            "valid": v['valid'],
            "routes": [{"node_ids": list(r.node_ids), "distance": r.distance} for r in sol.routes]
        }
    if super_dist_log:
        results_json["super_dist_comparison"] = []
        for entry in super_dist_log:
            results_json["super_dist_comparison"].append({
                "depth": entry["depth"],
                "n_super": entry["n_super"],
                "n_nodes": entry["n_nodes"],
                "centroid_total": sum(v for (a,b), v in entry["centroid"].items() if a != b),
                "real_total": sum(v for (a,b), v in entry["real"].items() if a != b),
            })

    out_path = os.path.join(os.path.dirname(__file__), 'exp2_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    print("\n" + "=" * 100)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 100)
