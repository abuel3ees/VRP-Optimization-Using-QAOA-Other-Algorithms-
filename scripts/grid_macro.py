#!/usr/bin/env python3
"""
Experiment 7: Grid-Based Macro-Routing with QAOA
==================================================
Instead of the K-Match / angular-sweep clustering used in exp1–6,
this experiment tests a **grid-based spatial decomposition**:

1. Overlay a grid on the geographic space of all customers.
2. Each non-empty grid cell becomes a "supernode" (centroid + summed demand).
3. Solve the macro-level VRP on supernodes with QAOA.
4. For each vehicle route through supernodes, solve the intra-cell TSP
   (visiting individual customers inside each cell) using QAOA.
5. Concatenate the intra-cell orderings into final routes.

Comparison against:
  - K-Match hierarchical QAOA (exp3)
  - Classical baselines: Clarke-Wright, Nearest-Neighbour, Sweep, OR-Tools

Key question:
  Does a simple, geometry-only grid decomposition produce competitive
  VRP solutions compared to the recursive K-Match hierarchy, and at
  what qubit cost?

Grid sizes tested:
  A. 3×3  (up to 9 supernodes  → macro-QAOA on ~9 nodes)
  B. 4×4  (up to 16 supernodes → macro-QAOA on ~16 nodes)
  C. 5×5  (up to 25 supernodes → macro-QAOA on ~25 nodes)
"""

import sys, os, time, math, itertools, json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

QAOA_VERBOSE = False

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES (same as exp3 for consistency)
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

def nearest_neighbor_route(node_ids, depot_id, dist_map):
    if not node_ids: return []
    unvisited = set(node_ids); route = []; current = depot_id
    while unvisited:
        nearest = min(unvisited, key=lambda nid: dist_map[(current, nid)])
        route.append(nearest); unvisited.remove(nearest); current = nearest
    return route

# ═══════════════════════════════════════════════════════════════════════════════
# GRID-BASED CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

def grid_clustering(nodes, depot, grid_rows, grid_cols):
    """
    Assign each customer node to a grid cell based on geographic position.

    Returns:
        cells: Dict[Tuple[int,int], List[Node]] — mapping from cell (r,c) to nodes
        cell_info: Dict[Tuple[int,int], dict] — centroid, demand, etc.
    """
    if not nodes:
        return {}, {}

    x_coords = [n.x for n in nodes]
    y_coords = [n.y for n in nodes]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Add small epsilon to avoid zero-size cells
    eps = 1e-6
    cell_width = (x_max - x_min + eps) / grid_cols
    cell_height = (y_max - y_min + eps) / grid_rows

    cells = defaultdict(list)
    for n in nodes:
        ci = min(int((n.x - x_min) / cell_width), grid_cols - 1)
        cj = min(int((n.y - y_min) / cell_height), grid_rows - 1)
        cells[(ci, cj)].append(n)

    # Remove empty cells and build info
    cells = {k: v for k, v in cells.items() if v}
    cell_info = {}
    for cell_key, cell_nodes in cells.items():
        cx = np.mean([n.x for n in cell_nodes])
        cy = np.mean([n.y for n in cell_nodes])
        total_demand = sum(n.demand for n in cell_nodes)
        cell_info[cell_key] = {
            'centroid': (cx, cy),
            'demand': total_demand,
            'n_customers': len(cell_nodes),
            'node_ids': [n.id for n in cell_nodes],
        }

    return cells, cell_info


# ═══════════════════════════════════════════════════════════════════════════════
# QAOA SOLVER (same as exp3 — for leaf sub-problems)
# ═══════════════════════════════════════════════════════════════════════════════

QAOA_STATS = {"success": 0, "fallback": 0}
QAOA_LOG = []
LEAF_SIZE = 4   # max nodes per direct QAOA call

def _classical_optimal_cost(node_ids, eff_k, depot_id, dist_map):
    if eff_k == 1:
        best = float("inf")
        for perm in itertools.permutations(node_ids):
            d = route_distance(list(perm), depot_id, dist_map)
            if d < best: best = d
        return best
    def _parts(ids, k):
        ids_l = list(ids); n = len(ids_l)
        if k == 1: yield (tuple(ids_l),); return
        if n == k: yield tuple((x,) for x in ids_l); return
        if n < k or k <= 0: return
        first = ids_l[0]; rest = ids_l[1:]
        for sub in _parts(tuple(rest), k - 1): yield ((first,),) + sub
        for sub in _parts(tuple(rest), k):
            for i in range(len(sub)): yield sub[:i] + ((first,) + sub[i],) + sub[i+1:]
    best = float("inf")
    for part in _parts(tuple(node_ids), eff_k):
        if len(part) != eff_k: continue
        for combo in itertools.product(*[itertools.permutations(b) for b in part]):
            total = sum(route_distance(list(p), depot_id, dist_map) for p in combo)
            if total < best: best = total
    return best


def solve_qaoa_leaf(nodes, k, depot, dist_map):
    """QAOA-based solver for small sub-problems (len(nodes) <= LEAF_SIZE)."""
    if not nodes:
        return VRPSolution(routes=[])

    node_ids = [n.id for n in nodes]
    node_map = {n.id: n for n in nodes}
    eff_k = min(k, len(node_ids))

    _PREFIX = f"  [QAOA leaf n={len(node_ids)} k={eff_k}]"

    def _classical_fallback(reason=""):
        QAOA_STATS["fallback"] += 1
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
            ids_l = list(ids); n = len(ids_l)
            if k == 1: yield (tuple(ids_l),); return
            if n == k: yield tuple((x,) for x in ids_l); return
            if n < k or k <= 0: return
            first = ids_l[0]; rest = ids_l[1:]
            for sub in _parts(tuple(rest), k - 1): yield ((first,),) + sub
            for sub in _parts(tuple(rest), k):
                for i in range(len(sub)): yield sub[:i] + ((first,) + sub[i],) + sub[i+1:]
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
    except ImportError as e:
        _sol = _classical_fallback(f"qiskit not installed: {e}")
        QAOA_LOG.append({
            "node_ids": node_ids, "k": eff_k, "n_qubits": 0,
            "outcome": "non-valid", "qaoa_cost": float("inf"),
            "optimal_cost": None, "gap_pct": None,
            "valid_frac": 0.0, "n_valid_unique": 0,
            "best_prob_rank": -1, "solver_used": "classical_fallback",
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
    _DECODE_K = 10
    _SEED     = 7

    import hashlib as _hl
    _leaf_seed = (_SEED + int(_hl.md5(str(sorted(node_ids)).encode()).hexdigest(), 16)) % (2**31)
    _ag.random_seed = _leaf_seed
    _np = np
    _np.random.seed(_leaf_seed)

    # ═══════════════════════════════════════════════════════════════════════
    # POSITION-INDEXED QUBO (k=1 TSP, subtour-free by construction)
    # Variables: p_{i}_{t}  = 1 if customer i (0-indexed) at position t
    # ═══════════════════════════════════════════════════════════════════════
    _N = len(node_ids)
    all_ids = [depot.id] + node_ids
    m1 = len(all_ids)

    dist_mat = _np.zeros((m1, m1), dtype=float)
    for qi, a in enumerate(all_ids):
        for qj, b in enumerate(all_ids):
            if a != b and (a, b) in dist_map:
                dist_mat[qi, qj] = dist_map[(a, b)]

    _gmax = float(GLOBAL_MAX_DIST) if GLOBAL_MAX_DIST > 0 else (
        float(_np.max(dist_mat)) if _np.max(dist_mat) > 0 else 1.0)
    dist_norm = dist_mat / _gmax

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

    if n_qubits > 20:
        _sol = _classical_fallback(f"n_qubits={n_qubits} > 20")
        QAOA_LOG.append({
            "node_ids": node_ids, "k": eff_k, "n_qubits": n_qubits,
            "outcome": "non-valid", "qaoa_cost": float("inf"),
            "optimal_cost": None, "gap_pct": None,
            "valid_frac": 0.0, "n_valid_unique": 0,
            "best_prob_rank": -1, "solver_used": "classical_fallback",
        })
        return _sol

    var_names = [v.name for v in qubo.variables]
    _pos_var_map = {}  # var_index -> (customer_idx, position)
    for vi, vn in enumerate(var_names):
        if vn.startswith("p_"):
            parts = vn.split("_")
            _pos_var_map[vi] = (int(parts[1]), int(parts[2]))

    ising_op, offset = qubo.to_ising()
    raw      = _np.array([abs(c) for _, c in ising_op.to_list()], dtype=float)
    op_scale = float(_np.max(raw)) if len(raw) > 0 and _np.max(raw) > 0 else 1.0
    ising_n  = ising_op / op_scale

    # ── decode helper (position-indexed) ──────────────────────────────────
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

            p = _np.zeros((_N, _N), dtype=int)
            for vi, (ci, ti) in _pos_var_map.items():
                if vi < len(bf):
                    p[ci, ti] = bf[vi]

            if not (_np.all(p.sum(axis=1) == 1) and _np.all(p.sum(axis=0) == 1)):
                continue

            perm = tuple(int(_np.argmax(p[:, t])) for t in range(_N))
            route_ids = [all_ids[pi + 1] for pi in perm]
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

    # ═══════════════════════════════════════════════════════════════════════
    # QAOA EXECUTION (Estimator-based, matching exp3)
    # ═══════════════════════════════════════════════════════════════════════
    backend_sv = _AerSim(method="statevector", seed_simulator=_leaf_seed)
    est        = _AerEst(run_options={"seed_simulator": _leaf_seed})

    all_results = []

    for ci, cfg in enumerate(_CONFIGS, 1):
        reps       = cfg["reps"]
        opt_name   = cfg["optimizer"].upper()
        maxiter    = cfg["maxiter"]
        n_restarts = cfg["restarts"]

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

        circ  = tqc.copy()
        if not circ.cregs: circ.measure_all()
        bound  = circ.assign_parameters(best_res.x, inplace=False)
        counts = backend_sv.run(bound, shots=_SHOTS).result().get_counts()

        topk_cost, mp_cost, global_cost, topk_result, vf, n_uniq, prob_rank = _decode_counts(counts)
        all_results.append((topk_cost, mp_cost, global_cost, topk_result, vf, n_uniq, prob_rank))

    # ── pick best across configs ──────────────────────────────────────────
    best_topk, best_mp, best_global, best_result, best_vf, best_n_uniq, best_prob_rank = \
        min(all_results, key=lambda t: t[0])

    opt_cost = _classical_optimal_cost(node_ids, eff_k, depot.id, dist_map)

    if best_result is not None:
        best_cost = route_distance(best_result, depot.id, dist_map)
        gap = (best_cost - opt_cost) / opt_cost * 100 if opt_cost > 0 else 0

        if abs(gap) < 0.01:
            outcome = "optimal"
        else:
            outcome = "feasible"

        QAOA_STATS["success"] += 1
        QAOA_LOG.append({
            "node_ids": node_ids, "k": eff_k, "n_qubits": n_qubits,
            "outcome": outcome, "qaoa_cost": best_cost, "optimal_cost": opt_cost,
            "gap_pct": round(gap, 4), "valid_frac": round(best_vf, 6),
            "n_valid_unique": best_n_uniq, "best_prob_rank": best_prob_rank,
            "solver_used": "QAOA",
        })

        load = sum(node_map[nid].demand for nid in best_result)
        return VRPSolution(routes=[Route(node_ids=list(best_result),
                                         distance=best_cost, load=load)])
    else:
        _sol = _classical_fallback("no valid QAOA solution found")
        QAOA_LOG.append({
            "node_ids": node_ids, "k": eff_k, "n_qubits": n_qubits,
            "outcome": "non-valid", "qaoa_cost": float("inf"),
            "optimal_cost": opt_cost, "gap_pct": None,
            "valid_frac": 0.0, "n_valid_unique": 0,
            "best_prob_rank": -1, "solver_used": "classical_fallback",
        })
        return _sol


# ═══════════════════════════════════════════════════════════════════════════════
# GRID-BASED VRP SOLVER (THE EXPERIMENT)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_grid_vrp(nodes, k, depot, dist_map, dist_matrix, grid_rows, grid_cols):
    """
    Grid-based VRP solver:
    1. Grid-cluster all nodes
    2. Build supernodes (one per cell)
    3. Solve macro-routing: assign supernodes to k vehicles using QAOA
    4. For each vehicle route, solve intra-cell TSP using QAOA
    5. Concatenate into final routes
    """
    t0 = time.time()
    node_map = {n.id: n for n in nodes}

    # ── Step 1: Grid clustering ───────────────────────────────────────────
    cells, cell_info = grid_clustering(nodes, depot, grid_rows, grid_cols)
    n_cells = len(cells)
    cell_keys = sorted(cells.keys())

    print(f"    Grid: {grid_rows}×{grid_cols} → {n_cells} non-empty cells")
    for ck in cell_keys:
        ci = cell_info[ck]
        print(f"      Cell {ck}: {ci['n_customers']} nodes, "
              f"demand={ci['demand']:.0f}, "
              f"centroid=({ci['centroid'][0]:.0f},{ci['centroid'][1]:.0f})")

    # ── Step 2: Create supernodes ─────────────────────────────────────────
    # Supernode IDs: 9000, 9001, ...  (avoid collision with real node IDs)
    super_nodes = []
    sn_to_cell = {}
    for idx, ck in enumerate(cell_keys):
        ci = cell_info[ck]
        sn_id = 9000 + idx
        sn = Node(id=sn_id, x=ci['centroid'][0], y=ci['centroid'][1],
                  demand=ci['demand'])
        super_nodes.append(sn)
        sn_to_cell[sn_id] = ck

    # Build supernode distance map using minimum real inter-cell distance
    sn_dist_map = {}
    all_sn_plus_depot = [depot] + super_nodes
    for a in all_sn_plus_depot:
        for b in all_sn_plus_depot:
            if a.id == b.id:
                sn_dist_map[(a.id, b.id)] = 0.0
                continue
            # Get real node IDs for each
            if a.id == depot.id:
                ids_a = [depot.id]
            else:
                ck_a = sn_to_cell[a.id]
                ids_a = cell_info[ck_a]['node_ids']
            if b.id == depot.id:
                ids_b = [depot.id]
            else:
                ck_b = sn_to_cell[b.id]
                ids_b = cell_info[ck_b]['node_ids']

            min_d = float('inf')
            for na_id in ids_a:
                for nb_id in ids_b:
                    d = dist_map.get((na_id, nb_id), float('inf'))
                    if d < min_d: min_d = d
            sn_dist_map[(a.id, b.id)] = min_d

    print(f"    Supernodes: {n_cells} (+ depot)")
    print(f"    Max supernode dist: {max(sn_dist_map.values()):.0f}")

    # ── Step 3: Assign supernodes to vehicles ─────────────────────────────
    # Use angular sweep to partition supernodes into k groups
    # (Like the classical baseline, but on supernodes)
    def sn_angle(sn):
        return math.atan2(sn.y - depot.y, sn.x - depot.x)

    sorted_sns = sorted(super_nodes, key=sn_angle)
    eff_k = min(k, n_cells)
    base_sz = n_cells // eff_k
    remainder = n_cells % eff_k
    vehicle_groups = []
    idx = 0
    for v in range(eff_k):
        sz = base_sz + (1 if v < remainder else 0)
        vehicle_groups.append(sorted_sns[idx:idx+sz])
        idx += sz

    print(f"\n    Vehicle assignments ({eff_k} vehicles):")
    for vi, group in enumerate(vehicle_groups):
        sn_ids = [sn.id for sn in group]
        n_real = sum(len(cells[sn_to_cell[sid]]) for sid in sn_ids)
        print(f"      V{vi+1}: {len(group)} supernodes ({n_real} real nodes) — "
              f"SN IDs: {sn_ids}")

    # ── Step 4: For each vehicle, solve intra-cell TSPs with QAOA ─────────
    routes = []

    for vi, group in enumerate(vehicle_groups):
        # Collect all real nodes for this vehicle
        vehicle_real_nodes = []
        for sn in group:
            ck = sn_to_cell[sn.id]
            vehicle_real_nodes.extend(cells[ck])

        if not vehicle_real_nodes:
            routes.append(Route(node_ids=[], distance=0.0, load=0.0))
            continue

        print(f"\n    V{vi+1}: solving TSP for {len(vehicle_real_nodes)} nodes...")

        # If small enough, solve directly with QAOA
        if len(vehicle_real_nodes) <= LEAF_SIZE:
            sub_sol = solve_qaoa_leaf(vehicle_real_nodes, 1, depot, dist_map)
            for r in sub_sol.routes:
                routes.append(r)
        else:
            # Solve each cell independently, then concatenate in supernode order
            # Order the supernodes for this vehicle (nearest-neighbour on supernodes)
            sn_ids_for_vehicle = [sn.id for sn in group]

            # NN ordering of supernodes
            if len(sn_ids_for_vehicle) > 1:
                ordered_sn = []
                unvisited_sn = set(sn_ids_for_vehicle)
                current_sn = min(unvisited_sn,
                                 key=lambda sid: sn_dist_map[(depot.id, sid)])
                while unvisited_sn:
                    ordered_sn.append(current_sn)
                    unvisited_sn.discard(current_sn)
                    if unvisited_sn:
                        current_sn = min(unvisited_sn,
                                         key=lambda sid: sn_dist_map[(current_sn, sid)])
            else:
                ordered_sn = sn_ids_for_vehicle

            # Solve intra-cell TSP for each cell
            cell_routes = {}
            for sn_id in ordered_sn:
                ck = sn_to_cell[sn_id]
                cell_nodes = cells[ck]

                if len(cell_nodes) <= LEAF_SIZE:
                    sub_sol = solve_qaoa_leaf(cell_nodes, 1, depot, dist_map)
                    cell_routes[sn_id] = sub_sol.routes[0].node_ids if sub_sol.routes else []
                else:
                    # Cell too large for single QAOA call — subdivide recursively
                    # Split into sub-groups of LEAF_SIZE
                    cell_node_ids = [n.id for n in cell_nodes]
                    # Use nearest-neighbour ordering then chunk
                    nn_order = nearest_neighbor_route(cell_node_ids, depot.id, dist_map)
                    sub_routes = []
                    for chunk_start in range(0, len(nn_order), LEAF_SIZE):
                        chunk_ids = nn_order[chunk_start:chunk_start+LEAF_SIZE]
                        chunk_nodes = [node_map[nid] for nid in chunk_ids]
                        sub_sol = solve_qaoa_leaf(chunk_nodes, 1, depot, dist_map)
                        if sub_sol.routes:
                            sub_routes.extend(sub_sol.routes[0].node_ids)
                        else:
                            sub_routes.extend(chunk_ids)
                    cell_routes[sn_id] = sub_routes

            # Concatenate cell orderings
            combined_ids = []
            for sn_id in ordered_sn:
                seg = cell_routes.get(sn_id, [])
                # Orient segment: check if connecting from last node is better forward or reversed
                if combined_ids and seg and len(seg) > 1:
                    anchor = combined_ids[-1]
                    d_fwd = dist_map.get((anchor, seg[0]), float('inf'))
                    d_rev = dist_map.get((anchor, seg[-1]), float('inf'))
                    if d_rev < d_fwd:
                        seg = list(reversed(seg))
                combined_ids.extend(seg)

            # Apply 2-opt improvement
            combined_ids = two_opt(combined_ids, depot.id, dist_map)

            dist_val = route_distance(combined_ids, depot.id, dist_map)
            load_val = sum(node_map[nid].demand for nid in combined_ids)
            routes.append(Route(node_ids=combined_ids, distance=dist_val, load=load_val))

    elapsed = time.time() - t0
    sol = VRPSolution(routes=routes)
    sol.solve_time = elapsed
    return sol


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSICAL BASELINES (copied from exp3 for reproducibility)
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
        mid = len(route) // 2
        return route[:mid], route[mid:]
    while len(result) < k:
        idx_l = max(range(len(result)), key=lambda i: len(result[i]))
        r = result[idx_l]
        if len(r) < 2: break
        left, right = _split_one(r)
        if not right: break
        result[idx_l] = left; result.append(right)
    return result

def greedy_vrp_no_opt(nodes, k, depot, dist_map):
    if not nodes: return VRPSolution(routes=[])
    node_map_local = {n.id: n for n in nodes}; node_ids = [n.id for n in nodes]
    routes_dict = {nid: [nid] for nid in node_ids}
    route_of = {nid: nid for nid in node_ids}
    savings = sorted([(dist_map[(depot.id,i)] + dist_map[(depot.id,j)] - dist_map[(i,j)], i, j)
                      for i in node_ids for j in node_ids if i < j], reverse=True)
    def route_load(r): return sum(node_map_local[nid].demand for nid in r)
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
        final = _split_routes_to_k(final, k, len(nodes), depot_id=depot.id, dist_map=dist_map, node_map=node_map_local)
    return VRPSolution(routes=[Route(node_ids=r, distance=route_distance(r, depot.id, dist_map),
                                     load=route_load(r)) for r in final])

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
        idx_rt = routing.Start(v); route = []
        while not routing.IsEnd(idx_rt):
            node_idx = manager.IndexToNode(idx_rt)
            if node_idx != 0: route.append(all_nodes[node_idx].id)
            idx_rt = assignment.Value(routing.NextVar(idx_rt))
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
    # Try multiple possible notebook locations
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'instance_12_final_with results.ipynb'),
        os.path.join(os.path.dirname(__file__), '..', 'final_approach.ipynb'),
    ]
    nb_path = None
    for c in candidates:
        if os.path.isfile(c):
            nb_path = c
            break
    if nb_path is None:
        raise RuntimeError(f"Could not find instance notebook. Tried: {candidates}")
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
    print("EXPERIMENT 7: Grid-Based Macro-Routing with QAOA")
    print("=" * 100)

    DIST_MATRIX, nodes, depot, N_NODES = load_instance()
    GLOBAL_MAX_DIST = float(np.max(DIST_MATRIX))
    K = 7
    dist_map_full = build_dist_map(nodes, depot, DIST_MATRIX)

    print(f"  Instance: RioClaroPostToy_50_0")
    print(f"  Nodes: {N_NODES}  |  Max dist: {GLOBAL_MAX_DIST:.3f}  |  K={K}  |  LEAF_SIZE={LEAF_SIZE}")

    # ── Previous experiment results (constants for comparison) ────────────
    PREV_RESULTS = {
        "K-Match (exp3)":       62177.45,
        "Clarke-Wright":        44144.84,
        "Nearest-Neighbour":    69132.32,
        "Sweep":                94644.60,
    }

    # ── Run grid variants ─────────────────────────────────────────────────
    GRID_CONFIGS = [
        ("Grid 3×3", 3, 3),
        ("Grid 4×4", 4, 4),
        ("Grid 5×5", 5, 5),
    ]

    results = {}

    for label, gr, gc in GRID_CONFIGS:
        print(f"\n{'─' * 80}")
        print(f"Running {label}...")
        print("─" * 80)

        QAOA_STATS["success"] = 0; QAOA_STATS["fallback"] = 0; QAOA_LOG.clear()

        t0 = time.time()
        sol = solve_grid_vrp(nodes, K, depot, dist_map_full, DIST_MATRIX, gr, gc)
        elapsed = time.time() - t0
        sol.solve_time = elapsed

        log_copy = list(QAOA_LOG)
        stats_copy = dict(QAOA_STATS)

        v = validate_solution(sol, nodes, depot, K, DIST_MATRIX)
        print(f"\n  Distance: {sol.total_distance:.2f}  |  Valid: {v['valid']}  |  "
              f"Time: {elapsed:.1f}s  |  QAOA calls: {len(log_copy)}  "
              f"(ok={stats_copy['success']}, fb={stats_copy['fallback']})")
        if not v['valid']:
            print(f"  ISSUES: {v['issues']}")

        results[label] = {
            "sol": sol, "time": elapsed, "valid": v,
            "qaoa_log": log_copy, "qaoa_stats": stats_copy,
            "grid": (gr, gc),
        }

    # ── Classical baselines ───────────────────────────────────────────────
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

    # ── TABLE 1: Grid clustering detail ───────────────────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 1: GRID CLUSTERING DETAIL")
    print("=" * 100)

    for label, gr, gc in GRID_CONFIGS:
        cells_t, cell_info_t = grid_clustering(nodes, depot, gr, gc)
        print(f"\n  [{label}] {len(cells_t)} non-empty cells out of {gr*gc} total")
        print(f"  {'Cell':>10s}  {'Nodes':>6s}  {'Demand':>8s}  {'Centroid':>20s}  Node IDs")
        print("  " + "-" * 80)
        for ck in sorted(cells_t.keys()):
            ci = cell_info_t[ck]
            ids_str = str(ci['node_ids'])
            if len(ids_str) > 35: ids_str = ids_str[:33] + ".."
            print(f"  {str(ck):>10s}  {ci['n_customers']:>6d}  {ci['demand']:>8.0f}  "
                  f"({ci['centroid'][0]:>8.0f},{ci['centroid'][1]:>8.0f})  {ids_str}")

    # ── TABLE 2: Per-route detail for each grid variant ───────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 2: PER-ROUTE DETAIL FOR EACH GRID VARIANT")
    print("=" * 100)

    for label, _, _ in GRID_CONFIGS:
        sol = results[label]["sol"]
        print(f"\n  [{label}]  Total: {sol.total_distance:.2f}  |  "
              f"Routes: {sol.num_vehicles_used}  |  σ={sol.distance_std:.2f}")
        print(f"  {'Route':>6s}  {'Nodes':>6s}  {'Distance':>12s}  {'Node IDs'}")
        print("  " + "-" * 80)
        for ri, r in enumerate(sol.routes):
            ids_str = str(sorted(r.node_ids))
            if len(ids_str) > 50: ids_str = ids_str[:48] + ".."
            print(f"  {ri+1:>6d}  {len(r.node_ids):>6d}  {r.distance:>12.2f}  {ids_str}")

    # ── TABLE 3: QAOA leaf performance ────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 3: QAOA LEAF PERFORMANCE BY GRID SIZE")
    print("=" * 100)

    for label, _, _ in GRID_CONFIGS:
        log = results[label]["qaoa_log"]
        n_total = len(log)
        n_opt = sum(1 for e in log if e["outcome"] == "optimal")
        n_feas = sum(1 for e in log if e["outcome"] == "feasible")
        n_nv = sum(1 for e in log if e["outcome"] == "non-valid")
        avg_qubits = np.mean([e["n_qubits"] for e in log]) if log else 0
        max_qubits = max([e["n_qubits"] for e in log]) if log else 0
        avg_valid = np.mean([e.get("valid_frac", 0) for e in log]) if log else 0
        print(f"\n  [{label}]")
        print(f"    Total QAOA calls: {n_total}")
        print(f"    Optimal: {n_opt}  |  Feasible: {n_feas}  |  Non-valid/Fallback: {n_nv}")
        print(f"    Avg qubits: {avg_qubits:.1f}  |  Max qubits: {max_qubits}")
        print(f"    Avg valid fraction: {avg_valid:.4f}")

    # ── TABLE 4: Full benchmark ───────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 4: FULL BENCHMARK — ALL ALGORITHMS")
    print(f"Instance: RioClaroPostToy_50_0  |  n=50  |  k={K}")
    print("=" * 100)

    valid_dists = [r["sol"].total_distance for r in results.values()
                   if r["valid"]["valid"]]
    best_dist = min(valid_dists) if valid_dists else float('inf')

    print(f"\n  {'Algorithm':<26s}  {'k':>4s}  {'Distance':>12s}  {'Std (σ)':>10s}  "
          f"{'CV%':>8s}  {'Time(s)':>10s}  {'Valid':>6s}  {'vs CW':>10s}")
    print("  " + "-" * 100)

    order = [lbl for lbl, _, _ in GRID_CONFIGS] + \
            ["Nearest-Neighbour", "Clarke-Wright", "Sweep", "OR-Tools (GLS)"]

    cw_dist = results['Clarke-Wright']['sol'].total_distance

    for name in order:
        r = results[name]
        sol = r["sol"]; elapsed = r["time"]; v = r["valid"]
        gap_cw = (sol.total_distance - cw_dist) / cw_dist * 100 if cw_dist > 0 else 0
        cv = (sol.distance_std / (sol.total_distance / sol.num_vehicles_used) * 100
              if sol.num_vehicles_used > 0 else 0)
        status = 'OK' if v['valid'] else 'FAIL'
        marker = ' ***' if abs(sol.total_distance - best_dist) < 1e-2 and v['valid'] else ''
        print(f"  {name:<26s}  {sol.num_vehicles_used:>4d}  {sol.total_distance:>12.2f}  "
              f"{sol.distance_std:>10.2f}  {cv:>7.1f}%  {elapsed:>10.2f}  "
              f"{status:>6s}  {gap_cw:>9.1f}%{marker}")

    # ── TABLE 5: Comparison with K-Match (exp3) ──────────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 5: GRID-BASED vs K-MATCH (EXP3) vs CLASSICAL")
    print("=" * 100)

    print(f"\n  {'Algorithm':<26s}  {'Distance':>12s}  {'vs CW':>10s}  {'vs K-Match':>12s}")
    print("  " + "-" * 66)

    km_dist = PREV_RESULTS["K-Match (exp3)"]

    all_compare = []
    for label, _, _ in GRID_CONFIGS:
        d = results[label]["sol"].total_distance
        all_compare.append((label, d))
    all_compare.append(("K-Match (exp3)", km_dist))
    all_compare.append(("Clarke-Wright", cw_dist))
    all_compare.append(("Nearest-Neighbour", PREV_RESULTS["Nearest-Neighbour"]))
    all_compare.append(("Sweep", PREV_RESULTS["Sweep"]))

    for name, d in all_compare:
        vs_cw = (d - cw_dist) / cw_dist * 100
        vs_km = (d - km_dist) / km_dist * 100
        print(f"  {name:<26s}  {d:>12.2f}  {vs_cw:>9.1f}%  {vs_km:>11.1f}%")

    # ── TABLE 6: Grid size analysis ───────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("TABLE 6: GRID SIZE SENSITIVITY ANALYSIS")
    print("=" * 100)

    print(f"\n  {'Grid':>8s}  {'Cells':>6s}  {'QAOA calls':>11s}  {'Max qubits':>11s}  "
          f"{'Distance':>12s}  {'σ':>10s}  {'Time(s)':>10s}  {'Valid':>6s}")
    print("  " + "-" * 88)

    for label, gr, gc in GRID_CONFIGS:
        r = results[label]
        log = r["qaoa_log"]
        n_cells = len(grid_clustering(nodes, depot, gr, gc)[0])
        max_q = max([e["n_qubits"] for e in log]) if log else 0
        sol = r["sol"]
        status = 'OK' if r['valid']['valid'] else 'FAIL'
        print(f"  {gr}×{gc:>5d}  {n_cells:>6d}  {len(log):>11d}  {max_q:>11d}  "
              f"{sol.total_distance:>12.2f}  {sol.distance_std:>10.2f}  "
              f"{r['time']:>10.2f}  {status:>6s}")

    # ── Save results ──────────────────────────────────────────────────────
    results_json = {
        "experiment": "Grid-Based Macro-Routing with QAOA",
        "instance": "RioClaroPostToy_50_0",
        "n_nodes": N_NODES, "k": K, "leaf_size": LEAF_SIZE,
        "previous_results": PREV_RESULTS,
        "algorithms": {}
    }

    for name in order:
        r = results[name]
        sol = r["sol"]
        entry = {
            "total_distance": sol.total_distance,
            "distance_std": sol.distance_std,
            "num_routes": sol.num_vehicles_used,
            "time_seconds": r["time"],
            "valid": r["valid"]["valid"],
            "routes": [{"node_ids": list(rt.node_ids), "distance": rt.distance}
                       for rt in sol.routes],
        }
        if "qaoa_log" in r:
            entry["qaoa_leaf_log"] = r["qaoa_log"]
            entry["qaoa_stats"] = r.get("qaoa_stats", {})
        if "grid" in r:
            entry["grid"] = r["grid"]
        results_json["algorithms"][name] = entry

    # Add comparison summary
    best_grid_label = min(
        [(lbl, results[lbl]["sol"].total_distance) for lbl, _, _ in GRID_CONFIGS],
        key=lambda x: x[1]
    )
    results_json["comparison"] = {
        "best_grid": best_grid_label[0],
        "best_grid_distance": best_grid_label[1],
        "k_match_distance": km_dist,
        "cw_distance": cw_dist,
        "grid_vs_k_match_pct": round((best_grid_label[1] - km_dist) / km_dist * 100, 3),
        "grid_vs_cw_pct": round((best_grid_label[1] - cw_dist) / cw_dist * 100, 3),
    }

    out_path = os.path.join(os.path.dirname(__file__), 'exp7_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    print("\n" + "=" * 100)
    print("EXPERIMENT 7 COMPLETE")
    print("=" * 100)
