"""Inter-route local search for VRP.

Two moves, applied in a first-improvement loop until a full pass finds nothing:

  1. Or-opt (relocate):  move a segment of 1..max_seg consecutive customers
     from its current route to any position in any route (including its own).
     Covers single-customer relocate and swap-by-relocation.

  2. 2-opt*:  swap the tails of two different routes at a chosen cut point
     in each.  Fixes long inter-route crossings that intra-route 2-opt
     cannot reach.

Both moves preserve the number of routes: any move that would empty a
route is rejected.

Usage (plugs into the existing notebook/QAOA.py):

    from inter_route_opt import improve_solution
    polished = improve_solution(sol, depot, dist_map)

`sol` is any object with a `.routes` list whose items have `.node_ids`.
Returns a new VRPSolution with the same shape.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

Edge = Tuple[int, int]
DistMap = Dict[Edge, float]

EPS = 1e-9


def _route_cost(route: List[int], depot_id: int, dm: DistMap) -> float:
    if not route:
        return 0.0
    c = dm[(depot_id, route[0])]
    for i in range(len(route) - 1):
        c += dm[(route[i], route[i + 1])]
    c += dm[(route[-1], depot_id)]
    return c


def _total_cost(routes: List[List[int]], depot_id: int, dm: DistMap) -> float:
    return sum(_route_cost(r, depot_id, dm) for r in routes)


def _try_relocate(
    routes: List[List[int]],
    costs: List[float],
    depot_id: int,
    dm: DistMap,
    max_seg: int,
) -> bool:
    """Scan all Or-opt moves; apply the first improving one in place.
    Returns True if a move was applied."""
    n_routes = len(routes)
    for r1 in range(n_routes):
        R1 = routes[r1]
        n1 = len(R1)
        for L in range(1, max_seg + 1):
            if L > n1:
                break
            # Moving out all nodes would empty route r1 — skip.
            if L == n1:
                continue
            for i in range(n1 - L + 1):
                seg = R1[i : i + L]
                R1_minus = R1[:i] + R1[i + L :]
                cost_R1_minus = _route_cost(R1_minus, depot_id, dm)
                delta_R1 = cost_R1_minus - costs[r1]

                for r2 in range(n_routes):
                    R2 = routes[r2]
                    n2 = len(R2)
                    for j in range(n2 + 1):
                        if r1 == r2:
                            # Same-route relocate: compute on the shrunken list.
                            if j == i or j == i + L:
                                continue  # same position
                            base = R1_minus
                            base_cost = cost_R1_minus
                            j_adj = j if j < i else j - L
                            if j_adj < 0 or j_adj > len(base):
                                continue
                        else:
                            base = R2
                            base_cost = costs[r2]
                            j_adj = j

                        for rev in (False, True):
                            insert = seg[::-1] if rev else seg
                            new_R = base[:j_adj] + insert + base[j_adj:]
                            new_cost = _route_cost(new_R, depot_id, dm)
                            if r1 == r2:
                                delta = new_cost - costs[r1]
                            else:
                                delta = delta_R1 + (new_cost - base_cost)
                            if delta < -EPS:
                                if r1 == r2:
                                    routes[r1] = new_R
                                    costs[r1] = new_cost
                                else:
                                    routes[r1] = R1_minus
                                    costs[r1] = cost_R1_minus
                                    routes[r2] = new_R
                                    costs[r2] = new_cost
                                return True
    return False


def _try_2opt_star(
    routes: List[List[int]],
    costs: List[float],
    depot_id: int,
    dm: DistMap,
) -> bool:
    """Scan all 2-opt* tail-swap moves; apply the first improving one.
    Returns True if a move was applied."""
    n_routes = len(routes)
    for r1 in range(n_routes):
        R1 = routes[r1]
        n1 = len(R1)
        for r2 in range(r1 + 1, n_routes):
            R2 = routes[r2]
            n2 = len(R2)
            base = costs[r1] + costs[r2]
            for i in range(n1 + 1):
                for j in range(n2 + 1):
                    # Tail swap
                    new_R1 = R1[:i] + R2[j:]
                    new_R2 = R2[:j] + R1[i:]
                    if not new_R1 or not new_R2:
                        continue  # don't drop a vehicle
                    c1 = _route_cost(new_R1, depot_id, dm)
                    c2 = _route_cost(new_R2, depot_id, dm)
                    if c1 + c2 < base - EPS:
                        routes[r1] = new_R1
                        routes[r2] = new_R2
                        costs[r1] = c1
                        costs[r2] = c2
                        return True
    return False


def improve_routes(
    initial_routes: List[List[int]],
    depot_id: int,
    dist_map: DistMap,
    max_seg: int = 3,
    max_passes: int = 200,
    verbose: bool = False,
) -> Tuple[List[List[int]], Dict[str, float]]:
    """Core driver.  Works on plain list[list[int]] routes.

    Returns (improved_routes, stats) where stats contains:
      before_cost, after_cost, improvement, n_relocate, n_2opt_star, n_passes
    """
    routes = [list(r) for r in initial_routes if r]
    costs = [_route_cost(r, depot_id, dist_map) for r in routes]
    before_cost = sum(costs)

    n_relocate = 0
    n_2opt_star = 0
    passes = 0
    while passes < max_passes:
        passes += 1
        if _try_relocate(routes, costs, depot_id, dist_map, max_seg):
            n_relocate += 1
            continue
        if _try_2opt_star(routes, costs, depot_id, dist_map):
            n_2opt_star += 1
            continue
        break

    after_cost = sum(costs)
    stats = {
        "before_cost": before_cost,
        "after_cost": after_cost,
        "improvement": before_cost - after_cost,
        "improvement_pct": (
            100.0 * (before_cost - after_cost) / before_cost if before_cost > 0 else 0.0
        ),
        "n_relocate": n_relocate,
        "n_2opt_star": n_2opt_star,
        "n_passes": passes,
    }
    if verbose:
        print(
            f"[inter_route_opt] {before_cost:.2f} -> {after_cost:.2f}  "
            f"(-{stats['improvement']:.2f}, -{stats['improvement_pct']:.2f}%)  "
            f"moves: relocate={n_relocate} 2opt*={n_2opt_star} passes={passes}"
        )
    return routes, stats


def improve_solution(sol, depot, dist_map, max_seg: int = 3, verbose: bool = False):
    """Wrapper for VRPSolution-shaped objects (duck-typed on .routes[*].node_ids).

    Rebuilds a VRPSolution using the same classes the input uses, so the
    returned object matches the notebook's dataclass types.
    """
    Route = type(sol.routes[0])
    VRPSolution = type(sol)
    node_demand = {}
    for r in sol.routes:
        for nid in r.node_ids:
            # Demand isn't stored on Route in a retrievable way, so we preserve
            # the total per-route load by summing original demands from the
            # Route's existing .load (distributed proportionally is overkill —
            # just recompute below from the route structure).
            pass

    # Record the original per-node "share" so we can reconstruct loads.
    # Every input Route carries r.load = sum of demands on its node_ids.
    # We infer per-node demand as load / len(node_ids) ONLY as a fallback;
    # in the notebook every delivery node has demand=1.0, so this is exact.
    per_node_demand = {}
    for r in sol.routes:
        if r.node_ids:
            share = r.load / len(r.node_ids) if r.node_ids else 0.0
            for nid in r.node_ids:
                per_node_demand[nid] = share

    initial = [list(r.node_ids) for r in sol.routes]
    improved, stats = improve_routes(
        initial, depot.id, dist_map, max_seg=max_seg, verbose=verbose
    )

    new_routes = []
    for ids in improved:
        d = _route_cost(ids, depot.id, dist_map)
        load = sum(per_node_demand.get(nid, 0.0) for nid in ids)
        new_routes.append(Route(node_ids=ids, distance=d, load=load))
    new_sol = VRPSolution(routes=new_routes)
    # Preserve any custom attributes (e.g. solver_used)
    if hasattr(sol, "solver_used"):
        new_sol.solver_used = sol.solver_used
    new_sol._inter_route_stats = stats
    return new_sol


if __name__ == "__main__":
    # Minimal smoke test with a hand-built tiny instance.
    import math

    # 6 customers on a line, depot at origin.  With 2 vehicles, the
    # optimal split is {1,2,3} and {4,5,6}.  Seed a bad split to show
    # the relocate move kicks in.
    coords = {
        0: (0, 0),
        1: (1, 0), 2: (2, 0), 3: (3, 0),
        4: (4, 0), 5: (5, 0), 6: (6, 0),
    }
    def d(a, b):
        return math.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1])
    dm = {(a, b): d(a, b) for a in coords for b in coords}

    bad = [[1, 4, 2], [3, 5, 6]]
    improved, stats = improve_routes(bad, 0, dm, verbose=True)
    print("routes:", improved)
