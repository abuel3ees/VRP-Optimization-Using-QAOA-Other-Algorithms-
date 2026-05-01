# Experiment 2: Impact of Real Inter-Cluster Distances on Super-Node QAOA

**Instance:** RioClaroPostToy_50_0 | **Nodes:** 50 | **Vehicles (k):** 7 | **Leaf Size:** 4

---

## 1. Algorithm Ranking

| Rank | Algorithm | Distance | σ (fairness) | Time (s) | Routes | Valid | Gap vs Best Valid |
|:----:|-----------|----------:|-------------:|---------:|:------:|:-----:|------------------:|
| 1 | Clarke-Wright | 44,144.84 | 9,343.05 | 0.00 | 7 | ✅ | 0.0% ★ |
| 2 | QAOA (centroid, base) | 62,707.26 | 2,741.09 | 126.81 | 7 | ✅ | +42.0% |
| 3 | QAOA (real dist, improved) | 63,002.45 | 2,812.82 | 152.17 | 7 | ✅ | +42.7% |
| 4 | Nearest-Neighbour | 69,132.32 | 3,174.63 | 0.00 | 7 | ✅ | +56.6% |
| 5 | Sweep | 94,644.60 | 4,883.98 | 0.00 | 7 | ✅ | +114.4% |
| 6 | OR-Tools (GLS) | 36,369.13 | 0.00 | 11.34 | 1 | ❌ | — |

> **Note:** OR-Tools returned 1 route instead of 7, violating the vehicle constraint. It is excluded from the valid ranking. Its distance (36,369.13) represents a TSP-like single-tour lower bound.

**Conclusion:** Clarke-Wright is the best valid algorithm at 44,144.84. Both QAOA variants rank 2nd and 3rd, outperforming Nearest-Neighbour (+9.3% better) and Sweep (+33.6% better). However, QAOA remains 42% behind Clarke-Wright. The quantum approach is competitive with simple heuristics but not yet with savings-based classical methods on this instance size.

---

## 2. QAOA Head-to-Head: Centroid vs Real Inter-Cluster Distances

| Metric | Centroid (Base) | Real Dist (Improved) | Winner |
|--------|----------------:|---------------------:|:------:|
| Total Distance | 62,707.26 | 63,002.45 | **Centroid** |
| Fairness (σ) | 2,741.09 | 2,812.82 | **Centroid** |
| Solve Time | 126.8 s | 152.2 s | **Centroid** |
| QAOA Optimal Leaves | 30/30 (100%) | 30/30 (100%) | Tie |
| QAOA Fallbacks | 0 | 0 | Tie |
| Num Routes | 7 | 7 | Tie |
| Shortest Route | 5,650.75 | 5,650.75 | Tie |
| Longest Route | 14,220.41 | 14,464.03 | **Centroid** |
| Route Range (max − min) | 8,569.67 | 8,813.28 | **Centroid** |

**Conclusion:** The centroid-based approximation wins on every non-tied metric. Real inter-cluster distances made the solution 0.5% worse (+295.19 units), increased route imbalance, and took 20% longer to solve. The centroid approximation, despite being geometrically simpler, produced a better QAOA solution. This suggests that centroid distances provide a smoother optimization landscape for QAOA, while real (minimum inter-cluster) distances introduce noise that can mislead the super-node routing decisions.

---

## 3. Per-Route Comparison

| Route | Centroid Dist | Real Dist | Δ (change) | Δ% | Changed? |
|:-----:|--------------:|----------:|-----------:|---:|:--------:|
| 1 | 8,246.59 | 8,246.59 | 0.00 | 0.0% | No |
| 2 | 5,650.75 | 5,650.75 | 0.00 | 0.0% | No |
| 3 | 10,646.84 | 10,698.52 | −51.68 | −0.5% | **Yes** |
| 4 | 7,427.35 | 7,427.35 | 0.00 | 0.0% | No |
| 5 | 6,349.55 | 6,349.44 | +0.11 | +0.0% | Negligible |
| 6 | 10,165.78 | 10,165.78 | 0.00 | 0.0% | No |
| 7 | 14,220.41 | 14,464.03 | −243.62 | −1.7% | **Yes** |
| **Total** | **62,707.26** | **63,002.45** | **−295.19** | **−0.5%** | |

**Conclusion:** Only 2 of 7 routes were affected by switching to real distances, and both got worse. Route 7 (the longest route) degraded by 1.7% (−243.62 units), accounting for 82.5% of the total degradation. Route 3 worsened by 0.5%. The remaining 5 routes were identical, indicating that the super-node routing change at the top level only propagated to clusters involved in the two most affected routes. The damage is concentrated in the longest route, worsening the already-poor fairness.

---

## 4. Super-Distance Accuracy Analysis (Centroid vs Real)

| Depth | #Super-nodes | #Nodes | Centroid Σ | Real Σ | Error | Error% | Direction |
|:-----:|:------------:|:------:|-----------:|-------:|------:|-------:|:---------:|
| 1 | 3 | 8 | 15,936.8 | 12,394.0 | +3,542.9 | +22.2% | Overest. |
| 1 | 3 | 6 | 11,162.5 | 13,341.9 | −2,179.4 | −19.5% | Underest. |
| 1 | 3 | 6 | 19,737.2 | 20,909.6 | −1,172.4 | −5.9% | Underest. |
| 1 | 3 | 8 | 15,482.4 | 13,279.7 | +2,202.7 | +14.2% | Overest. |
| 1 | 3 | 7 | 12,628.0 | 12,558.6 | +69.4 | +0.5% | Overest. |
| 1 | 3 | 6 | 17,773.9 | 17,798.4 | −24.5 | −0.1% | Underest. |
| 1 | 3 | 7 | 41,729.1 | 40,024.0 | +1,705.2 | +4.1% | Overest. |
| 0 | 8 | 50 | 184,556.5 | 129,190.6 | +55,365.9 | +30.0% | Overest. |
| | | | | | | | |
| **Total** | | | **319,006.5** | **259,496.8** | **+59,509.7** | **+18.7%** | |

| Summary Statistic | Value |
|--------------------|-------|
| Overestimates | 5 / 8 (62.5%) |
| Underestimates | 3 / 8 (37.5%) |
| Exact (< 1% error) | 2 / 8 (25.0%) |
| Max overestimate | +30.0% (depth 0, root level) |
| Max underestimate | −19.5% (depth 1) |
| Net bias | Overestimation (+18.7%) |

**Conclusion:** Centroid distances systematically overestimate inter-cluster distances, with a net +18.7% inflation across all recursion levels. The largest error occurs at the root level (depth 0), where centroid distances are 30% higher than real shortest-path distances across all 8 super-nodes. At the leaf level (depth 1), errors range from −19.5% to +22.2%, with 4 of 7 clusters overestimated and 3 underestimated. Despite this significant distortion, the centroid approximation still produces better QAOA solutions — suggesting that the overestimation acts as an implicit regularization that helps QAOA avoid poor super-node orderings, while real distances with their mixed over/underestimation create a noisier objective landscape.

---

## 5. QAOA Leaf Performance

| Variant | Total Leaves | Optimal | Feasible (suboptimal) | Non-valid | Avg Gap | Avg Valid% |
|---------|:------------:|:-------:|:---------------------:|:---------:|--------:|-----------:|
| Centroid (base) | 30 | 30 (100%) | 0 | 0 | 0.00% | ~10.0% |
| Real dist (improved) | 30 | 30 (100%) | 0 | 0 | 0.00% | ~10.1% |

**Conclusion:** Both QAOA variants achieved a perfect 100% optimal rate across all 30 leaf sub-problems (using top-k sampling over 5 optimizer configurations). The leaf-level QAOA performance is identical — the quality difference between centroid and real distances originates entirely at the super-node routing level, not at the intra-cluster TSP level. This confirms that the decomposition + multi-config QAOA strategy is highly robust for small sub-problems (≤ 4 nodes, ≤ 20 qubits).

---

## Final Conclusions

### Research Question
*Does replacing centroid-based Euclidean approximations with real (shortest-path) inter-cluster distances improve QAOA performance in the recursive decomposition framework?*

### Answer: **No.** Real distances made the solution 0.5% worse.

### Key Findings

1. **Centroid approximation is sufficient.** Despite overestimating inter-cluster distances by 18.7% on average, centroid-based routing produces a better final VRP solution (62,707 vs 63,002). The additional computational cost of computing real distances (+20% time) is not justified.

2. **Overestimation may help QAOA.** The centroid's systematic overestimation of distances appears to act as implicit regularization for the QAOA optimizer. By inflating distance differences between super-node orderings, it makes the objective landscape more separable, helping QAOA distinguish good from bad solutions.

3. **Leaf-level QAOA is unaffected.** All 30 QAOA leaf sub-problems achieved optimal solutions in both variants. The quality difference is entirely determined by super-node routing decisions at the top recursion level.

4. **Damage is concentrated.** Only 2 of 7 routes changed, and both worsened. Route 7 (the longest) absorbed 82.5% of the total degradation, suggesting that real distances particularly hurt routing decisions for distant, spread-out clusters.

5. **QAOA is competitive but not dominant.** Both QAOA variants outperform Nearest-Neighbour and Sweep but trail Clarke-Wright by ~42%. The quantum advantage lies in solution quality consistency (low σ) rather than raw distance optimization.

### Recommendation
For the recursive VRP decomposition framework, **retain centroid-based inter-cluster distances** as the default. The simpler approximation is faster, produces better solutions, and avoids the O(n²) shortest-path computation overhead. Future work should investigate whether weighted combinations of centroid and real distances could further improve super-node routing quality.
