"""
QUAV v2 - Quantum-assisted UAV path planning with QAOA (extended)
Name: Aditya Chattopadhyay | Reg. No.: 231080017 | Semester: VII

What is new compared to v1
  * Larger map with rectangular no-fly zones (UTM metres) + risk-aware edge costs
  * Graph pruning: only edges that lie on some START -> GOAL route become qubits
  * Explicit QUBO matrix (linear + quadratic terms) and gate / depth statistics
  * QAOA depth sweep p = 1..P_MAX with warm-started parameters
  * Simplified hardware-noise model (global depolarising fidelity) + finite shots
  * Baselines: Dijkstra, A*, RRT (continuous space)
  * Scaling study: qubits / two-qubit gates / parallel depth vs number of edges
  * One results figure (quav_results.png) and a printed summary table

Install:  pip install numpy scipy networkx utm matplotlib shapely
Run:      python quav_qaoa_v2.py
"""
import time
import numpy as np
import networkx as nx
import utm
from shapely.geometry import LineString, Point, box
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
ROWS, COLS = 5, 5
SPACING_M = 50.0
ORIGIN_LATLON = (19.0225, 72.8553)          # top-left corner (Mumbai)
START, GOAL = (0, 0), (ROWS - 1, COLS - 1)
USE_DIAGONALS = True

# No-fly zones in local metres: x = east, y = north (rows go DOWN => negative y)
NO_FLY = [box(40, -160, 160, -90),           # large central/lower block
          box(110, -60, 160, -20)]           # upper-right block
SAFETY_RADIUS = 60.0                         # risk grows inside this distance
RISK_ALPHA = 1.5                             # weight of the risk term

MAX_QUBITS = 18
P_MAX = 4
SHOTS = 4096
NOISE_2Q = [0.0, 1e-3, 3e-3, 1e-2]           # two-qubit gate error rates swept
NOISE_1Q_RATIO = 0.1                         # single-qubit error = ratio * 2q error
SEED = 11
RNG = np.random.default_rng(SEED)


# ----------------------------------------------------------------------
# 1. Environment -> graph
# ----------------------------------------------------------------------
def local_xy(node):
    i, j = node
    return (j * SPACING_M, -i * SPACING_M)


def risk_factor(p, q):
    """1 + alpha * proximity to the nearest no-fly zone, evaluated at the edge midpoint."""
    mid = Point((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    d = min(z.distance(mid) for z in NO_FLY)
    return 1.0 + RISK_ALPHA * max(0.0, 1.0 - d / SAFETY_RADIUS)


def build_graph(rows=ROWS, cols=COLS, zones=None, diagonals=USE_DIAGONALS):
    zones = NO_FLY if zones is None else zones
    G = nx.DiGraph()
    for i in range(rows):
        for j in range(cols):
            if not any(z.intersects(Point(local_xy((i, j)))) for z in zones):
                G.add_node((i, j))
    moves = [(0, 1), (1, 0)] + ([(1, 1)] if diagonals else [])
    for (i, j) in list(G.nodes):
        for di, dj in moves:
            v = (i + di, j + dj)
            if v not in G.nodes:
                continue
            p, q = local_xy((i, j)), local_xy(v)
            if any(z.intersects(LineString([p, q])) for z in zones):
                continue
            length = float(np.hypot(q[0] - p[0], q[1] - p[1]))
            G.add_edge((i, j), v, length=length,
                       w=length * (risk_factor(p, q) if zones is NO_FLY else 1.0))
    return G


def prune(G, start, goal):
    """Keep only edges that lie on at least one start->goal route."""
    fwd = nx.descendants(G, start) | {start}
    bwd = nx.ancestors(G, goal) | {goal}
    keep = fwd & bwd
    return G.subgraph(keep).copy()


def to_utm(node):
    e0, n0, zone, letter = utm.from_latlon(*ORIGIN_LATLON)
    x, y = local_xy(node)
    return (e0 + x, n0 + y), (zone, letter)


# ----------------------------------------------------------------------
# 2. QUBO:  min  sum w_e x_e + P * sum_v (out_v - in_v - b_v)^2
# ----------------------------------------------------------------------
def build_qubo(G, start, goal):
    edges = list(G.edges)
    n = len(edges)
    idx = {e: k for k, e in enumerate(edges)}
    w = np.array([G[u][v]["w"] for u, v in edges])
    P = 1.5 * w.sum() / max(1, len(G.nodes)) * 4         # penalty strength

    lin = w.copy()
    Q = np.zeros((n, n))                                  # upper-triangular quadratic part
    const = 0.0
    for v in G.nodes:
        a = {}
        for u in G.predecessors(v):
            a[idx[(u, v)]] = -1.0
        for s in G.successors(v):
            a[idx[(v, s)]] = +1.0
        b = 1.0 if v == start else (-1.0 if v == goal else 0.0)
        keys = list(a)
        for k in keys:
            lin[k] += P * (a[k] ** 2 - 2 * b * a[k])      # x^2 = x
        for ai in range(len(keys)):
            for bi in range(ai + 1, len(keys)):
                i, j = sorted((keys[ai], keys[bi]))
                Q[i, j] += 2 * P * a[keys[ai]] * a[keys[bi]]
        const += P * b ** 2
    return edges, lin, Q, const, P


def all_costs(lin, Q, const, n):
    ids = np.arange(2 ** n)
    bits = ((ids[:, None] >> np.arange(n)) & 1).astype(np.float64)
    cost = const + bits @ lin + ((bits @ Q) * bits).sum(1)
    return bits.astype(np.int8), cost


def feasible_mask(bits, edges, G, start, goal):
    """Vectorised check: a bitstring is a valid path iff every node satisfies flow conservation."""
    idx = {e: k for k, e in enumerate(edges)}
    nodes = list(G.nodes)
    M = np.zeros((len(nodes), len(edges)), dtype=np.int8)
    b = np.zeros(len(nodes), dtype=np.int8)
    for r, v in enumerate(nodes):
        for s in G.successors(v):
            M[r, idx[(v, s)]] += 1
        for u in G.predecessors(v):
            M[r, idx[(u, v)]] -= 1
        b[r] = 1 if v == start else (-1 if v == goal else 0)
    return np.all(bits @ M.T.astype(np.int8) == b, axis=1)


def gate_stats(Q, n, p):
    """Gate counts for p QAOA layers and a parallel-depth estimate per layer."""
    pairs = list(zip(*np.nonzero(Q)))
    H = nx.Graph(); H.add_nodes_from(range(n)); H.add_edges_from(pairs)
    colours = nx.coloring.greedy_color(nx.line_graph(H), strategy="largest_first") if pairs else {}
    n_colours = (max(colours.values()) + 1) if colours else 0
    return {
        "qubits": n,
        "rzz_per_layer": len(pairs),
        "cx_total": 2 * len(pairs) * p,
        "depth_sequential": p * (len(pairs) * 3 + 2),     # RZZ = CX-RZ-CX (3 layers) + RZ + RX
        "depth_parallel": p * (n_colours * 3 + 2),        # RZZs on disjoint qubits run together
    }


# ----------------------------------------------------------------------
# 3. QAOA (exact statevector in NumPy)
# ----------------------------------------------------------------------
def apply_mixer(psi, beta, n):
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(n):
        v = psi.reshape(2 ** (n - q - 1), 2, 2 ** q)
        a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :] = c * a0 + s * a1
        v[:, 1, :] = s * a0 + c * a1
    return psi


def qaoa_probs(params, cs, n, p):
    g, b = params[:p], params[p:]
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for k in range(p):
        psi = psi * np.exp(-1j * g[k] * cs)
        psi = apply_mixer(psi, b[k], n)
    return np.abs(psi) ** 2


def interp_params(params, p_old, p_new):
    g, b = params[:p_old], params[p_old:]
    x_old, x_new = np.linspace(0, 1, p_old), np.linspace(0, 1, p_new)
    return np.concatenate([np.interp(x_new, x_old, g), np.interp(x_new, x_old, b)])


def optimise(cs, n, p, x0_list, maxiter=250):
    best, evals = None, 0

    def energy(x):
        nonlocal evals
        evals += 1
        return float(np.dot(qaoa_probs(x, cs, n, p), cs))

    for x0 in x0_list:
        res = minimize(energy, x0, method="COBYLA", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    return best, evals


def apply_noise(probs, stats, e2):
    """Global depolarising approximation: p' = F*p + (1-F)/2^n."""
    e1 = NOISE_1Q_RATIO * e2
    n1q = stats["qubits"] * 2 * (stats["cx_total"] // max(1, 2 * stats["rzz_per_layer"]))  # RX + RZ per layer
    F = (1 - e2) ** stats["cx_total"] * (1 - e1) ** n1q
    return F * probs + (1 - F) / probs.size, F


# ----------------------------------------------------------------------
# 4. Decode, sample, baselines
# ----------------------------------------------------------------------
def decode(bits, edges, start, goal):
    chosen = [e for e, b in zip(edges, bits) if b]
    if not chosen:
        return None
    H = nx.DiGraph(chosen)
    if start in H and goal in H and nx.has_path(H, start, goal) \
            and len(chosen) == nx.shortest_path_length(H, start, goal):
        return nx.shortest_path(H, start, goal)
    return None


def path_cost(G, path, key="w"):
    return sum(G[a][b][key] for a, b in zip(path, path[1:]))


def sample_and_pick(probs, bits, cost, edges, G, start, goal, shots, rng):
    counts = rng.multinomial(shots, probs / probs.sum())
    sampled = np.nonzero(counts)[0]
    feasible, best = 0, None
    for k in sampled:
        path = decode(bits[k], edges, start, goal)
        if path:
            feasible += counts[k]
            c = path_cost(G, path)
            if best is None or c < best[0]:
                best = (c, path)
    return best, feasible / shots


def rrt(start_xy, goal_xy, iters=4000, step=35.0, goal_bias=0.1, rng=None):
    rng = rng or np.random.default_rng(SEED)
    xs = [0, (COLS - 1) * SPACING_M]
    ys = [-(ROWS - 1) * SPACING_M, 0]
    nodes, parent = [np.array(start_xy)], [-1]
    for _ in range(iters):
        tgt = np.array(goal_xy) if rng.random() < goal_bias else \
            np.array([rng.uniform(*xs), rng.uniform(*ys)])
        k = int(np.argmin([np.linalg.norm(nd - tgt) for nd in nodes]))
        d = tgt - nodes[k]
        new = nodes[k] + d / (np.linalg.norm(d) + 1e-9) * min(step, np.linalg.norm(d))
        if any(z.intersects(LineString([nodes[k], new])) for z in NO_FLY):
            continue
        nodes.append(new); parent.append(k)
        if np.linalg.norm(new - goal_xy) < step and \
                not any(z.intersects(LineString([new, goal_xy])) for z in NO_FLY):
            nodes.append(np.array(goal_xy)); parent.append(len(nodes) - 2)
            path, k = [], len(nodes) - 1
            while k != -1:
                path.append(nodes[k]); k = parent[k]
            path = path[::-1]
            return path, sum(np.linalg.norm(a - b) for a, b in zip(path, path[1:]))
    return None, None


# ----------------------------------------------------------------------
# 5. Scaling study (no simulation, just problem size)
# ----------------------------------------------------------------------
def scaling_study(sizes=range(3, 13)):
    rows = []
    for s in sizes:
        G = build_graph(s, s, zones=[], diagonals=True)
        G = prune(G, (0, 0), (s - 1, s - 1))
        edges = list(G.edges)
        Qn = np.zeros((len(edges), len(edges)))
        idx = {e: k for k, e in enumerate(edges)}
        for v in G.nodes:           # which edge pairs share a node (interaction graph)
            inc = [idx[(u, v)] for u in G.predecessors(v)] + [idx[(v, t)] for t in G.successors(v)]
            for a in range(len(inc)):
                for b in range(a + 1, len(inc)):
                    Qn[min(inc[a], inc[b]), max(inc[a], inc[b])] = 1
        st = gate_stats(Qn, len(edges), 1)
        rows.append((len(edges), st["rzz_per_layer"], st["depth_sequential"], st["depth_parallel"]))
    return np.array(rows)


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------
def main():
    t_all = time.perf_counter()
    G_full = build_graph()
    G = prune(G_full, START, GOAL)
    edges, lin, Q, const, P = build_qubo(G, START, GOAL)
    n = len(edges)
    (_, _), (zone, letter) = to_utm(START)
    print(f"Map {ROWS}x{COLS} | UTM {zone}{letter} | nodes {len(G_full.nodes)}->{len(G.nodes)} "
          f"| edges (qubits) {len(G_full.edges)}->{n} | penalty P={P:.0f}")
    if n > MAX_QUBITS:
        raise SystemExit(f"{n} qubits is too many for the NumPy simulator (max {MAX_QUBITS}). "
                         f"Add no-fly zones or shrink the grid.")

    bits, cost = all_costs(lin, Q, const, n)
    k_opt = int(np.argmin(cost))
    path_opt = decode(bits[k_opt], edges, START, GOAL)
    assert path_opt is not None, "QUBO optimum is not a valid path - increase the penalty"

    # --- classical baselines -------------------------------------------------
    t0 = time.perf_counter()
    p_dij = nx.dijkstra_path(G_full, START, GOAL, weight="w")
    t_dij = time.perf_counter() - t0
    c_dij = path_cost(G_full, p_dij)

    heur = lambda a, b: float(np.hypot(*(np.array(local_xy(a)) - np.array(local_xy(b)))))
    t0 = time.perf_counter()
    p_ast = nx.astar_path(G_full, START, GOAL, heuristic=heur, weight="w")
    t_ast = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_rrt, len_rrt = rrt(local_xy(START), local_xy(GOAL))
    t_rrt = time.perf_counter() - t0

    assert abs(path_cost(G, path_opt) - c_dij) < 1e-6, "QUBO optimum differs from Dijkstra"
    print(f"Dijkstra  : cost {c_dij:.1f} | length {path_cost(G_full, p_dij, 'length'):.1f} m | {t_dij*1e3:.2f} ms")
    print(f"A*        : cost {path_cost(G_full, p_ast):.1f} | {t_ast*1e3:.2f} ms")
    print(f"RRT       : length {len_rrt:.1f} m | {t_rrt*1e3:.0f} ms" if p_rrt else "RRT: no path")

    # --- QAOA depth sweep ----------------------------------------------------
    cs = (cost - cost.min()) / cost.std()
    rng = np.random.default_rng(SEED)
    feas = feasible_mask(bits, edges, G, START, GOAL)
    print(f"Feasible bitstrings: {feas.sum()} of {2 ** n} (valid START->GOAL paths)")
    sweep, params_by_p, probs_by_p = [], {}, {}
    prev, t_q = None, time.perf_counter()
    for p in range(1, P_MAX + 1):
        starts = [np.concatenate([rng.uniform(0, 1, p), rng.uniform(0, np.pi / 2, p)]) for _ in range(4 if p == 1 else 2)]
        if prev is not None:
            starts.insert(0, interp_params(prev, p - 1, p))
        res, ev = optimise(cs, n, p, starts)
        prev = res.x
        probs = qaoa_probs(res.x, cs, n, p)
        sweep.append(dict(p=p, energy=res.fun, evals=ev, p_opt=probs[k_opt], p_feas=probs[feas].sum()))
        params_by_p[p], probs_by_p[p] = res.x, probs
        print(f"  p={p}: <C>={res.fun:.3f}  P(opt)={probs[k_opt]:.4f}  P(feasible)={probs[feas].sum():.4f}  evals={ev}")
    t_q = time.perf_counter() - t_q
    uniform_opt = 1 / 2 ** n

    # --- noise sweep at the deepest p ---------------------------------------
    p_use = P_MAX
    stats = gate_stats(Q, n, p_use)
    print(f"Circuit (p={p_use}): {stats}")
    noise_rows = []
    for e2 in NOISE_2Q:
        pr, F = apply_noise(probs_by_p[p_use], stats, e2)
        best, feas_rate = sample_and_pick(pr, bits, cost, edges, G, START, GOAL, SHOTS, rng)
        noise_rows.append((e2, F, pr[k_opt], feas_rate, best[0] if best else np.nan,
                           best[1] if best else None))
    print("\n e2      fidelity  P(opt)    feasible%  best cost found")
    for e2, F, po, fr, bc, _ in noise_rows:
        print(f" {e2:<7.0e} {F:8.4f}  {po:8.5f}  {100*fr:8.2f}   {bc:.1f}")

    best_clean, _ = sample_and_pick(probs_by_p[p_use], bits, cost, edges, G, START, GOAL, SHOTS, rng)
    path_q = best_clean[1] if best_clean else None

    # --- scaling study --------------------------------------------------------
    sc = scaling_study()
    slope, icpt = np.polyfit(sc[:, 0], sc[:, 2], 1)
    r2 = np.corrcoef(sc[:, 0], sc[:, 2])[0, 1] ** 2
    print(f"\nScaling: sequential depth ~ {slope:.1f} * edges + {icpt:.0f}  (R^2 = {r2:.4f}) "
          f"| parallel depth stays {sc[:,3].min():.0f}-{sc[:,3].max():.0f} for p=1")

    # --- summary table --------------------------------------------------------
    print("\n=== SUMMARY ===")
    print(f"{'Method':<12}{'cost':>9}{'time (ms)':>12}")
    print(f"{'Dijkstra':<12}{c_dij:>9.1f}{t_dij*1e3:>12.2f}")
    print(f"{'A*':<12}{path_cost(G_full, p_ast):>9.1f}{t_ast*1e3:>12.2f}")
    print(f"{'QAOA (sim)':<12}{path_cost(G, path_q) if path_q else float('nan'):>9.1f}{t_q*1e3:>12.0f}")
    print(f"{'RRT (len)':<12}{len_rrt if len_rrt else float('nan'):>9.1f}{t_rrt*1e3:>12.0f}")

    # --- figure ---------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    # (a) map
    a = ax[0, 0]
    for z in NO_FLY:
        x, y = z.exterior.xy
        a.fill(x, y, color="red", alpha=.25)
    for u, v in G_full.edges:
        a.plot(*zip(local_xy(u), local_xy(v)), c="#dddddd", lw=1, zorder=1)
    for u, v in G.edges:
        a.plot(*zip(local_xy(u), local_xy(v)), c="#999999", lw=1.2, zorder=1)
    a.scatter(*zip(*[local_xy(v) for v in G_full.nodes]), c="k", s=15, zorder=2)
    if p_rrt:
        a.plot(*zip(*p_rrt), c="orange", lw=1.5, ls="--", label="RRT", zorder=3)
    a.plot(*zip(*[local_xy(v) for v in p_dij]), c="green", lw=6, alpha=.3, label="Dijkstra/A*", zorder=3)
    if path_q:
        a.plot(*zip(*[local_xy(v) for v in path_q]), c="blue", lw=2, label="QAOA", zorder=4)
    a.scatter(*local_xy(START), c="blue", s=90, zorder=5)
    a.scatter(*local_xy(GOAL), c="gold", marker="*", s=200, zorder=5)
    a.set_aspect("equal"); a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8)
    a.set_title(f"(a) Map: red = no-fly, grey = pruned edges, {n} qubits"); a.axis("off")
    # (b) p sweep
    b = ax[0, 1]
    ps = [s["p"] for s in sweep]
    b.plot(ps, [s["p_opt"] for s in sweep], "o-", label="P(optimal path)")
    b.plot(ps, [s["p_feas"] for s in sweep], "s-", label="P(any valid path)")
    b.axhline(uniform_opt, c="gray", ls=":", label="random guess (optimal)")
    b.set_xlabel("QAOA depth p"); b.set_ylabel("probability"); b.set_xticks(ps)
    b.set_title("(b) Depth sweep (noiseless)"); b.legend(fontsize=8); b.grid(alpha=.3)
    # (c) top outcomes
    c = ax[0, 2]
    order = np.argsort(-probs_by_p[p_use])[:12]
    cols = ["tab:blue" if decode(bits[k], edges, START, GOAL) else "lightgray" for k in order]
    c.bar(range(len(order)), probs_by_p[p_use][order], color=cols)
    c.set_xlabel("top-12 bitstrings (blue = valid path)"); c.set_ylabel("probability")
    c.set_title(f"(c) Output distribution, p={p_use}"); c.grid(alpha=.3, axis="y")
    # (d) noise
    d = ax[1, 0]
    labels = [f"{r[0]:.0e}" if r[0] else "0" for r in noise_rows]
    d.bar(labels, [100 * r[3] for r in noise_rows], color="tab:purple")
    for i, r in enumerate(noise_rows):
        d.text(i, 100 * r[3] + 1, f"F={r[1]:.2f}", ha="center", fontsize=8)
    d.set_ylim(0, 1.2 * max(100 * r[3] for r in noise_rows))
    d.set_xlabel("two-qubit gate error"); d.set_ylabel("valid shots (%)")
    d.set_title(f"(d) Noise: feasibility of {SHOTS} shots"); d.grid(alpha=.3, axis="y")
    # (e) scaling
    e = ax[1, 1]
    e.plot(sc[:, 0], sc[:, 2], "o-", label="sequential depth")
    e.plot(sc[:, 0], sc[:, 3], "s-", label="parallel depth (colouring)")
    e.plot(sc[:, 0], slope * sc[:, 0] + icpt, "k--", lw=1, label=f"linear fit R²={r2:.3f}")
    e.set_xlabel("edges = qubits"); e.set_ylabel("circuit depth (p=1)")
    e.set_title("(e) Scaling with problem size"); e.legend(fontsize=8); e.grid(alpha=.3)
    # (f) runtime
    f = ax[1, 2]
    names = ["Dijkstra", "A*", "RRT", "QAOA sim"]
    times = [t_dij * 1e3, t_ast * 1e3, t_rrt * 1e3, t_q * 1e3]
    f.bar(names, times, color=["tab:green", "tab:green", "tab:orange", "tab:blue"])
    f.set_yscale("log"); f.set_ylabel("runtime (ms, log)")
    f.set_title("(f) Runtime (QAOA = classical simulation, not a QPU)"); f.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig("quav_results.png", dpi=140)
    print(f"\nSaved quav_results.png | total time {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
