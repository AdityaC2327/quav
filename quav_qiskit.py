"""
QUAV on Qiskit - run the QAOA path planner on a simulator, a noisy IBM-Heron replica,
or a real IBM quantum computer.
Name: Aditya Chattopadhyay | Reg. No.: 231080017 | Semester: VII

Needs quav_qaoa_v2.py in the same folder (it provides the map, QUBO and decoder).

  pip install qiskit qiskit-aer qiskit-ibm-runtime numpy scipy networkx shapely utm matplotlib

Typical use
  python quav_qiskit.py --mode ideal                      # exact simulator
  python quav_qiskit.py --mode fake                       # noisy replica of ibm_fez (no account needed)
  export IBM_QUANTUM_TOKEN="your-api-key"                 # keep the key out of the code
  python quav_qiskit.py --mode ibm --dry-run              # connect, transpile, print cost. Uses NO QPU time
  python quav_qiskit.py --mode ibm --backend ibm_fez      # real run (uses QPU time)
  python quav_qiskit.py --mode ibm --retrieve <JOB_ID>    # re-read a finished job without re-running
  python quav_qiskit.py --compare                         # figure from all saved results
"""
import argparse
import glob
import json
import os
import time

import numpy as np
from shapely.geometry import box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import quav_qaoa_v2 as q

# ----------------------------------------------------------------------
# Problem instances (qubits = edges left after pruning)
# ----------------------------------------------------------------------
INSTANCES = {
    "small": dict(rows=4, cols=4, zones=[box(40, -110, 110, -40)]),                       # 14 qubits
    "large": dict(rows=5, cols=5, zones=[box(40, -160, 160, -90), box(110, -60, 160, -20)]),  # 18 qubits
}


def build_problem(name):
    cfg = INSTANCES[name]
    q.NO_FLY = cfg["zones"]                          # risk_factor() reads this global
    start, goal = (0, 0), (cfg["rows"] - 1, cfg["cols"] - 1)
    G_full = q.build_graph(cfg["rows"], cfg["cols"], zones=q.NO_FLY)
    G = q.prune(G_full, start, goal)
    edges, lin, Q, const, _ = q.build_qubo(G, start, goal)
    n = len(edges)
    bits, cost = q.all_costs(lin, Q, const, n)
    k_opt = int(np.argmin(cost))
    feas = q.feasible_mask(bits, edges, G, start, goal)
    return dict(G=G, G_full=G_full, start=start, goal=goal, edges=edges, lin=lin, Q=Q,
                const=const, n=n, bits=bits, cost=cost, k_opt=k_opt, feas=feas)


# ----------------------------------------------------------------------
# Training (classical simulation, cached) and circuit construction
# ----------------------------------------------------------------------
def train(pb, p, name):
    cache = f"params_{name}_p{p}.json"
    if os.path.exists(cache):
        return np.array(json.load(open(cache)))
    print(f"Training QAOA angles for p={p} on the simulator (cached to {cache}) ...")
    cs = (pb["cost"] - pb["cost"].min()) / pb["cost"].std()
    rng = np.random.default_rng(q.SEED)
    prev = None
    for k in range(1, p + 1):
        starts = [np.concatenate([rng.uniform(0, 1, k), rng.uniform(0, np.pi / 2, k)])
                  for _ in range(4 if k == 1 else 2)]
        if prev is not None:
            starts.insert(0, q.interp_params(prev, k - 1, k))
        res, _ = q.optimise(cs, pb["n"], k, starts)
        prev = res.x
        print(f"  p={k}: <C>={res.fun:.3f}")
    json.dump(prev.tolist(), open(cache, "w"))
    return prev


def ising_terms(pb):
    """x = (1 - Z)/2 turns the QUBO into h_i Z_i + J_ij Z_i Z_j (scaled like the NumPy simulator)."""
    scale = pb["cost"].std()
    h = -pb["lin"] / 2.0
    J = {}
    for i, j in zip(*np.nonzero(pb["Q"])):
        qij = pb["Q"][i, j]
        h[i] -= qij / 4
        h[j] -= qij / 4
        J[(int(i), int(j))] = qij / 4
    return h / scale, {k: v / scale for k, v in J.items()}


def qaoa_circuit(pb, params, measure=True):
    n, p = pb["n"], len(params) // 2
    gammas, betas = params[:p], params[p:]
    h, J = ising_terms(pb)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for (i, j), v in J.items():
            qc.rzz(2 * g * v, i, j)
        for i in range(n):
            qc.rz(2 * g * h[i], i)
        qc.rx(2 * b, range(n))
    if measure:
        qc.measure_all()
    return qc


def verify_circuit(pb, params):
    """The Qiskit circuit must reproduce the NumPy simulator exactly."""
    cs = (pb["cost"] - pb["cost"].min()) / pb["cost"].std()
    ref = q.qaoa_probs(params, cs, pb["n"], len(params) // 2)
    got = Statevector(qaoa_circuit(pb, params, measure=False)).probabilities()
    err = float(np.abs(ref - got).max())
    print(f"Circuit check vs NumPy simulator: max |dP| = {err:.2e}")
    assert err < 1e-8, "Qiskit circuit does not match the NumPy QAOA"


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------
def get_backend(args):
    if args.mode == "fake":
        from qiskit_ibm_runtime.fake_provider import FakeFez
        return FakeFez(), None
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    kw = dict(channel="ibm_quantum_platform")
    if token:
        kw["token"] = token
    if args.crn:
        kw["instance"] = args.crn
    try:
        service = QiskitRuntimeService(**kw)    # without a token it uses a saved account
    except Exception as e:
        raise SystemExit(f"Could not connect to IBM Quantum: {e}\n"
                         "Set your key first:  export IBM_QUANTUM_TOKEN=\"your-api-key\"  "
                         "(PowerShell: $env:IBM_QUANTUM_TOKEN=\"your-api-key\")")
    if args.backend:
        backend = service.backend(args.backend)
    else:
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=156)
    return backend, service


def run_sampler(isa, shots, mode, backend, dd):
    if mode == "ideal":
        job = StatevectorSampler(seed=q.SEED).run([(isa,)], shots=shots)
        return job.result()[0].data.meas.get_counts(), None
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    if dd:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    job = sampler.run([isa], shots=shots)
    jid = job.job_id() if callable(getattr(job, "job_id", None)) else None
    if mode == "ibm":
        print(f"Submitted job {jid}  (retrieve later with --retrieve {jid})")
    return job.result()[0].data.meas.get_counts(), jid


def analyse(pb, counts, shots):
    G, edges, bits = pb["G"], pb["edges"], pb["bits"]
    total = sum(counts.values())
    valid = optimal = 0
    best = None
    sum_cost = 0.0
    for bs, c in counts.items():
        k = int(bs, 2)                         # qubit 0 is the right-most character
        sum_cost += c * pb["cost"][k]
        if pb["feas"][k]:
            valid += c
            path = q.decode(bits[k], edges, pb["start"], pb["goal"])
            pc = q.path_cost(G, path)
            if k == pb["k_opt"]:
                optimal += c
            if best is None or pc < best[0]:
                best = (pc, path)
    return dict(shots=total, valid_frac=valid / total, optimal_frac=optimal / total,
                random_valid_frac=float(pb["feas"].mean()), mean_qubo_cost=sum_cost / total,
                best_cost=best[0] if best else None, best_path=best[1] if best else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ideal", "fake", "ibm"], default="ideal")
    ap.add_argument("--instance", choices=list(INSTANCES), default="small",
                    help="small = 14 qubits (default), large = 18 qubits")
    ap.add_argument("--layers", type=int, default=2, help="QAOA depth p")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--backend", help="e.g. ibm_fez (default: least busy Heron)")
    ap.add_argument("--ibm-instance", dest="ibm_instance", help="optional CRN of your IBM instance")
    ap.add_argument("--no-dd", action="store_true", help="turn off dynamical decoupling")
    ap.add_argument("--dry-run", action="store_true", help="transpile only, do not submit")
    ap.add_argument("--retrieve", help="IBM job id to read results from")
    ap.add_argument("--compare", action="store_true", help="plot all saved results and exit")
    args = ap.parse_args()
    args.problem, args.crn = args.instance, args.ibm_instance           # map size vs IBM instance

    if args.compare:
        return compare()

    pb = build_problem(args.problem)
    n = pb["n"]
    G_full = pb["G_full"]
    p_dij = __import__("networkx").dijkstra_path(G_full, pb["start"], pb["goal"], weight="w")
    c_dij = q.path_cost(G_full, p_dij)
    print(f"Instance '{args.problem}': {n} qubits | valid paths {int(pb['feas'].sum())} of {2 ** n} "
          f"| Dijkstra cost {c_dij:.1f}")

    params = train(pb, args.layers, args.problem)
    verify_circuit(pb, params)
    qc = qaoa_circuit(pb, params)

    backend, service = (None, None) if args.mode == "ideal" else get_backend(args)
    if backend is not None:
        print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                          seed_transpiler=q.SEED)
        isa = pm.run(qc)
        ops = dict(isa.count_ops())
        n2q = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx", "rzz"))
        print(f"Transpiled: depth {isa.depth()} | two-qubit gates {n2q} | ops {ops}")
        # crude success estimate from the median two-qubit error reported by the device
        if args.mode == "fake" or service is not None:
            try:
                errs = [v.error for k, v in backend.target["cz"].items() if v is not None and v.error]
                med = float(np.median(errs))
                print(f"Median CZ error {med:.2e} -> rough circuit fidelity ~ {(1 - med) ** n2q:.3f}")
            except Exception:
                pass
    else:
        isa = qc

    if args.dry_run:
        print("Dry run: nothing submitted.")
        return

    t0 = time.perf_counter()
    jid = None
    if args.retrieve:
        job = service.job(args.retrieve)
        counts = job.result()[0].data.meas.get_counts()
        jid = args.retrieve
    else:
        counts, jid = run_sampler(isa, args.shots, args.mode, backend, not args.no_dd)
    print(f"Finished in {time.perf_counter() - t0:.0f} s wall-clock")

    res = analyse(pb, counts, args.shots)
    res.update(mode=args.mode, instance=args.problem, layers=args.layers, qubits=n,
               backend=backend.name if backend else "statevector", job_id=jid,
               dijkstra_cost=c_dij)
    ideal = q.qaoa_probs(params, (pb["cost"] - pb["cost"].min()) / pb["cost"].std(), n, args.layers)
    res["ideal_valid_frac"] = float(ideal[pb["feas"]].sum())
    res["ideal_optimal_frac"] = float(ideal[pb["k_opt"]])

    print("\n=== RESULT ===")
    print(f"Valid-path shots   : {100 * res['valid_frac']:.2f}%  "
          f"(noiseless QAOA {100 * res['ideal_valid_frac']:.2f}%, random {100 * res['random_valid_frac']:.4f}%)")
    print(f"Optimal-path shots : {100 * res['optimal_frac']:.2f}%  "
          f"(noiseless QAOA {100 * res['ideal_optimal_frac']:.2f}%)")
    print(f"Best valid path    : cost {res['best_cost']} vs Dijkstra {c_dij:.1f}")
    print(f"Path               : {res['best_path']}")
    fname = f"results_{args.problem}_{args.mode}_p{args.layers}.json"
    json.dump(res, open(fname, "w"), indent=1, default=str)
    print(f"Saved {fname}")


def compare():
    files = sorted(glob.glob("results_*.json"))
    if not files:
        raise SystemExit("No results_*.json files yet. Run the script first.")
    rows = [json.load(open(f)) for f in files]
    labels = [f"{r['instance']} p={r['layers']}\n{r['backend'] if r['mode'] != 'ideal' else 'ideal sim'}" for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(rows))
    ax[0].bar(x, [100 * r["valid_frac"] for r in rows], color="tab:blue", label="measured")
    ax[0].plot(x, [100 * r["random_valid_frac"] for r in rows], "r_", ms=30, mew=2, label="random guessing")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("shots that form a valid path (%)"); ax[0].legend(); ax[0].grid(alpha=.3, axis="y")
    ax[0].set_title("Feasibility")
    ax[1].bar(x, [100 * r["optimal_frac"] for r in rows], color="tab:green")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("shots giving the optimal path (%)"); ax[1].grid(alpha=.3, axis="y")
    ax[1].set_title("Optimality")
    fig.tight_layout(); fig.savefig("quav_hardware_compare.png", dpi=140)
    print("Saved quav_hardware_compare.png from", files)


if __name__ == "__main__":
    main()
