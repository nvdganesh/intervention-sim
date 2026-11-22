"""
Experiment runner for the intervention pipeline study.

E1  recall x load sweep            -> r*(rho), tests H1
E1b volume-controlled replication  -> confound control for H1
E2  alert budget vs unbudgeted     -> tests H2
E3  AUC -> saturation -> success    -> tests H3 (mediation)
E4  context vs recall improvement  -> tests H4

Checkpointed and resumable: completed cells are appended to results/cells.jsonl
and skipped on subsequent invocations. Run repeatedly until it reports "ALL DONE".

    python3 experiments.py [seconds_budget]

Deterministic given seeds; resuming does not change results.
"""

import json
import math
import os
import sys
import time
from multiprocessing import Pool
from statistics import NormalDist

from intervention_sim import Config, simulate, mean_ci, calibrate_lambda

ND = NormalDist()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
CELLS = os.path.join(OUT, "cells.jsonl")

N_REPS = 15
HORIZON = 10000.0
WARMUP = 2000.0
LOADS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
RECALLS = [round(0.10 + 0.05 * i, 2) for i in range(18)]  # 0.10 .. 0.95


def base(**kw) -> Config:
    c = Config(horizon=HORIZON, warmup=WARMUP)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _run(args):
    cfg_kw, seed = args
    c = base(**cfg_kw)
    c.seed = seed
    r = simulate(c)
    return {
        "success_per_hr": r.success_rate_per_hr,
        "action_rate": r.action_rate,
        "mean_wait": r.mean_wait,
        "p95_wait": r.wait_pct(0.95),
        "saturation": r.saturation,
        "utilization": r.utilization,
        "mean_trust": r.mean_trust,
        "delivered_precision": r.delivered_precision,
        "delivered": r.delivered,
        "reneged": r.reneged,
        "dropped_budget": r.dropped_budget,
        "missed_tp": r.missed_tp,
        "tp_events": r.tp_events,
        "wasted_time": r.wasted_time,
    }


# ---------------------------------------------------------------------------
# Cell specification -- the full experiment grid, built deterministically
# ---------------------------------------------------------------------------


def solve_a_for(recall, alert_frac, pi):
    """Discriminative power `a` giving `recall` at a fixed alert fraction."""
    f = (alert_frac - pi * recall) / (1 - pi)
    if not (1e-6 < f < 0.999):
        return None
    tau = -ND.inv_cdf(f)
    return ND.inv_cdf(recall) + tau


def build_grid():
    grid = []

    # --- E1: recall x load -------------------------------------------------
    for load in LOADS:
        lam = calibrate_lambda(base(), load)
        for r in RECALLS:
            grid.append(
                {"exp": "e1", "load": load, "recall": r, "cfg": {"lam": lam, "recall": r}}
            )

    # --- E1b: volume-controlled -------------------------------------------
    for load in [0.75, 1.25]:
        ref = base()
        lam = calibrate_lambda(ref, load)
        ref.lam = lam
        target_frac = ref.pi * 0.5 + (1 - ref.pi) * ref.fpr()
        for r in RECALLS:
            a = solve_a_for(r, target_frac, ref.pi)
            if a is None:
                continue
            grid.append(
                {
                    "exp": "e1b",
                    "load": load,
                    "recall": r,
                    "a": a,
                    "auc": ND.cdf(a / math.sqrt(2)),
                    "cfg": {"lam": lam, "recall": r, "a": a},
                }
            )

    # --- E2: alert budget --------------------------------------------------
    for load in LOADS:
        lam = calibrate_lambda(base(), load)
        for budget in [None, 10, 5, 3, 2, 1]:
            grid.append(
                {
                    "exp": "e2",
                    "load": load,
                    "budget": budget,
                    "cfg": {"lam": lam, "recall": 0.7, "budget": budget},
                }
            )

    # --- E3: mediation -----------------------------------------------------
    for load in [0.5, 1.0, 1.5]:
        lam = calibrate_lambda(base(), load)
        for a in [1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8]:
            grid.append(
                {
                    "exp": "e3",
                    "load": load,
                    "a": a,
                    "auc": ND.cdf(a / math.sqrt(2)),
                    "cfg": {"lam": lam, "recall": 0.7, "a": a},
                }
            )

    # --- E4: context vs recall --------------------------------------------
    for load in [0.5, 0.75, 1.0, 1.25]:
        lam = calibrate_lambda(base(), load)
        for ctx in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for r in [0.3, 0.5, 0.7]:
                grid.append(
                    {
                        "exp": "e4",
                        "load": load,
                        "context": ctx,
                        "recall": r,
                        "cfg": {"lam": lam, "recall": r, "context": ctx},
                    }
                )

    for i, cell in enumerate(grid):
        cell["id"] = f"{cell['exp']}_{i:04d}"
    return grid


def done_ids():
    if not os.path.exists(CELLS):
        return set()
    ids = set()
    with open(CELLS) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    return ids


def run_cell(pool, spec):
    jobs = [(spec["cfg"], 1000 + k) for k in range(N_REPS)]
    runs = pool.map(_run, jobs)
    out = {k: v for k, v in spec.items() if k != "cfg"}
    out["cfg"] = spec["cfg"]
    for key in runs[0]:
        m, hw = mean_ci([r[key] for r in runs])
        out[key] = {"mean": m, "ci": hw}
    out["_raw_success"] = [r["success_per_hr"] for r in runs]
    out["_raw_sat"] = [r["saturation"] for r in runs]
    return out


def main():
    budget_s = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0
    grid = build_grid()
    have = done_ids()
    todo = [c for c in grid if c["id"] not in have]
    print(f"grid={len(grid)} done={len(have)} todo={len(todo)}")
    if not todo:
        print("ALL DONE")
        return

    t0 = time.time()
    n = 0
    with Pool(2) as pool, open(CELLS, "a") as fh:
        for spec in todo:
            if time.time() - t0 > budget_s:
                break
            cell = run_cell(pool, spec)
            fh.write(json.dumps(cell) + "\n")
            fh.flush()
            n += 1
    left = len(todo) - n
    print(f"completed {n} cells this run, {left} remaining")
    if left == 0:
        print("ALL DONE")


if __name__ == "__main__":
    main()
