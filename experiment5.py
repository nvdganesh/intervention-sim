"""
E5: decouple operator capacity from model quality.

E3 could not identify mediation because saturation was a near-deterministic
function of AUC (|r| > 0.94). Here capacity (number of operators) is varied
independently of model quality, giving two separate sources of variation in
saturation and allowing the marginal return to model quality to be estimated
at different capacity levels.

Checkpointed like experiments.py. Run repeatedly until "ALL DONE".
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
CELLS = os.path.join(OUT, "cells_e5.jsonl")

N_REPS = 15
HORIZON = 10000.0
WARMUP = 2000.0


def base(**kw):
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
        "saturation": r.saturation,
        "action_rate": r.action_rate,
        "mean_wait": r.mean_wait,
        "mean_trust": r.mean_trust,
        "utilization": r.utilization,
        "delivered": r.delivered,
        "missed_tp": r.missed_tp,
    }


def build_grid():
    grid = []
    # fixed event stream; capacity varied independently of model quality
    ref = base()
    lam = calibrate_lambda(ref, 1.0)  # load 1.0 at the 2-operator reference
    for n_ops in [1, 2, 3, 4, 6, 8]:
        for a in [1.6, 1.9, 2.2, 2.5]:
            grid.append(
                {
                    "exp": "e5",
                    "n_operators": n_ops,
                    "a": a,
                    "auc": ND.cdf(a / math.sqrt(2)),
                    "cfg": {"lam": lam, "recall": 0.7, "a": a, "n_operators": n_ops},
                }
            )
    for i, c in enumerate(grid):
        c["id"] = f"e5_{i:04d}"
    return grid


def done_ids():
    if not os.path.exists(CELLS):
        return set()
    out = set()
    for line in open(CELLS):
        line = line.strip()
        if line:
            out.add(json.loads(line)["id"])
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
            runs = pool.map(_run, [(spec["cfg"], 1000 + k) for k in range(N_REPS)])
            out = {k: v for k, v in spec.items() if k != "cfg"}
            out["cfg"] = spec["cfg"]
            for key in runs[0]:
                m, hw = mean_ci([r[key] for r in runs])
                out[key] = {"mean": m, "ci": hw}
            fh.write(json.dumps(out) + "\n")
            fh.flush()
            n += 1
    left = len(todo) - n
    print(f"completed {n}, {left} remaining")
    if left == 0:
        print("ALL DONE")


if __name__ == "__main__":
    main()
