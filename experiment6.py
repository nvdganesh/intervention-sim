"""
E6: ablation and sensitivity.

The central question: does the H1 saturation result depend on the trust
feedback loop -- the most speculative component of the model? If the inverted-U
disappears when trust is disabled, the finding is an artefact of an assumed
behavioural mechanism and must be reported as such.

Variants:
  baseline        as published
  no_trust        trust loop disabled, compliance fixed at 0.6
  deadline        hard actionability deadline instead of exponential decay
  theta60         slower value decay (60 min instead of 30)
  pi001           rare events (base rate 0.01 instead of 0.05)
  pi020           common events (base rate 0.20)

Checkpointed. Run repeatedly until "ALL DONE".
"""

import json
import math
import os
import sys
import time
from multiprocessing import Pool

from intervention_sim import Config, simulate, mean_ci, calibrate_lambda

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
CELLS = os.path.join(OUT, "cells_e6.jsonl")

N_REPS = 15
HORIZON = 10000.0
WARMUP = 2000.0
LOADS = [0.5, 1.5]
RECALLS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

VARIANTS = {
    "baseline": {},
    "no_trust": {"trust_enabled": False, "fixed_compliance": 0.6},
    "deadline": {"decay_mode": "deadline"},
    "theta60": {"theta": 60.0},
    "pi001": {"pi": 0.01},
    "pi020": {"pi": 0.20},
}


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
        "reneged": r.reneged,
        "delivered": r.delivered,
        "missed_tp": r.missed_tp,
    }


def build_grid():
    grid = []
    for vname, vkw in VARIANTS.items():
        for load in LOADS:
            # calibrate arrival rate under this variant's own parameters
            lam = calibrate_lambda(base(**vkw), load)
            for r in RECALLS:
                cfg = {"lam": lam, "recall": r}
                cfg.update(vkw)
                grid.append(
                    {"exp": "e6", "variant": vname, "load": load, "recall": r, "cfg": cfg}
                )
    for i, c in enumerate(grid):
        c["id"] = f"e6_{i:04d}"
    return grid


def done_ids():
    if not os.path.exists(CELLS):
        return set()
    return {json.loads(l)["id"] for l in open(CELLS) if l.strip()}


def main():
    budget_s = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0
    grid = build_grid()
    have = done_ids()
    todo = [c for c in grid if c["id"] not in have]
    print(f"grid={len(grid)} done={len(have)} todo={len(todo)}")
    if not todo:
        print("ALL DONE")
        return
    t0, n = time.time(), 0
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
