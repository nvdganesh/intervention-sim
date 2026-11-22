"""
Validation suite for the intervention pipeline simulator.

These are correctness checks, not experiments. Every check must pass before
any result in the paper is treated as meaningful.
"""

import math
from statistics import NormalDist

from intervention_sim import Config, simulate, replicate, agg, calibrate_lambda

ND = NormalDist()
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, PASS if ok else FAIL, detail))
    print(f"[{PASS if ok else FAIL}] {name}  {detail}")


# --- 1. ROC arithmetic ------------------------------------------------------
cfg = Config(a=1.8, recall=0.5)
emp_r, emp_f, n = 0, 0, 200000
import random

rng = random.Random(7)
tau = cfg.threshold()
tp = fp = pos = neg = 0
for _ in range(n):
    actionable = rng.random() < 0.5
    s = rng.gauss(cfg.a, 1) if actionable else rng.gauss(0, 1)
    if actionable:
        pos += 1
        tp += s > tau
    else:
        neg += 1
        fp += s > tau
check(
    "ROC recall matches target",
    abs(tp / pos - cfg.recall) < 0.01,
    f"empirical={tp/pos:.4f} target={cfg.recall}",
)
check(
    "ROC FPR matches closed form",
    abs(fp / neg - cfg.fpr()) < 0.01,
    f"empirical={fp/neg:.4f} closed={cfg.fpr():.4f}",
)

# --- 2. Event conservation --------------------------------------------------
cfg = Config()
cfg.lam = calibrate_lambda(cfg, 0.8)
r = simulate(cfg)
check(
    "TPs alerted <= TP events",
    r.tp_alerted <= r.tp_events,
    f"{r.tp_alerted} <= {r.tp_events}",
)
check(
    "successes <= acted",
    r.successes <= r.acted,
    f"{r.successes} <= {r.acted}",
)
check(
    "acted <= delivered",
    r.acted <= r.delivered,
    f"{r.acted} <= {r.delivered}",
)
check(
    "observation window equals horizon minus warmup",
    abs(r.obs_time - (cfg.horizon - cfg.warmup)) < 50,
    f"obs={r.obs_time:.0f} expected={cfg.horizon-cfg.warmup:.0f}",
)

# --- 3. Recall realized in simulation matches configured recall -------------
realized = r.tp_alerted / r.tp_events
check(
    "realized recall matches config",
    abs(realized - cfg.recall) < 0.03,
    f"realized={realized:.3f} config={cfg.recall}",
)

# --- 4. Utilization bounded -------------------------------------------------
check("utilization in [0,1]", 0.0 <= r.utilization <= 1.0, f"util={r.utilization:.3f}")

# --- 5. Low load => short waits; high load => long waits --------------------
lo = Config()
lo.lam = calibrate_lambda(lo, 0.25)
rlo = simulate(lo)
hi = Config()
hi.lam = calibrate_lambda(hi, 1.6)
rhi = simulate(hi)
check(
    "waiting increases with load",
    rhi.mean_wait > rlo.mean_wait,
    f"low={rlo.mean_wait:.2f}min high={rhi.mean_wait:.2f}min",
)
check(
    "saturation increases with load",
    rhi.saturation > rlo.saturation,
    f"low={rlo.saturation:.2f} high={rhi.saturation:.2f}",
)
check(
    "reneging occurs only under congestion",
    rlo.reneged == 0 and rhi.reneged > 0,
    f"low={rlo.reneged} high={rhi.reneged}",
)

# --- 6. Trust loop does not collapse to the floor everywhere ----------------
# (design doc section 4.6 sanity requirement)
trusts = {}
for pi in (0.02, 0.05, 0.15, 0.30):
    c = Config(pi=pi)
    c.lam = calibrate_lambda(c, 0.8)
    rr = simulate(c)
    trusts[pi] = rr.trust_final
spread = max(trusts.values()) - min(trusts.values())
check(
    "trust responds to base rate (no universal collapse)",
    spread > 0.10 and max(trusts.values()) > 0.3,
    f"trust by pi={ {k: round(v,3) for k,v in trusts.items()} }",
)

# --- 7. Trust tracks realized precision ------------------------------------
c = Config(pi=0.15)
c.lam = calibrate_lambda(c, 0.6)
rr = simulate(c)
observed_precision = rr.delivered_precision
check(
    "mean trust approximates realized precision of inspected alerts",
    abs(rr.mean_trust - observed_precision) < 0.05,
    f"mean_trust={rr.mean_trust:.3f} precision={observed_precision:.3f} "
    f"(terminal snapshot was {rr.trust_final:.3f} -- noisy by design)",
)

# --- 8. Budget actually caps concurrency -----------------------------------
c = Config(budget=3)
c.lam = calibrate_lambda(c, 1.5)
rb = simulate(c)
check(
    "alert budget caps saturation",
    rb.saturation <= 3.0 / c.n_operators + 0.05,
    f"sat={rb.saturation:.3f} cap={3.0/c.n_operators:.3f}",
)
check("budget drops alerts under load", rb.dropped_budget > 0, f"dropped={rb.dropped_budget}")

# --- 9. Determinism ---------------------------------------------------------
c1 = Config(seed=42)
c1.lam = calibrate_lambda(c1, 0.8)
c2 = Config(seed=42)
c2.lam = c1.lam
check(
    "same seed reproduces identical output",
    simulate(c1).successes == simulate(c2).successes,
)

# --- 10. Replication CI half-width within tolerance -------------------------
c = Config()
c.lam = calibrate_lambda(c, 0.8)
reps = replicate(c, n_reps=20)
m, hw = agg(reps, "success_rate_per_hr")
check(
    "CI half-width < 2% of mean at 20 reps",
    hw / m < 0.02,
    f"mean={m:.3f} halfwidth={hw:.4f} ({100*hw/m:.2f}%)",
)

# --- 11. Warm-up adequacy: split-half stability -----------------------------
c_short = Config(horizon=11000, warmup=2000)
c_short.lam = calibrate_lambda(c_short, 0.8)
c_long = Config(horizon=20000, warmup=2000)
c_long.lam = c_short.lam
a1, _ = agg(replicate(c_short, 10), "success_rate_per_hr")
a2, _ = agg(replicate(c_long, 10), "success_rate_per_hr")
check(
    "steady state reached (short vs long horizon agree)",
    abs(a1 - a2) / a2 < 0.03,
    f"short={a1:.3f} long={a2:.3f} diff={100*abs(a1-a2)/a2:.2f}%",
)

# --- summary ----------------------------------------------------------------
print()
n_fail = sum(1 for _, s, _ in results if s == FAIL)
print(f"{len(results) - n_fail}/{len(results)} checks passed")
if n_fail:
    print("FAILURES:")
    for name, s, detail in results:
        if s == FAIL:
            print(f"  - {name}: {detail}")
    raise SystemExit(1)
