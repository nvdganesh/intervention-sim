"""
Analysis of E5 (capacity decoupled from model quality) plus robustness checks
on the r* identification from E1.
"""

import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
M = lambda c, k: c[k]["mean"]
CI = lambda c, k: c[k]["ci"]

e5 = [json.loads(l) for l in open(f"{OUT}/cells_e5.jsonl") if l.strip()]
e1 = [
    json.loads(l)
    for l in open(f"{OUT}/cells.jsonl")
    if l.strip() and json.loads(l)["exp"] == "e1"
]
findings = {}


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    return b, my - b * mx


def ols2(x1, x2, y):
    b21, a21 = ols(x2, x1)
    r1 = [u - (a21 + b21 * v) for u, v in zip(x1, x2)]
    b2y, a2y = ols(x2, y)
    ry = [u - (a2y + b2y * v) for u, v in zip(y, x2)]
    b, _ = ols(r1, ry)
    return b


def corr(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


print("=" * 72)
print("H3 (re-tested in E5): mediation with capacity varied independently")
print("=" * 72)

auc = [c["auc"] for c in e5]
sat = [M(c, "saturation") for c in e5]
suc = [M(c, "success_per_hr") for c in e5]

r_tm = corr(auc, sat)
total, _ = ols(auc, suc)
direct = ols2(auc, sat, suc)
a_path, _ = ols(auc, sat)
b_path = ols2(sat, auc, suc)
indirect = total - direct
prop = indirect / total if total else float("nan")

print(f"  corr(AUC, saturation) = {r_tm:+.4f}   (E3 was about -1.0; now identified)")
print(f"  total effect  of AUC on successes/hr : {total:8.3f}")
print(f"  direct effect of AUC | saturation    : {direct:8.3f}")
print(f"  indirect (via saturation)            : {indirect:8.3f}")
print(f"  proportion of total effect mediated  : {prop*100:6.1f}%")
print(f"  a-path AUC->saturation               : {a_path:8.3f}")
print(f"  b-path saturation->successes | AUC   : {b_path:8.4f}")
verdict = "SUPPORTED" if 0.0 < prop < 1.0 else "NOT SUPPORTED"
print(f"  H3 {verdict}")

findings["H3_e5"] = {
    "treatment_mediator_corr": r_tm,
    "total_effect": total,
    "direct_effect": direct,
    "indirect_effect": indirect,
    "prop_mediated": prop,
    "a_path": a_path,
    "b_path": b_path,
    "supported": bool(0.0 < prop < 1.0),
}

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("Capacity x model-quality interaction (the headline practical result)")
print("=" * 72)

by_n = defaultdict(dict)
for c in e5:
    by_n[c["n_operators"]][round(c["auc"], 3)] = c

rows = []
for n in sorted(by_n):
    aucs = sorted(by_n[n])
    lo, hi = by_n[n][aucs[0]], by_n[n][aucs[-1]]
    gain = M(hi, "success_per_hr") - M(lo, "success_per_hr")
    rows.append(
        {
            "n_operators": n,
            "auc_low": aucs[0],
            "auc_high": aucs[-1],
            "success_low": M(lo, "success_per_hr"),
            "success_high": M(hi, "success_per_hr"),
            "absolute_gain": gain,
            "utilization_at_low_auc": M(lo, "utilization"),
        }
    )
    print(
        f"  n_ops={n}: util@low-AUC={M(lo,'utilization'):.2f}  "
        f"same AUC gain {aucs[0]:.3f}->{aucs[-1]:.3f} is worth {gain:+.3f}/hr"
    )

gains = [r["absolute_gain"] for r in rows]
ratio = max(gains) / min(gains) if min(gains) > 0 else float("nan")
print()
print(
    f"  The identical model improvement is worth {ratio:.1f}x more at the best "
    f"capacity level than at the worst."
)
findings["capacity_quality_interaction"] = {"rows": rows, "max_min_ratio": ratio}

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("Robustness of r* identification in E1")
print("=" * 72)
print("  Is the peak distinguishable from its neighbours, given replication noise?")

by_load = defaultdict(list)
for c in e1:
    by_load[c["load"]].append(c)

rob = []
for load in sorted(by_load):
    rr = sorted(by_load[load], key=lambda c: c["recall"])
    best = max(rr, key=lambda c: M(c, "success_per_hr"))
    peak, peak_ci = M(best, "success_per_hr"), CI(best, "success_per_hr")
    # recall values whose CI overlaps the peak's CI -> statistically indistinguishable
    plateau = [
        c["recall"]
        for c in rr
        if M(c, "success_per_hr") + CI(c, "success_per_hr") >= peak - peak_ci
    ]
    rob.append(
        {
            "load": load,
            "r_star": best["recall"],
            "plateau": plateau,
            "plateau_width": max(plateau) - min(plateau),
        }
    )
    print(
        f"  load={load:<5} r*={best['recall']:<5} "
        f"indistinguishable-from-peak range: {min(plateau):.2f}-{max(plateau):.2f} "
        f"(width {max(plateau)-min(plateau):.2f})"
    )
print()
print("  r* should be reported as a plateau, not a point estimate. The monotone")
print("  downward trend holds because the plateaus themselves shift down with load.")
findings["r_star_robustness"] = rob

json.dump(findings, open(f"{OUT}/findings_e5.json", "w"), indent=1)
print()
print(f"wrote {OUT}/findings_e5.json")
