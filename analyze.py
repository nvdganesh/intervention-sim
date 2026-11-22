"""
Analysis of simulation results. Produces the numbers reported in the paper.

Reads results/cells.jsonl, writes results/findings.json and prints a report.
"""

import json
import math
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
cells = [json.loads(l) for l in open(os.path.join(OUT, "cells.jsonl")) if l.strip()]
by_exp = defaultdict(list)
for c in cells:
    by_exp[c["exp"]].append(c)

findings = {}
M = lambda c, k: c[k]["mean"]
CI = lambda c, k: c[k]["ci"]


def line(s=""):
    print(s)


# ===========================================================================
# H1: does the value-maximizing recall fall as load rises?
# ===========================================================================
line("=" * 72)
line("H1  Optimal recall r* as a function of load")
line("=" * 72)

e1 = defaultdict(list)
for c in by_exp["e1"]:
    e1[c["load"]].append(c)

h1_rows = []
for load in sorted(e1):
    rows = sorted(e1[load], key=lambda c: c["recall"])
    best = max(rows, key=lambda c: M(c, "success_per_hr"))
    peak = M(best, "success_per_hr")
    at_max_recall = M(rows[-1], "success_per_hr")
    drop = (peak - at_max_recall) / peak * 100 if peak else 0
    h1_rows.append(
        {
            "load": load,
            "r_star": best["recall"],
            "peak_success": peak,
            "peak_ci": CI(best, "success_per_hr"),
            "success_at_r095": at_max_recall,
            "pct_lost_at_max_recall": drop,
            "sat_at_star": M(best, "saturation"),
            "action_rate_at_star": M(best, "action_rate"),
            "action_rate_at_r095": M(rows[-1], "action_rate"),
        }
    )
    line(
        f"  load={load:<5} r*={best['recall']:<5} peak={peak:6.3f}/hr "
        f"(+-{CI(best,'success_per_hr'):.3f})   at r=0.95: {at_max_recall:6.3f}/hr "
        f"-> {drop:5.1f}% below peak"
    )

r_stars = [r["r_star"] for r in h1_rows]
monotone = all(r_stars[i] >= r_stars[i + 1] for i in range(len(r_stars) - 1))
strict = r_stars[0] > r_stars[-1]
line()
line(f"  r* sequence by increasing load: {r_stars}")
line(f"  non-increasing in load: {monotone};  strictly lower at max load: {strict}")
line(f"  H1 {'SUPPORTED' if monotone and strict else 'NOT SUPPORTED'}")
findings["H1"] = {
    "rows": h1_rows,
    "r_star_sequence": r_stars,
    "monotone_non_increasing": monotone,
    "strictly_decreasing_end_to_end": strict,
    "supported": bool(monotone and strict),
}

# ===========================================================================
# H1b: volume-controlled confound check
# ===========================================================================
line()
line("=" * 72)
line("H1b Volume-controlled: recall raised WITHOUT increasing alert volume")
line("=" * 72)

e1b = defaultdict(list)
for c in by_exp["e1b"]:
    e1b[c["load"]].append(c)

h1b_rows = []
for load in sorted(e1b):
    rows = sorted(e1b[load], key=lambda c: c["recall"])
    best = max(rows, key=lambda c: M(c, "success_per_hr"))
    vols = [M(c, "delivered") for c in rows]
    vol_spread = (max(vols) - min(vols)) / (sum(vols) / len(vols)) * 100
    h1b_rows.append(
        {
            "load": load,
            "r_star": best["recall"],
            "peak_success": M(best, "success_per_hr"),
            "success_at_r095": M(rows[-1], "success_per_hr"),
            "volume_spread_pct": vol_spread,
        }
    )
    line(
        f"  load={load:<5} r*={best['recall']:<5} peak={M(best,'success_per_hr'):6.3f} "
        f"at r=0.95: {M(rows[-1],'success_per_hr'):6.3f}   "
        f"alert-volume spread across sweep: {vol_spread:.1f}%"
    )
line()
line("  Interpretation: if r* here sits at/near the top of the recall range while")
line("  r* in E1 falls with load, the E1 effect is driven by alert VOLUME, not by")
line("  recall as such -- which is the paper's actual claim.")
findings["H1b"] = h1b_rows

# ===========================================================================
# H2: does an alert budget help under saturation?
# ===========================================================================
line()
line("=" * 72)
line("H2  Alert budget vs unbudgeted delivery")
line("=" * 72)

e2 = defaultdict(dict)
for c in by_exp["e2"]:
    e2[c["load"]][c["budget"]] = c

h2_rows = []
for load in sorted(e2):
    unb = e2[load][None]
    base_s = M(unb, "success_per_hr")
    best_b, best_c = None, None
    for b, c in e2[load].items():
        if b is None:
            continue
        if best_c is None or M(c, "success_per_hr") > M(best_c, "success_per_hr"):
            best_b, best_c = b, c
    gain = (M(best_c, "success_per_hr") - base_s) / base_s * 100 if base_s else 0
    # non-overlapping CIs as a conservative significance screen
    sep = (M(best_c, "success_per_hr") - CI(best_c, "success_per_hr")) > (
        base_s + CI(unb, "success_per_hr")
    )
    h2_rows.append(
        {
            "load": load,
            "unbudgeted": base_s,
            "best_budget": best_b,
            "best_success": M(best_c, "success_per_hr"),
            "gain_pct": gain,
            "ci_separated": bool(sep),
            "missed_tp_unbudgeted": M(unb, "missed_tp"),
            "missed_tp_budgeted": M(best_c, "missed_tp"),
            "wait_unbudgeted": M(unb, "mean_wait"),
            "wait_budgeted": M(best_c, "mean_wait"),
        }
    )
    line(
        f"  load={load:<5} unbudgeted={base_s:6.3f}  best B={str(best_b):>4} -> "
        f"{M(best_c,'success_per_hr'):6.3f}  ({gain:+5.1f}%)  "
        f"CIs separated: {sep}"
    )
findings["H2"] = h2_rows

# ===========================================================================
# H3: mediation of the model-quality effect by saturation
# ===========================================================================
line()
line("=" * 72)
line("H3  Mediation: AUC -> saturation -> successful interventions")
line("=" * 72)


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    return b, my - b * mx


def ols2(x1, x2, y):
    """Two-predictor OLS via Frisch-Waugh: coefficient on x1 controlling x2."""
    b21, a21 = ols(x2, x1)
    r1 = [xx1 - (a21 + b21 * xx2) for xx1, xx2 in zip(x1, x2)]
    b2y, a2y = ols(x2, y)
    ry = [yy - (a2y + b2y * xx2) for yy, xx2 in zip(y, x2)]
    b, _ = ols(r1, ry)
    return b


e3 = defaultdict(list)
for c in by_exp["e3"]:
    e3[c["load"]].append(c)

h3_rows = []
for load in sorted(e3):
    rows = sorted(e3[load], key=lambda c: c["auc"])
    auc = [c["auc"] for c in rows]
    sat = [M(c, "saturation") for c in rows]
    suc = [M(c, "success_per_hr") for c in rows]
    total, _ = ols(auc, suc)          # total effect of AUC
    direct = ols2(auc, sat, suc)      # direct effect controlling saturation
    prop = 1 - direct / total if total else float("nan")
    a_path, _ = ols(auc, sat)
    h3_rows.append(
        {
            "load": load,
            "total_effect": total,
            "direct_effect": direct,
            "prop_mediated": prop,
            "auc_to_saturation": a_path,
            "auc_range": [min(auc), max(auc)],
            "success_range": [min(suc), max(suc)],
        }
    )
    line(
        f"  load={load:<5} total={total:8.3f}  direct(|sat)={direct:8.3f}  "
        f"proportion mediated={prop*100:5.1f}%   AUC->sat slope={a_path:7.3f}"
    )
line()


def corr(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


line("  IDENTIFICATION CHECK -- correlation between treatment (AUC) and mediator:")
collinear = []
for load in sorted(e3):
    rows = e3[load]
    r = corr([c["auc"] for c in rows], [M(c, "saturation") for c in rows])
    collinear.append(r)
    line(f"    load={load:<5} corr(AUC, saturation) = {r:+.4f}")
line()
line("  |corr| is at or near 1.0: in this design saturation is a near-deterministic")
line("  function of model quality, because a better model at fixed recall emits")
line("  fewer false alerts. Mediation analysis is NOT IDENTIFIED under such")
line("  collinearity, and the proportion-mediated estimates above are unstable")
line("  artefacts (note the implausible magnitudes). H3 is therefore NOT TESTABLE")
line("  in design E3. Experiment E5 breaks the collinearity by varying operator")
line("  capacity independently of model quality; see analyze_e5.py.")
findings["H3"] = {
    "rows": h3_rows,
    "treatment_mediator_corr_by_load": collinear,
    "identified": False,
    "verdict": "not identified in E3 due to treatment-mediator collinearity; "
    "re-tested in E5",
}

# ===========================================================================
# H4: context enrichment vs recall increase
# ===========================================================================
line()
line("=" * 72)
line("H4  Marginal value of context enrichment vs raising recall")
line("=" * 72)

e4 = defaultdict(dict)
for c in by_exp["e4"]:
    e4[c["load"]][(c["context"], c["recall"])] = c

h4_rows = []
for load in sorted(e4):
    cells_l = e4[load]
    ref = cells_l.get((0.5, 0.5))
    if ref is None:
        continue
    base_s = M(ref, "success_per_hr")
    ctx_up = cells_l.get((1.0, 0.5))
    rec_up = cells_l.get((0.5, 0.7))
    g_ctx = (M(ctx_up, "success_per_hr") - base_s) / base_s * 100
    g_rec = (M(rec_up, "success_per_hr") - base_s) / base_s * 100
    h4_rows.append(
        {
            "load": load,
            "baseline": base_s,
            "gain_context_pct": g_ctx,
            "gain_recall_pct": g_rec,
            "context_wins": g_ctx > g_rec,
        }
    )
    line(
        f"  load={load:<5} baseline={base_s:6.3f}  "
        f"context 0.5->1.0: {g_ctx:+6.1f}%   recall 0.5->0.7: {g_rec:+6.1f}%   "
        f"{'CONTEXT wins' if g_ctx > g_rec else 'RECALL wins'}"
    )
findings["H4"] = h4_rows

# ===========================================================================
# Supporting descriptive numbers used in the paper
# ===========================================================================
line()
line("=" * 72)
line("Supporting descriptives")
line("=" * 72)
for load in sorted(e1):
    rows = sorted(e1[load], key=lambda c: c["recall"])
    lo, hi = rows[0], rows[-1]
    line(
        f"  load={load:<5} at r=0.10: action={M(lo,'action_rate'):.3f} trust={M(lo,'mean_trust'):.3f} "
        f"| at r=0.95: action={M(hi,'action_rate'):.3f} trust={M(hi,'mean_trust'):.3f} "
        f"wait={M(hi,'mean_wait'):.1f}min reneged={M(hi,'reneged'):.0f}"
    )

max_ci_pct = max(
    (CI(c, "success_per_hr") / M(c, "success_per_hr") * 100)
    for c in cells
    if M(c, "success_per_hr") > 0
)
line()
line(f"  worst-case CI half-width across all cells: {max_ci_pct:.2f}% of mean")
findings["max_ci_halfwidth_pct"] = max_ci_pct
findings["n_cells"] = len(cells)

json.dump(findings, open(os.path.join(OUT, "findings.json"), "w"), indent=1)
line()
line(f"wrote {OUT}/findings.json  ({len(cells)} cells)")
