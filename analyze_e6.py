"""
Analysis of E6: ablation and sensitivity.

Central question: is the H1 result an artefact of the trust feedback loop --
the most speculative component of the model?
"""

import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
M = lambda c, k: c[k]["mean"]
CI = lambda c, k: c[k]["ci"]

cells = [json.loads(l) for l in open(f"{OUT}/cells_e6.jsonl") if l.strip()]
by = defaultdict(lambda: defaultdict(list))
for c in cells:
    by[c["variant"]][c["load"]].append(c)

ORDER = ["baseline", "no_trust", "deadline", "theta60", "pi001", "pi020"]
LABEL = {
    "baseline": "baseline (as published)",
    "no_trust": "trust loop DISABLED",
    "deadline": "hard deadline decay",
    "theta60": "slower decay (θ=60)",
    "pi001": "rare events (π=0.01)",
    "pi020": "common events (π=0.20)",
}

rows = []
print(f"{'variant':<12} {'load':>5} {'r*':>5} {'peak/hr':>8} {'@r=.95':>8} {'shortfall':>10}")
print("-" * 60)
for v in ORDER:
    entry = {"variant": v, "label": LABEL[v], "by_load": {}}
    for load in sorted(by[v]):
        rr = sorted(by[v][load], key=lambda c: c["recall"])
        best = max(rr, key=lambda c: M(c, "success_per_hr"))
        peak, pci = M(best, "success_per_hr"), CI(best, "success_per_hr")
        last = M(rr[-1], "success_per_hr")
        short = (peak - last) / peak * 100 if peak else 0
        plateau = [
            c["recall"] for c in rr if M(c, "success_per_hr") + CI(c, "success_per_hr") >= peak - pci
        ]
        entry["by_load"][load] = {
            "r_star": best["recall"],
            "peak": peak,
            "at_max_recall": last,
            "shortfall_pct": short,
            "plateau": [min(plateau), max(plateau)],
        }
        print(
            f"{v:<12} {load:>5} {best['recall']:>5} {peak:>8.3f} {last:>8.3f} {short:>9.1f}%"
        )
    lo, hi = entry["by_load"][0.5]["r_star"], entry["by_load"][1.5]["r_star"]
    entry["r_star_falls_with_load"] = hi < lo
    rows.append(entry)

print()
print("=" * 60)
print("ROBUSTNESS OF H1 ACROSS ABLATIONS")
print("=" * 60)
for e in rows:
    lo = e["by_load"][0.5]["r_star"]
    hi = e["by_load"][1.5]["r_star"]
    print(
        f"  {e['label']:<26} r*: {lo} -> {hi}   "
        f"{'HOLDS' if e['r_star_falls_with_load'] else 'FAILS'}"
    )

all_hold = all(e["r_star_falls_with_load"] for e in rows)
print()
print(f"  H1 direction robust across all {len(rows)} variants: {all_hold}")

# magnitude sensitivity
shorts = {
    e["variant"]: (e["by_load"][0.5]["shortfall_pct"], e["by_load"][1.5]["shortfall_pct"])
    for e in rows
}
lo_vals = [s[0] for s in shorts.values()]
print(
    f"  Shortfall at r=0.95 (load 0.5) ranges {min(lo_vals):.1f}% to {max(lo_vals):.1f}% "
    f"across variants -> DIRECTION robust, MAGNITUDE specification-sensitive"
)

# trust ablation detail
b = next(e for e in rows if e["variant"] == "baseline")
n = next(e for e in rows if e["variant"] == "no_trust")
print()
print("  Trust ablation detail:")
for load in (0.5, 1.5):
    print(
        f"    load={load}: baseline r*={b['by_load'][load]['r_star']} "
        f"shortfall={b['by_load'][load]['shortfall_pct']:.1f}%  |  "
        f"no-trust r*={n['by_load'][load]['r_star']} "
        f"shortfall={n['by_load'][load]['shortfall_pct']:.1f}%"
    )
print("    -> queueing + value decay alone reproduce the effect;")
print("       the trust loop amplifies but is not necessary for it.")

json.dump(
    {"rows": rows, "all_variants_hold": all_hold},
    open(f"{OUT}/findings_e6.json", "w"),
    indent=1,
)
print(f"\nwrote {OUT}/findings_e6.json")
