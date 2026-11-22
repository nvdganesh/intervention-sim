"""Generate the paper's figures from the simulation results."""

import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

M = lambda c, k: c[k]["mean"]
CI = lambda c, k: c[k]["ci"]

cells = [json.loads(l) for l in open(f"{OUT}/cells.jsonl") if l.strip()]
e5 = [json.loads(l) for l in open(f"{OUT}/cells_e5.jsonl") if l.strip()]
by_exp = defaultdict(list)
for c in cells:
    by_exp[c["exp"]].append(c)

def save(fig, path_png):
    """Write both a PNG preview and a vector PDF for LaTeX."""
    fig.savefig(path_png)
    fig.savefig(path_png.replace(".png", ".pdf"))


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

# ---------------------------------------------------------------- Figure 1
fig, ax = plt.subplots(figsize=(6.0, 3.6))
byload = defaultdict(list)
for c in by_exp["e1"]:
    byload[c["load"]].append(c)
cmap = plt.get_cmap("viridis")
loads = sorted(byload)
for i, load in enumerate(loads):
    rows = sorted(byload[load], key=lambda c: c["recall"])
    x = [c["recall"] for c in rows]
    y = [M(c, "success_per_hr") for c in rows]
    e = [CI(c, "success_per_hr") for c in rows]
    col = cmap(i / max(1, len(loads) - 1))
    ax.errorbar(x, y, yerr=e, marker="o", ms=3, lw=1.4, color=col, label=f"ρ={load}")
    bi = max(range(len(y)), key=lambda k: y[k])
    ax.plot(x[bi], y[bi], marker="*", ms=14, color=col, mec="black", mew=0.6, zorder=5)
ax.set_xlabel("Model recall (threshold operating point)")
ax.set_ylabel("Successful interventions / hour")
ax.set_title("Value-maximising recall falls as offered load rises\n(★ = optimum)", fontsize=10)
ax.legend(fontsize=7, ncol=2)
save(fig, f"{FIG}/fig1_recall_load.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
b_by = defaultdict(list)
for c in by_exp["e1b"]:
    b_by[c["load"]].append(c)
for ax, load in zip(axes, sorted(b_by)):
    rows_n = sorted(byload[load], key=lambda c: c["recall"])
    rows_c = sorted(b_by[load], key=lambda c: c["recall"])
    ax.plot(
        [c["recall"] for c in rows_n],
        [M(c, "success_per_hr") for c in rows_n],
        "o-", ms=3, lw=1.4, label="threshold lowered\n(alert volume grows)",
    )
    ax.plot(
        [c["recall"] for c in rows_c],
        [M(c, "success_per_hr") for c in rows_c],
        "s--", ms=3, lw=1.4, label="model improved\n(volume held constant)",
    )
    ax.set_title(f"ρ = {load}", fontsize=9)
    ax.set_xlabel("Recall")
    ax.legend(fontsize=6.5)
axes[0].set_ylabel("Successful interventions / hour")
fig.suptitle(
    "Recall only destroys value when it increases alert volume", fontsize=10, y=1.02
)
save(fig, f"{FIG}/fig2_volume_control.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
fig, ax = plt.subplots(figsize=(6.0, 3.4))
e2 = defaultdict(dict)
for c in by_exp["e2"]:
    e2[c["load"]][c["budget"]] = c
budgets = [None, 10, 5, 3, 2, 1]
labels = ["none", "10", "5", "3", "2", "1"]
w = 0.13
xs = range(len(sorted(e2)))
for j, (b, lab) in enumerate(zip(budgets, labels)):
    ys = [M(e2[load][b], "success_per_hr") for load in sorted(e2)]
    es = [CI(e2[load][b], "success_per_hr") for load in sorted(e2)]
    ax.bar(
        [x + (j - 2.5) * w for x in xs], ys, w, yerr=es, capsize=1.5,
        label=f"B={lab}", color=plt.get_cmap("plasma")(j / 5), edgecolor="white", lw=0.4,
    )
ax.set_xticks(list(xs))
ax.set_xticklabels([str(l) for l in sorted(e2)])
ax.set_xlabel("Offered load ρ")
ax.set_ylabel("Successful interventions / hour")
ax.set_title("Alert budgeting under saturation (recall fixed at 0.7)", fontsize=10)
ax.legend(fontsize=7, ncol=3)
save(fig, f"{FIG}/fig3_budget.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
fig, ax = plt.subplots(figsize=(6.0, 3.4))
by_n = defaultdict(dict)
for c in e5:
    by_n[c["n_operators"]][round(c["auc"], 3)] = c
aucs = sorted({round(c["auc"], 3) for c in e5})
cmap2 = plt.get_cmap("coolwarm")
for i, n in enumerate(sorted(by_n)):
    ys = [M(by_n[n][a], "success_per_hr") for a in aucs]
    es = [CI(by_n[n][a], "success_per_hr") for a in aucs]
    ax.errorbar(
        aucs, ys, yerr=es, marker="o", ms=4, lw=1.5,
        color=cmap2(i / (len(by_n) - 1)), label=f"{n} operator{'s' if n>1 else ''}",
    )
ax.set_xlabel("Model quality (AUC)")
ax.set_ylabel("Successful interventions / hour")
ax.set_title(
    "The same model improvement is worth ~11× more when capacity exists", fontsize=10
)
ax.legend(fontsize=7, ncol=2)
save(fig, f"{FIG}/fig4_capacity_quality.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 5
fig, ax = plt.subplots(figsize=(6.0, 3.4))
e4 = defaultdict(dict)
for c in by_exp["e4"]:
    e4[c["load"]][(c["context"], c["recall"])] = c
loads4 = sorted(e4)
gc, gr = [], []
for load in loads4:
    base = M(e4[load][(0.5, 0.5)], "success_per_hr")
    gc.append((M(e4[load][(1.0, 0.5)], "success_per_hr") - base) / base * 100)
    gr.append((M(e4[load][(0.5, 0.7)], "success_per_hr") - base) / base * 100)
x = range(len(loads4))
ax.bar([i - 0.19 for i in x], gc, 0.38, label="context 0.5 → 1.0", color="#2a9d8f")
ax.bar([i + 0.19 for i in x], gr, 0.38, label="recall 0.5 → 0.7", color="#e76f51")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels([str(l) for l in loads4])
ax.set_xlabel("Offered load ρ")
ax.set_ylabel("Change in successful interventions (%)")
ax.set_title("Context enrichment vs raising recall", fontsize=10)
ax.legend(fontsize=8)
save(fig, f"{FIG}/fig5_context_vs_recall.png")
plt.close(fig)

print("figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    print("  ", f)
