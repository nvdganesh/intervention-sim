# intervention-sim

Simulation artifact for *"Alert Saturation Bounds the Operational Value of
Predictive Models: A Simulation Study of Human-in-the-Loop Intervention
Pipelines."*

No proprietary or field data is used. All parameters are synthetic.

## Contents

| File | Purpose |
|---|---|
| `intervention_sim.py` | The discrete-event simulator (no third-party dependencies) |
| `validate.py` | 18-check correctness suite — run this first |
| `experiments.py` | Experiments E1, E1b, E2, E3, E4 (checkpointed/resumable) |
| `experiment5.py` | Experiment E5: capacity decoupled from model quality |
| `experiment6.py` | Experiment E6: ablation and sensitivity (incl. trust loop off) |
| `analyze.py` | Hypothesis tests H1, H1b, H2, H3 (E3), H4 |
| `analyze_e5.py` | H3 re-test in E5, capacity×quality interaction, r\* robustness |
| `analyze_e6.py` | Ablation: does the result depend on the trust model? |
| `figures.py` | Figures 1–5 (requires matplotlib) |
| `results/` | Raw per-cell output (JSONL) and computed findings (JSON) |
| `figures/` | Generated figures (PNG) |

## Reproducing

```bash
python3 validate.py          # must print 18/18 checks passed

# checkpointed: re-run until each prints ALL DONE
python3 experiments.py 300
python3 experiment5.py 300
python3 experiment6.py 300

python3 analyze.py
python3 analyze_e5.py
python3 analyze_e6.py
python3 figures.py           # needs matplotlib
```

Total compute: roughly 12 minutes on 2 cores for all 405 configurations
(6,075 runs at 15 replications each). Results are deterministic given seeds;
resuming a partial run does not change output.

`experiments.py` and `experiment5.py` take an optional time budget in seconds
and append completed cells to `results/*.jsonl`, skipping work already done.
Delete those files to force a clean re-run.

## Model summary

Poisson event arrivals → binormal-ROC scoring (discriminative power `a` and
threshold `τ` varied independently) → optional alert budget as admission control
→ FIFO or priority queue → N operators with triage plus LogNormal work time →
exponential value decay in waiting time with reneging → EWMA trust feedback
governing compliance.

Separating `a` from `τ` is what allows *model quality* and *threshold policy* to
be varied independently; this underpins the volume-controlled confound check
(experiment E1b) that determines how the headline result must be stated.

## Ablation result (why the finding is not an artifact)

`experiment6.py` re-runs the recall sweep under six specifications. The direction
of the headline result — optimal recall falls as load rises — holds in **all
six**, including one with the trust feedback loop switched off entirely. Queueing
plus time-limited actionability reproduce the effect on their own; the trust loop
amplifies it but is not required. Magnitude, however, is specification-sensitive
(shortfall at recall 0.95 ranges 55.5%–90.4% across variants), so magnitudes
should be read as indicative rather than transferable.

## Known limitations

- Not calibrated to any real deployment; parameters chosen for plausibility.
- Single queue, homogeneous operators, stationary arrivals.
- Trust is a scalar EWMA — a severe simplification (but see ablation above).
- Precision is weak (CI half-width up to 13.2%) in collapsed-regime cells with
  near-zero throughput. No quantitative claim in the paper rests on those cells.

## License

MIT.
