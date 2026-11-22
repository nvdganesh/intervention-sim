"""
Discrete-event simulation of a human-in-the-loop AI intervention pipeline.

Models the path: operational event -> model score -> alert admission control
-> operator queue -> triage -> action -> outcome, with (a) value decay while
queued, (b) reneging, and (c) an endogenous trust feedback loop in which
operator compliance responds to recently experienced alert veracity.

No proprietary or field data is used. All parameters are synthetic.

Author: V D Ganesh Nandigam
License: MIT
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Optional

ND = NormalDist()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # --- event generation ---
    lam: float = 10.0           # events per hour entering the pipeline
    pi: float = 0.05            # base rate: P(event genuinely actionable)

    # --- scoring model (binormal ROC) ---
    a: float = 1.8              # discriminative power; AUC = Phi(a/sqrt(2))
    recall: float = 0.5         # target TPR; threshold derived from a & recall

    # --- decision orchestration ---
    budget: Optional[int] = None   # max alerts concurrently in system; None = unlimited
    priority: bool = False         # serve highest score first instead of FIFO

    # --- operators ---
    n_operators: int = 2
    triage_min: float = 1.0        # minutes always spent assessing an alert
    work_mu: float = 1.7           # LogNormal mu for the action itself (minutes)
    work_sigma: float = 0.5

    # --- value decay / reneging ---
    theta: float = 30.0            # actionability decay constant (minutes)
    value_cutoff: float = 0.2      # renege when decayed value falls below this
    decay_mode: str = "exponential"  # "exponential" | "deadline"
    deadline: float = 48.3         # minutes; used when decay_mode == "deadline"
                                   # 48.3 = theta*ln(1/cutoff), so the reneging
                                   # horizon matches the exponential default

    # --- ablation switches ---
    trust_enabled: bool = True     # False -> compliance fixed at fixed_compliance
    fixed_compliance: float = 0.6  # used when trust_enabled is False

    # --- context quality ---
    context: float = 0.5           # c in [0,1]
    context_speedup: float = 0.35  # max fractional reduction in work time
    context_success_gain: float = 0.35  # max additive gain to action success

    base_success: float = 0.55     # P(action succeeds) at c = 0

    # --- trust dynamics ---
    alpha: float = 0.9             # EWMA inertia
    trust_floor: float = 0.15      # compliance never falls below this
    trust_init: float = 0.6

    # --- run control ---
    horizon: float = 20000.0       # minutes of simulated time
    warmup: float = 2000.0         # discarded transient
    seed: int = 0

    def threshold(self) -> float:
        """Score threshold achieving the target recall under the binormal model."""
        r = min(max(self.recall, 1e-4), 1 - 1e-4)
        return self.a - ND.inv_cdf(r)

    def fpr(self) -> float:
        return 1.0 - ND.cdf(self.threshold())

    def auc(self) -> float:
        return ND.cdf(self.a / math.sqrt(2.0))

    def alert_rate(self) -> float:
        """Expected alerts per hour before admission control."""
        return self.lam * (self.pi * self.recall + (1 - self.pi) * self.fpr())


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _Ev:
    t: float
    seq: int
    kind: str = field(compare=False)
    payload: object = field(default=None, compare=False)


@dataclass
class Alert:
    created: float
    is_tp: bool
    score: float


@dataclass
class Result:
    successes: int = 0            # true positives resolved successfully
    delivered: int = 0            # alerts admitted to the queue
    acted: int = 0                # alerts the operator chose to act on
    dropped_budget: int = 0       # suppressed by admission control
    reneged: int = 0              # abandoned in queue (value decayed out)
    tp_events: int = 0            # genuinely actionable events occurring
    tp_alerted: int = 0           # of those, how many produced an alert
    wait_sum: float = 0.0
    wait_n: int = 0
    waits: list = field(default_factory=list)
    busy_time: float = 0.0        # operator-minutes consumed
    wasted_time: float = 0.0      # operator-minutes spent on false positives
    insys_area: float = 0.0       # time-integral of alerts in system
    obs_time: float = 0.0
    trust_final: float = 0.0      # terminal EWMA snapshot (noisy; diagnostic only)
    trust_sum: float = 0.0        # for the time-average trust level
    trust_n: int = 0

    # derived -------------------------------------------------------------
    @property
    def success_rate_per_hr(self) -> float:
        return self.successes / (self.obs_time / 60.0) if self.obs_time else 0.0

    @property
    def action_rate(self) -> float:
        return self.acted / self.delivered if self.delivered else 0.0

    @property
    def mean_wait(self) -> float:
        return self.wait_sum / self.wait_n if self.wait_n else 0.0

    def wait_pct(self, q: float) -> float:
        if not self.waits:
            return 0.0
        s = sorted(self.waits)
        i = min(int(q * len(s)), len(s) - 1)
        return s[i]

    @property
    def saturation(self) -> float:
        """Mean alerts in system per operator."""
        return self.insys_area / self.obs_time if self.obs_time else 0.0

    @property
    def mean_trust(self) -> float:
        """Average operator trust across the observation window."""
        return self.trust_sum / self.trust_n if self.trust_n else 0.0

    @property
    def delivered_precision(self) -> float:
        return self.tp_alerted / self.delivered if self.delivered else 0.0

    @property
    def missed_tp(self) -> int:
        return self.tp_events - self.successes

    n_operators: int = 1          # recorded so utilization can be normalized

    @property
    def utilization(self) -> float:
        """Fraction of total operator capacity consumed, in [0, 1]."""
        denom = self.obs_time * self.n_operators
        return self.busy_time / denom if denom else 0.0


def simulate(cfg: Config) -> Result:
    rng = random.Random(cfg.seed)
    res = Result()
    res.n_operators = cfg.n_operators

    tau = cfg.threshold()
    c = cfg.context
    p_success = min(1.0, cfg.base_success + cfg.context_success_gain * c)
    work_scale = 1.0 - cfg.context_speedup * c

    def value_of(w: float) -> float:
        """Remaining fraction of an alert's value after waiting w minutes."""
        if cfg.decay_mode == "deadline":
            return 1.0 if w < cfg.deadline else 0.0
        return math.exp(-w / cfg.theta)

    queue: list = []            # pending alerts
    free_ops = cfg.n_operators
    trust = cfg.trust_init

    evq: list = []
    seq = 0

    def push(t: float, kind: str, payload=None):
        nonlocal seq
        seq += 1
        heapq.heappush(evq, _Ev(t, seq, kind, payload))

    last_t = 0.0
    in_system = 0

    def advance(t: float):
        nonlocal last_t
        if t > cfg.warmup:
            start = max(last_t, cfg.warmup)
            if t > start:
                res.insys_area += in_system * (t - start)
                res.obs_time += t - start
        last_t = t

    push(rng.expovariate(cfg.lam / 60.0), "arrival")

    def try_dispatch(now: float):
        """Start service on queued alerts while operators are free."""
        nonlocal free_ops, in_system
        while free_ops > 0 and queue:
            if cfg.priority:
                idx = max(range(len(queue)), key=lambda i: queue[i].score)
            else:
                idx = 0
            al = queue.pop(idx)

            wait = now - al.created
            if value_of(wait) < cfg.value_cutoff:
                # decayed past usefulness before anyone could look at it
                res.reneged += 1
                in_system -= 1
                continue

            free_ops -= 1
            if now > cfg.warmup:
                res.wait_sum += wait
                res.wait_n += 1
                res.waits.append(wait)
            push(now + cfg.triage_min, "triage_done", (al, wait))

    while evq:
        ev = heapq.heappop(evq)
        now = ev.t
        if now > cfg.horizon:
            break
        advance(now)

        if ev.kind == "arrival":
            actionable = rng.random() < cfg.pi
            score = rng.gauss(cfg.a, 1.0) if actionable else rng.gauss(0.0, 1.0)
            if now > cfg.warmup and actionable:
                res.tp_events += 1

            if score > tau:  # model raises this event
                if now > cfg.warmup and actionable:
                    res.tp_alerted += 1
                if cfg.budget is not None and in_system >= cfg.budget:
                    if now > cfg.warmup:
                        res.dropped_budget += 1
                else:
                    queue.append(Alert(now, actionable, score))
                    in_system += 1
                    if now > cfg.warmup:
                        res.delivered += 1
            push(now + rng.expovariate(cfg.lam / 60.0), "arrival")
            try_dispatch(now)

        elif ev.kind == "triage_done":
            al, wait = ev.payload
            if now > cfg.warmup:
                res.busy_time += cfg.triage_min
                if not al.is_tp:
                    res.wasted_time += cfg.triage_min

            # trust updates on every alert the operator inspects
            if cfg.trust_enabled:
                trust = cfg.alpha * trust + (1 - cfg.alpha) * (1.0 if al.is_tp else 0.0)
                p_act = cfg.trust_floor + (1 - cfg.trust_floor) * trust
            else:
                p_act = cfg.fixed_compliance   # ablation: no trust feedback
            if now > cfg.warmup:
                res.trust_sum += trust if cfg.trust_enabled else p_act
                res.trust_n += 1

            if rng.random() < p_act:
                if now > cfg.warmup:
                    res.acted += 1
                work = rng.lognormvariate(cfg.work_mu, cfg.work_sigma) * work_scale
                push(now + work, "work_done", (al, wait, work, p_success))
            else:
                free_ops += 1
                in_system -= 1
                try_dispatch(now)

        elif ev.kind == "work_done":
            al, wait, work, ps = ev.payload
            if now > cfg.warmup:
                res.busy_time += work
                if not al.is_tp:
                    res.wasted_time += work
            value = value_of(wait)
            if al.is_tp and value >= cfg.value_cutoff and rng.random() < ps * value:
                if now > cfg.warmup:
                    res.successes += 1
            free_ops += 1
            in_system -= 1
            try_dispatch(now)

    advance(min(cfg.horizon, last_t if last_t else cfg.horizon))
    res.trust_final = trust
    if cfg.n_operators:
        res.insys_area /= cfg.n_operators  # saturation expressed per operator
    return res


# ---------------------------------------------------------------------------
# Replication helpers
# ---------------------------------------------------------------------------


def replicate(cfg: Config, n_reps: int = 20, base_seed: int = 1000):
    out = []
    for k in range(n_reps):
        c = Config(**{**cfg.__dict__})
        c.seed = base_seed + k
        out.append(simulate(c))
    return out


def mean_ci(vals):
    """Mean and 95% CI half-width (normal approximation)."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, 1.96 * math.sqrt(var / n)


def agg(results, attr):
    return mean_ci([getattr(r, attr) for r in results])


def calibrate_lambda(cfg: Config, target_load: float, ref_recall: float = 0.5) -> float:
    """
    Choose lambda so that offered operator load equals target_load when the
    model runs at ref_recall. Offered load = alert_rate * E[service] / n_ops.
    """
    ref = Config(**{**cfg.__dict__})
    ref.recall = ref_recall
    e_work = math.exp(ref.work_mu + ref.work_sigma ** 2 / 2) * (
        1 - ref.context_speedup * ref.context
    )
    e_service = ref.triage_min + 0.6 * e_work   # 0.6 ~ nominal compliance
    per_event = ref.pi * ref.recall + (1 - ref.pi) * ref.fpr()
    denom = per_event * (e_service / 60.0) / ref.n_operators
    return target_load / denom if denom > 0 else cfg.lam
