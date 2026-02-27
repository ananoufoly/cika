
""" 
TRIBUS Monte Carlo Stress Test (Interactive, TRIBUS-specific)
===========================================================

Purpose
-------
This is a *risk stress* model (operational + behavioral + governance), not an accounting engine.
It simulates many groups (G) over their cycle length and estimates probabilities of:
- Hard / Soft collapse
- Escalation
- Dispute dynamics
- Matelas (safety buffer) usage and insufficiency
- Cost and revenue-at-risk

What this edition adds
----------------------
1) Guided inputs: each parameter prompt includes a brief explanation.
2) Universal (regime-independent) tension dashboard: GREEN / AMBER / RED + an overall score.
3) Conservative thresholds: triggers AMBER/RED earlier as an early-warning framework.
4) Optimized group-size input: you define a *distribution* of group sizes (population mix), not one
   probability per group.
5) Scenario presets: existing/public × (central, peak_load, major_incident).

Run:
  python tribus_st.py

Outputs:
- Tension dashboard (overall)
- Summary metrics with tension flags
- Cost stats (mean / p50 / p95 / p99)
- Breakdown by group size
- Optional CSV exports
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

import numpy as np
import math
import copy
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-x))


def _print_help(lines: List[str]):
  for ln in lines:
  print(f"  - {ln}")


def prompt_str(msg: str, default: str = "", help_text: Optional[str] = None) -> str:
  if help_text:
  _print_help([help_text])
out = input(f"{msg} [{default}]: ").strip()
return out if out else default


def prompt_int(msg: str, default: int, help_text: Optional[str] = None) -> int:
  if help_text:
  _print_help([help_text])
out = input(f"{msg} [{default}]: ").strip()
return int(out) if out else default


def prompt_float(msg: str, default: float, help_text: Optional[str] = None) -> float:
  if help_text:
  _print_help([help_text])
out = input(f"{msg} [{default}]: ").strip()
return float(out) if out else default


def prompt_bool(msg: str, default: bool, help_text: Optional[str] = None) -> bool:
  if help_text:
  _print_help([help_text])
d = "y" if default else "n"
out = input(f"{msg} [y/n, default={d}]: ").strip().lower()
if not out:
  return default
return out in ("y", "yes", "1", "true", "t")


def parse_ints(csv: str, default):
  s = csv.strip()
if not s:
  return list(default)
return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_floats(csv: str, default):
  s = csv.strip()
if not s:
  return list(default)
return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_size_prob_pairs(s: str) -> Tuple[List[int], List[float]]:
  """Parse size:prob pairs like '10:0.25,12:0.35,20:0.25,30:0.15'."""
s = s.strip()
if not s:
  return [], []
parts = [p.strip() for p in s.split(",") if p.strip()]
sizes, probs = [], []
for p in parts:
  if ":" not in p:
  raise ValueError(f"Invalid pair '{p}'. Expected format size:prob.")
a, b = p.split(":", 1)
sizes.append(int(a.strip()))
probs.append(float(b.strip()))
return sizes, probs


def normalize_probs(probs: List[float]) -> List[float]:
  arr = np.array(probs, dtype=float)
if arr.size == 0:
  return []
if np.any(arr < 0):
  raise ValueError("Probabilities/weights must be non-negative.")
s = float(arr.sum())
if s <= 0:
  raise ValueError("Sum of probabilities/weights must be > 0.")
return (arr / s).tolist()


def gen_range_sizes(n_min: int, n_max: int, step: int) -> List[int]:
  if step <= 0:
  raise ValueError("step must be >= 1")
if n_max < n_min:
  n_min, n_max = n_max, n_min
return list(range(n_min, n_max + 1, step))


def weights_for_sizes(sizes: List[int], scheme: str) -> List[float]:
  """Generate weights for sizes according to a scheme (then normalize)."""
if not sizes:
  return []
scheme = scheme.lower()

if scheme == "uniform":
  return normalize_probs([1.0] * len(sizes))

if scheme == "triangular":
  xs = np.array(sizes, dtype=float)
center = np.median(xs)
dist = np.abs(xs - center)
w = (dist.max() - dist + 1.0).tolist()
return normalize_probs(w)

if scheme == "zipf":
  order = np.argsort(sizes)
ranks = np.empty(len(sizes))
ranks[order] = np.arange(1, len(sizes) + 1)
w = (1.0 / ranks).tolist()
return normalize_probs(w)

raise ValueError("Unknown weighting scheme. Use uniform/triangular/zipf.")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class Config:
  # Simulation scale
  G: int = 10000
seed: int = 42

# Group sizes distribution (population mix)
N_values: Tuple[int, ...] = (10, 12, 20, 30)
N_probs: Tuple[float, ...] = (0.25, 0.35, 0.25, 0.15)
T_equals_N: bool = True

# Regime: "existing" (baseline) vs "public" (stressful environment)
regime: str = "existing"

# --- Operational incidents (per group-period)
p_NF: float = 0.01
p_PR: float = 0.003
p_SM: float = 0.003
p_PI: float = 0.001

# --- Resolution time (hours) for critical incidents: LogNormal(log-hours)
tau_logn_mu: float = 3.0
tau_logn_sigma: float = 0.7

# --- SLA threshold (hours)
SLA_hours: float = 48.0

# --- Behavior: missed payments + recovery
a_m: float = -2.2
b_m: float = -0.6
u_m_dispute: float = 0.4
v_m_notif: float = 0.2

a_r: float = 1.0
b_r: float = 0.5
u_r_dispute: float = 0.5

# --- Public regime only (cash&run)
enable_cash_run: bool = False
a_cr: float = -5.0
b_cr: float = -0.4
u_cr_paid_early: float = 2.3

# --- Dispute process (Poisson intensity)
a_d: float = -1.5
b_d: float = -0.5
u_d_arrears: float = 0.25
v_d_incident: float = 0.9

# Closure baseline (fraction closed per period)
close_rate_base: float = 0.55

# --- Governance
theta_D_to_pivot: int = 2
theta_D_to_vote: int = 3
p_vote_resolve: float = 0.75
p_pivot_resolve_boost: float = 0.15
chef_inactivity_prob: float = 0.01
pivot_inactivity_prob: float = 0.005

allow_soft_collapse_on_sla_breach: bool = True
allow_soft_collapse_on_unresolved_vote: bool = True
hard_on_PI_and_SLA: bool = True

# --- Matelas
matelas_enabled: bool = True
matelas_contrib_per_member: float = 0.5
matelas_max_multiple_of_c: float = 1.0
matelas_cover_prob: float = 0.85
matelas_dispute_reduction: float = 0.25

# --- Escalation
a_e: float = -6.0
b_e: float = -0.8
u_e_disputes: float = 0.5
k_sla: float = 0.08

# --- Costs
c_ticket: float = 4.0
tickets_per_dispute: float = 1.0
tickets_per_incident: float = 2.0
c_eng_hour: float = 60.0
h0_eng: float = 2.0
h1_eng_per_hour: float = 0.08

goodwill_enabled: bool = False
c_goodwill: float = 15.0

# Revenue-at-risk proxy
LTV: float = 25.0
# Baseline revenue (Blended ARPU) per member per month for % metrics
rev_per_member_per_month: float = 1.0
a_c: float = -3.0
u_c_failure: float = 1.6
v_c_escalation: float = 1.2


# ---------------------------------------------------------------------
# Scenario presets (starting priors)
# ---------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Dict[str, object]]] = {
  "existing": {
    "central": {
      "p_NF": 0.004, "p_PR": 0.0015, "p_SM": 0.0015, "p_PI": 0.0003,
      "SLA_hours": 8.0, "tau_logn_mu": 0.693, "tau_logn_sigma": 0.843,
      "close_rate_base": 0.55, "chef_inactivity_prob": 0.005, "pivot_inactivity_prob": 0.002,
      "theta_D_to_pivot": 2, "theta_D_to_vote": 3,
      "p_vote_resolve": 0.78, "p_pivot_resolve_boost": 0.15,
      "matelas_enabled": True, "matelas_contrib_per_member": 0.5, "matelas_max_multiple_of_c": 1.0,
      "matelas_cover_prob": 0.85, "matelas_dispute_reduction": 0.25,
    },
    "peak_load": {
      "p_NF": 0.008, "p_PR": 0.0030, "p_SM": 0.0030, "p_PI": 0.0006,
      "SLA_hours": 8.0, "tau_logn_mu": 1.386, "tau_logn_sigma": 0.843,
      "close_rate_base": 0.50, "chef_inactivity_prob": 0.008, "pivot_inactivity_prob": 0.003,
      "theta_D_to_pivot": 2, "theta_D_to_vote": 3,
      "p_vote_resolve": 0.75, "p_pivot_resolve_boost": 0.12,
      "matelas_enabled": True, "matelas_contrib_per_member": 0.5, "matelas_max_multiple_of_c": 1.0,
      "matelas_cover_prob": 0.82, "matelas_dispute_reduction": 0.25,
    },
    "major_incident": {
      "p_NF": 0.006, "p_PR": 0.0025, "p_SM": 0.0025, "p_PI": 0.0012,
      "SLA_hours": 8.0, "tau_logn_mu": 2.485, "tau_logn_sigma": 1.089,
      "close_rate_base": 0.48, "chef_inactivity_prob": 0.01, "pivot_inactivity_prob": 0.005,
      "theta_D_to_pivot": 2, "theta_D_to_vote": 3,
      "p_vote_resolve": 0.70, "p_pivot_resolve_boost": 0.12,
      "matelas_enabled": True, "matelas_contrib_per_member": 0.5, "matelas_max_multiple_of_c": 1.0,
      "matelas_cover_prob": 0.78, "matelas_dispute_reduction": 0.25,
    },
  },
  "public": {
    "central": {
      "p_NF": 0.006, "p_PR": 0.0022, "p_SM": 0.0022, "p_PI": 0.0005,
      "SLA_hours": 8.0, "tau_logn_mu": 0.693, "tau_logn_sigma": 0.900,
      "close_rate_base": 0.50, "chef_inactivity_prob": 0.01, "pivot_inactivity_prob": 0.005,
      "theta_D_to_pivot": 2, "theta_D_to_vote": 3,
      "p_vote_resolve": 0.65, "p_pivot_resolve_boost": 0.12,
      "matelas_enabled": True, "matelas_contrib_per_member": 0.6, "matelas_max_multiple_of_c": 1.0,
      "matelas_cover_prob": 0.85, "matelas_dispute_reduction": 0.30,
    },
    "peak_load": {
      "p_NF": 0.012, "p_PR": 0.0044, "p_SM": 0.0044, "p_PI": 0.0010,
      "SLA_hours": 8.0, "tau_logn_mu": 1.386, "tau_logn_sigma": 0.900,
      "close_rate_base": 0.45, "chef_inactivity_prob": 0.015, "pivot_inactivity_prob": 0.007,
      "theta_D_to_pivot": 2, "theta_D_to_vote": 3,
      "p_vote_resolve": 0.62, "p_pivot_resolve_boost": 0.10,
      "matelas_enabled": True, "matelas_contrib_per_member": 0.6, "matelas_max_multiple_of_c": 1.0,
      "matelas_cover_prob": 0.80, "matelas_dispute_reduction": 0.30,
    },
    "major_incident": {
      "p_NF": 0.010, "p_PR": 0.0035, "p_SM": 0.0035, "p_PI": 0.0020,
      "SLA_hours": 8.0, "tau_logn_mu": 2.485, "tau_logn_sigma": 1.150,
      "close_rate_base": 0.40, "chef_inactivity_prob": 0.02, "pivot_inactivity_prob": 0.01,
      "theta_D_to_pivot": 2, "theta_D_to_vote": 3,
      "p_vote_resolve": 0.55, "p_pivot_resolve_boost": 0.08,
      "matelas_enabled": True, "matelas_contrib_per_member": 0.6, "matelas_max_multiple_of_c": 1.0,
      "matelas_cover_prob": 0.75, "matelas_dispute_reduction": 0.30,
    },
  },
}


def apply_preset(cfg: Config, preset_name: str) -> None:
  """Apply a preset in-place. preset_name format: '<regime>.<scenario>' e.g. 'existing.central'."""
if "." not in preset_name:
  raise ValueError("preset_name must be like 'existing.central'")
reg, scen = preset_name.split(".", 1)
reg = reg.lower().strip()
scen = scen.lower().strip()
if reg not in PRESETS or scen not in PRESETS[reg]:
  raise ValueError(f"Unknown preset '{preset_name}'.")
for k, v in PRESETS[reg][scen].items():
  setattr(cfg, k, v)
cfg.regime = reg


# ---------------------------------------------------------------------
# Regime presets (structural shifts)
# ---------------------------------------------------------------------

def apply_availability_target(cfg: Config) -> Config:
  """Apply simple scaling for stricter availability targets.

    If availability_target >= 99.95, halve incident likelihood and median restore time
    to reflect a ~2x tighter monthly downtime budget vs 99.9.
    """
target = getattr(cfg, '_availability_target', '')
try:
  t = float(str(target).strip())
except Exception:
  return cfg
factor = 1.0
if t >= 99.95:
  factor = 0.5
for k in ('p_NF','p_PR','p_SM','p_PI'):
  setattr(cfg, k, max(0.0, min(1.0, float(getattr(cfg,k)) * factor)))
if factor != 1.0:
  cfg.tau_logn_mu = float(cfg.tau_logn_mu) + math.log(factor)
return cfg

def apply_overrides(cfg: Config, overrides: dict) -> Config:
  if not overrides:
  return cfg
for k,v in overrides.items():
  setattr(cfg,k,v)
return cfg

def apply_regime_preset(cfg: Config) -> Config:
  """Adjust parameters to reflect 'existing' vs 'public' regime defaults."""
if cfg.regime.lower() == "existing":
  cfg.enable_cash_run = False
# Existing: slightly better recovery, slightly fewer disputes
cfg.a_r = 1.1
cfg.a_d = -1.6
cfg.p_vote_resolve = max(cfg.p_vote_resolve, 0.78)
return cfg

if cfg.regime.lower() == "public":
  cfg.enable_cash_run = True
# Public: higher missed payments, lower recovery, higher disputes, more escalation risk
cfg.a_m = -1.8
cfg.a_r = min(cfg.a_r, 0.6)
cfg.a_d = -1.2
cfg.a_e = -5.5
cfg.p_vote_resolve = min(cfg.p_vote_resolve, 0.65)
cfg.chef_inactivity_prob = max(cfg.chef_inactivity_prob, 0.015)
return cfg

raise ValueError("Unknown regime. Use 'existing' or 'public'.")


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

def simulate(cfg: Config) -> Dict[str, pd.DataFrame]:
  cfg = apply_regime_preset(cfg)
rng = np.random.default_rng(cfg.seed)

# Draw group sizes
N = rng.choice(cfg.N_values, size=cfg.G, p=cfg.N_probs)
T = N.copy() if cfg.T_equals_N else np.full(cfg.G, int(np.mean(cfg.N_values)))

# Latent factor
Z = rng.normal(0.0, 1.0, size=cfg.G)

# States
ACTIVE, HARD, SOFT, COMPLETED = 0, 1, 2, 3
state = np.full(cfg.G, ACTIVE, dtype=np.int8)

# Governance stages
CHEF, PIVOT, VOTE = 0, 1, 2
stage = np.full(cfg.G, CHEF, dtype=np.int8)
stage_time = np.zeros(cfg.G, dtype=np.int16)

# Group-level counters/states
arrears = np.zeros(cfg.G, dtype=np.int32)
disputes = np.zeros(cfg.G, dtype=np.int32)
any_escalation = np.zeros(cfg.G, dtype=np.int8)
any_failure = np.zeros(cfg.G, dtype=np.int8)
hard_collapse = np.zeros(cfg.G, dtype=np.int8)
soft_collapse = np.zeros(cfg.G, dtype=np.int8)

# Matelas balance
matelas = np.zeros(cfg.G, dtype=float)
matelas_used_count = np.zeros(cfg.G, dtype=np.int16)
matelas_insufficient_count = np.zeros(cfg.G, dtype=np.int16)

# Governance KPIs
pivot_triggered = np.zeros(cfg.G, dtype=np.int8)
vote_triggered = np.zeros(cfg.G, dtype=np.int8)
vote_failed = np.zeros(cfg.G, dtype=np.int8)

# Cost accumulators
cost_support = np.zeros(cfg.G)
cost_ops = np.zeros(cfg.G)
cost_goodwill = np.zeros(cfg.G)

maxT = int(T.max())

for t in range(1, maxT + 1):
  alive = (state == ACTIVE) & (t <= T)
if not np.any(alive):
  break
idx = np.where(alive)[0]

# Fund matelas
if cfg.matelas_enabled:
  cap = cfg.matelas_max_multiple_of_c * N[idx] * 1.0
matelas[idx] = np.minimum(matelas[idx] + cfg.matelas_contrib_per_member * N[idx], cap)

# Incidents
NF = rng.random(idx.size) < cfg.p_NF
PR = rng.random(idx.size) < cfg.p_PR
SM = rng.random(idx.size) < cfg.p_SM
PI = rng.random(idx.size) < cfg.p_PI

X = PR | SM | PI

tau = np.zeros(idx.size)
if np.any(X):
  tau[X] = rng.lognormal(cfg.tau_logn_mu, cfg.tau_logn_sigma, int(X.sum()))

SLA_breach = X & (tau > cfg.SLA_hours)

# Missed payments
p_m = sigmoid(
  cfg.a_m + cfg.b_m * Z[idx] + cfg.u_m_dispute * (disputes[idx] > 0) + cfg.v_m_notif * NF
)
missed = rng.random(idx.size) < p_m

# Matelas cover
matelas_used = np.zeros(idx.size, dtype=bool)
if cfg.matelas_enabled and np.any(missed):
  can_cover = (matelas[idx] >= 1.0) & missed
try_cover = can_cover & (rng.random(idx.size) < cfg.matelas_cover_prob)
if np.any(try_cover):
  tmp_m = matelas[idx]
tmp_m[try_cover] -= 1.0
matelas[idx] = tmp_m
matelas_used[try_cover] = True
matelas_used_count[idx[try_cover]] += 1

insufficient = missed & (~can_cover)
if np.any(insufficient):
  matelas_insufficient_count[idx[insufficient]] += 1

eff_missed = missed & (~matelas_used)

# Recovery
p_r = sigmoid(cfg.a_r + cfg.b_r * Z[idx] - cfg.u_r_dispute * (disputes[idx] > 0))
recover = rng.random(idx.size) < p_r
arrears[idx] = np.maximum(arrears[idx] + eff_missed.astype(int) - recover.astype(int), 0)

# Optional cash&run (public)
if cfg.enable_cash_run:
  paid_early = (t <= (T[idx] // 2))
p_cr = sigmoid(cfg.a_cr + cfg.b_cr * Z[idx] + cfg.u_cr_paid_early * paid_early.astype(int))
cashrun = rng.random(idx.size) < p_cr
arrears[idx] += cashrun.astype(int) * 2

# Governance stage transitions
to_pivot = (disputes[idx] >= cfg.theta_D_to_pivot) & (stage[idx] == CHEF)
if np.any(to_pivot):
  stage[idx[to_pivot]] = PIVOT
stage_time[idx[to_pivot]] = 0
pivot_triggered[idx[to_pivot]] = 1

to_vote = (disputes[idx] >= cfg.theta_D_to_vote) & (stage[idx] != VOTE)
if np.any(to_vote):
  stage[idx[to_vote]] = VOTE
stage_time[idx[to_vote]] = 0
vote_triggered[idx[to_vote]] = 1

stage_time[idx] += 1

# Dispute arrivals intensity
lam = np.exp(cfg.a_d + cfg.b_d * Z[idx] + cfg.u_d_arrears * arrears[idx] + cfg.v_d_incident * X.astype(int))
if cfg.matelas_enabled:
  lam = lam * (1.0 - cfg.matelas_dispute_reduction * matelas_used.astype(float))
new_disputes = rng.poisson(lam)

# Dispute closure
close_rate = np.full(idx.size, cfg.close_rate_base, dtype=float)
close_rate[stage[idx] == PIVOT] = np.minimum(0.95, close_rate[stage[idx] == PIVOT] + cfg.p_pivot_resolve_boost)

vote_mask = (stage[idx] == VOTE)
vote_resolve = np.zeros(idx.size, dtype=bool)
if np.any(vote_mask):
  vote_resolve[vote_mask] = rng.random(int(vote_mask.sum())) < cfg.p_vote_resolve

resolved_idx = np.where(vote_resolve)[0]
if resolved_idx.size > 0:
  disputes[idx[resolved_idx]] = (disputes[idx[resolved_idx]] * 0.2).astype(int)

unresolved = vote_mask & (~vote_resolve)
if np.any(unresolved):
  vote_failed[idx[unresolved]] = 1

closed = (rng.random(idx.size) < close_rate) * disputes[idx]
disputes[idx] = np.maximum(disputes[idx] - closed, 0) + new_disputes

# Inactivity (governance op risk)
action_needed = (disputes[idx] > 0)
inactive = np.zeros(idx.size, dtype=bool)
chef_mask = (stage[idx] == CHEF) & action_needed
pivot_mask = (stage[idx] == PIVOT) & action_needed
if np.any(chef_mask):
  inactive[chef_mask] = rng.random(int(chef_mask.sum())) < cfg.chef_inactivity_prob
if np.any(pivot_mask):
  inactive[pivot_mask] = rng.random(int(pivot_mask.sum())) < cfg.pivot_inactivity_prob

# Escalation probability
p_esc0 = sigmoid(cfg.a_e + cfg.b_e * Z[idx] + cfg.u_e_disputes * disputes[idx])
p_esc_sla = np.zeros(idx.size)
p_esc_sla[X] = 1.0 - np.exp(-cfg.k_sla * np.maximum(0.0, tau[X] - cfg.SLA_hours))
p_esc = np.maximum(p_esc0, p_esc_sla)
Esc = rng.random(idx.size) < p_esc
any_escalation[idx] = np.maximum(any_escalation[idx], Esc.astype(np.int8))

# Collapse logic
hard_trigger = np.zeros(idx.size, dtype=bool)
if cfg.hard_on_PI_and_SLA:
  hard_trigger = (PI & (tau > cfg.SLA_hours))

# Soft collapse triggers are combined (OR), not overwritten
soft_trigger = np.zeros(idx.size, dtype=bool)
soft_trigger |= inactive
if cfg.allow_soft_collapse_on_sla_breach:
  soft_trigger |= SLA_breach
if cfg.allow_soft_collapse_on_unresolved_vote:
  soft_trigger |= ((stage[idx] == VOTE) & (vote_failed[idx] == 1) & (stage_time[idx] >= 2))

hard_idx = idx[hard_trigger]
soft_idx = idx[soft_trigger & (~hard_trigger)]
if hard_idx.size > 0:
  state[hard_idx] = HARD
hard_collapse[hard_idx] = 1
any_failure[hard_idx] = 1
if soft_idx.size > 0:
  state[soft_idx] = SOFT
soft_collapse[soft_idx] = 1
any_failure[soft_idx] = 1

# Costs
n_tickets = cfg.tickets_per_dispute * new_disputes + cfg.tickets_per_incident * X.astype(int)
cost_support[idx] += n_tickets * cfg.c_ticket
eng_hours = cfg.h0_eng + cfg.h1_eng_per_hour * tau
cost_ops[idx] += X.astype(int) * eng_hours * cfg.c_eng_hour
if cfg.goodwill_enabled:
  cost_goodwill[idx] += Esc.astype(int) * cfg.c_goodwill

# Mark completed when cycle ends
end_cycle = (state == ACTIVE) & (t == T)
state[end_cycle] = COMPLETED

# Revenue-at-risk
p_churn = sigmoid(cfg.a_c + cfg.u_c_failure * any_failure + cfg.v_c_escalation * any_escalation)
revenue_at_risk = p_churn * cfg.LTV

total_cost = cost_support + cost_ops + cost_goodwill + revenue_at_risk
# Baseline revenue (Blended ARPU) and normalized impact metrics
base_revenue = N * T * cfg.rev_per_member_per_month
base_rev_safe = np.maximum(base_revenue, 1e-12)
support_cost_pct = cost_support / base_rev_safe
ops_cost_pct = cost_ops / base_rev_safe
goodwill_cost_pct = cost_goodwill / base_rev_safe
rev_at_risk_pct = revenue_at_risk / base_rev_safe
total_impact_pct = total_cost / base_rev_safe

df = pd.DataFrame(
  {
    "Regime": cfg.regime,
    "N": N,
    "HardCollapse": hard_collapse,
    "SoftCollapse": soft_collapse,
    "PlatformFailure": any_failure,
    "AnyEscalation": any_escalation,
    "MatelasUsedCount": matelas_used_count,
    "MatelasInsufficientCount": matelas_insufficient_count,
    "PivotTriggered": pivot_triggered,
    "VoteTriggered": vote_triggered,
    "VoteFailedFlag": vote_failed,
    "CostSupport": cost_support,
    "CostOps": cost_ops,
    "CostGoodwill": cost_goodwill,
    "RevenueAtRisk": revenue_at_risk,
    "TotalCost": total_cost,
    "BaseRevenue": base_revenue,
    "SupportCostPct": support_cost_pct,
    "OpsCostPct": ops_cost_pct,
    "GoodwillCostPct": goodwill_cost_pct,
    "RevenueAtRiskPct": rev_at_risk_pct,
    "TotalImpactPct": total_impact_pct,
  }
)

# Summary metrics
def stats_series(s: pd.Series) -> Dict[str, float]:
  return {
    "mean": float(s.mean()),
    "p50": float(s.quantile(0.50)),
    "p95": float(s.quantile(0.95)),
    "p99": float(s.quantile(0.99)),
  }

metrics = {
  "P(HardCollapse)": df["HardCollapse"].mean(),
  "P(SoftCollapse)": df["SoftCollapse"].mean(),
  "P(PlatformFailure)": df["PlatformFailure"].mean(),
  "P(AnyEscalation)": df["AnyEscalation"].mean(),
  "AvgMatelasUses": df["MatelasUsedCount"].mean(),
  "P(MatelasInsufficient>0)": (df["MatelasInsufficientCount"] > 0).mean(),
  "P(PivotTriggered)": df["PivotTriggered"].mean(),
  "P(VoteTriggered)": df["VoteTriggered"].mean(),
  "P(VoteFailedFlag)": df["VoteFailedFlag"].mean(),
}

df_metrics = pd.DataFrame([metrics]).T.reset_index()
df_metrics.columns = ["Metric", "Value"]

cost_stats = {k: stats_series(df[k]) for k in ["CostSupport", "CostOps", "CostGoodwill", "RevenueAtRisk", "TotalCost", "RevenueAtRiskPct", "TotalImpactPct"]}
df_cost = pd.DataFrame(cost_stats).T.reset_index().rename(columns={"index": "CostComponent"})

byN = df.groupby("N").agg(
  Groups=("N", "size"),
  P_Hard=("HardCollapse", "mean"),
  P_Soft=("SoftCollapse", "mean"),
  P_Fail=("PlatformFailure", "mean"),
  P_Esc=("AnyEscalation", "mean"),
  AvgMatelasUses=("MatelasUsedCount", "mean"),
  P_MatelasInsuff=("MatelasInsufficientCount", lambda s: (s > 0).mean()),
  P_Pivot=("PivotTriggered", "mean"),
  P_Vote=("VoteTriggered", "mean"),
  CostMean=("TotalCost", "mean"),
  CostP95=("TotalCost", lambda s: s.quantile(0.95)),
  CostP99=("TotalCost", lambda s: s.quantile(0.99)),
  ImpactPctMean=("TotalImpactPct", "mean"),
  ImpactPctP95=("TotalImpactPct", lambda s: s.quantile(0.95)),
  ImpactPctP99=("TotalImpactPct", lambda s: s.quantile(0.99)),
  RevRiskPctMean=("RevenueAtRiskPct", "mean"),
  RevRiskPctP95=("RevenueAtRiskPct", lambda s: s.quantile(0.95)),
  RevRiskPctP99=("RevenueAtRiskPct", lambda s: s.quantile(0.99)),
).reset_index()

return {"raw": df, "metrics": df_metrics, "cost": df_cost, "byN": byN}


# ---------------------------------------------------------------------
# Tension interpretation (universal + conservative)
# ---------------------------------------------------------------------

def _zone(val: float, green_max: float, amber_max: float) -> str:
  if val <= green_max:
  return "GREEN"
if val <= amber_max:
  return "AMBER"
return "RED"


def add_tension_flags(df_metrics: pd.DataFrame) -> pd.DataFrame:
  """Add Zone + Interpretation columns using universal conservative thresholds."""
df = df_metrics.copy()

rules = {
  "P(HardCollapse)": (0.002, 0.01),
  "P(SoftCollapse)": (0.005, 0.02),
  "P(PlatformFailure)": (0.01, 0.04),
  "P(AnyEscalation)": (0.05, 0.15),
  "P(MatelasInsufficient>0)": (0.02, 0.08),
  "P(VoteFailedFlag)": (0.01, 0.04),
  "P(VoteTriggered)": (0.07, 0.15),
  "P(PivotTriggered)": (0.07, 0.15),
  "AvgMatelasUses": (1.0, 2.0),
}

zones, notes = [], []
for _, row in df.iterrows():
  m = row["Metric"]
v = float(row["Value"])
if m in rules:
  g, a = rules[m]
z = _zone(v, g, a)
else:
  z = "NA"

zones.append(z)
if z == "GREEN":
  notes.append("No obvious tension signal")
elif z == "AMBER":
  notes.append("Watchlist: early warning tension")
elif z == "RED":
  notes.append("High tension: investigate / mitigate")
else:
  notes.append("No rule")

df["Zone"] = zones
df["Interpretation"] = notes
return df


def overall_tension(df_metrics_flagged: pd.DataFrame) -> Dict[str, object]:
  red = int((df_metrics_flagged["Zone"] == "RED").sum())
amber = int((df_metrics_flagged["Zone"] == "AMBER").sum())
score = int(2 * red + 1 * amber)

if score <= 1:
  overall = "GREEN"
msg = "Stable: metrics look healthy under this scenario."
elif score <= 4:
  overall = "AMBER"
msg = "Moderate tension: early warning signals detected; consider mitigations."
else:
  overall = "RED"
msg = "High tension: risk stress suggests instability; action recommended."

return {"OverallZone": overall, "Score": score, "Message": msg, "RedCount": red, "AmberCount": amber}


# ---------------------------------------------------------------------
# Interactive config (guided)
# ---------------------------------------------------------------------

def interactive_config() -> Config:
  cfg = Config()
print("\n=== TRIBUS Monte Carlo (TRIBUS-specific, Interactive) ===\n")

cfg.regime = prompt_str(
  "Regime ('existing' or 'public')",
  cfg.regime,
  help_text="Select the environment. 'public' is a more stressful environment with cash&run enabled.",
).lower()

cfg.G = prompt_int(
  "Number of groups G",
  cfg.G,
  help_text="Monte Carlo sample size. Use 2000 for quick tuning, 10000+ for stable estimates.",
)

cfg.seed = prompt_int(
  "Random seed",
  cfg.seed,
  help_text="Fix seed for reproducible results.",
)

cfg.rev_per_member_per_month = prompt_float(
  "Blended ARPU per member per month (for % metrics)",
  cfg.rev_per_member_per_month,
  help_text="Blended ARPU = subscription + usage/transaction revenue per active member per month. Set to 1.0 for pure % outputs.",
)

availability_target = prompt_str(
  "Availability target for documentation (e.g., 99.9 or 99.95)",
  "99.9",
  help_text="If >= 99.95 we apply transparent scaling (halve incident rates and median restore time).",
).strip()
cfg._availability_target = availability_target

cfg._run_suite = prompt_bool(
  "Run full scenario suite for this regime (central + stresses)?",
  True,
  help_text="If yes: runs central, peak_load, major_incident for the chosen regime for direct comparison.",
)

use_preset = prompt_bool(
  "Use a scenario preset?",
  True,
  help_text=(
    "Presets provide CENTRAL vs two stress styles: peak_load and major_incident. "
    "You can override any parameter after applying a preset."
  ),
)

if use_preset:
  print("\nAvailable presets:")
print("  - existing.central | existing.peak_load | existing.major_incident")
print("  - public.central   | public.peak_load   | public.major_incident")
preset_name = prompt_str(
  "Choose preset",
  f"{cfg.regime}.central",
  help_text="Format: <regime>.<scenario> (e.g., public.peak_load)",
)
apply_preset(cfg, preset_name)
print(f"\nApplied preset: {preset_name}")

print("\n-- Group size distribution (population mix) --")
_print_help([
  "You are NOT entering a probability per group.",
  "You define the distribution of group sizes in the population.",
  "N_probs means: P(a randomly drawn group has size N).",
  "The simulator draws G groups from this mix.",
])

mode = prompt_str(
  "Group-size input mode (list / pairs / range)",
  "range",
  help_text=(
    "list: enter sizes and (optional) probabilities; blank probs => uniform. "
    "pairs: enter size:prob pairs. range: generate sizes from min..max with a weighting."
  ),
).lower()

if mode == "pairs":
  s = prompt_str(
    "Enter size:prob pairs",
    "10:0.25,12:0.35,20:0.25,30:0.15",
    help_text="Example: 10:0.25,12:0.35,20:0.25,30:0.15 (auto-normalized).",
  )
sizes, probs = parse_size_prob_pairs(s)
probs = normalize_probs(probs)

elif mode == "range":
  n_min = prompt_int("N min", 10, help_text="Minimum group size in the population mix.")
n_max = prompt_int("N max", 30, help_text="Maximum group size in the population mix.")
step = prompt_int("Step", 1, help_text="Step for generated sizes (e.g., 1 or 2).")
scheme = prompt_str(
  "Weighting (uniform / triangular / zipf)",
  "zipf",
  help_text="uniform: equal. triangular: peak at median. zipf: heavier weight on smaller sizes.",
)
sizes = gen_range_sizes(n_min, n_max, step)
probs = weights_for_sizes(sizes, scheme)

else:
  Nv = prompt_str(
    "Group sizes N (comma-separated)",
    ",".join(map(str, cfg.N_values)),
    help_text="Example: 10,12,20,30 or a long list (e.g., 10,11,12,...,50).",
  )
sizes = parse_ints(Nv, cfg.N_values)
Np = prompt_str(
  "Probabilities for each N (comma-separated; optional)",
  "",
  help_text="Optional. Blank => uniform weights. If provided, length must match N list.",
)
if Np.strip() == "":
  probs = normalize_probs([1.0] * len(sizes))
else:
  probs_raw = parse_floats(Np, [1.0] * len(sizes))
if len(probs_raw) != len(sizes):
  raise ValueError("Length of probabilities must match length of N list.")
probs = normalize_probs(probs_raw)

cfg.N_values = tuple(sizes)
cfg.N_probs = tuple(probs)

cfg.T_equals_N = prompt_bool(
  "Cycle length T equals N?",
  cfg.T_equals_N,
  help_text="If yes: classic ROSCA length = N; else: uses average cycle length.",
)

# Optional overrides (common ones)
cfg._overrides = {}
override = prompt_bool(
  "Override key operational parameters?",
  False,
  help_text="If you chose a preset, keep this 'no' unless you want to tweak incidents/SLA/governance.",
)

if override:
  print("\n-- Operational incidents & SLA --")
cfg.SLA_hours = prompt_float("SLA hours", cfg.SLA_hours, help_text="SLA threshold (hours).")
cfg.p_NF = prompt_float("p_NF notification failure", cfg.p_NF)
cfg.p_PR = prompt_float("p_PR reconciliation mismatch", cfg.p_PR)
cfg.p_SM = prompt_float("p_SM state mismatch", cfg.p_SM)
cfg.p_PI = prompt_float("p_PI payout integrity breach", cfg.p_PI)
cfg.tau_logn_mu = prompt_float("LogNormal mu (log-hours)", cfg.tau_logn_mu)
cfg.tau_logn_sigma = prompt_float("LogNormal sigma (log-hours)", cfg.tau_logn_sigma)

print("\n-- Governance --")
cfg.theta_D_to_pivot = prompt_int("Disputes threshold to trigger PIVOT", cfg.theta_D_to_pivot)
cfg.theta_D_to_vote = prompt_int("Disputes threshold to trigger VOTE", cfg.theta_D_to_vote)
cfg.p_vote_resolve = prompt_float("Vote resolve probability", cfg.p_vote_resolve)
cfg.p_pivot_resolve_boost = prompt_float("Pivot closure rate boost", cfg.p_pivot_resolve_boost)
cfg.close_rate_base = prompt_float("close_rate_base", cfg.close_rate_base)
cfg.chef_inactivity_prob = prompt_float("Chef inactivity probability", cfg.chef_inactivity_prob)
cfg.pivot_inactivity_prob = prompt_float("Pivot inactivity probability", cfg.pivot_inactivity_prob)

print("\n-- Matelas --")
cfg.matelas_enabled = prompt_bool("Enable matelas?", cfg.matelas_enabled)
if cfg.matelas_enabled:
  cfg.matelas_contrib_per_member = prompt_float("Matelas contrib per member per period", cfg.matelas_contrib_per_member)
cfg.matelas_max_multiple_of_c = prompt_float("Matelas cap multiple", cfg.matelas_max_multiple_of_c)
cfg.matelas_cover_prob = prompt_float("Matelas cover probability", cfg.matelas_cover_prob)
cfg.matelas_dispute_reduction = prompt_float("Dispute reduction when matelas used", cfg.matelas_dispute_reduction)

print("\nConfig ready.\n")

print("--- Scenario summary (traceability) ---")
print(f"Regime={cfg.regime} | G={cfg.G} | seed={cfg.seed} | T_equals_N={cfg.T_equals_N}")
print(f"Group size mix: {len(cfg.N_values)} sizes; min={min(cfg.N_values)}, max={max(cfg.N_values)}")
print(f"Incidents: p_NF={cfg.p_NF}, p_PR={cfg.p_PR}, p_SM={cfg.p_SM}, p_PI={cfg.p_PI}")
print(f"SLA_hours={cfg.SLA_hours} | tau_logn_mu={cfg.tau_logn_mu} | tau_logn_sigma={cfg.tau_logn_sigma}")
print(f"Governance: close_rate_base={cfg.close_rate_base}, chef_inact={cfg.chef_inactivity_prob}, pivot_inact={cfg.pivot_inactivity_prob}")
if cfg.matelas_enabled:
  print(f"Matelas: contrib={cfg.matelas_contrib_per_member}, cap_mult={cfg.matelas_max_multiple_of_c}, cover_prob={cfg.matelas_cover_prob}, dispute_reduction={cfg.matelas_dispute_reduction}")
else:
  print("Matelas: disabled")

return cfg


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------



# ---------------------------------------------------------------------
# HTML dashboard (interactive charts)
# ---------------------------------------------------------------------

def _compute_byN_revenue(raw_df: pd.DataFrame) -> pd.DataFrame | None:
  """Compute RevenueAtRisk mean/p95/p99 by group size N from raw simulation output."""
if raw_df is None or not {"N", "RevenueAtRisk"}.issubset(raw_df.columns):
  return None
g = raw_df.groupby("N")["RevenueAtRisk"]
out = g.agg(
  RevMean="mean",
  RevP95=lambda s: s.quantile(0.95),
  RevP99=lambda s: s.quantile(0.99),
).reset_index()
return out


def _compute_byN_cost_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame | None:
  """Compute TotalCost mean/p95/p99 by N if byN does not provide it."""
if raw_df is None or not {"N", "TotalCost"}.issubset(raw_df.columns):
  return None
g = raw_df.groupby("N")["TotalCost"]
out = g.agg(
  CostMean="mean",
  CostP95=lambda s: s.quantile(0.95),
  CostP99=lambda s: s.quantile(0.99),
).reset_index()
return out


def generate_html_dashboard(res: Dict[str, pd.DataFrame], cfg: Config, out_file: str, title: str | None = None, cost_log_scale: bool = False) -> str:
  """Generate a standalone HTML report with interactive charts.

    Charts:
      1) Failure probabilities vs N: P_Hard, P_Soft, P_Fail
      2) Total cost vs N: mean, p95, p99 (from byN if available, else computed from raw)
      3) Revenue-at-risk vs N: mean, p95, p99 (computed from raw)

    The HTML embeds Plotly JS so it can be opened offline.
    """
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

byN = res.get("byN").copy()
raw = res.get("raw")
metrics = res.get("metrics")
cost = res.get("cost")

byN = byN.sort_values("N")

# Detect column variants
p_hard_col = "P_Hard" if "P_Hard" in byN.columns else "P_HardCollapse" if "P_HardCollapse" in byN.columns else None
p_soft_col = "P_Soft" if "P_Soft" in byN.columns else "P_SoftCollapse" if "P_SoftCollapse" in byN.columns else None
p_fail_col = "P_Fail" if "P_Fail" in byN.columns else "P_PlatformFailure" if "P_PlatformFailure" in byN.columns else None

# Costs
have_cost_cols = {"CostMean", "CostP95", "CostP99"}.issubset(byN.columns)
cost_byN = byN[["N"]].copy()
if have_cost_cols:
  cost_byN = cost_byN.join(byN[["CostMean", "CostP95", "CostP99"]])
else:
  c = _compute_byN_cost_from_raw(raw)
if c is not None:
  cost_byN = cost_byN.merge(c, on="N", how="left")

rev_byN = _compute_byN_revenue(raw)

if title is None:
  title = f"TRIBUS Monte Carlo Dashboard — regime={cfg.regime} | G={cfg.G} | seed={cfg.seed}"

# Figure 1: failure probabilities
fig_prob = go.Figure()
if p_hard_col:
  fig_prob.add_trace(go.Scatter(x=byN["N"], y=byN[p_hard_col], mode="lines+markers", name="P(HardCollapse)"))
if p_soft_col:
  fig_prob.add_trace(go.Scatter(x=byN["N"], y=byN[p_soft_col], mode="lines+markers", name="P(SoftCollapse)"))
if p_fail_col:
  fig_prob.add_trace(go.Scatter(x=byN["N"], y=byN[p_fail_col], mode="lines+markers", name="P(PlatformFailure)"))
fig_prob.update_layout(
  title="Failure probabilities by group size (N)",
  xaxis_title="Group size N",
  yaxis_title="Probability",
  hovermode="x unified",
  template="plotly_white",
)

# Figure 2: costs
fig_cost = go.Figure()
if {"CostMean", "CostP95", "CostP99"}.issubset(cost_byN.columns):
  fig_cost.add_trace(go.Scatter(x=cost_byN["N"], y=cost_byN["CostMean"], mode="lines+markers", name="TotalCost mean"))
fig_cost.add_trace(go.Scatter(x=cost_byN["N"], y=cost_byN["CostP95"], mode="lines+markers", name="TotalCost p95"))
fig_cost.add_trace(go.Scatter(x=cost_byN["N"], y=cost_byN["CostP99"], mode="lines+markers", name="TotalCost p99"))
fig_cost.update_layout(
  title="Total cost by group size (N)",
  xaxis_title="Group size N",
  yaxis_title="Cost (model units)",
  hovermode="x unified",
  template="plotly_white",
)
fig_cost.update_yaxes(type="log" if cost_log_scale else "linear")

# Figure 3: revenue-at-risk
fig_rev = None
if rev_byN is not None:
  fig_rev = go.Figure()
fig_rev.add_trace(go.Scatter(x=rev_byN["N"], y=rev_byN["RevMean"], mode="lines+markers", name="RevenueAtRisk mean"))
fig_rev.add_trace(go.Scatter(x=rev_byN["N"], y=rev_byN["RevP95"], mode="lines+markers", name="RevenueAtRisk p95"))
fig_rev.add_trace(go.Scatter(x=rev_byN["N"], y=rev_byN["RevP99"], mode="lines+markers", name="RevenueAtRisk p99"))
fig_rev.update_layout(
  title="Revenue-at-risk by group size (N)",
  xaxis_title="Group size N",
  yaxis_title="Revenue-at-risk (model units)",
  hovermode="x unified",
  template="plotly_white",
)

# Summary tables
summary_html = ""
if metrics is not None:
  summary_html += "<h3>Summary metrics</h3>" + metrics.to_html(index=False)
if cost is not None:
  summary_html += "<h3>Cost stats</h3>" + cost.to_html(index=False)

byn_html = "<h3>Breakdown by N</h3>" + byN.to_html(index=False)

parts = [
  f"<h1>{title}</h1>",
  "<p><em>Interactive charts generated directly by TRIBUS Monte Carlo. Plotly JS is embedded for offline viewing.</em></p>",
  summary_html,
  "<h2>Charts</h2>",
  pio.to_html(fig_prob, full_html=False, include_plotlyjs=True),
  pio.to_html(fig_cost, full_html=False, include_plotlyjs=False),
]
if fig_rev is not None:
  parts.append(pio.to_html(fig_rev, full_html=False, include_plotlyjs=False))
parts.append(byn_html)

html = "\n".join(parts)
Path(out_file).write_text(html, encoding="utf-8")
return out_file
def main():
  cfg = interactive_config()

# Run either a single scenario or a full suite (central + stresses)
if getattr(cfg, "_run_suite", False):
  scenarios = ["central", "peak_load", "major_incident"]
suite_results = {}
all_metrics = []
all_cost = []
all_byN = []

for scen in scenarios:
  cfg_s = copy.deepcopy(cfg)
apply_preset(cfg_s, f"{cfg.regime}.{scen}")
cfg_s._availability_target = getattr(cfg, "_availability_target", "99.9")
apply_availability_target(cfg_s)
apply_overrides(cfg_s, getattr(cfg, "_overrides", {}))

res = simulate(cfg_s)
metrics_flagged = add_tension_flags(res["metrics"])
metrics_flagged.insert(0, "Scenario", scen)

cost_df = res["cost"].copy()
cost_df.insert(0, "Scenario", scen)

byN_df = res["byN"].copy()
byN_df.insert(0, "Scenario", scen)

suite_results[scen] = {"cfg": cfg_s, "res": res, "metrics_flagged": metrics_flagged}
all_metrics.append(metrics_flagged)
all_cost.append(cost_df)
all_byN.append(byN_df)

df_metrics_flagged = pd.concat(all_metrics, ignore_index=True)
df_cost_all = pd.concat(all_cost, ignore_index=True)
df_byN_all = pd.concat(all_byN, ignore_index=True)
overall_by_scenario = {sc: overall_tension(suite_results[sc]["metrics_flagged"]) for sc in scenarios}

print() 
print("=== Scenario Suite: Tension Dashboard (universal, conservative) ===")
for scen, ov in overall_by_scenario.items():
  print(
    f"Scenario={scen} | Overall zone: {ov['OverallZone']} | score={ov['Score']} "
    f"(RED={ov['RedCount']}, AMBER={ov['AmberCount']})"
  )
print(ov["Message"])

print() 
print("=== Summary Metrics (with tension flags) — ALL scenarios ===")
print(df_metrics_flagged.to_string(index=False))

print() 
print("=== Cost Stats (mean / p50 / p95 / p99) — ALL scenarios ===")
print(df_cost_all.to_string(index=False))

print() 
print("=== Breakdown by Group Size N — ALL scenarios ===")
print(df_byN_all.to_string(index=False))

# Headline comparison view
try:
  pivot = df_metrics_flagged.pivot(index="Metric", columns="Scenario", values="Value")
headline = [
  m
  for m in [
    "P(PlatformFailure)",
    "P(HardCollapse)",
    "P(SoftCollapse)",
    "P(AnyEscalation)",
    "MeanTotalImpactPct",
    "MeanRevenueAtRiskPct",
  ]
  if m in pivot.index
]
if headline:
  print() 
print("=== Headline comparison (Value by scenario) ===")
print(pivot.loc[headline].to_string())
except Exception:
  pass

save = prompt_bool(
  "Save suite outputs to CSV?",
  True,
  help_text="Exports combined metrics/cost/byN plus per-scenario raw CSV files.",
)
if save:
  prefix = f"tribus_mc_{cfg.regime}_suite"
df_metrics_flagged.to_csv(f"{prefix}_metrics.csv", index=False)
df_cost_all.to_csv(f"{prefix}_cost.csv", index=False)
df_byN_all.to_csv(f"{prefix}_byN.csv", index=False)
for scen in scenarios:
  suite_results[scen]["res"]["raw"].to_csv(f"{prefix}_{scen}_raw.csv", index=False)
print(f"Saved suite: {prefix}_metrics.csv, {prefix}_cost.csv, {prefix}_byN.csv + per-scenario raw CSVs")

make_dash = prompt_bool(
  "Generate HTML dashboards for each scenario?",
  True,
  help_text="Creates one HTML report per scenario (central, peak_load, major_incident).",
)
if make_dash:
  prefix = f"tribus_mc_{cfg.regime}_suite"
for scen in scenarios:
  out_html = f"{prefix}_{scen}_dashboard.html"
generate_html_dashboard(
  suite_results[scen]["res"],
  suite_results[scen]["cfg"],
  out_html,
  cost_log_scale=False,
)
print(f"Saved HTML dashboard: {out_html}")

else:
  res = simulate(cfg)
metrics_flagged = add_tension_flags(res["metrics"])
overall = overall_tension(metrics_flagged)

print() 
print("=== Tension Dashboard (universal, conservative) ===")
print(
  f"Overall zone: {overall['OverallZone']} | score={overall['Score']} "
  f"(RED={overall['RedCount']}, AMBER={overall['AmberCount']})"
)
print(overall["Message"])

print() 
print("=== Summary Metrics (with tension flags) ===")
print(metrics_flagged.to_string(index=False))

print() 
print("=== Cost Stats (mean / p50 / p95 / p99) ===")
print(res["cost"].to_string(index=False))

print() 
print("=== Breakdown by Group Size N ===")
print(res["byN"].to_string(index=False))

save = prompt_bool(
  "Save outputs to CSV?",
  True,
  help_text="Exports raw + metrics + cost + byN as CSV files.",
)
if save:
  prefix = f"tribus_mc_{cfg.regime}"
res["raw"].to_csv(f"{prefix}_raw.csv", index=False)
metrics_flagged.to_csv(f"{prefix}_metrics.csv", index=False)
res["cost"].to_csv(f"{prefix}_cost.csv", index=False)
res["byN"].to_csv(f"{prefix}_byN.csv", index=False)
print(f"Saved: {prefix}_raw.csv, {prefix}_metrics.csv, {prefix}_cost.csv, {prefix}_byN.csv")

make_dash = prompt_bool(
  "Generate HTML dashboard?",
  True,
  help_text="Creates an interactive HTML report with charts by group size N.",
)
if make_dash:
  prefix = f"tribus_mc_{cfg.regime}"
out_html = f"{prefix}_dashboard.html"
generate_html_dashboard(res, cfg, out_html, cost_log_scale=False)
print(f"Saved HTML dashboard: {out_html}")

print() 
print("Done.")
print()
if __name__ == "__main__":
  main()
