"""tribus_combined_risk_model.py
TRIBUS combined Macro–Ops stress → contribution revenue risk model.

Replaces risk_analysis.py with these fixes and improvements:
- All indentation errors corrected (original had return statements and class
  fields at column 0 inside function/class bodies, causing SyntaxErrors).
- CombinedConfig dataclass fields properly indented as class attributes.
- OPTIMIZATION: q_base (sigmoid of a constant) is computed once as a scalar
  before the monthly loop instead of allocating an (n_sims,) array of
  identical values every month.
- Prompt helpers imported from cli_prompts.
- Logic, math and all scenario combinations are identical to the original.

What this simulates
-------------------
Correlated macro shock (M) and ops shock (O) drive:
  - Per-member payment probability
  - Recovery rate
  - Churn / joining rate
  - Group failure hazards (calibrated from tribus_stress_engine via PlatformCalibrator)

Outputs a 2×3 scenario matrix (macro: baseline/adverse × ops: central/peak/major)
with revenue, arrears and group/user survival statistics.

Run:
  python tribus_combined_risk_model.py

Requires tribus_stress_engine.py in the same folder (used as platform calibrator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple
import importlib.util

import numpy as np
import pandas as pd

from cli_prompts import ask_bool, ask_float, ask_int, ask_str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def monthly_hazard_from_cycle_prob(p_cycle: float, cycle_len: int) -> float:
    """Convert P(>=1 event over cycle_len periods) into per-period hazard."""
    p_cycle = float(np.clip(p_cycle, 0.0, 1.0))
    cycle_len = max(int(cycle_len), 1)
    return 1.0 - (1.0 - p_cycle) ** (1.0 / cycle_len)


def draw_correlated_normals(
    rng: np.random.Generator, n: int, rho: float,
    m_shift: float = 0.0, o_shift: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw (M, O) correlated normals with Corr=rho and optional mean shifts."""
    rho = float(np.clip(rho, -0.999, 0.999))
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    M = z1 + m_shift
    O = rho * z1 + np.sqrt(1.0 - rho**2) * z2 + o_shift
    return M, O


def sample_mean_severity_per_sim(
    rng: np.random.Generator, sev_pool: np.ndarray, counts: np.ndarray,
) -> np.ndarray:
    """For each simulation i, sample counts[i] severities and return per-sim mean.

    Robust to zero-count entries (including trailing zeros).
    """
    counts_int = counts.astype(int)
    total = int(counts_int.sum())
    if total <= 0 or sev_pool.size == 0:
        return np.zeros_like(counts, dtype=float)

    draws = rng.choice(sev_pool, size=total, replace=True)

    nz = counts_int > 0
    counts_nz = counts_int[nz]
    cs = np.cumsum(counts_nz)
    starts = np.concatenate(([0], cs[:-1]))

    sums_nz = np.add.reduceat(draws, starts)
    means = np.zeros_like(counts, dtype=float)
    means[nz] = sums_nz / counts_nz
    return means


def loss_from_severity(
    sev_mean: np.ndarray, impact_mode: str, gamma: np.ndarray,
) -> np.ndarray:
    s = np.clip(gamma * np.clip(sev_mean, 0.0, None), 0.0, None)
    if impact_mode == "revenue_loss":
        return np.minimum(s, 1.0)
    return 1.0 - np.exp(-np.minimum(s, 5.0))


# ---------------------------------------------------------------------------
# Load platform engine module
# ---------------------------------------------------------------------------

def load_platform_engine(path: str = "tribus_stress_engine.py"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find platform engine: {path}")
    spec = importlib.util.spec_from_file_location("tribus_platform_engine", str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPS_SCENARIOS = ["central", "peak_load", "major_incident"]
MACRO_SCENARIOS = ["baseline", "adverse"]


@dataclass
class CombinedConfig:
    # Simulation
    n_sims: int = 5000
    months: int = 12
    seed: int = 42

    # Portfolio / size mix
    total_groups: int = 100
    N_values: Tuple[int, ...] = (10, 12, 15, 20, 25, 30)
    N_probs: Tuple[float, ...] = (0.25, 0.15, 0.10, 0.15, 0.15, 0.20)

    # Monetization (C=1 normalized)
    tau_take_rate: float = 0.015
    penalty_lambda: float = 0.025
    penalty_realization: float = 0.30

    # Macro behavioural targets (baseline)
    p_eff0: float = 0.95
    q0_baseline: float = 0.995
    r0: float = 0.55
    j0: float = 0.002

    # Macro sensitivities
    b_payM: float = -0.9
    b_recM: float = 0.7
    b_jM: float = 0.9

    # Macro adverse shift
    macro_shift_adverse: float = 0.9

    # Macro–Ops correlation
    rho_MO: float = 0.35
    ops_shift_by_regime: Optional[Dict[str, float]] = None

    # Ops intensity scaling with O
    kappa_fail_O: float = 0.60
    kappa_sev_O: float = 0.50

    # Severity -> collection impairment mapping
    impact_mode: str = "economic_loss"
    gamma_impact_to_q: float = 1.0
    q_floor: float = 0.0

    # Soft collapse -> death weight
    soft_to_death_weight: float = 0.35

    # Arrears writeoff on group death
    kappa_arrears_group: float = 1.5

    # Platform engine controls
    platform_regime: str = "existing"
    platform_G: int = 5000
    platform_seed: int = 42
    platform_rev_per_member_per_month: float = 1.0

    def __post_init__(self):
        if self.ops_shift_by_regime is None:
            self.ops_shift_by_regime = {
                "central": 0.0,
                "peak_load": 0.25,
                "major_incident": 0.50,
            }


# ---------------------------------------------------------------------------
# Platform calibration
# ---------------------------------------------------------------------------

class PlatformCalibrator:
    def __init__(self, platform_mod, cfg: CombinedConfig):
        self.pm = platform_mod
        self.cfg = cfg
        self.cache: Dict[str, Dict] = {}

    def run_ops_scenario(self, ops_scenario: str) -> Dict:
        if ops_scenario in self.cache:
            return self.cache[ops_scenario]

        pcfg = self.pm.Config()
        pcfg.regime = self.cfg.platform_regime
        pcfg.G = self.cfg.platform_G
        pcfg.seed = self.cfg.platform_seed
        pcfg.rev_per_member_per_month = self.cfg.platform_rev_per_member_per_month
        pcfg.N_values = tuple(self.cfg.N_values)
        pcfg.N_probs = tuple(self.cfg.N_probs)
        pcfg.T_equals_N = True

        self.pm.apply_preset(pcfg, f"{pcfg.regime}.{ops_scenario}")
        pcfg._availability_target = "99.9"
        self.pm.apply_availability_target(pcfg)

        res = self.pm.simulate(pcfg)
        raw = res["raw"].copy()
        byN = res["byN"].copy()

        vote_failed_byN = (
            raw.groupby("N")["VoteFailedFlag"].mean()
            .rename("P_VoteFailed").reset_index()
        )
        byN = byN.merge(vote_failed_byN, on="N", how="left")

        sev_samples: Dict[int, np.ndarray] = {
            int(n): raw.loc[raw["N"] == n, "TotalImpactPct"].to_numpy(copy=True)
            for n in sorted(raw["N"].unique())
        }

        out = {"raw": raw, "byN": byN, "sev_samples": sev_samples}
        self.cache[ops_scenario] = out
        return out

    def get_size_maps(self, ops_scenario: str) -> Dict[int, Dict[str, float]]:
        byN = self.run_ops_scenario(ops_scenario)["byN"]
        maps: Dict[int, Dict[str, float]] = {}
        for _, row in byN.iterrows():
            n = int(row["N"])
            maps[n] = {
                "h_fail0": monthly_hazard_from_cycle_prob(float(row.get("P_Fail", 0.0)), n),
                "h_hard0": monthly_hazard_from_cycle_prob(float(row.get("P_Hard", 0.0)), n),
                "h_soft0": monthly_hazard_from_cycle_prob(float(row.get("P_Soft", 0.0)), n),
            }
        return maps


# ---------------------------------------------------------------------------
# Combined simulation
# ---------------------------------------------------------------------------

def simulate_combined(
    cfg: CombinedConfig,
    calibrator: PlatformCalibrator,
    macro_scenario: str,
    ops_scenario: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    N_vals = np.array(cfg.N_values, dtype=int)
    probs = np.array(cfg.N_probs, dtype=float)
    probs = probs / probs.sum()

    n_groups_size = np.floor(cfg.total_groups * probs).astype(int)
    rem = cfg.total_groups - int(n_groups_size.sum())
    if rem > 0:
        n_groups_size[np.argsort(-probs)[:rem]] += 1

    alive_groups = np.tile(n_groups_size, (cfg.n_sims, 1)).astype(int)
    arrears = np.zeros(cfg.n_sims, dtype=float)
    users = (alive_groups * N_vals).sum(axis=1).astype(float)

    a_pay = logit(min(cfg.p_eff0 / max(cfg.q0_baseline, 1e-6), 0.999999))
    a_q0 = logit(cfg.q0_baseline)
    a_rec = logit(min(cfg.r0 / max(cfg.q0_baseline, 1e-6), 0.999999))
    a_j = logit(cfg.j0)

    # OPTIMIZATION: q_base is sigmoid of a constant — compute scalar once,
    # not sigmoid(a_q0 * np.ones(n_sims)) (new array allocation) every month.
    q_base_scalar = float(1.0 / (1.0 + np.exp(-a_q0)))

    m_shift = 0.0 if macro_scenario == "baseline" else cfg.macro_shift_adverse
    o_shift = float(cfg.ops_shift_by_regime.get(ops_scenario, 0.0))

    size_haz0 = calibrator.get_size_maps(ops_scenario)
    sev_pools = calibrator.run_ops_scenario(ops_scenario)["sev_samples"]

    rev_total = np.zeros(cfg.n_sims, dtype=float)

    for _ in range(cfg.months):
        M, O = draw_correlated_normals(rng, cfg.n_sims, cfg.rho_MO, m_shift=m_shift, o_shift=o_shift)

        p_user = sigmoid(a_pay + cfg.b_payM * M)
        # q_base_scalar broadcasts correctly — no array allocation needed
        r_base = sigmoid(a_rec - cfg.b_recM * M)
        j = sigmoid(a_j + cfg.b_jM * M)

        paid_total = np.zeros(cfg.n_sims, dtype=float)
        miss_total = np.zeros(cfg.n_sims, dtype=float)

        for jn, n in enumerate(N_vals):
            g_alive = alive_groups[:, jn]
            if np.all(g_alive == 0):
                continue

            hz0 = size_haz0.get(int(n), {"h_fail0": 0.0, "h_hard0": 0.0, "h_soft0": 0.0})

            scale = np.exp(cfg.kappa_fail_O * O)
            h_fail = np.clip(hz0["h_fail0"] * scale, 0.0, 0.50)
            h_hard = np.clip(hz0["h_hard0"] * scale, 0.0, 0.30)
            h_soft = np.clip(hz0["h_soft0"] * scale, 0.0, 0.40)

            g_fail = rng.binomial(g_alive, h_fail)
            g_nonfail = g_alive - g_fail

            members_fail = g_fail.astype(float) * float(n)
            members_nonfail = g_nonfail.astype(float) * float(n)
            members_total = members_fail + members_nonfail

            sev_pool = sev_pools.get(int(n), np.array([], dtype=float))
            sev_mean = sample_mean_severity_per_sim(rng, sev_pool, g_fail)

            gamma_vec = cfg.gamma_impact_to_q * np.exp(cfg.kappa_sev_O * O)
            loss_vec = loss_from_severity(sev_mean, cfg.impact_mode, gamma_vec)
            q_fail_vec = np.clip(1.0 - loss_vec, cfg.q_floor, 1.0)

            p_eff_nonfail = np.clip(p_user * q_base_scalar, 0.0, 1.0)
            p_eff_fail = np.clip(p_user * q_base_scalar * q_fail_vec, 0.0, 1.0)

            paid_nonfail = rng.binomial(members_nonfail.astype(int), p_eff_nonfail)
            paid_fail = rng.binomial(members_fail.astype(int), p_eff_fail)
            paid = paid_nonfail + paid_fail
            miss = members_total - paid

            paid_total += paid
            miss_total += miss

            g_hard = rng.binomial(g_alive, h_hard)
            g_soft = rng.binomial(np.maximum(g_alive - g_hard, 0), h_soft)
            deaths_float = g_hard + cfg.soft_to_death_weight * g_soft
            deaths_int = np.floor(deaths_float).astype(int)
            deaths_int += (rng.random(cfg.n_sims) < (deaths_float - deaths_int)).astype(int)
            deaths_int = np.minimum(deaths_int, g_alive)

            alive_groups[:, jn] = g_alive - deaths_int

            frac_dead = deaths_int / np.maximum(g_alive, 1)
            arrears *= (1.0 - np.clip(cfg.kappa_arrears_group * frac_dead, 0.0, 1.0))

        arrears += miss_total

        denom = np.maximum(users * p_user * q_base_scalar, 1.0)
        q_eff = np.clip(paid_total / denom, 0.0, 1.0)
        r_eff = np.clip(r_base * q_eff, 0.0, 1.0)
        rec = rng.binomial(np.maximum(arrears, 0).astype(int), r_eff)
        arrears -= rec

        cr_users = rng.binomial(np.maximum(users, 0).astype(int), np.clip(j, 0.0, 1.0))
        users = np.maximum(users - cr_users, 0.0)
        share_cr = cr_users / np.maximum(users + cr_users, 1.0)
        arrears *= (1.0 - share_cr)

        rev_fee = cfg.tau_take_rate * (paid_total + rec)
        rev_pen = cfg.penalty_lambda * cfg.penalty_realization * miss_total
        rev_total += rev_fee + rev_pen

    groups_end = alive_groups.sum(axis=1).astype(float)

    return pd.DataFrame({
        "Macro": macro_scenario,
        "Ops": ops_scenario,
        "RevenueTotal": rev_total,
        "ArrearsEnd": arrears,
        "GroupsEnd": groups_end,
        "UsersEnd": users,
    })


# ---------------------------------------------------------------------------
# Summarize a single scenario DataFrame
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame, baseline_ref: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    out: Dict[str, float] = {
        "rev_mean": float(df["RevenueTotal"].mean()),
        "rev_p50": float(df["RevenueTotal"].quantile(0.50)),
        "rev_p05": float(df["RevenueTotal"].quantile(0.05)),
        "arrears_p50": float(df["ArrearsEnd"].quantile(0.50)),
        "arrears_p95": float(df["ArrearsEnd"].quantile(0.95)),
        "groups_p50": float(df["GroupsEnd"].quantile(0.50)),
        "groups_p05": float(df["GroupsEnd"].quantile(0.05)),
        "users_p50": float(df["UsersEnd"].quantile(0.50)),
        "users_p05": float(df["UsersEnd"].quantile(0.05)),
    }
    if baseline_ref is not None:
        base_med = float(baseline_ref["RevenueTotal"].quantile(0.50))
        out["p_rev_drawdown_20"] = float((df["RevenueTotal"] < 0.80 * base_med).mean())
        out["p_rev_drawdown_40"] = float((df["RevenueTotal"] < 0.60 * base_med).mean())
        out["RaR95_vs_baseMed"] = float(base_med - out["rev_p05"])
    else:
        out["p_rev_drawdown_20"] = float("nan")
        out["p_rev_drawdown_40"] = float("nan")
        out["RaR95_vs_baseMed"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Full 2×3 scenario suite
# ---------------------------------------------------------------------------

def run_suite(cfg: CombinedConfig, calibrator: PlatformCalibrator) -> pd.DataFrame:
    df_base = simulate_combined(cfg, calibrator, "baseline", "central")
    rows = [{"Macro": "baseline", "Ops": "central", **summarize(df_base, baseline_ref=df_base)}]

    for macro in MACRO_SCENARIOS:
        for ops in OPS_SCENARIOS:
            if macro == "baseline" and ops == "central":
                continue
            df = simulate_combined(cfg, calibrator, macro, ops)
            rows.append({"Macro": macro, "Ops": ops, **summarize(df, baseline_ref=df_base)})

    out = pd.DataFrame(rows)
    out.insert(0, "Scenario", out["Macro"] + "__" + out["Ops"])
    return out


# ---------------------------------------------------------------------------
# Interactive entry point
# ---------------------------------------------------------------------------

def interactive() -> None:
    print("\n=== TRIBUS Combined Risk Model (Macro–Ops → Contribution Revenue Risk, C=1) ===\n")

    cfg = CombinedConfig()

    cfg.n_sims = ask_int("Number of simulations", cfg.n_sims)
    cfg.months = ask_int("Horizon (months)", cfg.months)
    cfg.seed = ask_int("Random seed", cfg.seed)
    cfg.total_groups = ask_int("Total groups in portfolio", cfg.total_groups)

    Nv = ask_str("Group sizes N (comma-separated)", ",".join(map(str, cfg.N_values)))
    Np = ask_str("Group size probabilities (comma-separated)", ",".join(map(str, cfg.N_probs)))
    N_values = [int(x.strip()) for x in Nv.split(",") if x.strip()]
    N_probs_raw = [float(x.strip()) for x in Np.split(",") if x.strip()]
    if len(N_probs_raw) != len(N_values):
        raise ValueError("Length of probabilities must match length of sizes.")
    total = sum(N_probs_raw)
    cfg.N_values = tuple(N_values)
    cfg.N_probs = tuple(p / total for p in N_probs_raw)

    cfg.tau_take_rate = ask_float("Take rate tau (e.g., 0.015 = 1.5%)", cfg.tau_take_rate)
    cfg.penalty_lambda = ask_float("Penalty lambda on MISS (fraction of contribution)", cfg.penalty_lambda)
    cfg.penalty_realization = ask_float("Penalty realization rate (0-1)", cfg.penalty_realization)

    if ask_bool("Adjust Macro–Ops correlation settings?", True):
        cfg.rho_MO = ask_float("Correlation rho between macro shock M and ops shock O", cfg.rho_MO)
        cfg.kappa_fail_O = ask_float("Ops hazard amplification kappa_fail_O", cfg.kappa_fail_O)
        cfg.kappa_sev_O = ask_float("Severity amplification kappa_sev_O", cfg.kappa_sev_O)

    if ask_bool("Adjust macro adverse shift?", False):
        cfg.macro_shift_adverse = ask_float("Macro adverse shift (mean M)", cfg.macro_shift_adverse)

    cfg.impact_mode = ask_str("Impact mapping mode (economic_loss/revenue_loss)", cfg.impact_mode)
    cfg.gamma_impact_to_q = ask_float("Severity scaling gamma (TotalImpactPct -> loss)", cfg.gamma_impact_to_q)

    cfg.platform_regime = ask_str("Platform engine regime (existing/public)", cfg.platform_regime).lower()
    cfg.platform_G = ask_int("Platform engine Monte Carlo groups G (calibration)", cfg.platform_G)
    cfg.platform_seed = ask_int("Platform engine seed", cfg.platform_seed)

    platform_mod = load_platform_engine()
    calibrator = PlatformCalibrator(platform_mod, cfg)

    print("\nCalibrating ops regimes from platform stress engine ...\n")
    for ops in OPS_SCENARIOS:
        calibrator.run_ops_scenario(ops)

    print("Running combined 2×3 suite (macro × ops) ...\n")
    table = run_suite(cfg, calibrator)

    pd.set_option("display.max_columns", None)
    print("=== Combined Scenario Summary (normalized revenue; multiply by contribution to scale) ===")
    print(table.to_string(index=False))

    if ask_bool("Save outputs to CSV?", True):
        out_file = "tribus_combined_risk_model_suite.csv"
        table.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")

    print("\nDone.\n")


if __name__ == "__main__":
    interactive()
