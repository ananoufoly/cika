"""rosca_score_engine.py
ROSCA-based credit score simulation engine — population edition.

Design
------
Members are NOT pre-labelled.  Each member's behavioural parameters are drawn
independently from population-level distributions.  A macro environment can
apply systemic and within-group shocks.  A true PD is derived from the raw
drawn parameters (the oracle ground truth).  The 5-pillar score is then
computed from noisy meeting data alone.

This lets us ask: does the score correctly rank real risk, across different
parameter settings and stress regimes?

Pillars (unchanged from spec)
------------------------------
  pdis  — Payment discipline          (max 35)
  ordr  — Order & post-payout         (max 15)
  gov   — Governance & enforcement    (max 20)
  liq   — Liquidity stress (bidding)  (max 15)
  soc   — Social capital              (max 15)
  Total score S* ∈ [0, 100]

Usage
-----
>>> from rosca_score_engine import (
...     PopulationParams, MacroEnvironment, ScoreParams,
...     generate_population, run_sensitivity,
... )
>>> pop   = PopulationParams(n_groups=20)
>>> macro = MacroEnvironment(stress_level=0.20, within_group_corr=0.30)
>>> params = ScoreParams()
>>> result = generate_population(pop, macro, params, seed=42)
>>> print(result.member_df[["mid", "gid", "true_pd", "score"]].head(20))
>>> print(result.validation)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, fields, replace as dc_replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _log1p_safe(x: float) -> float:
    return float(np.log1p(max(0.0, x)))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no scipy needed)."""
    n = len(x)
    if n < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d2 = ((rx - ry) ** 2).sum()
    return float(1.0 - 6.0 * d2 / (n * (n ** 2 - 1)))


# ---------------------------------------------------------------------------
# 1. Score parameters  (unchanged from spec)
# ---------------------------------------------------------------------------

@dataclass
class ScoreParams:
    """All tunable hyperparameters for the 5-pillar score.

    Defaults are the recommended starting values from Tables 2-4 of the spec.
    """
    # pdis
    a: float = 0.80
    c_otr: float = 0.85
    k_otr: float = 12.0
    a_al: float = 0.70
    a_ls: float = 0.60
    c_rc: float = 0.70
    k_rc: float = 10.0
    # ordr
    a_slip: float = 0.80
    # gov
    k_rules: float = 12.0
    a_san: float = 0.60
    # liq
    q0: float = 0.50
    k_q: float = 10.0
    v_ref: float = 0.10
    a_v: float = 0.80
    # soc
    w_rep: float = 5.0
    w_cent: float = 4.0
    w_endf: float = 3.0
    w_ends: float = 3.0


# ---------------------------------------------------------------------------
# 2. Population parameters  (govern member & group distributions)
# ---------------------------------------------------------------------------

@dataclass
class PopulationParams:
    """Distributions from which groups and members are randomly drawn.

    Intuitive knobs:
      p_ontime_mean  — average on-time payment rate in the population [0,1]
      p_ontime_conc  — concentration (higher = tighter spread around mean)
      The Beta(α, β) for p_ontime is derived as:
        α = mean * conc,  β = (1 - mean) * conc
    """
    # ---- group structure ---------------------------------------------------
    n_groups: int = 20
    group_size_min: int = 6
    group_size_max: int = 20
    rtype_bidding_prob: float = 0.50    # prob group uses bidding allocation
    rules_prob: float = 0.75            # prob group has formal written rules
    san_rate_min: float = 0.10
    san_rate_max: float = 1.20
    num_cycles_min: int = 1
    num_cycles_max: int = 3

    # ---- payment discipline ------------------------------------------------
    p_ontime_mean: float = 0.80         # population mean on-time rate
    p_ontime_conc: float = 9.0          # Beta concentration (spread control)

    dlate_lognorm_mu: float = 1.6       # log-mean of days-late when late
    dlate_lognorm_sigma: float = 0.8    # log-std  of days-late when late

    # ---- post-payout slip --------------------------------------------------
    post_slip_mean: float = 0.08        # mean slip tendency in population
    post_slip_conc: float = 8.0         # Beta concentration

    # ---- bidding behaviour (only for bidding groups) -----------------------
    bid_agg_mean: float = 0.22          # mean bid aggressiveness
    bid_agg_conc: float = 7.0           # Beta concentration
    bid_vol_min: float = 0.02
    bid_vol_max: float = 0.30

    # ---- social capital (Bernoulli probs) ----------------------------------
    p_rep: float = 0.45
    p_cent: float = 0.30
    p_endf: float = 0.25
    p_ends: float = 0.15

    # ---- surety (categorical: none / weak / strong) ------------------------
    p_sure_none: float = 0.40
    p_sure_weak: float = 0.35
    # p_sure_strong = 1 - none - weak

    def _beta_params(self, mean: float, conc: float) -> Tuple[float, float]:
        mean = float(np.clip(mean, 1e-4, 1 - 1e-4))
        conc = max(conc, 0.5)
        return mean * conc, (1 - mean) * conc


# ---------------------------------------------------------------------------
# 3. Macro environment
# ---------------------------------------------------------------------------

@dataclass
class MacroEnvironment:
    """Systemic and within-group shocks.

    stress_level:
        Baseline macro stress [0, 1].  Reduces effective p_ontime for everyone
        via a logit-space downward shift.  0 = benign, 1 = severe.

    within_group_corr:
        Fraction of variance that is common within a group [0, 1].
        0 = fully idiosyncratic.  1 = entire group moves together.

    shock_windows:
        List of (first_meeting_no, last_meeting_no, severity_multiplier).
        During these meetings, p_ontime is multiplied by the severity.
        E.g., [(5, 8, 0.70)] = meetings 5-8 are 30% worse than baseline.
    """
    stress_level: float = 0.0
    within_group_corr: float = 0.20
    shock_windows: List[Tuple[int, int, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 4. Member profile  (raw drawn parameters — the oracle)
# ---------------------------------------------------------------------------

@dataclass
class MemberProfile:
    """Behavioural parameters drawn for one member.  No archetype label."""
    mid: str
    gid: str
    n: int                   # group size
    aord: int                # 1-based allocation order in first cycle
    rtype: str
    rules: bool
    san_rate: float
    num_cycles: int

    # Drawn parameters (ground truth)
    p_ontime_raw: float
    dlate_mu: float
    post_slip_prob: float
    bid_aggressiveness: float
    bid_volatility: float
    rep: bool
    cent: bool
    endf: bool
    ends: bool
    sure_str: str            # none | weak | strong

    # Oracle true PD (set after drawing)
    true_pd: float = 0.0


# ---------------------------------------------------------------------------
# 5. True PD  (oracle label — derived from raw parameters, NOT from score)
# ---------------------------------------------------------------------------

def _compute_true_pd(
    profile: MemberProfile,
    macro_stress: float,
    rng: np.random.Generator,
) -> float:
    """Logistic true PD from raw behavioural parameters + macro.

    Coefficients are chosen to give a realistic PD range (~2 – 60%).
    The score does NOT see true_pd — it only sees the noisy meeting data.
    """
    sure_val = {"none": 0.0, "weak": 0.5, "strong": 1.0}.get(profile.sure_str, 0.0)
    social = float(profile.rep + profile.cent + profile.endf + profile.ends) / 4.0

    logit_pd = (
        -3.50
        + 3.00 * (1.0 - profile.p_ontime_raw)    # discipline (inverted → high p → low PD)
        + 1.50 * profile.bid_aggressiveness        # liquidity stress
        + 1.20 * profile.post_slip_prob            # moral hazard
        + 1.50 * macro_stress                      # macro environment
        - 0.60 * social                            # social capital (protective)
        - 0.50 * sure_val                          # surety (protective)
        + rng.normal(0.0, 0.30)                    # idiosyncratic noise
    )
    return float(_sigmoid(logit_pd))


# ---------------------------------------------------------------------------
# 6. Effective p_ontime  (applies macro shocks per meeting)
# ---------------------------------------------------------------------------

def _effective_p_ontime(
    p_raw: float,
    macro: MacroEnvironment,
    meeting_no: int,
    group_shock: float,    # N(0,1), drawn once per meeting per group
    idio_shock: float,     # N(0,1), drawn per member × meeting
) -> float:
    """Combine baseline, macro stress, within-group correlation, and shocks."""
    corr = float(np.clip(macro.within_group_corr, 0.0, 1.0))
    agg_shock = np.sqrt(corr) * group_shock + np.sqrt(1.0 - corr) * idio_shock

    # Macro stress shifts logit baseline downward
    logit_base = _logit(p_raw) - macro.stress_level * 1.5

    # Aggregate shock adds noise (scale 0.4 = moderate volatility)
    logit_noisy = logit_base + agg_shock * 0.40
    p_eff = _sigmoid(logit_noisy)

    # Apply shock windows (multiplicative severity)
    for t0, t1, sev in macro.shock_windows:
        if t0 <= meeting_no <= t1:
            p_eff *= float(sev)

    return float(np.clip(p_eff, 0.01, 0.99))


# ---------------------------------------------------------------------------
# 7. Meeting data generator for one member
# ---------------------------------------------------------------------------

def _generate_member_meetings(
    profile: MemberProfile,
    macro: MacroEnvironment,
    group_shocks: Dict[int, float],    # meeting_no → group shock value
    rng: np.random.Generator,
    base_date: pd.Timestamp,
    K_min: int = 6,
) -> List[dict]:
    n = profile.n
    meetings_per_cycle = max(K_min, n)

    # Allocation dates — one per cycle (aord position within cycle)
    adate_by_cycle: Dict[int, pd.Timestamp] = {}
    for cid in range(1, profile.num_cycles + 1):
        meeting_no = (cid - 1) * meetings_per_cycle + profile.aord
        adate_by_cycle[cid] = base_date + pd.DateOffset(months=meeting_no - 1)

    total_meetings = meetings_per_cycle * profile.num_cycles
    T_i = base_date + pd.DateOffset(months=total_meetings - 1)

    # Bid attempts (bidding only)
    bid_disc_by_cycle: Dict[int, float] = {}
    if profile.rtype == "bidding":
        for cid in range(1, profile.num_cycles + 1):
            d = rng.normal(profile.bid_aggressiveness,
                           max(profile.bid_volatility, 0.01))
            bid_disc_by_cycle[cid] = float(np.clip(d, 0.0, 0.99))

    rows: List[dict] = []
    for cid in range(1, profile.num_cycles + 1):
        adate_i = adate_by_cycle[cid]
        tdec = int(adate_i <= T_i)

        for m in range(1, meetings_per_cycle + 1):
            meeting_no = (cid - 1) * meetings_per_cycle + m
            mdate = base_date + pd.DateOffset(months=meeting_no - 1)
            if mdate > T_i:
                continue

            # Post-payout slip effect
            is_post_payout = (mdate > adate_i)
            slip_factor = profile.post_slip_prob if is_post_payout else 0.0

            # Effective p_ontime with macro + slip
            g_shock = group_shocks.get(meeting_no, 0.0)
            idio = rng.standard_normal()
            p_eff = _effective_p_ontime(
                profile.p_ontime_raw * (1.0 - 0.5 * slip_factor),
                macro, meeting_no, g_shock, idio,
            )

            ont = int(rng.random() < p_eff)
            if ont:
                dlate = 0
            else:
                raw_late = rng.lognormal(
                    mean=np.log(max(profile.dlate_mu, 1.0)),
                    sigma=0.8,
                )
                dlate = max(1, int(round(raw_late)))

            # Sanction (group-level Poisson-ish)
            san_this = int(rng.random() < profile.san_rate / 6.0)

            # Bid
            bid_flag = 0
            disc = 0.0
            if profile.rtype == "bidding" and m == profile.aord and cid <= profile.num_cycles:
                bid_flag = 1
                disc = bid_disc_by_cycle.get(cid, 0.0)

            rows.append({
                "mid": profile.mid,
                "gid": profile.gid,
                "cid": cid,
                "meeting_no": meeting_no,
                "mdate": mdate,
                "T_i": T_i,
                "rtype": profile.rtype,
                "rules": int(profile.rules),
                "aord": profile.aord,
                "adate": adate_i,
                "tdec": tdec,
                "ont": ont,
                "dlate": dlate,
                "san_flag": san_this,
                "bid": bid_flag,
                "disc": disc,
                "rep": int(profile.rep),
                "cent": int(profile.cent),
                "end_f": int(profile.endf),
                "end_s": int(profile.ends),
                "sure_str": profile.sure_str,
                "_n": n,
                "_p_ontime_raw": profile.p_ontime_raw,
                "_true_pd": profile.true_pd,
            })

    return rows


# ---------------------------------------------------------------------------
# 8. Score computer  (unchanged formulas from spec)
# ---------------------------------------------------------------------------

def _max_consec_late(ont_arr: np.ndarray) -> int:
    max_s = cur = 0
    for v in ont_arr:
        cur = cur + 1 if v == 0 else 0
        max_s = max(max_s, cur)
    return max_s


def compute_score(member_rows: pd.DataFrame, params: ScoreParams) -> dict:
    """Compute the 5-pillar credit score for one member from meeting data."""
    df = member_rows.sort_values("meeting_no").reset_index(drop=True)
    if df.empty:
        return {k: 0.0 for k in
                ["score","s_pdis","s_ordr","s_gov","s_liq","s_soc","otr","al","ls","rc","slip"]}

    n        = int(df["_n"].iloc[0])
    rtype    = df["rtype"].iloc[0]
    rules    = int(df["rules"].iloc[0])
    tdec     = int(df["tdec"].iloc[0])
    aord     = int(df["aord"].iloc[0])
    sure_str = df["sure_str"].iloc[0]
    rep      = int(df["rep"].iloc[0])
    cent     = int(df["cent"].iloc[0])
    end_f    = int(df["end_f"].iloc[0])
    end_s    = int(df["end_s"].iloc[0])

    K = len(df)
    a = params.a
    ms = np.arange(1, K + 1)
    w = a ** (K - ms)
    W = float(w.sum()) or 1.0

    ont_arr   = df["ont"].values.astype(float)
    dlate_arr = df["dlate"].values.astype(float)

    otr = float((w * ont_arr).sum() / W)
    al  = float((w * np.maximum(0, dlate_arr)).sum() / W)
    ls  = _max_consec_late(ont_arr.astype(int))
    cur = ((dlate_arr > 0) & (dlate_arr < 7)).astype(float)
    rc  = float((w * cur).sum() / W)

    S_otr = 18.0 * _sigmoid(params.k_otr * (otr - params.c_otr))
    S_al  = 6.0  * np.exp(-params.a_al * _log1p_safe(al))
    S_ls  = 8.0  * np.exp(-params.a_ls * max(0, ls - 1))
    S_rc  = 7.0  * _sigmoid(params.k_rc * (rc - params.c_rc))
    s_pdis = min(35.0, (35.0 / 39.0) * (S_otr + S_al + S_ls + S_rc))

    # ordr
    ratio = aord / n
    if ratio <= 1/3:
        b_ord, bucket = 0.3, "early"
    elif ratio <= 2/3:
        b_ord, bucket = 0.6, "mid"
    else:
        b_ord, bucket = 1.0, "late"

    slip = 0
    adate = df["adate"].iloc[0]
    post_df = df[df["mdate"] > adate].sort_values("meeting_no")
    if not post_df.empty:
        streak = 0
        for v in post_df["ont"].values:
            if v == 0:
                streak += 1
                if streak >= 2:
                    slip = 1
                    break
            else:
                streak = 0

    s_ordr = 15.0 * b_ord * (1.0 - params.a_slip * slip) if tdec else 15.0 * b_ord

    # gov
    san6 = float(df["san_flag"].iloc[-5:].sum())   # last 6 meetings
    S_rules = 5.0 * _sigmoid(params.k_rules * (rules - 0.5))
    S_san   = 6.0 * np.exp(-params.a_san * san6)
    S_sure  = {"none": 0.0, "weak": 3.0, "strong": 6.0}.get(sure_str, 0.0)
    s_gov   = (20.0 / 17.0) * (S_rules + S_san + S_sure)

    # liq
    s_liq = 0.0
    if rtype == "bidding":
        bids = df[df["bid"] == 1]["disc"].values.astype(float)
        if len(bids) >= 1:
            q_rank = float(df["_disc_q_rank"].iloc[0]) if "_disc_q_rank" in df.columns else 0.5
            last6  = bids[-6:] if len(bids) >= 6 else bids
            iqr    = float(np.percentile(last6, 75) - np.percentile(last6, 25)) if len(last6) > 1 else 0.0
            S_lvl  = 6.0 * (1.0 - _sigmoid(params.k_q * (q_rank - params.q0)))
            S_vol  = 6.0 * np.exp(-params.a_v * _log1p_safe(iqr / params.v_ref))
            s_liq  = (15.0 / 12.0) * (S_lvl + S_vol)

    # soc
    s_soc = params.w_rep * rep + params.w_cent * cent + params.w_endf * end_f + params.w_ends * end_s

    score = s_pdis + s_ordr + s_gov + s_liq + s_soc

    return {
        "score":  round(score, 3),
        "s_pdis": round(s_pdis, 3),
        "s_ordr": round(s_ordr, 3),
        "s_gov":  round(s_gov, 3),
        "s_liq":  round(s_liq, 3),
        "s_soc":  round(s_soc, 3),
        "otr": round(otr, 4), "al": round(al, 4),
        "ls": ls, "rc": round(rc, 4), "slip": slip,
        "bucket": bucket, "san6": san6, "b_ord": b_ord, "tdec": tdec, "K": K,
    }


# ---------------------------------------------------------------------------
# 9. Validation metrics
# ---------------------------------------------------------------------------

def _compute_validation(member_df: pd.DataFrame) -> Dict[str, Any]:
    scores   = member_df["score"].values.astype(float)
    true_pds = member_df["true_pd"].values.astype(float)

    # Spearman rank correlation between score and (1 - true_pd)
    # Higher score should predict lower PD
    rho = _spearman(scores, 1.0 - true_pds)

    # Score by true-PD decile
    decile_labels = pd.qcut(true_pds, q=5, labels=["PD_Q1","PD_Q2","PD_Q3","PD_Q4","PD_Q5"])
    by_decile = (
        member_df.assign(_pd_decile=decile_labels)
        .groupby("_pd_decile")["score"]
        .agg(["mean", "std", "count"])
        .round(2)
    )

    # KS-style: mean score for bottom-PD half vs top-PD half
    median_pd = float(np.median(true_pds))
    hi_pd = scores[true_pds >= median_pd]
    lo_pd = scores[true_pds <  median_pd]
    separation = float(lo_pd.mean() - hi_pd.mean()) if len(hi_pd) and len(lo_pd) else float("nan")

    return {
        "spearman_rho": round(rho, 4),
        "score_separation": round(separation, 2),   # mean(score | low PD) - mean(score | high PD)
        "score_mean": round(float(scores.mean()), 2),
        "score_std":  round(float(scores.std()), 2),
        "true_pd_mean": round(float(true_pds.mean()), 4),
        "true_pd_std":  round(float(true_pds.std()), 4),
        "n_members": len(member_df),
        "score_by_pd_quintile": by_decile,
    }


# ---------------------------------------------------------------------------
# 10. Simulation result
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    member_df:  pd.DataFrame          # one row per member
    meeting_df: pd.DataFrame          # full meeting-level data
    validation: Dict[str, Any]        # discrimination metrics


# ---------------------------------------------------------------------------
# 11. Population generator  (main entry point)
# ---------------------------------------------------------------------------

def generate_population(
    pop: PopulationParams,
    macro: MacroEnvironment,
    params: ScoreParams,
    seed: int = 42,
    K_min: int = 6,
) -> SimulationResult:
    """
    Generate `pop.n_groups` random groups, draw member profiles, apply macro
    environment, generate meeting data, compute scores, and compute validation
    metrics against true PD.

    Parameters
    ----------
    pop    : PopulationParams — controls group/member distributions
    macro  : MacroEnvironment — controls systemic shocks
    params : ScoreParams — the scoring hyperparameters to evaluate
    seed   : int — reproducibility
    K_min  : int — minimum meetings per cycle (spec: ≥ 6)
    """
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2024-01-01")

    all_profiles: List[MemberProfile] = []
    all_meetings: List[pd.DataFrame]  = []

    # ----- draw groups -------------------------------------------------------
    for g_idx in range(pop.n_groups):
        gid = f"G{g_idx+1:02d}"

        n          = int(rng.integers(pop.group_size_min, pop.group_size_max + 1))
        rtype      = "bidding" if rng.random() < pop.rtype_bidding_prob else "random"
        rules      = rng.random() < pop.rules_prob
        san_rate   = float(rng.uniform(pop.san_rate_min, pop.san_rate_max))
        num_cycles = int(rng.integers(pop.num_cycles_min, pop.num_cycles_max + 1))

        # Allocation order: random shuffle of 1..n (per cycle → use first cycle as aord)
        aord_list = list(rng.permutation(n) + 1)   # 1-based

        meetings_per_cycle = max(K_min, n)
        total_meetings = meetings_per_cycle * num_cycles

        # Pre-draw group-level meeting shocks (one per meeting per group)
        group_shocks = {
            m: float(rng.standard_normal())
            for m in range(1, total_meetings + 1)
        }

        # ----- draw members --------------------------------------------------
        a_otr, b_otr = pop._beta_params(pop.p_ontime_mean, pop.p_ontime_conc)
        a_slip, b_slip = pop._beta_params(pop.post_slip_mean, pop.post_slip_conc)
        a_bid, b_bid   = pop._beta_params(pop.bid_agg_mean, pop.bid_agg_conc)

        group_profiles: List[MemberProfile] = []
        for m_idx in range(n):
            mid = f"{gid}_M{m_idx+1:02d}"
            aord = int(aord_list[m_idx])

            p_ontime_raw  = float(np.clip(rng.beta(a_otr, b_otr), 0.01, 0.99))
            dlate_mu      = float(np.clip(rng.lognormal(pop.dlate_lognorm_mu,
                                                         pop.dlate_lognorm_sigma), 1.0, 60.0))
            post_slip     = float(np.clip(rng.beta(a_slip, b_slip), 0.0, 1.0))
            bid_agg       = float(np.clip(rng.beta(a_bid, b_bid), 0.0, 0.99))
            bid_vol       = float(rng.uniform(pop.bid_vol_min, pop.bid_vol_max))

            rep   = rng.random() < pop.p_rep
            cent  = rng.random() < pop.p_cent
            endf  = rng.random() < pop.p_endf
            ends  = rng.random() < pop.p_ends

            r_sure = rng.random()
            if r_sure < pop.p_sure_none:
                sure_str = "none"
            elif r_sure < pop.p_sure_none + pop.p_sure_weak:
                sure_str = "weak"
            else:
                sure_str = "strong"

            profile = MemberProfile(
                mid=mid, gid=gid, n=n, aord=aord, rtype=rtype,
                rules=rules, san_rate=san_rate, num_cycles=num_cycles,
                p_ontime_raw=p_ontime_raw, dlate_mu=dlate_mu,
                post_slip_prob=post_slip, bid_aggressiveness=bid_agg,
                bid_volatility=bid_vol,
                rep=rep, cent=cent, endf=endf, ends=ends, sure_str=sure_str,
            )
            profile.true_pd = _compute_true_pd(profile, macro.stress_level, rng)
            group_profiles.append(profile)

        all_profiles.extend(group_profiles)

        # ----- generate meeting rows -----------------------------------------
        g_rows: List[dict] = []
        for profile in group_profiles:
            g_rows.extend(
                _generate_member_meetings(
                    profile, macro, group_shocks, rng, base_date, K_min,
                )
            )

        if not g_rows:
            continue

        mtg = pd.DataFrame(g_rows)
        mtg = mtg.sort_values(["mid", "meeting_no"]).reset_index(drop=True)

        # san6 rolling window
        san6_vals: List[int] = []
        for _, grp in mtg.groupby("mid", sort=False):
            sv = grp["san_flag"].values
            san6_vals.extend(
                int(sv[max(0, i-5):i+1].sum()) for i in range(len(sv))
            )
        mtg["san6"] = san6_vals

        # Disc quantile rank within group (for liq pillar)
        if rtype == "bidding":
            bid_only = mtg[mtg["bid"] == 1]
            member_mean_disc = bid_only.groupby("mid")["disc"].mean()
            ranks = member_mean_disc.rank(pct=True) if len(member_mean_disc) > 1 \
                    else member_mean_disc * 0 + 0.5
            mtg["_disc_q_rank"] = mtg["mid"].map(ranks).fillna(0.5)
        else:
            mtg["_disc_q_rank"] = 0.5

        all_meetings.append(mtg)

    # ----- score all members -------------------------------------------------
    meeting_df = pd.concat(all_meetings, ignore_index=True) if all_meetings else pd.DataFrame()

    score_rows: List[dict] = []
    for mid_val, member_rows in meeting_df.groupby("mid", sort=False):
        sd = compute_score(member_rows, params)
        true_pd = float(member_rows["_true_pd"].iloc[0])
        p_raw   = float(member_rows["_p_ontime_raw"].iloc[0])
        gid_val = member_rows["gid"].iloc[0]
        rtype_v = member_rows["rtype"].iloc[0]
        sd.update({
            "mid": mid_val, "gid": gid_val, "rtype": rtype_v,
            "true_pd": true_pd, "p_ontime_raw": round(p_raw, 4),
        })
        score_rows.append(sd)

    member_df = pd.DataFrame(score_rows)
    front = ["mid", "gid", "rtype", "true_pd", "p_ontime_raw",
             "score", "s_pdis", "s_ordr", "s_gov", "s_liq", "s_soc"]
    rest  = [c for c in member_df.columns if c not in front]
    member_df = member_df[front + rest].reset_index(drop=True)

    validation = _compute_validation(member_df)

    return SimulationResult(
        member_df=member_df,
        meeting_df=meeting_df,
        validation=validation,
    )


# ---------------------------------------------------------------------------
# 12. Parameter sensitivity  (sweeps ScoreParams)
# ---------------------------------------------------------------------------

def run_sensitivity(
    base_params: ScoreParams,
    param_grid: Dict[str, list],
    pop: PopulationParams,
    macro: MacroEnvironment,
    seed: int = 42,
    max_combinations: int = 200,
) -> pd.DataFrame:
    """Sweep a ScoreParams grid and collect score + validation per combo.

    Returns one row per (param_combo × member).
    """
    valid_fields = {f.name for f in fields(ScoreParams)}
    for k in param_grid:
        if k not in valid_fields:
            raise ValueError(f"'{k}' is not a valid ScoreParams field.")

    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    if len(combos) > max_combinations:
        rng = np.random.default_rng(seed)
        idx    = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[i] for i in sorted(idx)]

    rows: List[pd.DataFrame] = []
    for combo in combos:
        overrides = dict(zip(keys, combo))
        p = dc_replace(base_params, **overrides)
        result = generate_population(pop, macro, p, seed=seed)
        df = result.member_df.copy()
        for k, v in overrides.items():
            df[f"param_{k}"] = v
        df["_rho"]        = result.validation["spearman_rho"]
        df["_separation"] = result.validation["score_separation"]
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 13. CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ROSCA population simulation")
    parser.add_argument("--n-groups",     type=int,   default=20)
    parser.add_argument("--stress",       type=float, default=0.0,
                        help="Macro stress level [0,1]")
    parser.add_argument("--corr",         type=float, default=0.20,
                        help="Within-group correlation [0,1]")
    parser.add_argument("--p-ontime",     type=float, default=0.80,
                        help="Population mean on-time rate")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    pop    = PopulationParams(n_groups=args.n_groups, p_ontime_mean=args.p_ontime)
    macro  = MacroEnvironment(stress_level=args.stress, within_group_corr=args.corr)
    params = ScoreParams()

    result = generate_population(pop, macro, params, seed=args.seed)
    df = result.member_df

    print(f"\n{'='*70}")
    print(f"Population simulation  |  {len(df)} members across {args.n_groups} groups")
    print(f"{'='*70}")
    print(df[["mid","gid","true_pd","p_ontime_raw","score",
              "s_pdis","s_ordr","s_gov","s_liq","s_soc"]].to_string(index=False))
    print(f"\n{'─'*70}")
    print("Validation metrics:")
    v = result.validation
    print(f"  Spearman ρ (score vs 1-PD) : {v['spearman_rho']:+.4f}")
    print(f"  Score separation (lo-PD - hi-PD): {v['score_separation']:+.2f} pts")
    print(f"  Mean score : {v['score_mean']:.2f}  ±  {v['score_std']:.2f}")
    print(f"  Mean true PD : {v['true_pd_mean']:.2%}  ±  {v['true_pd_std']:.2%}")
    print(f"\nScore by true-PD quintile:\n{v['score_by_pd_quintile'].to_string()}")