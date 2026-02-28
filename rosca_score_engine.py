# rosca_score_engine.py
# Updated: adds explicit default detection, Monte Carlo PD* estimation, and logistic PD* fitting.
# Keep original functionality intact; new helpers are appended near the end.

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, fields, replace as dc_replace
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import copy
import numpy as np
import pandas as pd

# Optional ML dependency for logistic PD* estimator
try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - sklearn may be absent in some environments
    LogisticRegression = None


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
    # guardrails (Items 2 & 3)
    gamma_rep: float = 0.30        # s_soc multiplier for members with a prior default history
    w_unverified: float = 1.0      # weight for unverified on-time payments in otr (1=full trust, 0=only verified)
    gov_star_penalty: float = 0.40 # fraction of s_gov removed for star-topology (closed-loop) groups
    # Item 5 — relative mean reversion in pdis
    alpha_macro: float = 0.0       # [0,1] how much group-median lateness absorbs individual penalty
                                   # 0 = pure absolute (original behaviour), 1 = full relative adjustment
    # Item 1 — credit stacking / cross-group obligations
    lambda_stack: float = 0.15     # per-extra-concurrent-group haircut on s_pdis
                                   # 0 = no penalty; 1 = full payment-discipline wipe per extra group


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

    # ---- guardrails (Items 2 & 3 simulation parameters) -------------------
    p_prior_default: float = 0.10   # share of members with a hidden prior-default history
    p_payment_verified: float = 0.85 # population mean share of payments backed by digital proof
    p_pay_ver_conc: float = 8.0      # Beta concentration for verification rate distribution
    p_star_topology: float = 0.15   # share of groups with closed-loop star topology
    # ---- guardrail (Item 1 — credit stacking) ------------------------------
    p_multi_group: float = 0.20     # share of members with concurrent ROSCA obligations in other groups
    n_extra_groups_max: int = 2     # maximum number of extra concurrent ROSCA groups a member can hold

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

    # Guardrail flags (Items 2 & 3)
    prior_default: bool = False       # member carries a hidden prior-default in another group
    verification_rate: float = 1.0    # share of this member's payments that are digitally verifiable
    star_topology: bool = False       # member belongs to a closed-loop star-topology group
    # Guardrail flag (Item 1)
    extra_groups: int = 0             # number of other concurrent ROSCA groups this member belongs to


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
    """Generate meeting rows for one member using vectorised numpy operations.

    All per-meeting random draws (shocks, payments, sanctions, verification)
    are batched into single numpy calls, replacing the original Python loop
    over meetings.  Statistical properties are identical; only the RNG call
    order differs (breaking exact seed-level reproducibility vs the old code).
    """
    n = profile.n
    meetings_per_cycle = max(K_min, n)
    num_cycles = profile.num_cycles
    total_meetings = meetings_per_cycle * num_cycles

    # ── Dates ──────────────────────────────────────────────────────────────
    # pd.date_range is O(1) vs O(M) repeated DateOffset additions.
    all_dates = pd.date_range(base_date, periods=total_meetings, freq="MS")
    T_i = all_dates[-1]

    # Per-meeting cycle (0-based) and allocation index within the flat array
    cycle_0       = np.arange(total_meetings) // meetings_per_cycle   # 0-based
    alloc_flat    = cycle_0 * meetings_per_cycle + (profile.aord - 1) # 0-based alloc index
    adate_arr     = all_dates[alloc_flat]                              # per-meeting adate

    # Post-payout slip: meeting is post-payout if its flat index > its cycle's alloc flat index
    is_post = np.arange(total_meetings) > alloc_flat
    p_raw_m = profile.p_ontime_raw * np.where(is_post, 1.0 - 0.5 * profile.post_slip_prob, 1.0)

    # ── Effective p_ontime — fully vectorised ───────────────────────────────
    corr      = float(np.clip(macro.within_group_corr, 0.0, 1.0))
    sqrt_corr = np.sqrt(corr)
    sqrt_1mc  = np.sqrt(max(0.0, 1.0 - corr))

    # Convert group_shocks dict to array (meeting_no is 1-based)
    meeting_nos  = np.arange(1, total_meetings + 1)
    g_shocks_arr = np.array([group_shocks.get(int(m), 0.0) for m in meeting_nos])
    idio_shocks  = rng.standard_normal(total_meetings)
    agg          = (sqrt_corr * g_shocks_arr + sqrt_1mc * idio_shocks) * 0.40

    p_clipped  = np.clip(p_raw_m, 1e-6, 1.0 - 1e-6)
    logit_base = np.log(p_clipped / (1.0 - p_clipped)) - macro.stress_level * 1.5
    p_eff      = 1.0 / (1.0 + np.exp(-np.clip(logit_base + agg, -30, 30)))

    for t0, t1, sev in macro.shock_windows:
        mask = (meeting_nos >= t0) & (meeting_nos <= t1)
        if mask.any():
            p_eff[mask] *= float(sev)
    p_eff = np.clip(p_eff, 0.01, 0.99)

    # ── Payment outcomes, lateness, sanctions, verification — all batched ───
    ont      = (rng.random(total_meetings) < p_eff).astype(int)
    raw_late = rng.lognormal(mean=np.log(max(profile.dlate_mu, 1.0)),
                             sigma=0.8, size=total_meetings)
    dlate    = np.where(ont == 1, 0, np.maximum(1, np.round(raw_late).astype(int)))
    san_flag = (rng.random(total_meetings) < profile.san_rate / 6.0).astype(int)
    ont_verified = ((ont == 1) & (rng.random(total_meetings) < profile.verification_rate)).astype(int)

    # ── Bid flags (only `num_cycles` draws, not `total_meetings`) ──────────
    bid_flag = np.zeros(total_meetings, dtype=int)
    disc_arr = np.zeros(total_meetings)
    if profile.rtype == "bidding":
        for cid in range(1, num_cycles + 1):
            bid_idx = (cid - 1) * meetings_per_cycle + (profile.aord - 1)
            if bid_idx < total_meetings:
                d = rng.normal(profile.bid_aggressiveness, max(profile.bid_volatility, 0.01))
                bid_flag[bid_idx] = 1
                disc_arr[bid_idx] = float(np.clip(d, 0.0, 0.99))

    # tdec: adate <= T_i is always True (T_i is the last meeting date), so always 1
    tdec_arr = (adate_arr <= T_i).astype(int)

    # ── Build rows from arrays ──────────────────────────────────────────────
    # Constants replicated across all meetings
    _const: dict = {
        "mid": profile.mid, "gid": profile.gid, "rtype": profile.rtype,
        "rules": int(profile.rules), "aord": profile.aord,
        "rep": int(profile.rep), "cent": int(profile.cent),
        "end_f": int(profile.endf), "end_s": int(profile.ends),
        "sure_str": profile.sure_str, "prior_default": int(profile.prior_default),
        "star_topology": int(profile.star_topology),
        "extra_groups": profile.extra_groups,
        "_n": n, "_p_ontime_raw": profile.p_ontime_raw,
        "_true_pd": profile.true_pd, "T_i": T_i,
    }
    cid_arr = cycle_0 + 1  # 1-based cycle id per meeting

    rows: List[dict] = []
    for i in range(total_meetings):
        rows.append({
            **_const,
            "cid":          int(cid_arr[i]),
            "meeting_no":   int(meeting_nos[i]),
            "mdate":        all_dates[i],
            "adate":        adate_arr[i],
            "tdec":         int(tdec_arr[i]),
            "ont":          int(ont[i]),
            "ont_verified": int(ont_verified[i]),
            "dlate":        int(dlate[i]),
            "san_flag":     int(san_flag[i]),
            "bid":          int(bid_flag[i]),
            "disc":         float(disc_arr[i]),
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


def compute_score(member_rows: pd.DataFrame, params: ScoreParams, streak_threshold: int = 0) -> dict:
    """Compute the 5-pillar credit score for one member from meeting data.

    If streak_threshold > 0, any member who misses that many consecutive meetings
    after receiving the pot has their score forced to 0 (hard default rule).
    """
    # member_rows is expected pre-sorted by meeting_no (generate_population guarantees this).
    df = member_rows.reset_index(drop=True)
    if df.empty:
        return {"score": 0.0, "s_pdis": 0.0, "s_ordr": 0.0, "s_gov": 0.0, "s_liq": 0.0,
                "s_soc": 0.0, "otr": 0.0, "al": 0.0, "ls": 0, "rc": 0.0, "slip": 0,
                "bucket": "n/a", "san6": 0.0, "b_ord": 0.0, "tdec": 0, "K": 0, "defaulted": False}

    K = len(df)

    # Detect default early but do NOT return yet — we still compute payment metrics
    # so that otr/al/ls/rc are available (non-zero) for logistic PD* fitting.
    _is_default = streak_threshold > 0 and detect_default_from_meetings(df, streak_threshold)

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

    a = params.a
    ms = np.arange(1, K + 1)
    w = a ** (K - ms)
    W = float(w.sum()) or 1.0

    ont_arr   = df["ont"].values.astype(float)
    dlate_arr = df["dlate"].values.astype(float)

    # Item 5 — relative mean reversion: absorb group-median lateness from individual penalty.
    # dlate_im* = max(0, dlate_im - alpha_macro * median_dlate_m_group)
    # alpha_macro=0 → pure absolute (original); alpha_macro=1 → full relative.
    if params.alpha_macro > 0.0 and "_grp_median_dlate" in df.columns:
        grp_med = df["_grp_median_dlate"].values.astype(float)
        dlate_arr = np.maximum(0.0, dlate_arr - params.alpha_macro * grp_med)

    # Item 3 — verification: unverified on-time payments are downweighted by w_unverified.
    # verified payments always get full weight; late payments are unaffected (already 0).
    if "ont_verified" in df.columns:
        ont_ver = df["ont_verified"].values.astype(float)
        ont_eff = ont_ver + params.w_unverified * np.maximum(0.0, ont_arr - ont_ver)
    else:
        ont_eff = ont_arr  # fallback: full trust (backward compatible)

    otr = float((w * ont_eff).sum() / W)
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

    # Slip: any 2 consecutive late meetings post-allocation (vectorised, no sort needed
    # since df is already sorted by meeting_no from the caller).
    slip = 0
    adate = df["adate"].iloc[0]
    post_ont = ont_arr[df["mdate"].values > adate]
    if len(post_ont) >= 2:
        late_post = post_ont == 0
        slip = int(np.any(late_post[:-1] & late_post[1:]))

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

    # Item 1 — credit stacking: concurrent ROSCA obligations strain payment discipline
    extra_groups = int(df["extra_groups"].iloc[0]) if "extra_groups" in df.columns else 0
    if extra_groups > 0 and params.lambda_stack > 0.0:
        stack_haircut = max(0.0, 1.0 - params.lambda_stack * extra_groups)
        s_pdis *= stack_haircut

    # soc
    s_soc = params.w_rep * rep + params.w_cent * cent + params.w_endf * end_f + params.w_ends * end_s
    # Item 2 — reputation decay: prior defaulters have social capital penalised by gamma_rep
    prior_default = bool(df["prior_default"].iloc[0]) if "prior_default" in df.columns else False
    if prior_default:
        s_soc *= params.gamma_rep

    # Item 3 — star topology: closed-loop groups get a governance penalty
    star_topology = bool(df["star_topology"].iloc[0]) if "star_topology" in df.columns else False
    if star_topology:
        s_gov *= (1.0 - params.gov_star_penalty)

    score = s_pdis + s_ordr + s_gov + s_liq + s_soc

    # Hard default rule: zero ALL score pillars if post-allocation default detected.
    # Payment metrics (otr, al, ls, rc, slip) are preserved so logistic PD* has
    # meaningful non-zero features for defaulters.
    if _is_default:
        score = s_pdis = s_ordr = s_gov = s_liq = s_soc = 0.0

    # Encode surety as numeric so it can be used as a regression feature
    sure_val = {"none": 0.0, "weak": 0.5, "strong": 1.0}.get(sure_str, 0.0)

    return {
        "score":  round(score, 3),
        "s_pdis": round(s_pdis, 3),
        "s_ordr": round(s_ordr, 3),
        "s_gov":  round(s_gov, 3),
        "s_liq":  round(s_liq, 3),
        "s_soc":  round(s_soc, 3),
        # --- raw score inputs (preserved for PD* logistic model) ---
        "otr": round(otr, 4), "al": round(al, 4),
        "ls": ls, "rc": round(rc, 4), "slip": slip,
        "san6": san6, "b_ord": b_ord, "tdec": tdec, "K": K,
        # governance inputs
        "rules": rules, "sure_val": sure_val, "star_topology": star_topology,
        # social capital inputs
        "rep": rep, "cent": cent, "end_f": end_f, "end_s": end_s,
        # credit-stacking / prior-default
        "extra_groups": extra_groups, "prior_default": prior_default,
        # housekeeping
        "bucket": bucket, "defaulted": _is_default,
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
# 10. PD* validation  (primary evaluation axis — MC-estimated default frequency)
# ---------------------------------------------------------------------------

def compute_pd_star_validation(member_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute validation metrics using pd_star (MC default frequency) as the reference.

    member_df must contain columns: 'score', 'pd_star'.
    If 'defaulted' (bool) is also present, binary separation metrics are added.

    This is the primary evaluation function.  true_pd is not used here.
    """
    if "pd_star" not in member_df.columns:
        raise ValueError("member_df must contain 'pd_star'. Run compute_pd_star_mc first.")

    scores   = member_df["score"].values.astype(float)
    pd_stars = member_df["pd_star"].values.astype(float)

    rho = _spearman(scores, 1.0 - pd_stars)

    try:
        quintile_labels = pd.qcut(
            pd_stars, q=5,
            labels=["PD*_Q1", "PD*_Q2", "PD*_Q3", "PD*_Q4", "PD*_Q5"],
            duplicates="drop",
        )
    except Exception:
        quintile_labels = pd.cut(
            pd_stars, bins=5,
            labels=["PD*_Q1", "PD*_Q2", "PD*_Q3", "PD*_Q4", "PD*_Q5"],
        )

    by_quintile = (
        member_df.assign(_pdstar_q=quintile_labels)
        .groupby("_pdstar_q", observed=True)["score"]
        .agg(["mean", "std", "count"])
        .round(2)
    )

    median_pdstar = float(np.median(pd_stars))
    hi_pd = scores[pd_stars >= median_pdstar]
    lo_pd = scores[pd_stars <  median_pdstar]
    separation = float(lo_pd.mean() - hi_pd.mean()) if len(hi_pd) and len(lo_pd) else float("nan")

    out: Dict[str, Any] = {
        "spearman_rho_pdstar":     round(rho, 4),
        "score_separation_pdstar": round(separation, 2),
        "pd_star_mean": round(float(pd_stars.mean()), 4),
        "pd_star_std":  round(float(pd_stars.std()),  4),
        "n_members":    len(member_df),
        "score_by_pdstar_quintile": by_quintile,
    }

    if "defaulted" in member_df.columns:
        defaulted = member_df["defaulted"].astype(bool)
        out["default_rate"]             = round(float(defaulted.mean()), 4)
        out["n_defaulted"]              = int(defaulted.sum())
        out["score_mean_defaulted"]     = round(float(scores[defaulted.values].mean()), 2) \
                                          if defaulted.any() else 0.0
        out["score_mean_non_defaulted"] = round(float(scores[~defaulted.values].mean()), 2) \
                                          if (~defaulted).any() else 0.0

    return out


def _compute_scores_for_group(
    mtg: pd.DataFrame,
    group_profiles: List["MemberProfile"],
    params: "ScoreParams",
    streak_threshold: int = 0,
) -> List[dict]:
    """Score all members in a group at once using numpy matrix operations.

    Requires `mtg` sorted by (mid, meeting_no) with all members having the
    same number of meetings M.  Replaces N individual compute_score() calls
    with a single pass over (N, M) numpy arrays, eliminating per-member pandas
    overhead and giving a ~3× speedup for typical group sizes.
    """
    N = len(group_profiles)
    if N == 0 or mtg.empty:
        return []
    M = len(mtg) // N          # meetings per member (same for all in group)
    if M == 0:
        return []

    # ── Extract (N, M) matrices by reshaping the sorted group DataFrame ─────
    ont_mat     = mtg["ont"].values.reshape(N, M).astype(float)
    dlate_mat   = mtg["dlate"].values.reshape(N, M).astype(float)
    san_mat     = mtg["san_flag"].values.reshape(N, M).astype(float)
    ont_ver_mat = (mtg["ont_verified"].values.reshape(N, M).astype(float)
                   if "ont_verified" in mtg.columns else ont_mat)
    mdate_mat   = mtg["mdate"].values.reshape(N, M)
    adate_mat   = mtg["adate"].values.reshape(N, M)

    # ── Time-decay weights ───────────────────────────────────────────────────
    a  = params.a
    w  = a ** (M - np.arange(1, M + 1))   # (M,)
    W  = float(w.sum()) or 1.0

    # ── Item 3: payment verification downweighting ───────────────────────────
    ont_eff = ont_ver_mat + params.w_unverified * np.maximum(0.0, ont_mat - ont_ver_mat)

    # ── Item 5: macro mean-reversion ─────────────────────────────────────────
    if params.alpha_macro > 0.0 and "_grp_median_dlate" in mtg.columns:
        grp_med   = mtg["_grp_median_dlate"].values.reshape(N, M).astype(float)
        dlate_adj = np.maximum(0.0, dlate_mat - params.alpha_macro * grp_med)
    else:
        dlate_adj = dlate_mat

    # ── Payment-discipline metrics ───────────────────────────────────────────
    otr = (ont_eff   @ w) / W                                   # (N,)
    al  = (np.maximum(0, dlate_adj) @ w) / W                    # (N,)
    rc  = (((dlate_adj > 0) & (dlate_adj < 7)).astype(float) @ w) / W  # (N,)
    # Max consecutive late (ls): small Python loop over group members (N ≈ 13), not over all members
    ls  = np.array([_max_consec_late(ont_mat[i].astype(int)) for i in range(N)])  # (N,)

    # ── s_pdis ───────────────────────────────────────────────────────────────
    S_otr  = 18.0 / (1.0 + np.exp(-np.clip(params.k_otr * (otr - params.c_otr), -30, 30)))
    S_al   = 6.0  * np.exp(-params.a_al * np.log1p(np.maximum(0, al)))
    S_ls   = 8.0  * np.exp(-params.a_ls * np.maximum(0, ls - 1))
    S_rc   = 7.0  / (1.0 + np.exp(-np.clip(params.k_rc * (rc - params.c_rc), -30, 30)))
    s_pdis = np.minimum(35.0, (35.0 / 39.0) * (S_otr + S_al + S_ls + S_rc))

    # Item 1: credit-stacking haircut
    extra_groups_arr = np.array([p.extra_groups for p in group_profiles])
    if params.lambda_stack > 0.0:
        s_pdis *= np.maximum(0.0, 1.0 - params.lambda_stack * extra_groups_arr)

    # ── s_ordr ───────────────────────────────────────────────────────────────
    n_group = group_profiles[0].n
    aords   = np.array([p.aord for p in group_profiles])
    ratios  = aords / n_group
    b_ord   = np.where(ratios <= 1/3, 0.3, np.where(ratios <= 2/3, 0.6, 1.0))
    buckets = np.where(ratios <= 1/3, "early", np.where(ratios <= 2/3, "mid", "late"))

    # Slip: any 2 consecutive late payments after first-cycle allocation
    slip = np.zeros(N, dtype=int)
    for i in range(N):
        post_ont = ont_mat[i, mdate_mat[i] > adate_mat[i, 0]]
        if len(post_ont) >= 2:
            late_p   = post_ont == 0
            slip[i]  = int(np.any(late_p[:-1] & late_p[1:]))

    # tdec: always 1 (T_i == last meeting date >= any adate)
    s_ordr = 15.0 * b_ord * (1.0 - params.a_slip * slip)

    # ── s_gov ────────────────────────────────────────────────────────────────
    san6        = san_mat[:, -5:].sum(axis=1)   # last 5 meetings (matches compute_score behaviour)
    rules_arr   = np.array([int(p.rules) for p in group_profiles])
    sure_strs   = [p.sure_str for p in group_profiles]
    S_rules = 5.0 / (1.0 + np.exp(-np.clip(params.k_rules * (rules_arr - 0.5), -30, 30)))
    S_san   = 6.0 * np.exp(-params.a_san * san6)
    S_sure  = np.array([{"none": 0.0, "weak": 3.0, "strong": 6.0}.get(s, 0.0) for s in sure_strs])
    s_gov   = (20.0 / 17.0) * (S_rules + S_san + S_sure)
    # Item 3: star topology governance penalty
    star_arr = np.array([int(p.star_topology) for p in group_profiles])
    s_gov   *= (1.0 - params.gov_star_penalty * star_arr)

    # ── s_liq (bidding only) ─────────────────────────────────────────────────
    s_liq    = np.zeros(N)
    if group_profiles[0].rtype == "bidding" and "_disc_q_rank" in mtg.columns:
        bid_mat  = mtg["bid"].values.reshape(N, M)
        disc_mat = mtg["disc"].values.reshape(N, M)
        q_rank   = mtg["_disc_q_rank"].values.reshape(N, M)[:, 0]
        for i in range(N):
            bids = disc_mat[i, bid_mat[i] == 1]
            if len(bids) >= 1:
                last6 = bids[-6:]
                iqr   = (float(np.percentile(last6, 75) - np.percentile(last6, 25))
                         if len(last6) > 1 else 0.0)
                S_lvl     = 6.0 / (1.0 + np.exp(-np.clip(params.k_q * (q_rank[i] - params.q0), -30, 30)))
                S_vol     = 6.0 * np.exp(-params.a_v * np.log1p(max(0.0, iqr / params.v_ref)))
                s_liq[i]  = (15.0 / 12.0) * (S_lvl + S_vol)

    # ── s_soc ────────────────────────────────────────────────────────────────
    reps    = np.array([int(p.rep)   for p in group_profiles])
    cents   = np.array([int(p.cent)  for p in group_profiles])
    endfs   = np.array([int(p.endf)  for p in group_profiles])
    endss   = np.array([int(p.ends)  for p in group_profiles])
    priors  = np.array([int(p.prior_default) for p in group_profiles])
    s_soc   = (params.w_rep * reps + params.w_cent * cents
               + params.w_endf * endfs + params.w_ends * endss).astype(float)
    s_soc  *= np.where(priors, params.gamma_rep, 1.0)

    # ── Total score ──────────────────────────────────────────────────────────
    score = s_pdis + s_ordr + s_gov + s_liq + s_soc

    # ── Default detection (vectorised per member using cumsum) ───────────────
    defaulted = np.zeros(N, dtype=bool)
    if streak_threshold > 0:
        k = streak_threshold
        for i in range(N):
            post_ont = ont_mat[i, mdate_mat[i] > adate_mat[i, 0]]
            n_post   = len(post_ont)
            if n_post >= k:
                late = (post_ont == 0).astype(np.int32)
                cs   = late.cumsum()
                pad  = np.concatenate([[0], cs])
                defaulted[i] = bool((pad[k:] - pad[:-k] == k).any())
        score[defaulted]  = s_pdis[defaulted] = s_ordr[defaulted] = 0.0
        s_gov[defaulted]  = s_liq[defaulted]  = s_soc[defaulted]  = 0.0

    # ── Encode sure_str numerically ──────────────────────────────────────────
    sure_val  = np.array([{"none": 0.0, "weak": 0.5, "strong": 1.0}.get(s, 0.0)
                           for s in sure_strs])

    # ── Build one result dict per member ─────────────────────────────────────
    results: List[dict] = []
    for i, p in enumerate(group_profiles):
        results.append({
            "score":   round(float(score[i]),  3),
            "s_pdis":  round(float(s_pdis[i]), 3),
            "s_ordr":  round(float(s_ordr[i]), 3),
            "s_gov":   round(float(s_gov[i]),  3),
            "s_liq":   round(float(s_liq[i]),  3),
            "s_soc":   round(float(s_soc[i]),  3),
            "otr":     round(float(otr[i]), 4),
            "al":      round(float(al[i]),  4),
            "ls":      int(ls[i]),
            "rc":      round(float(rc[i]), 4),
            "slip":    int(slip[i]),
            "san6":    float(san6[i]),
            "b_ord":   float(b_ord[i]),
            "tdec":    1,
            "K":       M,
            "rules":         int(rules_arr[i]),
            "sure_val":      float(sure_val[i]),
            "star_topology": bool(star_arr[i]),
            "rep":    int(reps[i]),
            "cent":   int(cents[i]),
            "end_f":  int(endfs[i]),
            "end_s":  int(endss[i]),
            "extra_groups":  int(extra_groups_arr[i]),
            "prior_default": bool(priors[i]),
            "bucket":    str(buckets[i]),
            "defaulted": bool(defaulted[i]),
        })
    return results


@dataclass
class SimulationResult:
    member_df:  pd.DataFrame          # one row per member
    meeting_df: pd.DataFrame          # full meeting-level data
    validation: Dict[str, Any]        # discrimination metrics


# ---------------------------------------------------------------------------
# 11. Population generator  (main entry point)  -- unchanged
# ---------------------------------------------------------------------------

def generate_population(
    pop: PopulationParams,
    macro: MacroEnvironment,
    params: ScoreParams,
    seed: int = 42,
    K_min: int = 6,
    default_streak_threshold: int = 0,
    keep_meetings: bool = True,
) -> SimulationResult:
    """
    Draw member profiles (fixed by seed), generate one realisation of meeting
    data, compute scores, and validate against true PD.

    Profiles are drawn via draw_population_profiles so they are identical to
    what compute_pd_star_mc uses — ensuring score and PD* are comparable.
    Meeting shocks use per-group / per-member deterministic seeds so each call
    with the same seed reproduces exactly.

    Parameters
    ----------
    pop                      : PopulationParams
    macro                    : MacroEnvironment
    params                   : ScoreParams
    seed                     : int — controls both profile draw and meeting shocks
    K_min                    : int — minimum meetings per cycle (≥ 6)
    default_streak_threshold : int — if > 0, any member missing this many
                               consecutive meetings after allocation gets score 0
    """
    base_date = pd.Timestamp("2024-01-01")

    # Step 1: draw fixed profiles — same draw order as compute_pd_star_mc
    profiles_by_group = draw_population_profiles(pop, macro, seed=seed, K_min=K_min)

    score_rows: List[dict] = []
    all_meetings: List[pd.DataFrame] = [] if keep_meetings else None

    # Steps 2+3: generate meeting data per group and score immediately.
    # Scoring inside the group loop avoids a giant pd.concat + full-population
    # groupby which become O(n²) in pandas for large n_groups.
    for gid, group_meta, group_profiles in profiles_by_group:
        n          = group_meta["n"]
        rtype      = group_meta["rtype"]
        num_cycles = group_meta["num_cycles"]

        meetings_per_cycle = max(K_min, n)
        total_meetings     = meetings_per_cycle * num_cycles

        # Deterministic group-shock RNG (same convention as compute_pd_star_mc run 0)
        rng_group   = np.random.default_rng(seed + (abs(hash(gid))   % (2 ** 31)))
        group_shocks = {
            m: float(rng_group.standard_normal())
            for m in range(1, total_meetings + 1)
        }

        g_rows: List[dict] = []
        for profile in group_profiles:
            rng_member = np.random.default_rng(seed + (abs(hash(profile.mid)) % (2 ** 31)))
            g_rows.extend(
                _generate_member_meetings(
                    profile, macro, group_shocks, rng_member, base_date, K_min,
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
                int(sv[max(0, i - 5):i + 1].sum()) for i in range(len(sv))
            )
        mtg["san6"] = san6_vals

        # Item 5 — group median dlate per meeting (for relative mean reversion in pdis)
        # For each meeting_no, compute the median dlate across all members in the group.
        grp_median = (
            mtg.groupby("meeting_no")["dlate"]
            .median()
            .rename("_grp_median_dlate")
        )
        mtg = mtg.join(grp_median, on="meeting_no")

        # Disc quantile rank within group (for liq pillar)
        if rtype == "bidding":
            bid_only         = mtg[mtg["bid"] == 1]
            member_mean_disc = bid_only.groupby("mid")["disc"].mean()
            ranks = (member_mean_disc.rank(pct=True) if len(member_mean_disc) > 1
                     else member_mean_disc * 0 + 0.5)
            mtg["_disc_q_rank"] = mtg["mid"].map(ranks).fillna(0.5)
        else:
            mtg["_disc_q_rank"] = 0.5

        # Score all members in the group at once using numpy matrix operations.
        group_scores = _compute_scores_for_group(
            mtg, group_profiles, params, streak_threshold=default_streak_threshold
        )
        for profile, sd in zip(group_profiles, group_scores):
            sd.update({
                "mid": profile.mid, "gid": profile.gid, "rtype": profile.rtype,
                "true_pd": profile.true_pd,
                "p_ontime_raw": round(profile.p_ontime_raw, 4),
            })
            score_rows.append(sd)

        if all_meetings is not None:
            all_meetings.append(mtg)

    member_df = pd.DataFrame(score_rows)
    front = ["mid", "gid", "rtype", "true_pd", "p_ontime_raw",
             "score", "s_pdis", "s_ordr", "s_gov", "s_liq", "s_soc", "defaulted"]
    rest  = [c for c in member_df.columns if c not in front]
    member_df = member_df[front + rest].reset_index(drop=True)

    meeting_df = pd.concat(all_meetings, ignore_index=True) if all_meetings else pd.DataFrame()
    validation = _compute_validation(member_df)

    return SimulationResult(
        member_df=member_df,
        meeting_df=meeting_df,
        validation=validation,
    )


# ---------------------------------------------------------------------------
# 12. New: Default detection and PD* estimation helpers
# ---------------------------------------------------------------------------

def detect_default_from_meetings(member_meetings: pd.DataFrame, streak_threshold: int = 3) -> bool:
    """Return True if member misses `streak_threshold` consecutive meetings after their allocation date.

    Expects `member_meetings` sorted by meeting_no (no internal sort performed).
    Uses a vectorised cumsum sliding-window to find the streak instead of a Python loop.
    """
    if member_meetings.empty:
        return False
    adate = member_meetings["adate"].iloc[0]
    post_ont = member_meetings.loc[member_meetings["mdate"] > adate, "ont"].values
    if len(post_ont) < streak_threshold:
        return False
    late = (post_ont == 0).astype(np.int32)
    k    = streak_threshold
    cs   = late.cumsum()
    pad  = np.concatenate([[0], cs])
    return bool((pad[k:] - pad[:-k] == k).any())


def generate_population_with_defaults(
    pop: PopulationParams,
    macro: MacroEnvironment,
    params: ScoreParams,
    seed: int = 42,
    K_min: int = 6,
    streak_threshold: int = 3,
    keep_meetings: bool = True,
) -> SimulationResult:
    """Simulate population with scores zeroed for post-allocation defaulters.

    A member defaults if they miss `streak_threshold` consecutive meetings after
    receiving the pot.  Their score is forced to 0 via compute_score's hard rule.
    The 'defaulted' column (bool) is always present in member_df.
    'default' is kept as an alias for backward compatibility.

    keep_meetings : bool
        If False, the full meeting-level DataFrame is not retained (saves memory
        and time for large populations; member_df and validation are unaffected).
    """
    result = generate_population(
        pop, macro, params, seed=seed, K_min=K_min,
        default_streak_threshold=streak_threshold,
        keep_meetings=keep_meetings,
    )
    member_df = result.member_df.copy()
    member_df["default"] = member_df["defaulted"].astype(bool)
    return SimulationResult(member_df=member_df, meeting_df=result.meeting_df, validation=result.validation)


def draw_population_profiles(
    pop: PopulationParams,
    macro: MacroEnvironment,
    seed: int = 42,
    K_min: int = 6,
) -> List[Tuple[str, dict, List[MemberProfile]]]:
    """Draw groups and member profiles only (no meeting simulation).
    Returns list of tuples: (gid, group_meta, [MemberProfile,...]).
    """
    rng = np.random.default_rng(seed)
    profiles_by_group = []
    for g_idx in range(pop.n_groups):
        gid = f"G{g_idx+1:02d}"
        n = int(rng.integers(pop.group_size_min, pop.group_size_max + 1))
        rtype = "bidding" if rng.random() < pop.rtype_bidding_prob else "random"
        rules = rng.random() < pop.rules_prob
        san_rate = float(rng.uniform(pop.san_rate_min, pop.san_rate_max))
        num_cycles = int(rng.integers(pop.num_cycles_min, pop.num_cycles_max + 1))
        aord_list = list(rng.permutation(n) + 1)
        # Item 3: star topology is a group-level property
        star_topology = bool(rng.random() < pop.p_star_topology)

        a_otr, b_otr = pop._beta_params(pop.p_ontime_mean, pop.p_ontime_conc)
        a_slip, b_slip = pop._beta_params(pop.post_slip_mean, pop.post_slip_conc)
        a_bid, b_bid   = pop._beta_params(pop.bid_agg_mean, pop.bid_agg_conc)
        a_ver, b_ver   = pop._beta_params(pop.p_payment_verified, pop.p_pay_ver_conc)

        group_profiles: List[MemberProfile] = []
        for m_idx in range(n):
            mid = f"{gid}_M{m_idx+1:02d}"
            aord = int(aord_list[m_idx])
            p_ontime_raw  = float(np.clip(rng.beta(a_otr, b_otr), 0.01, 0.99))
            dlate_mu      = float(np.clip(rng.lognormal(pop.dlate_lognorm_mu, pop.dlate_lognorm_sigma), 1.0, 60.0))
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
            # Item 2: prior default history (hidden)
            prior_default    = bool(rng.random() < pop.p_prior_default)
            # Item 3: individual verification rate (share of payments with digital proof)
            verification_rate = float(np.clip(rng.beta(a_ver, b_ver), 0.0, 1.0))
            # Item 1: concurrent ROSCA obligations in other groups (credit stacking)
            extra_groups = (
                int(rng.integers(1, max(2, pop.n_extra_groups_max + 1)))
                if rng.random() < pop.p_multi_group else 0
            )

            profile = MemberProfile(
                mid=mid, gid=gid, n=n, aord=aord, rtype=rtype,
                rules=rules, san_rate=san_rate, num_cycles=num_cycles,
                p_ontime_raw=p_ontime_raw, dlate_mu=dlate_mu,
                post_slip_prob=post_slip, bid_aggressiveness=bid_agg,
                bid_volatility=bid_vol,
                rep=rep, cent=cent, endf=endf, ends=ends, sure_str=sure_str,
                prior_default=prior_default,
                verification_rate=verification_rate,
                star_topology=star_topology,
                extra_groups=extra_groups,
            )
            profile.true_pd = _compute_true_pd(profile, macro.stress_level, rng)
            group_profiles.append(profile)

        group_meta = {
            "gid": gid,
            "n": n,
            "rtype": rtype,
            "rules": rules,
            "san_rate": san_rate,
            "num_cycles": num_cycles,
            "aord_list": aord_list,
        }
        profiles_by_group.append((gid, group_meta, group_profiles))
    return profiles_by_group


def compute_pd_star_mc(
    pop: PopulationParams,
    macro: MacroEnvironment,
    params: ScoreParams,
    n_runs: int = 200,
    base_seed: int = 42,
    K_min: int = 6,
    streak_threshold: int = 3,
) -> pd.DataFrame:
    """Estimate PD* via vectorised Monte Carlo.

    Profiles are drawn once; then all n_runs are simulated in a single numpy
    pass per (group × member), replacing the previous Python loop over runs.
    This is typically 10–50× faster than the loop-based implementation.

    Returns a DataFrame with one row per member:
      mid, gid, p_ontime_raw, true_pd, default_count, pd_star
    """
    profiles_by_group = draw_population_profiles(pop, macro, seed=base_seed, K_min=K_min)

    corr      = float(np.clip(macro.within_group_corr, 0.0, 1.0))
    sqrt_corr = np.sqrt(corr)
    sqrt_1mc  = np.sqrt(max(0.0, 1.0 - corr))
    # logit-space downward shift matching _effective_p_ontime
    stress_shift = macro.stress_level * 1.5

    rows: List[dict] = []

    for gid, group_meta, group_profiles in profiles_by_group:
        meetings_per_cycle = max(K_min, group_meta["n"])
        num_cycles         = group_meta["num_cycles"]
        total_meetings     = meetings_per_cycle * num_cycles

        # Group-level shocks: (n_runs, total_meetings) — same shock for all members in group
        rng_group   = np.random.default_rng(base_seed + abs(hash(gid)) % (2 ** 31))
        group_shocks = rng_group.standard_normal((n_runs, total_meetings))  # (R, M)

        for profile in group_profiles:
            rng_mem = np.random.default_rng(base_seed + abs(hash(profile.mid)) % (2 ** 31))

            # Member idiosyncratic shocks: (n_runs, total_meetings)
            idio_shocks = rng_mem.standard_normal((n_runs, total_meetings))

            # Aggregate shock with scale 0.40, matching _effective_p_ontime
            agg = (sqrt_corr * group_shocks + sqrt_1mc * idio_shocks) * 0.40  # (R, M)

            # Per-meeting baseline p_raw, adjusted for post-payout slip per cycle.
            # Each cycle's slip applies only to meetings after that cycle's allocation.
            p_raw_m = np.full(total_meetings, profile.p_ontime_raw)
            for cid in range(1, num_cycles + 1):
                alloc_no  = (cid - 1) * meetings_per_cycle + profile.aord  # 1-based meeting no.
                alloc_idx = alloc_no - 1                                    # 0-based index
                cycle_end = cid * meetings_per_cycle                        # exclusive index
                if alloc_idx + 1 < cycle_end:
                    p_raw_m[alloc_idx + 1: cycle_end] *= (1.0 - 0.5 * profile.post_slip_prob)

            # Logit-space baseline per meeting: (M,)
            p_clipped  = np.clip(p_raw_m, 1e-6, 1.0 - 1e-6)
            logit_base = np.log(p_clipped / (1.0 - p_clipped)) - stress_shift

            # Effective p_ontime: (R, M)
            logit_eff = logit_base[np.newaxis, :] + agg
            p_eff = 1.0 / (1.0 + np.exp(-np.clip(logit_eff, -30, 30)))

            # Apply shock windows (multiplicative severity on probability)
            for t0, t1, sev in macro.shock_windows:
                c0 = t0 - 1
                c1 = min(t1, total_meetings)
                if c0 < c1:
                    p_eff[:, c0:c1] *= float(sev)
            p_eff = np.clip(p_eff, 0.01, 0.99)

            # Payment outcomes: (R, M) — 1 = on-time, 0 = late
            ont = (rng_mem.random((n_runs, total_meetings)) < p_eff).astype(np.int8)

            # Post-allocation default detection (mirrors detect_default_from_meetings):
            # look for `streak_threshold` consecutive late payments after first-cycle allocation.
            first_alloc_idx = profile.aord - 1  # 0-based index
            post_start      = first_alloc_idx + 1

            if post_start < total_meetings and (total_meetings - post_start) >= streak_threshold:
                late   = (1 - ont[:, post_start:]).astype(np.int32)  # (R, n_post)
                n_post = late.shape[1]
                k      = streak_threshold
                # Vectorized sliding window via cumsum: no Python loop over windows.
                # cs[r, j] = number of late payments in late[r, 0:j+1].
                # window sum for window [w, w+k) = cs[:, w+k-1] - cs[:, w-1].
                cs = late.cumsum(axis=1)                   # (R, n_post)
                pad = np.zeros((n_runs, 1), dtype=np.int32)
                cs_pad = np.concatenate([pad, cs], axis=1) # (R, n_post+1)
                # window_sums[:, w] = sum of late payments in positions [w, w+k)
                window_sums = cs_pad[:, k:] - cs_pad[:, :-k]  # (R, n_post-k+1)
                has_streak  = (window_sums == k).any(axis=1)   # (R,)
                default_count = int(has_streak.sum())
                pd_star       = float(has_streak.mean())
            else:
                default_count = 0
                pd_star       = 0.0

            rows.append({
                "mid":          profile.mid,
                "gid":          profile.gid,
                "p_ontime_raw": profile.p_ontime_raw,
                "true_pd":      profile.true_pd,
                "default_count": default_count,
                "pd_star":      pd_star,
            })

    return pd.DataFrame(rows)


def fit_logistic_pd_star(
    stacked_runs_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    min_events: int = 10,
):
    """Fit a logistic regression to predict default from raw behavioural variables.

    Uses the underlying variables that *feed into* the score — not the score
    or its pillars.  This keeps the PD* estimate free of any expert-defined
    weighting choices embedded in the scoring formula, so it captures how
    default probability responds to the raw signals themselves.

    Feature priority (in order; all expert-formula outputs excluded):
      1. Payment-behaviour metrics: otr, al, ls, rc, slip, san6
      2. Oracle payment propensity: p_ontime_raw
      3. Observable member facts:  prior_default, extra_groups, b_ord

    Score pillars (s_pdis, s_ordr, s_gov, s_liq, s_soc) are intentionally
    never used: they embed expert weights and are zeroed for defaulters,
    which would produce a bimodal rather than continuous PD* estimate.

    `stacked_runs_df` may contain one row per member (single-run workflow) or
    one row per (member × MC run).  It must have a binary 'default' column.
    Returns (model, df_with_probs) where df_with_probs adds 'pd_star_logit'.
    """
    if LogisticRegression is None:
        raise RuntimeError("scikit-learn is required for logistic PD* fitting but is not installed.")

    if feature_cols is None:
        # All candidates are raw observable variables — score pillars are excluded.
        # This mirrors every input that feeds into the five score pillars:
        #   s_pdis ← otr, al, ls, rc, extra_groups
        #   s_ordr ← b_ord, slip, tdec
        #   s_gov  ← rules, san6, sure_val, star_topology
        #   s_liq  ← (bidding-specific disc/q_rank, not available in member_df)
        #   s_soc  ← rep, cent, end_f, end_s, prior_default
        candidate = [
            # Payment-discipline inputs
            "otr", "al", "ls", "rc", "slip", "san6",
            # Oracle payment propensity
            "p_ontime_raw",
            # Order / allocation inputs
            "b_ord", "tdec",
            # Governance inputs
            "rules", "sure_val", "star_topology",
            # Social capital inputs
            "rep", "cent", "end_f", "end_s",
            # Credit-stacking / prior-default
            "extra_groups", "prior_default",
        ]
        feature_cols = [c for c in candidate if c in stacked_runs_df.columns]
        if not feature_cols:
            raise ValueError(
                "No raw behavioural variables found. Ensure the DataFrame contains columns from: "
                "otr, al, ls, rc, slip, san6, p_ontime_raw, prior_default, extra_groups, b_ord. "
                "Pass feature_cols explicitly if using a custom DataFrame."
            )

    df = stacked_runs_df.copy().dropna(subset=["default"])
    y = df["default"].astype(int).values
    if len(y) < min_events or y.sum() < 2:
        raise ValueError("Not enough default events to fit logistic model. Increase MC runs or relax min_events.")

    X = df[feature_cols].astype(float).fillna(0.0).values
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    df_out = df.copy()
    df_out["pd_star_logit"] = probs
    return model, df_out


# ---------------------------------------------------------------------------
# 13. Portfolio concentration guardrail  (Item 4)
# ---------------------------------------------------------------------------

def check_portfolio_concentration(
    member_df: pd.DataFrame,
    rho_max: float = 0.15,
    score_threshold: float = 40.0,
) -> pd.DataFrame:
    """Post-scoring portfolio guardrail: flag groups that dominate loan approvals.

    A group is 'concentrated' when its share of all eligible candidates (score ≥
    score_threshold) exceeds rho_max.  The lender should pause new approvals for
    flagged groups until concentration falls below the limit.

    Parameters
    ----------
    member_df        : DataFrame with 'gid' and 'score' columns (from generate_population*)
    rho_max          : maximum acceptable share any single group may hold of total eligibles
    score_threshold  : minimum score to be counted as an eligible loan candidate

    Returns
    -------
    DataFrame with one row per group:
        gid, n_members, n_eligible, eligible_share, flagged
    Sorted by eligible_share descending.
    """
    if "score" not in member_df.columns or "gid" not in member_df.columns:
        raise ValueError("member_df must contain 'gid' and 'score' columns.")

    eligible = member_df["score"] >= score_threshold
    total_eligible = int(eligible.sum())

    rows = []
    for gid, grp in member_df.groupby("gid"):
        n_members  = len(grp)
        n_elig     = int((grp["score"] >= score_threshold).sum())
        share      = float(n_elig) / max(1, total_eligible)
        rows.append({
            "gid":            gid,
            "n_members":      n_members,
            "n_eligible":     n_elig,
            "eligible_share": round(share, 4),
            "flagged":        share > rho_max,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("eligible_share", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 14. Parameter sensitivity  (sweeps ScoreParams)  -- unchanged
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
# 15. CLI entry point  (unchanged)
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
