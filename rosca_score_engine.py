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
from sklearn.linear_model import LogisticRegression

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
    n = len(x)
    if n < 3: return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d2 = ((rx - ry) ** 2).sum()
    return float(1.0 - 6.0 * d2 / (n * (n ** 2 - 1)))

# ---------------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------------

@dataclass
class ScoreParams:
    """Tunable hyperparameters for the 5-pillar score."""
    a: float = 0.80
    c_otr: float = 0.85; k_otr: float = 12.0
    a_al: float = 0.70; a_ls: float = 0.60
    c_rc: float = 0.70; k_rc: float = 10.0
    a_slip: float = 0.80
    k_rules: float = 12.0; a_san: float = 0.60
    q0: float = 0.50; k_q: float = 10.0
    v_ref: float = 0.10; a_v: float = 0.80
    w_rep: float = 5.0; w_cent: float = 4.0
    w_endf: float = 3.0; w_ends: float = 3.0

@dataclass
class PopulationParams:
    """Distributions for random member/group drawing."""
    n_groups: int = 20
    group_size_min: int = 6; group_size_max: int = 20
    rtype_bidding_prob: float = 0.50
    rules_prob: float = 0.75
    san_rate_min: float = 0.10; san_rate_max: float = 1.20
    num_cycles_min: int = 1; num_cycles_max: int = 3
    p_ontime_mean: float = 0.80; p_ontime_conc: float = 9.0
    dlate_lognorm_mu: float = 1.6; dlate_lognorm_sigma: float = 0.8
    post_slip_mean: float = 0.08; post_slip_conc: float = 8.0
    bid_agg_mean: float = 0.22; bid_agg_conc: float = 7.0
    bid_vol_min: float = 0.02; bid_vol_max: float = 0.30
    p_rep: float = 0.45; p_cent: float = 0.30; p_endf: float = 0.25; p_ends: float = 0.15
    p_sure_none: float = 0.40; p_sure_weak: float = 0.35

    def _beta_params(self, mean: float, conc: float) -> Tuple[float, float]:
        mean = float(np.clip(mean, 1e-4, 1 - 1e-4))
        conc = max(conc, 0.5)
        return mean * conc, (1 - mean) * conc

@dataclass
class MacroEnvironment:
    stress_level: float = 0.0
    within_group_corr: float = 0.20
    shock_windows: List[Tuple[int, int, float]] = field(default_factory=list)

@dataclass
class MemberProfile:
    mid: str; gid: str; n: int; aord: int; rtype: str; rules: bool; san_rate: float; num_cycles: int
    p_ontime_raw: float; dlate_mu: float; post_slip_prob: float
    bid_aggressiveness: float; bid_volatility: float
    rep: bool; cent: bool; endf: bool; ends: bool; sure_str: str
    true_pd: float = 0.0

# ---------------------------------------------------------------------------
# 2. Oracle Logic (True PD)
# ---------------------------------------------------------------------------

def _compute_true_pd(profile: MemberProfile, macro_stress: float, rng: np.random.Generator) -> float:
    sure_val = {"none": 0.0, "weak": 0.5, "strong": 1.0}.get(profile.sure_str, 0.0)
    social = float(profile.rep + profile.cent + profile.endf + profile.ends) / 4.0
    logit_pd = (
        -3.50 
        + 3.00 * (1.0 - profile.p_ontime_raw) 
        + 1.50 * profile.bid_aggressiveness 
        + 1.20 * profile.post_slip_prob 
        + 1.50 * macro_stress 
        - 0.60 * social 
        - 0.50 * sure_val 
        + rng.normal(0.0, 0.30)
    )
    return float(_sigmoid(logit_pd))

def _effective_p_ontime(p_raw: float, macro: MacroEnvironment, meeting_no: int, group_shock: float, idio_shock: float) -> float:
    corr = float(np.clip(macro.within_group_corr, 0.0, 1.0))
    agg_shock = np.sqrt(corr) * group_shock + np.sqrt(1.0 - corr) * idio_shock
    logit_base = _logit(p_raw) - macro.stress_level * 1.5
    logit_noisy = logit_base + agg_shock * 0.40
    p_eff = _sigmoid(logit_noisy)
    for t0, t1, sev in macro.shock_windows:
        if t0 <= meeting_no <= t1: p_eff *= float(sev)
    return float(np.clip(p_eff, 0.01, 0.99))

# ---------------------------------------------------------------------------
# 3. Data Generation (With Hard Default Logic)
# ---------------------------------------------------------------------------

def _generate_member_meetings(profile: MemberProfile, macro: MacroEnvironment, group_shocks: Dict[int, float], rng: np.random.Generator, base_date: pd.Timestamp, K_min: int = 6) -> List[dict]:
    n = profile.n
    meetings_per_cycle = max(K_min, n)
    adate_by_cycle = {cid: base_date + pd.DateOffset(months=((cid - 1) * meetings_per_cycle + profile.aord) - 1) for cid in range(1, profile.num_cycles + 1)}
    total_meetings = meetings_per_cycle * profile.num_cycles
    T_i = base_date + pd.DateOffset(months=total_meetings - 1)

    # State for Hard Default Sequence
    consecutive_late_post_payout = 0
    has_defaulted = False
    
    rows: List[dict] = []
    for cid in range(1, profile.num_cycles + 1):
        adate_i = adate_by_cycle[cid]
        for m in range(1, meetings_per_cycle + 1):
            meeting_no = (cid - 1) * meetings_per_cycle + m
            mdate = base_date + pd.DateOffset(months=meeting_no - 1)
            
            if mdate > T_i or has_defaulted: 
                continue

            is_post_payout = (mdate > adate_i)
            p_eff = _effective_p_ontime(
                profile.p_ontime_raw * (1.0 - 0.5 * (profile.post_slip_prob if is_post_payout else 0.0)), 
                macro, meeting_no, group_shocks.get(meeting_no, 0.0), rng.standard_normal()
            )
            
            ont = int(rng.random() < p_eff)
            dlate = 0 if ont else max(1, int(round(rng.lognormal(np.log(max(profile.dlate_mu, 1.0)), 0.8))))

            # --- SEQUENCE TRIGGER: Hard Default Logic ---
            if is_post_payout:
                consecutive_late_post_payout = consecutive_late_post_payout + 1 if ont == 0 else 0
                if consecutive_late_post_payout >= 3:
                    has_defaulted = True

            rows.append({
                "mid": profile.mid, "gid": profile.gid, "cid": cid, "meeting_no": meeting_no, "mdate": mdate, "adate": adate_i, "ont": ont, "dlate": dlate, 
                "has_defaulted": int(has_defaulted),
                "rtype": profile.rtype, "rules": int(profile.rules), "aord": profile.aord, "san_flag": int(rng.random() < profile.san_rate / 6.0),
                "bid": 1 if (profile.rtype == "bidding" and m == profile.aord) else 0,
                "disc": float(np.clip(rng.normal(profile.bid_aggressiveness, max(profile.bid_volatility, 0.01)), 0, 0.99)) if (profile.rtype == "bidding" and m == profile.aord) else 0.0,
                "rep": int(profile.rep), "cent": int(profile.cent), "end_f": int(profile.endf), "end_s": int(profile.ends), "sure_str": profile.sure_str,
                "_n": n, "_true_pd": profile.true_pd, "tdec": int(adate_i <= T_i)
            })
    return rows

# ---------------------------------------------------------------------------
# 4. Scoring Logic (With Hard Zero Penalty)
# ---------------------------------------------------------------------------

def _max_consec_late(ont_arr: np.ndarray) -> int:
    max_s = cur = 0
    for v in ont_arr:
        cur = cur + 1 if v == 0 else 0
        max_s = max(max_s, cur)
    return max_s

def compute_score(member_rows: pd.DataFrame, params: ScoreParams) -> dict:
    df = member_rows.sort_values("meeting_no").reset_index(drop=True)
    if df.empty: return {k: 0.0 for k in ["score","s_pdis","s_ordr","s_gov","s_liq","s_soc"]}

    # Sequence Observation
    is_defaulter = df["has_defaulted"].max() > 0

    n, rtype, rules, tdec, aord, sure_str = int(df["_n"].iloc[0]), df["rtype"].iloc[0], int(df["rules"].iloc[0]), int(df["tdec"].iloc[0]), int(df["aord"].iloc[0]), df["sure_str"].iloc[0]
    rep, cent, end_f, end_s = int(df["rep"].iloc[0]), int(df["cent"].iloc[0]), int(df["end_f"].iloc[0]), int(df["end_s"].iloc[0])

    K = len(df); w = params.a ** (K - np.arange(1, K + 1)); W = float(w.sum()) or 1.0
    ont_arr, dlate_arr = df["ont"].values.astype(float), df["dlate"].values.astype(float)

    otr = float((w * ont_arr).sum() / W); al = float((w * np.maximum(0, dlate_arr)).sum() / W); ls = _max_consec_late(ont_arr.astype(int))
    rc = float((w * ((dlate_arr > 0) & (dlate_arr < 7)).astype(float)).sum() / W)

    # Pillars
    s_pdis = min(35.0, (35.0 / 39.0) * (18.0 * _sigmoid(params.k_otr * (otr - params.c_otr)) + 6.0 * np.exp(-params.a_al * _log1p_safe(al)) + 8.0 * np.exp(-params.a_ls * max(0, ls - 1)) + 7.0 * _sigmoid(params.k_rc * (rc - params.c_rc))))
    s_ordr = 15.0 * (aord / n) * (1.0 - params.a_slip * (1 if any(df[df["mdate"] > df["adate"]]["ont"] == 0) else 0)) if tdec else 15.0 * (aord / n)
    s_gov = (20.0 / 17.0) * (5.0 * _sigmoid(params.k_rules * (rules - 0.5)) + 6.0 * np.exp(-params.a_san * float(df["san_flag"].iloc[-5:].sum())) + {"none": 0.0, "weak": 3.0, "strong": 6.0}.get(sure_str, 0.0))
    
    s_liq = 0.0
    if rtype == "bidding":
        bids = df[df["bid"] == 1]["disc"].values.astype(float)
        if len(bids) >= 1:
            q_rank = float(df["_disc_q_rank"].iloc[0]) if "_disc_q_rank" in df.columns else 0.5
            iqr = float(np.percentile(bids, 75) - np.percentile(bids, 25)) if len(bids) > 1 else 0.0
            s_liq = (15.0 / 12.0) * (6.0 * (1.0 - _sigmoid(params.k_q * (q_rank - params.q0))) + 6.0 * np.exp(-params.a_v * _log1p_safe(iqr / params.v_ref)))

    s_soc = params.w_rep * rep + params.w_cent * cent + params.w_endf * end_f + params.w_ends * end_s
    
    # --- HARD PENALTY: Score goes to zero if defaulted ---
    final_score = 0.0 if is_defaulter else (s_pdis + s_ordr + s_gov + s_liq + s_soc)

    return {
        "score": round(final_score, 3), 
        "is_defaulter": int(is_defaulter), 
        "s_pdis": s_pdis, "s_ordr": s_ordr, "s_gov": s_gov, "s_liq": s_liq, "s_soc": s_soc,
        "otr": otr, "al": al, "ls": ls, "rc": rc
    }

# ---------------------------------------------------------------------------
# 5. Sequence 2 Calibration (Regression Layer)
# ---------------------------------------------------------------------------

def run_calibration(member_df: pd.DataFrame) -> Tuple[Dict[str, float], np.ndarray]:
    """Logistic Regression to find the 'True Coefficients' of each pillar."""
    features = ['s_pdis', 's_ordr', 's_gov', 's_liq', 's_soc']
    X, y = member_df[features], member_df['is_defaulter']
    
    if y.nunique() < 2: 
        return {f: 0.0 for f in features}, np.zeros(len(member_df))
    
    lr = LogisticRegression(max_iter=1000).fit(X, y)
    calibrated_weights = dict(zip(features, lr.coef_[0]))
    true_pd_star = lr.predict_proba(X)[:, 1]
    
    return calibrated_weights, true_pd_star

# ---------------------------------------------------------------------------
# 6. Simulation Entry Point
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    member_df: pd.DataFrame
    meeting_df: pd.DataFrame
    calibrated_weights: Dict[str, float]
    rho_star: float

def generate_population(pop: PopulationParams, macro: MacroEnvironment, params: ScoreParams, seed: int = 42) -> SimulationResult:
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2024-01-01")
    all_profiles, all_meetings = [], []

    # Sequence 1: Generate Data
    for g_idx in range(pop.n_groups):
        gid = f"G{g_idx+1:02d}"
        n = int(rng.integers(pop.group_size_min, pop.group_size_max + 1))
        rtype = "bidding" if rng.random() < pop.rtype_bidding_prob else "random"
        rules, san_rate, num_cycles = rng.random() < pop.rules_prob, float(rng.uniform(pop.san_rate_min, pop.san_rate_max)), int(rng.integers(pop.num_cycles_min, pop.num_cycles_max + 1))
        aord_list = list(rng.permutation(n) + 1)
        group_shocks = {m: float(rng.standard_normal()) for m in range(1, (max(6, n) * num_cycles) + 1)}
        
        a_otr, b_otr = pop._beta_params(pop.p_ontime_mean, pop.p_ontime_conc)
        a_slip, b_slip = pop._beta_params(pop.post_slip_mean, pop.post_slip_conc)
        a_bid, b_bid = pop._beta_params(pop.bid_agg_mean, pop.bid_agg_conc)

        for m_idx in range(n):
            mid = f"{gid}_M{m_idx+1:02d}"
            p_ontime_raw = float(np.clip(rng.beta(a_otr, b_otr), 0.01, 0.99))
            dlate_mu = float(np.clip(rng.lognormal(pop.dlate_lognorm_mu, pop.dlate_lognorm_sigma), 1.0, 60.0))
            post_slip, bid_agg, bid_vol = float(np.clip(rng.beta(a_slip, b_slip), 0.0, 1.0)), float(np.clip(rng.beta(a_bid, b_bid), 0.0, 0.99)), float(rng.uniform(pop.bid_vol_min, pop.bid_vol_max))
            rep, cent, endf, ends = (rng.random() < p) for p in [pop.p_rep, pop.p_cent, pop.p_endf, pop.p_ends]
            
            r_sure = rng.random()
            sure_str = "none" if r_sure < pop.p_sure_none else "weak" if r_sure < (pop.p_sure_none + pop.p_sure_weak) else "strong"

            profile = MemberProfile(mid=mid, gid=gid, n=n, aord=aord_list[m_idx], rtype=rtype, rules=rules, san_rate=san_rate, num_cycles=num_cycles, p_ontime_raw=p_ontime_raw, dlate_mu=dlate_mu, post_slip_prob=post_slip, bid_aggressiveness=bid_agg, bid_volatility=bid_vol, rep=rep, cent=cent, endf=endf, ends=ends, sure_str=sure_str)
            profile.true_pd = _compute_true_pd(profile, macro.stress_level, rng)
            
            g_rows = _generate_member_meetings(profile, macro, group_shocks, rng, base_date)
            if g_rows:
                mtg = pd.DataFrame(g_rows)
                if rtype == "bidding":
                    mtg["_disc_q_rank"] = 0.5 # Placeholder for rank
                all_meetings.append(mtg)

    meeting_df = pd.concat(all_meetings, ignore_index=True)
    score_rows = []
    for mid_val, m_rows in meeting_df.groupby("mid", sort=False):
        sd = compute_score(m_rows, params)
        sd.update({"mid": mid_val, "true_pd_oracle": m_rows["_true_pd"].iloc[0]})
        score_rows.append(sd)

    member_df = pd.DataFrame(score_rows)

    # Sequence 2: ML Calibration
    weights, pd_star = run_calibration(member_df)
    member_df['true_pd_star'] = pd_star
    
    # Correlation between original score and the optimized PD*
    rho_star = _spearman(member_df['score'], 1.0 - member_df['true_pd_star'])
    
    return SimulationResult(member_df=member_df, meeting_df=meeting_df, calibrated_weights=weights, rho_star=rho_star)

# ---------------------------------------------------------------------------
# 7. CLI Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pop = PopulationParams(n_groups=25, p_ontime_mean=0.82)
    macro = MacroEnvironment(stress_level=0.15)
    params = ScoreParams()

    result = generate_population(pop, macro, params)
    
    print("\n" + "="*70)
    print("CALIBRATED SIMULATION COMPLETE")
    print("="*70)
    print(f"Spearman ρ (Score vs 1-PD*): {result.rho_star:+.4f}")
    print("\nCalibrated Pillar Weights (Feature Importance):")
    for k, v in result.calibrated_weights.items():
        print(f"  {k:6} : {v:+.4f}")
    
    print("\nSample Population Result:")
    print(result.member_df[['mid', 'score', 'is_defaulter', 'true_pd_star']].head(10).to_string(index=False))