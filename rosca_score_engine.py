"""
rosca_score_engine.py
Sequence 2 Edition: Hard Defaults & ML Calibration.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from sklearn.linear_model import LogisticRegression

# --- Math Helpers ---
def _sigmoid(x: float) -> float: return float(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))
def _logit(p: float) -> float: 
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))
def _log1p_safe(x: float) -> float: return float(np.log1p(max(0.0, x)))
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 3: return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    d2 = ((rx - ry) ** 2).sum()
    return float(1.0 - 6.0 * d2 / (n * (n ** 2 - 1)))

# --- Parameters ---
@dataclass
class ScoreParams:
    a: float = 0.80; c_otr: float = 0.85; k_otr: float = 12.0
    a_al: float = 0.70; a_ls: float = 0.60; c_rc: float = 0.70; k_rc: float = 10.0
    a_slip: float = 0.80; k_rules: float = 12.0; a_san: float = 0.60
    q0: float = 0.50; k_q: float = 10.0; v_ref: float = 0.10; a_v: float = 0.80
    w_rep: float = 5.0; w_cent: float = 4.0; w_endf: float = 3.0; w_ends: float = 3.0

@dataclass
class PopulationParams:
    n_groups: int = 20; group_size_min: int = 6; group_size_max: int = 20
    rtype_bidding_prob: float = 0.50; rules_prob: float = 0.75
    p_ontime_mean: float = 0.80; p_ontime_conc: float = 9.0
    post_slip_mean: float = 0.08; bid_agg_mean: float = 0.22
    p_rep: float = 0.45; p_cent: float = 0.30; p_endf: float = 0.25; p_ends: float = 0.15
    san_rate_min: float = 0.10; san_rate_max: float = 1.20; num_cycles_min: int = 1; num_cycles_max: int = 3
    dlate_lognorm_mu: float = 1.6; dlate_lognorm_sigma: float = 0.8
    bid_vol_min: float = 0.02; bid_vol_max: float = 0.30; p_sure_none: float = 0.40; p_sure_weak: float = 0.35

    def _beta_params(self, mean: float, conc: float) -> Tuple[float, float]:
        mean = float(np.clip(mean, 1e-4, 1 - 1e-4))
        return mean * conc, (1 - mean) * conc

@dataclass
class MacroEnvironment:
    stress_level: float = 0.0; within_group_corr: float = 0.20
    shock_windows: List[Tuple[int, int, float]] = field(default_factory=list)

@dataclass
class MemberProfile:
    mid: str; gid: str; n: int; aord: int; rtype: str; rules: bool; san_rate: float; num_cycles: int
    p_ontime_raw: float; dlate_mu: float; post_slip_prob: float; bid_aggressiveness: float
    bid_volatility: float; rep: bool; cent: bool; endf: bool; ends: bool; sure_str: str; true_pd: float = 0.0

# --- Engine Logic ---
def _compute_true_pd(profile: MemberProfile, macro_stress: float, rng: np.random.Generator) -> float:
    sure_val = {"none": 0.0, "weak": 0.5, "strong": 1.0}.get(profile.sure_str, 0.0)
    social = float(profile.rep + profile.cent + profile.endf + profile.ends) / 4.0
    logit_pd = (-3.50 + 3.00 * (1.0 - profile.p_ontime_raw) + 1.50 * profile.bid_aggressiveness + 1.20 * profile.post_slip_prob + 1.50 * macro_stress - 0.60 * social - 0.50 * sure_val + rng.normal(0.0, 0.30))
    return float(_sigmoid(logit_pd))

def _effective_p_ontime(p_raw: float, macro: MacroEnvironment, meeting_no: int, group_shock: float, idio_shock: float) -> float:
    corr = float(np.clip(macro.within_group_corr, 0.0, 1.0))
    agg_shock = np.sqrt(corr) * group_shock + np.sqrt(1.0 - corr) * idio_shock
    logit_base = _logit(p_raw) - macro.stress_level * 1.5
    return _sigmoid(logit_base + agg_shock * 0.40)

def _generate_member_meetings(profile: MemberProfile, macro: MacroEnvironment, group_shocks: Dict[int, float], rng: np.random.Generator, base_date: pd.Timestamp) -> List[dict]:
    n = profile.n
    meetings_per_cycle = max(6, n)
    adate_by_cycle = {cid: base_date + pd.DateOffset(months=((cid - 1) * meetings_per_cycle + profile.aord) - 1) for cid in range(1, profile.num_cycles + 1)}
    total_meetings = meetings_per_cycle * profile.num_cycles
    T_i = base_date + pd.DateOffset(months=total_meetings - 1)
    
    consecutive_late_post_payout = 0
    has_defaulted = False
    rows = []

    for cid in range(1, profile.num_cycles + 1):
        adate_i = adate_by_cycle[cid]
        for m in range(1, meetings_per_cycle + 1):
            meeting_no = (cid - 1) * meetings_per_cycle + m
            mdate = base_date + pd.DateOffset(months=meeting_no - 1)
            if mdate > T_i or has_defaulted: continue
            
            is_post_payout = (mdate > adate_i)
            p_eff = _effective_p_ontime(profile.p_ontime_raw * (1.0 - 0.5 * (profile.post_slip_prob if is_post_payout else 0.0)), macro, meeting_no, group_shocks.get(meeting_no, 0.0), rng.standard_normal())
            ont = int(rng.random() < p_eff)
            dlate = 0 if ont else max(1, int(round(rng.lognormal(np.log(max(profile.dlate_mu, 1.0)), 0.8))))

            if is_post_payout:
                consecutive_late_post_payout = consecutive_late_post_payout + 1 if ont == 0 else 0
                if consecutive_late_post_payout >= 3: has_defaulted = True

            rows.append({
                "mid": profile.mid, "gid": profile.gid, "meeting_no": meeting_no, "mdate": mdate, "adate": adate_i, "ont": ont, "dlate": dlate, 
                "has_defaulted": int(has_defaulted), "rtype": profile.rtype, "rules": int(profile.rules), "aord": profile.aord, 
                "san_flag": int(rng.random() < profile.san_rate / 6.0), "bid": 1 if (profile.rtype == "bidding" and m == profile.aord) else 0,
                "disc": float(np.clip(rng.normal(profile.bid_aggressiveness, max(profile.bid_volatility, 0.01)), 0, 0.99)) if (profile.rtype == "bidding" and m == profile.aord) else 0.0,
                "rep": int(profile.rep), "cent": int(profile.cent), "end_f": int(profile.endf), "end_s": int(profile.ends), "sure_str": profile.sure_str,
                "_n": n, "_true_pd": profile.true_pd, "tdec": int(adate_i <= T_i)
            })
    return rows

def compute_score(member_rows: pd.DataFrame, params: ScoreParams) -> dict:
    df = member_rows.sort_values("meeting_no")
    if df.empty: return {k: 0.0 for k in ["score", "s_pdis", "s_ordr", "s_gov", "s_liq", "s_soc", "is_defaulter"]}
    
    is_defaulter = df["has_defaulted"].max() > 0
    ont_arr = df["ont"].values.astype(float)
    W = (params.a ** (len(df) - np.arange(1, len(df) + 1))).sum() or 1.0
    otr = (ont_arr * (params.a ** (len(df) - np.arange(1, len(df) + 1)))).sum() / W
    
    s_pdis = 35.0 * _sigmoid(params.k_otr * (otr - params.c_otr))
    s_ordr = 15.0 * (df["aord"].iloc[0] / df["_n"].iloc[0])
    s_gov = 20.0 * (0.5 * int(df["rules"].iloc[0]) + 0.5 * ({"none":0.0, "weak":0.5, "strong":1.0}[df["sure_str"].iloc[0]]))
    s_liq = 15.0 if df["rtype"].iloc[0] == "bidding" else 0.0
    s_soc = min(15.0, params.w_rep * df["rep"].iloc[0] + params.w_cent * df["cent"].iloc[0] + params.w_endf * df["end_f"].iloc[0])

    raw_score = s_pdis + s_ordr + s_gov + s_liq + s_soc
    return {"score": 0.0 if is_defaulter else round(raw_score, 3), "is_defaulter": int(is_defaulter), "s_pdis": s_pdis, "s_ordr": s_ordr, "s_gov": s_gov, "s_liq": s_liq, "s_soc": s_soc}

@dataclass
class SimulationResult:
    member_df: pd.DataFrame; meeting_df: pd.DataFrame; calibrated_weights: Dict[str, float]; rho_star: float

def generate_population(pop: PopulationParams, macro: MacroEnvironment, params: ScoreParams, seed: int = 42) -> SimulationResult:
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2024-01-01")
    all_meetings = []
    
    # Draw Groups and Members
    for g_idx in range(pop.n_groups):
        gid = f"G{g_idx+1:02d}"
        n = int(rng.integers(pop.group_size_min, pop.group_size_max + 1))
        rtype = "bidding" if rng.random() < pop.rtype_bidding_prob else "random"
        aord_list = list(rng.permutation(n) + 1)
        group_shocks = {m: float(rng.standard_normal()) for m in range(1, 100)}
        a_otr, b_otr = pop._beta_params(pop.p_ontime_mean, pop.p_ontime_conc)
        
        for m_idx in range(n):
            profile = MemberProfile(mid=f"{gid}_M{m_idx+1:02d}", gid=gid, n=n, aord=aord_list[m_idx], rtype=rtype, rules=True, san_rate=0.5, num_cycles=2, p_ontime_raw=float(rng.beta(a_otr, b_otr)), dlate_mu=1.6, post_slip_prob=0.08, bid_aggressiveness=0.22, bid_volatility=0.05, rep=True, cent=False, endf=False, ends=False, sure_str="weak")
            profile.true_pd = _compute_true_pd(profile, macro.stress_level, rng)
            all_meetings.append(pd.DataFrame(_generate_member_meetings(profile, macro, group_shocks, rng, base_date)))

    meeting_df = pd.concat(all_meetings)
    scores = [compute_score(m_rows, params) for _, m_rows in meeting_df.groupby("mid")]
    member_df = pd.DataFrame(scores)
    
    # Calibration
    features = ['s_pdis', 's_ordr', 's_gov', 's_liq', 's_soc']
    X, y = member_df[features], member_df['is_defaulter']
    weights, pd_star = {f: 0.0 for f in features}, np.zeros(len(member_df))
    if y.nunique() >= 2:
        lr = LogisticRegression().fit(X, y)
        weights = dict(zip(features, lr.coef_[0]))
        pd_star = lr.predict_proba(X)[:, 1]
    
    member_df['true_pd_star'] = pd_star
    return SimulationResult(member_df, meeting_df, weights, _spearman(member_df['score'], 1.0 - pd_star))