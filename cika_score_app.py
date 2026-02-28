# cika_score_app.py
"""
CIKA — Credit Score Simulator
Streamlit app: simulate, score, estimate default risk (PD*), and validate.

Primary workflow (fast):
  1. Score + defaults  →  zero-out members who missed N meetings after the pot
  2. Fit logistic PD*  →  model-predicted default probability per member (seconds)
  3. Validate          →  compare score ranking to PD* ranking

Optional (slower):
  • Run MC PD*  →  model-free frequency estimate via 100-200 re-simulations
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from rosca_score_engine import (
    MacroEnvironment,
    PopulationParams,
    ScoreParams,
    generate_population,
    generate_population_with_defaults,
    compute_pd_star_mc,
    fit_logistic_pd_star,
    compute_pd_star_validation,
    check_portfolio_concentration,
)

try:
    from sklearn.metrics import roc_auc_score, brier_score_loss
    SKLEARN = True
except Exception:
    SKLEARN = False


# ── Schema  (section, type, default, friendly label, help text) ───────────────
SCHEMA = {
    # Population
    "n_groups":           ("pop",    int,   20,   "Number of savings groups",
                           "How many ROSCA groups to simulate. More groups = more members and more robust results."),
    "group_size_min":     ("pop",    int,   6,    "Min. members per group",
                           "Smallest group size allowed."),
    "group_size_max":     ("pop",    int,   20,   "Max. members per group",
                           "Largest group size allowed."),
    "rtype_bidding_prob": ("pop",    float, 0.50, "% groups that auction the pot",
                           "0 = all groups use random order, 1 = all groups auction. 0.5 = half and half."),
    "rules_prob":         ("pop",    float, 0.75, "% groups with written rules",
                           "Share of groups that operate with formal bylaws."),
    "p_ontime_mean":      ("pop",    float, 0.80, "Average on-time payment rate",
                           "Typical share of meetings where members pay on time. 0.80 = 80% on time."),
    "p_ontime_conc":      ("pop",    float, 9.0,  "How spread out is payment discipline?",
                           "Higher = members are more similar to each other. Lower = more variation."),
    "post_slip_mean":     ("pop",    float, 0.08, "Post-payout reliability drop",
                           "Average probability that a member becomes less reliable after receiving the pot."),
    "bid_agg_mean":       ("pop",    float, 0.22, "Avg. bidding aggressiveness",
                           "How early and aggressively members bid in auction-style groups (0 = never early, 1 = always)."),
    "p_rep":              ("pop",    float, 0.45, "% with repeat-cycle history",
                           "Share of members who have participated in at least one previous ROSCA."),
    "p_cent":             ("pop",    float, 0.30, "% network-central members",
                           "Share of members who are well-connected in the community."),
    "p_endf":             ("pop",    float, 0.25, "% with group-leader endorsement",
                           "Share of members vouched for by a trusted foreman or group leader."),
    # Macro
    "stress_level":       ("macro",  float, 0.0,  "Economic stress level (0 = calm, 1 = severe)",
                           "Simulates a difficult environment. Higher values reduce everyone's payment reliability."),
    "within_group_corr":  ("macro",  float, 0.20, "How correlated are shocks within a group?",
                           "0 = each member is affected independently. 1 = the whole group is hit together."),
    # Score shape
    "a":        ("score", float, 0.80, "Recency weight",
                 "How much recent meetings count vs old ones. 0.8 = recent payments weighted more."),
    "c_otr":    ("score", float, 0.85, "On-time rate threshold",
                 "On-time rate above which the payment score rises steeply."),
    "k_otr":    ("score", float, 12.0, "On-time rate sensitivity",
                 "Steepness of the score jump around the on-time threshold."),
    "a_al":     ("score", float, 0.70, "Lateness penalty strength",
                 "How much average days-late reduces the payment score."),
    "a_ls":     ("score", float, 0.60, "Late-streak penalty strength",
                 "How much a consecutive run of late payments reduces the score."),
    "a_slip":   ("score", float, 0.80, "Post-payout slip penalty",
                 "How much the order score is cut when a member slips after receiving the pot."),
    "k_rules":  ("score", float, 12.0, "Formal-rules impact",
                 "How strongly having written rules boosts the governance score."),
    "a_san":    ("score", float, 0.60, "Sanction fade rate",
                 "How fast past sanctions fade from the governance score."),
    "q0":       ("score", float, 0.50, "Bid level midpoint",
                 "The bid discount around which the liquidity score is neutral."),
    "k_q":      ("score", float, 10.0, "Bid aggressiveness sensitivity",
                 "How steeply bid aggressiveness affects the liquidity score."),
    "a_v":      ("score", float, 0.80, "Bid volatility penalty",
                 "How much erratic bidding reduces the liquidity score."),
    "w_rep":    ("score", float, 5.0,  "Points for repeat-cycle history",
                 "Social score points awarded for having a repeat-participation record."),
    "w_cent":   ("score", float, 4.0,  "Points for network centrality",
                 "Social score points for being well-connected in the community."),
    "w_endf":   ("score", float, 3.0,  "Points for leader endorsement",
                 "Social score points for a foreman or leader endorsement."),
    "w_ends":   ("score", float, 3.0,  "Points for senior endorsement",
                 "Social score points for a senior-member endorsement."),
    # Default rule
    "streak_threshold": ("pdstar", int,   3,   "Default: consecutive missed meetings after pot",
                         "If a member misses this many meetings IN A ROW after getting the pot, their score is set to 0."),
    "mc_runs":          ("pdstar", int,   200, "MC simulations (optional, slow)",
                         "Number of re-simulations used for the optional model-free PD* estimate. More = accurate but slower."),
    # Guardrails — population prevalence
    "p_prior_default":    ("pop",   float, 0.10, "% with hidden prior-default history",
                           "Share of members who have defaulted in another group before but are not disclosing it. "
                           "Set to 0 to turn this off. Higher = more reputation-scrubbing risk in the portfolio."),
    "p_payment_verified": ("pop",   float, 0.85, "% payments backed by digital proof",
                           "Average share of each member's payments that are verifiable (e.g. mobile-money receipt). "
                           "1.0 = all payments are verified. 0 = nothing is verifiable (pure self-report)."),
    "p_star_topology":    ("pop",   float, 0.15, "% groups with closed-loop structure",
                           "Share of groups that are isolated 'star' networks with no external connections — "
                           "higher fraud risk. Set to 0 to disable. Penalises governance score for those groups."),
    "p_multi_group":      ("pop",   float, 0.20, "% members in multiple concurrent ROSCAs",
                           "Share of members who simultaneously belong to another ROSCA group (credit stacking). "
                           "These members carry cross-group payment obligations that strain their discipline score. "
                           "Set to 0 to disable."),
    "n_extra_groups_max": ("pop",   int,   2,    "Max extra concurrent ROSCA groups (stacking cap)",
                           "Upper bound on how many additional ROSCAs a stacked member can belong to. "
                           "Each extra group applies a lambda_stack haircut to the payment discipline score."),
    # Guardrails — portfolio concentration
    "rho_max":            ("portf", float, 0.15, "Portfolio concentration limit per group",
                           "Maximum share any single group may hold of all eligible loan candidates. "
                           "Groups exceeding this share are flagged for approval suspension. 0.15 = 15%."),
    "score_threshold":    ("portf", float, 40.0, "Eligibility score threshold",
                           "Minimum score for a member to count as an eligible loan candidate "
                           "in the portfolio concentration check."),
    # Guardrails — score penalties
    "gamma_rep":          ("score", float, 0.30, "Reputation decay for prior defaulters",
                           "Multiplier applied to the social capital score when a member has a prior-default flag. "
                           "0 = social score is wiped out entirely. 1 = no penalty (disable the rule)."),
    "w_unverified":       ("score", float, 1.00, "Trust weight for unverified payments",
                           "How much to count on-time payments that have no digital proof. "
                           "1.0 = full trust (ignore verification). 0 = only verified payments count."),
    "gov_star_penalty":   ("score", float, 0.40, "Governance cut for closed-loop groups",
                           "Fraction of the governance score removed when a group is flagged as star-topology. "
                           "0.4 = 40% reduction. 0 = no penalty (disable the rule)."),
    "lambda_stack":       ("score", float, 0.15, "Credit-stacking haircut per extra group",
                           "Fraction of the payment discipline score removed for each extra concurrent ROSCA. "
                           "0.15 = 15% cut per extra group. 0 = no penalty (disable). "
                           "Applied only to members with extra_groups > 0."),
    "alpha_macro":        ("score", float, 0.00, "Macro shock absorption (mean reversion)",
                           "Absorbs systemic shocks: late payments are discounted by this fraction of the group's "
                           "median lateness that meeting. 0 = no adjustment (default). "
                           "1 = a member is only penalised for lateness above the group median."),
    # Global
    "seed": ("global", int, 42, "Random seed",
             "Change this number to get a different random population with the same settings."),
}

DEFAULTS = {k: v[2] for k, v in SCHEMA.items()}

UI_GROUPS = {
    "Group structure": ["n_groups", "group_size_min", "group_size_max",
                        "rtype_bidding_prob", "rules_prob"],
    "Member behaviour": ["p_ontime_mean", "p_ontime_conc", "post_slip_mean",
                         "bid_agg_mean", "p_rep", "p_cent", "p_endf"],
    "Economic environment": ["stress_level", "within_group_corr"],
    "Default rule": ["streak_threshold"],
    "Risk guardrails": ["p_prior_default", "p_payment_verified", "p_star_topology",
                        "p_multi_group", "n_extra_groups_max",
                        "rho_max", "score_threshold"],
    "Score formula (advanced)": ["a", "c_otr", "k_otr", "a_al", "a_ls", "a_slip",
                                  "k_rules", "a_san", "q0", "k_q", "a_v",
                                  "w_rep", "w_cent", "w_endf", "w_ends",
                                  "gamma_rep", "w_unverified", "gov_star_penalty",
                                  "lambda_stack", "alpha_macro"],
    "Randomness": ["seed"],
    "MC PD* (optional)": ["mc_runs"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_configs(vals):
    pop = PopulationParams(
        n_groups=int(vals["n_groups"]),
        group_size_min=int(vals["group_size_min"]),
        group_size_max=max(int(vals["group_size_max"]), int(vals["group_size_min"]) + 1),
        rtype_bidding_prob=float(vals["rtype_bidding_prob"]),
        rules_prob=float(vals["rules_prob"]),
        p_ontime_mean=float(vals["p_ontime_mean"]),
        p_ontime_conc=float(vals["p_ontime_conc"]),
        post_slip_mean=float(vals["post_slip_mean"]),
        bid_agg_mean=float(vals["bid_agg_mean"]),
        p_rep=float(vals["p_rep"]),
        p_cent=float(vals["p_cent"]),
        p_endf=float(vals["p_endf"]),
        p_prior_default=float(vals["p_prior_default"]),
        p_payment_verified=float(vals["p_payment_verified"]),
        p_star_topology=float(vals["p_star_topology"]),
        p_multi_group=float(vals["p_multi_group"]),
        n_extra_groups_max=int(vals["n_extra_groups_max"]),
    )
    macro = MacroEnvironment(
        stress_level=float(vals["stress_level"]),
        within_group_corr=float(vals["within_group_corr"]),
    )
    score_kwargs = {k: float(vals[k]) for k in SCHEMA if SCHEMA[k][0] == "score"}
    params = ScoreParams(**score_kwargs)
    return pop, macro, params


def _hist(df, col, bins=35, title=None, color=None):
    enc = dict(
        x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins)),
        y=alt.Y("count()"),
        tooltip=[alt.Tooltip(f"{col}:Q", format=".2f"), alt.Tooltip("count()", title="n")],
    )
    if color:
        enc["color"] = color
    return (alt.Chart(df).mark_bar(opacity=0.7)
            .encode(**enc)
            .properties(height=260, title=title or col))


def _scatter_score_pdstar(df):
    has_def = "defaulted" in df.columns
    color = (
        alt.Color("defaulted:N",
                  scale=alt.Scale(domain=[False, True], range=["#3b82f6", "#ef4444"]),
                  legend=alt.Legend(title="Defaulted (score=0)"))
        if has_def else alt.value("#3b82f6")
    )
    tt = ["mid", "gid",
          alt.Tooltip("score:Q", format=".1f"),
          alt.Tooltip("pd_star:Q", format=".3f"),
          alt.Tooltip("true_pd:Q", format=".3f"),
          ] + (["defaulted"] if has_def else [])
    return (
        alt.Chart(df)
        .mark_circle(opacity=0.45, size=50)
        .encode(
            x=alt.X("pd_star:Q", title="PD*  (estimated default probability)",
                    scale=alt.Scale(domain=[0, max(1.0, float(df["pd_star"].max()) + 0.02)])),
            y=alt.Y("score:Q", title="Score  (0–100)",
                    scale=alt.Scale(domain=[0, 100])),
            color=color,
            tooltip=tt,
        )
        .properties(height=400,
                    title="Score vs PD*  ·  negative slope = good discrimination  ·  defaulters cluster bottom-right")
    )


def _quintile_bar(qt_df):
    df = qt_df.reset_index()
    df.columns = ["quintile", "mean_score", "std_score", "count"]
    COLORS = ["#22c55e", "#86efac", "#facc15", "#f97316", "#ef4444"]
    domain = list(df["quintile"].astype(str))
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("quintile:N", sort=None,
                    title="PD* Quintile  (Q1 = safest members, Q5 = riskiest)"),
            y=alt.Y("mean_score:Q", title="Mean Score",
                    scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("quintile:N",
                            scale=alt.Scale(domain=domain, range=COLORS[:len(domain)]),
                            legend=None),
            tooltip=["quintile",
                     alt.Tooltip("mean_score:Q", format=".1f", title="Mean score"),
                     alt.Tooltip("std_score:Q",  format=".1f", title="Std"),
                     alt.Tooltip("count:Q",       title="Members")],
        )
        .properties(height=290,
                    title="Mean Score by PD* Quintile  ·  should fall steadily Q1 → Q5")
    )


def _score_dist_by_default(df):
    if "defaulted" not in df.columns:
        return _hist(df, "score", title="Score distribution")
    return (
        alt.Chart(df)
        .mark_bar(opacity=0.55)
        .encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="Score"),
            y=alt.Y("count()", stack=None),
            color=alt.Color(
                "defaulted:N",
                scale=alt.Scale(domain=[False, True], range=["#3b82f6", "#ef4444"]),
                legend=alt.Legend(title="Defaulted"),
            ),
            tooltip=[alt.Tooltip("score:Q", bin=True), "count()"],
        )
        .properties(height=290,
                    title="Score distribution  ·  defaulters (red) should spike at 0")
    )


def _run_logistic_pdstar(member_df: pd.DataFrame):
    """Fit logistic PD* on the 5 score pillars and return member_df + pd_star column."""
    try:
        _, df_out = fit_logistic_pd_star(member_df)
        member_df = member_df.copy()
        # merge predictions back; df_out has same index
        member_df["pd_star"] = df_out["pd_star_logit"].values
        return member_df, None
    except Exception as e:
        return member_df, str(e)


def _merge_mc_pdstar(result, mc_df):
    df = result.member_df.copy()
    mp = mc_df.set_index("mid")["pd_star"].to_dict()
    df["pd_star"] = df["mid"].map(mp)
    return df.dropna(subset=["pd_star"])


# ── Page layout ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CIKA — Score", layout="wide")
st.title("CIKA — Credit Score")
st.caption(
    "Scores savings-group members on 5 pillars (payment discipline, allocation order, "
    "governance, liquidity, social capital). Members who miss payments consistently "
    "**after receiving the pot** are scored 0.  "
    "Click **Full analysis** to get everything in one step."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
preset = st.sidebar.selectbox(
    "Quick scenario",
    ["Default", "High economic stress", "Low payment discipline", "Strict scoring"],
    help="Pre-fills parameters for a common scenario. You can still adjust individual values.",
)
vals = {}
for group_name, keys in UI_GROUPS.items():
    advanced = group_name in ("Score formula (advanced)", "Randomness", "MC PD* (optional)")
    with st.sidebar.expander(group_name, expanded=not advanced):
        for k in keys:
            sec, typ, default, label, desc = SCHEMA[k]
            v0 = default
            if preset == "High economic stress"   and k == "stress_level":  v0 = 0.70
            if preset == "Low payment discipline" and k == "p_ontime_mean": v0 = 0.60
            if preset == "Strict scoring"         and k in ("a_al", "a_ls", "a_slip"):
                v0 = round(default * 1.3, 4)
            if typ is int:
                vals[k] = st.number_input(label, value=v0, step=1, help=desc, key=k)
            elif typ is float:
                vals[k] = st.number_input(label, value=v0, step=0.01, format="%.4f", help=desc, key=k)
            else:
                vals[k] = st.text_input(label, value=str(v0), help=desc, key=k)

# ── Action buttons ────────────────────────────────────────────────────────────
st.subheader("Run")
c1, c2, c3 = st.columns(3)
with c1:
    full_btn    = st.button("Full analysis ▶", type="primary",
                            help="Score + default rule + logistic PD* — one click, results in seconds.")
with c2:
    run_def_btn = st.button("Score + defaults only",
                            help="Simulate scores and apply the default rule. No PD* fitting.")
with c3:
    run_mc_btn  = st.button("Add MC PD* (optional)",
                            help="Model-free PD* estimate via Monte Carlo re-simulation (slower). "
                                 "Adds a comparison tab vs the logistic estimate.")

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [("last_result", None), ("last_mc", None),
                ("logit_df", None),    # member_df with pd_star from logistic
                ("merged_df", None),   # primary merged (logistic preferred, else MC)
                ("pdstar_val", None), ("pdstar_source", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _finalise(result, mc_df=None):
    """Fit logistic PD* (always attempted); also store MC if provided.
    Primary merged_df uses logistic PD* when available, falls back to MC.
    """
    # Always try logistic
    logit_member_df, err = _run_logistic_pdstar(result.member_df)
    if err is None:
        st.session_state["logit_df"] = logit_member_df
        merged = logit_member_df
        source = "logistic (score pillars)"
    elif mc_df is not None:
        st.warning(f"Logistic PD* failed ({err}). Falling back to MC PD*.")
        merged = _merge_mc_pdstar(result, mc_df)
        source = "MC frequency"
    else:
        st.error(
            f"**Logistic PD* could not be fitted.**  \n"
            f"Error: `{err}`  \n"
            "Check that **scikit-learn** is installed (`pip install scikit-learn`). "
            "The Score Overview and Benchmark tabs below still show your simulation results."
        )
        merged = None
        source = None

    if merged is not None:
        try:
            pv = compute_pd_star_validation(merged)
            st.session_state.update({"merged_df": merged, "pdstar_val": pv,
                                      "pdstar_source": source})
        except Exception as e:
            st.error(f"PD* validation failed: {e}")
            st.session_state.update({"merged_df": merged, "pdstar_val": None,
                                      "pdstar_source": source})


# ── Run handlers ──────────────────────────────────────────────────────────────
if full_btn:
    pop, macro, params = build_configs(vals)
    streak = int(vals["streak_threshold"])
    with st.spinner(f"Simulating population + applying default rule (≥{streak} missed meetings → score 0) + fitting logistic PD*…"):
        result = generate_population_with_defaults(
            pop, macro, params, seed=int(vals["seed"]), streak_threshold=streak)
    st.session_state.update({"last_result": result, "last_mc": None,
                              "merged_df": None, "pdstar_val": None})
    _finalise(result)
    n_def = int(result.member_df["defaulted"].sum()) if "defaulted" in result.member_df.columns else "?"
    src   = st.session_state.get("pdstar_source", "?")
    st.success(
        f"Done — {len(result.member_df)} members, **{n_def} defaulters** zeroed, "
        f"PD* fitted via **{src}**."
    )

if run_def_btn:
    pop, macro, params = build_configs(vals)
    streak = int(vals["streak_threshold"])
    with st.spinner(f"Simulating population (default rule: {streak} missed meetings → score 0)…"):
        result = generate_population_with_defaults(
            pop, macro, params, seed=int(vals["seed"]), streak_threshold=streak)
    st.session_state.update({"last_result": result, "merged_df": None,
                              "pdstar_val": None, "pdstar_source": None})
    _finalise(result)
    n_def = int(result.member_df["defaulted"].sum()) if "defaulted" in result.member_df.columns else "?"
    st.success(f"Done — {len(result.member_df)} members, **{n_def} defaulters** zeroed.")

if run_mc_btn:
    pop, macro, params = build_configs(vals)
    n_runs = int(vals["mc_runs"])
    streak = int(vals["streak_threshold"])
    with st.spinner(f"Running {n_runs} MC simulations — this takes a moment…"):
        mc_df = compute_pd_star_mc(pop, macro, params,
                                   n_runs=n_runs, base_seed=int(vals["seed"]),
                                   K_min=6, streak_threshold=streak)
    st.session_state["last_mc"] = mc_df
    result = st.session_state.get("last_result")
    if result is not None:
        # If logistic is already done, keep it as primary; MC goes into last_mc only
        if st.session_state.get("logit_df") is None:
            merged = _merge_mc_pdstar(result, mc_df)
            pv     = compute_pd_star_validation(merged)
            st.session_state.update({"merged_df": merged, "pdstar_val": pv,
                                      "pdstar_source": "MC frequency"})
    st.success(f"MC PD* done — mean PD* = **{mc_df['pd_star'].mean():.4f}**  "
               f"· See **Logistic vs MC comparison** at the bottom of the PD* Validation tab.")

# ── Results ───────────────────────────────────────────────────────────────────
result  = st.session_state.get("last_result")
mc_df   = st.session_state.get("last_mc")
merged  = st.session_state.get("merged_df")
pv      = st.session_state.get("pdstar_val")
source  = st.session_state.get("pdstar_source")

if result is None:
    st.info("Configure settings in the sidebar and click **Full analysis** to get started.")
    st.stop()

st.divider()
tab_pdstar, tab_score, tab_bench, tab_data = st.tabs([
    "📊 PD* Validation  ← primary results",
    "🎯 Score Overview",
    "🔬 Benchmark (true PD)",
    "📋 Raw data",
])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 0 — PD* Validation
# ──────────────────────────────────────────────────────────────────────────────
with tab_pdstar:
    if merged is None or pv is None:
        st.info(
            "PD* not yet estimated. Run **Full analysis** — it fits a logistic model "
            "in seconds and produces all the results below."
        )
    else:
        rho    = pv["spearman_rho_pdstar"]
        sep    = pv["score_separation_pdstar"]
        n_def  = pv.get("n_defaulted")
        def_rt = pv.get("default_rate")
        sc_d   = pv.get("score_mean_defaulted")
        sc_nd  = pv.get("score_mean_non_defaulted")

        st.caption(f"PD* source: **{source}**  ·  "
                   "PD* = estimated probability of default per member  ·  "
                   "Score should rank members opposite to PD*.")

        # ── Headline metrics ──────────────────────────────────────────────────────
        st.subheader("How well does the score predict default risk?")
        m1, m2, m3, m4 = st.columns(4)
        rho_delta = "🟢 strong" if rho >= 0.40 else ("🟡 moderate" if rho >= 0.20 else "🔴 weak")
        with m1:
            st.metric(
                "Spearman ρ  (score vs 1 − PD*)",
                f"{rho:+.4f}",
                delta=rho_delta,
                help="Rank correlation: how consistently does a higher score correspond to lower PD*? "
                     "Values above +0.40 are strong. Values near 0 mean the score doesn't rank risk.",
            )
        with m2:
            st.metric(
                "Score gap  (non-defaulter − defaulter)",
                f"{sep:+.2f} pts",
                help="Average score of non-defaulters minus average score of defaulters. "
                     "A large positive gap means the score clearly separates the two groups.",
            )
        with m3:
            if n_def is not None:
                st.metric(
                    "Members scored 0  (defaulted)",
                    f"{n_def}  out of {pv['n_members']}",
                    delta=f"{def_rt:.1%} default rate",
                    help="Members who triggered the post-payout default rule "
                         "(missed ≥ N consecutive meetings after receiving the pot).",
                )
        with m4:
            if sc_d is not None and sc_nd is not None:
                st.metric(
                    "Avg score: non-defaulted / defaulted",
                    f"{sc_nd:.1f}  /  {sc_d:.1f}",
                    help="Mean score in each group. "
                         "Defaulters should cluster near 0.",
                )

        # AUC / Brier
        if SKLEARN and "defaulted" in merged.columns:
            try:
                y_true  = merged["defaulted"].astype(int).values
                y_score = merged["score"].values.astype(float)
                smin, smax = y_score.min(), y_score.max()
                y_norm = (y_score - smin) / (smax - smin + 1e-9)
                auc   = roc_auc_score(y_true, y_norm) if y_true.sum() > 0 else float("nan")
                brier = brier_score_loss(y_true, y_norm)
                a1, a2, a3 = st.columns(3)
                auc_d = "🟢 good" if auc >= 0.70 else ("🟡 moderate" if auc >= 0.60 else "🔴 low")
                with a1:
                    st.metric("AUC  (score separates defaulters)", f"{auc:.4f}", delta=auc_d,
                              help="Area under the ROC curve. "
                                   "0.5 = random, 1.0 = perfect separation. Target: > 0.70.")
                with a2:
                    st.metric("Brier score  (lower = better)", f"{brier:.4f}",
                              help="Mean squared error between normalised score and binary default label. "
                                   "0 = perfect, 0.25 = random.")
                with a3:
                    st.metric("Members in analysis", pv["n_members"])
            except Exception:
                pass

        st.divider()

        # ── Score by PD* quintile ─────────────────────────────────────────────────
        st.subheader("Score by PD* quintile")
        st.caption(
            "Members are sorted into 5 equal groups by PD* (estimated default probability). "
            "Q1 = safest (lowest PD*),  Q5 = riskiest (highest PD*). "
            "The score should fall steadily from Q1 to Q5."
        )
        qt_df = pv["score_by_pdstar_quintile"]
        qa, qb = st.columns([3, 2])
        with qa:
            st.altair_chart(_quintile_bar(qt_df), use_container_width=True)
        with qb:
            qt_show = qt_df.reset_index().copy()
            qt_show.columns = ["PD* Quintile", "Mean Score", "Std", "Members"]
            st.dataframe(qt_show, use_container_width=True, hide_index=True)

        st.divider()

        # ── Scatter + distribution ────────────────────────────────────────────────
        s1, s2 = st.columns(2)
        with s1:
            st.altair_chart(_scatter_score_pdstar(merged), use_container_width=True)
        with s2:
            st.altair_chart(_score_dist_by_default(merged), use_container_width=True)

        # ── PD* histogram ─────────────────────────────────────────────────────────
        with st.expander("PD* distribution"):
            st.caption("Distribution of estimated default probabilities across all members.")
            st.altair_chart(
                _hist(merged, "pd_star", bins=30,
                      title=f"PD* histogram  ·  mean = {pv['pd_star_mean']:.4f}"),
                use_container_width=True,
            )

        # ── Logistic vs MC comparison (shown when both are available) ─────────────
        logit_df = st.session_state.get("logit_df")
        mc_state = st.session_state.get("last_mc")
        result_s = st.session_state.get("last_result")

        if logit_df is not None and mc_state is not None and result_s is not None:
            st.divider()
            st.subheader("Diagnostic: Logistic PD* vs MC PD*")
            st.caption(
                "Comparing the two PD* estimation methods for the same population. "
                "They should agree closely — high correlation confirms the logistic model is reliable."
            )

            mc_map = mc_state.set_index("mid")["pd_star"].to_dict()
            comp_df = logit_df[["mid", "gid", "score", "defaulted", "pd_star"]
                                if "defaulted" in logit_df.columns
                                else ["mid", "gid", "score", "pd_star"]].copy()
            comp_df = comp_df.rename(columns={"pd_star": "pd_star_logit"})
            comp_df["pd_star_mc"] = comp_df["mid"].map(mc_map)
            comp_df = comp_df.dropna(subset=["pd_star_mc"])

            if not comp_df.empty:
                from rosca_score_engine import _spearman as _sp
                rho_ll    = _sp(comp_df["pd_star_logit"].values, comp_df["pd_star_mc"].values)
                mean_diff = float((comp_df["pd_star_logit"] - comp_df["pd_star_mc"]).abs().mean())

                x1, x2, x3 = st.columns(3)
                with x1:
                    c = "🟢 high agreement" if rho_ll >= 0.70 else ("🟡 moderate" if rho_ll >= 0.40 else "🔴 diverge")
                    st.metric("Spearman ρ  (logistic vs MC)", f"{rho_ll:+.4f}", delta=c,
                              help="How well the two PD* methods agree on member ranking. "
                                   "High = logistic is a reliable proxy for MC.")
                with x2:
                    st.metric("Mean absolute difference", f"{mean_diff:.4f}",
                              help="Average absolute gap between logistic PD* and MC PD* per member.")
                with x3:
                    st.metric("Members compared", len(comp_df))

                scat = (
                    alt.Chart(comp_df)
                    .mark_circle(opacity=0.5, size=40)
                    .encode(
                        x=alt.X("pd_star_mc:Q",   title="MC PD* (model-free)",   scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("pd_star_logit:Q", title="Logistic PD* (fitted)", scale=alt.Scale(domain=[0, 1])),
                        tooltip=["mid", "gid",
                                 alt.Tooltip("pd_star_logit:Q", format=".3f"),
                                 alt.Tooltip("pd_star_mc:Q",    format=".3f"),
                                 alt.Tooltip("score:Q",         format=".1f")],
                    )
                    .properties(height=340,
                                title="Logistic PD* vs MC PD*  ·  diagonal = perfect agreement")
                )
                diag = (alt.Chart(pd.DataFrame({"x": [0, 1]}))
                        .mark_line(strokeDash=[4, 2], color="gray")
                        .encode(x="x:Q", y="x:Q"))
                st.altair_chart(scat + diag, use_container_width=True)

                with st.expander("Full comparison table (score + logistic PD* + MC PD*)"):
                    display_cols = ["mid", "gid", "score", "pd_star_logit", "pd_star_mc"]
                    if "defaulted" in comp_df.columns:
                        display_cols.append("defaulted")
                    st.dataframe(
                        comp_df[display_cols]
                        .sort_values("pd_star_logit", ascending=False)
                        .reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                    )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Score Overview
# ──────────────────────────────────────────────────────────────────────────────
with tab_score:
    df = result.member_df
    v  = result.validation

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Members simulated", v["n_members"])
        st.metric("Mean score", f"{v['score_mean']:.1f} ± {v['score_std']:.1f}")
        if "defaulted" in df.columns:
            n_z = int(df["defaulted"].sum())
            st.metric("Members scored 0  (defaulted)", f"{n_z}  ({n_z / len(df):.1%})")
    with col2:
        st.altair_chart(
            _score_dist_by_default(df) if "defaulted" in df.columns
            else _hist(df, "score", title="Score distribution"),
            use_container_width=True,
        )

    st.subheader("Average contribution of each scoring pillar")
    st.caption(
        "Payment discipline (s_pdis) + Allocation order (s_ordr) + Governance (s_gov) + "
        "Liquidity / bidding (s_liq) + Social capital (s_soc) = total score (max ≈ 85)."
    )
    pillars = ["s_pdis", "s_ordr", "s_gov", "s_liq", "s_soc"]
    pillar_labels = {
        "s_pdis": "Payment discipline",
        "s_ordr": "Allocation order",
        "s_gov":  "Governance",
        "s_liq":  "Liquidity / bidding",
        "s_soc":  "Social capital",
    }
    pil_df = df[pillars].mean().reset_index()
    pil_df.columns = ["pillar", "mean"]
    pil_df["label"] = pil_df["pillar"].map(pillar_labels)
    bar = (
        alt.Chart(pil_df)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=None, title="Pillar"),
            y=alt.Y("mean:Q", title="Mean score contribution",
                    scale=alt.Scale(domain=[0, 35])),
            color=alt.Color("label:N", legend=None),
            tooltip=["label", alt.Tooltip("mean:Q", format=".2f")],
        )
        .properties(height=240)
    )
    st.altair_chart(bar, use_container_width=True)

    st.subheader("Best and worst-scoring members")
    disp_cols = (["mid", "gid", "rtype", "score", "s_pdis", "s_ordr", "s_gov", "s_liq", "s_soc"]
                 + (["defaulted"] if "defaulted" in df.columns else [])
                 + ["true_pd", "p_ontime_raw"])
    st.caption("Top 10 by score")
    st.dataframe(df.nlargest(10, "score")[disp_cols], use_container_width=True, hide_index=True)
    st.caption("Bottom 10 by score")
    st.dataframe(df.nsmallest(10, "score")[disp_cols], use_container_width=True, hide_index=True)

    # ── Item 4 — Portfolio concentration guardrail ────────────────────────────
    st.divider()
    st.subheader("Portfolio concentration guardrail")
    rho_max_v       = float(vals.get("rho_max", 0.15))
    score_thresh_v  = float(vals.get("score_threshold", 40.0))
    st.caption(
        f"A group is **flagged** when it accounts for more than **{rho_max_v:.0%}** of all eligible "
        f"candidates (score ≥ {score_thresh_v:.0f}). Flagged groups should have new loan approvals "
        "paused until concentration falls back below the limit."
    )
    try:
        conc_df = check_portfolio_concentration(
            df, rho_max=rho_max_v, score_threshold=score_thresh_v
        )
        n_flagged = int(conc_df["flagged"].sum())
        n_eligible_total = int(conc_df["n_eligible"].sum())

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.metric("Groups flagged", f"{n_flagged} / {len(conc_df)}",
                      delta="⚠ review required" if n_flagged > 0 else "✓ all within limit",
                      delta_color="inverse" if n_flagged > 0 else "normal")
        with cc2:
            st.metric("Total eligible candidates", n_eligible_total,
                      help=f"Members with score ≥ {score_thresh_v:.0f}")
        with cc3:
            largest_share = float(conc_df["eligible_share"].max()) if len(conc_df) else 0.0
            st.metric("Largest group share", f"{largest_share:.1%}",
                      delta="above limit" if largest_share > rho_max_v else "within limit",
                      delta_color="inverse" if largest_share > rho_max_v else "normal")

        # Colour bar chart: flagged groups in red
        conc_plot = conc_df.copy()
        conc_plot["status"] = conc_plot["flagged"].map({True: "Flagged", False: "OK"})
        conc_bar = (
            alt.Chart(conc_plot)
            .mark_bar()
            .encode(
                x=alt.X("gid:N", sort="-y", title="Group"),
                y=alt.Y("eligible_share:Q", title="Share of eligible candidates",
                        axis=alt.Axis(format=".0%")),
                color=alt.Color("status:N",
                                scale=alt.Scale(domain=["Flagged", "OK"],
                                                range=["#ef4444", "#3b82f6"]),
                                legend=alt.Legend(title="Status")),
                tooltip=["gid", "n_members",
                         alt.Tooltip("n_eligible:Q",     title="Eligible"),
                         alt.Tooltip("eligible_share:Q", title="Share", format=".1%"),
                         "flagged"],
            )
            .properties(height=260,
                        title=f"Group share of eligible candidates  ·  limit = {rho_max_v:.0%}")
        )
        limit_line = (
            alt.Chart(pd.DataFrame({"y": [rho_max_v]}))
            .mark_rule(strokeDash=[4, 2], color="red")
            .encode(y="y:Q")
        )
        st.altair_chart(conc_bar + limit_line, use_container_width=True)

        with st.expander("Full concentration table"):
            st.dataframe(conc_df, use_container_width=True, hide_index=True)

    except Exception as _ce:
        st.warning(f"Portfolio concentration check failed: {_ce}")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Benchmark vs true PD (oracle)
# ──────────────────────────────────────────────────────────────────────────────
with tab_bench:
    v = result.validation
    st.caption(
        "The **true PD** is an unobservable oracle value computed directly from each member's "
        "underlying simulated parameters — not from the score. It serves as a ground-truth "
        "benchmark to confirm the score's direction is correct. "
        "In a real deployment you would use PD* from the main tab instead."
    )

    b1, b2, b3 = st.columns(3)
    with b1: st.metric("Spearman ρ vs true PD",        f"{v['spearman_rho']:+.4f}")
    with b2: st.metric("Score separation (true PD)",   f"{v['score_separation']:+.2f} pts")
    with b3: st.metric("Mean true PD",                 f"{v['true_pd_mean']:.2%}")

    st.subheader("Score by true-PD quintile (oracle reference)")
    qt = v["score_by_pd_quintile"].reset_index()
    qt.columns = ["PD Quintile", "Mean Score", "Std", "Members"]
    st.dataframe(qt, use_container_width=True, hide_index=True)

    sc = (
        alt.Chart(result.member_df)
        .mark_circle(opacity=0.45, size=40)
        .encode(
            x=alt.X("true_pd:Q", title="True PD (oracle — not observable in practice)"),
            y=alt.Y("score:Q",   title="Score", scale=alt.Scale(domain=[0, 100])),
            tooltip=["mid", "gid", alt.Tooltip("score:Q", format=".1f"),
                     alt.Tooltip("true_pd:Q", format=".3f"), "p_ontime_raw"],
        )
        .properties(height=360, title="Score vs true PD  ·  oracle benchmark only")
    )
    st.altair_chart(sc, use_container_width=True)

    if mc_df is not None:
        st.subheader("MC PD* (model-free estimate)")
        st.metric("Mean MC PD*", f"{mc_df['pd_star'].mean():.4f}")
        st.dataframe(mc_df.sort_values("pd_star", ascending=False).head(20),
                     use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — Raw data
# ──────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.subheader("Member-level scores")
    st.dataframe(result.member_df, use_container_width=True)
    if merged is not None:
        st.subheader("Combined — score + PD*")
        st.dataframe(merged.sort_values("pd_star", ascending=False), use_container_width=True)
    if mc_df is not None:
        st.subheader("MC PD* estimates")
        st.dataframe(mc_df.sort_values("pd_star", ascending=False), use_container_width=True)


# ── Footer diagnostics ────────────────────────────────────────────────────────
st.divider()
d1, d2, d3 = st.columns(3)
with d1: st.metric("Groups simulated",    int(vals["n_groups"]))
with d2: st.metric("Default threshold",   f"{int(vals['streak_threshold'])} missed meetings in a row")
with d3: st.metric("Random seed",         int(vals["seed"]))

with st.expander("What does each term mean?"):
    st.markdown("""
| Term | Meaning |
|---|---|
| **Score** | A number from 0 to ~85 summarising a member's creditworthiness based on their payment history, governance, and social factors. |
| **Defaulted** | A member who missed ≥ N consecutive meetings **after** receiving the pot. Their score is set to 0 as a hard penalty. |
| **PD*** | Estimated probability of default for each member, computed by fitting a logistic model on the 5 score pillars. |
| **Spearman ρ** | Measures how consistently a higher score corresponds to a lower PD*. +1 = perfect, 0 = no relationship. Values above +0.40 are strong. |
| **AUC** | How well the score separates defaulters from non-defaulters. 0.5 = random, 1.0 = perfect. |
| **PD* Quintile Q1–Q5** | Members split into 5 equal groups by PD*. Q1 = lowest risk (should have highest score). Q5 = highest risk (should have lowest score). |
| **On-time payment rate** | Share of meetings where a member pays on time. The single strongest driver of both the score and default risk. |
| **Logistic PD*** | A logistic regression is fitted on the 5 score pillars to predict binary default outcomes. The predicted probabilities are PD*. |
| **Credit stacking** | A member who belongs to multiple ROSCA groups simultaneously carries cross-group payment obligations. Their payment discipline score is reduced by lambda_stack per extra group. |
| **Portfolio concentration** | A group is flagged when it holds more than rho_max share of all eligible loan candidates. High concentration means a single group default could hit the entire portfolio. |
| **Prior default** | A member who defaulted in a previous ROSCA but did not disclose it. Detected indirectly via reputation signals; penalises the social capital score. |
| **Star topology** | A closed-loop group with no external connections, which increases Sybil/fraud risk. Penalises the governance score. |
""")
