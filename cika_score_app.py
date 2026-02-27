# rosca_score_app_extended.py
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
)

# Optional sklearn metrics
try:
    from sklearn.metrics import roc_auc_score, brier_score_loss, calibration_curve
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -------------------------------
# Schema and UI groups
# -------------------------------
SCHEMA = {
    # Population
    "n_groups": ("pop", int, 20, "Number of ROSCA groups simulated."),
    "group_size_min": ("pop", int, 6, "Minimum members per group."),
    "group_size_max": ("pop", int, 20, "Maximum members per group."),
    "rtype_bidding_prob": ("pop", float, 0.50, "Probability a group uses bidding."),
    "rules_prob": ("pop", float, 0.75, "Probability a group has formal rules."),
    "p_ontime_mean": ("pop", float, 0.80, "Average on-time payment rate."),
    "p_ontime_conc": ("pop", float, 9.0, "Concentration of on-time behavior."),
    "post_slip_mean": ("pop", float, 0.08, "Tendency to slip after receiving the pot."),
    "bid_agg_mean": ("pop", float, 0.22, "Aggressiveness in bidding."),
    "p_rep": ("pop", float, 0.45, "Probability of repeat participation."),
    "p_cent": ("pop", float, 0.30, "Probability of network centrality."),
    "p_endf": ("pop", float, 0.25, "Probability of foreman endorsement."),

    # Macro
    "stress_level": ("macro", float, 0.0, "Systemic stress level (0–1)."),
    "within_group_corr": ("macro", float, 0.20, "Correlation of shocks within a group."),

    # Score
    "a": ("score", float, 0.80, "Time-decay factor."),
    "c_otr": ("score", float, 0.85, "On-time sigmoid center."),
    "k_otr": ("score", float, 12.0, "On-time sigmoid slope."),
    "a_al": ("score", float, 0.70, "Penalty for average lateness."),
    "a_ls": ("score", float, 0.60, "Penalty for late streaks."),
    "a_slip": ("score", float, 0.80, "Penalty for post-payout slips."),
    "k_rules": ("score", float, 12.0, "Rules sigmoid slope."),
    "a_san": ("score", float, 0.60, "Sanction decay."),
    "q0": ("score", float, 0.50, "Bid centering."),
    "k_q": ("score", float, 10.0, "Bid slope."),
    "a_v": ("score", float, 0.80, "Bid volatility penalty."),
    "w_rep": ("score", float, 5.0, "Weight: repeat participation."),
    "w_cent": ("score", float, 4.0, "Weight: centrality."),
    "w_endf": ("score", float, 3.0, "Weight: foreman endorsement."),
    "w_ends": ("score", float, 3.0, "Weight: senior endorsement."),

    # Defaults / PD*
    "streak_threshold": ("pdstar", int, 3, "Missed meetings after allocation → default threshold"),
    "mc_runs": ("pdstar", int, 200, "Monte Carlo runs for PD* estimation"),

    # Global
    "seed": ("global", int, 42, "Random seed."),
}

DEFAULTS = {k: v[2] for k, v in SCHEMA.items()}

UI_GROUPS = {
    "Population structure & behavior": [
        "n_groups", "group_size_min", "group_size_max",
        "rtype_bidding_prob", "rules_prob",
        "p_ontime_mean", "p_ontime_conc",
        "post_slip_mean", "bid_agg_mean",
        "p_rep", "p_cent", "p_endf",
    ],
    "Macro environment": ["stress_level", "within_group_corr"],
    "Scoring logic": [
        "a", "c_otr", "k_otr",
        "a_al", "a_ls", "a_slip",
        "k_rules", "a_san",
        "q0", "k_q", "a_v",
        "w_rep", "w_cent", "w_endf", "w_ends",
    ],
    "PD* and defaults": ["streak_threshold", "mc_runs"],
    "Randomness": ["seed"],
}

# -------------------------------
# Helpers
# -------------------------------
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
    )
    macro = MacroEnvironment(
        stress_level=float(vals["stress_level"]),
        within_group_corr=float(vals["within_group_corr"]),
    )
    score_kwargs = {k: float(vals[k]) for k in SCHEMA if SCHEMA[k][0] == "score"}
    params = ScoreParams(**score_kwargs)
    return pop, macro, params

def hist_chart(df, col, bins=30, title=None):
    chart = alt.Chart(df).mark_bar().encode(
        alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins)),
        y='count()',
        tooltip=[alt.Tooltip(f"{col}:Q", format=".2f"), alt.Tooltip("count()", title="count")]
    ).properties(height=240, title=title)
    return chart

def calibration_plot(y_true, y_prob, n_bins=10, title="Calibration"):
    if not SKLEARN_AVAILABLE:
        return None
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    df = pd.DataFrame({"pred": prob_pred, "true": prob_true})
    base = alt.Chart(df).mark_line(point=True).encode(x="pred:Q", y="true:Q").properties(height=240, title=title)
    line = alt.Chart(pd.DataFrame({"x":[0,1]})).mark_line(strokeDash=[4,2]).encode(x="x:Q", y="x:Q")
    return base + line

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="ROSCA Score Simulator — Extended", layout="wide")
st.title("ROSCA Credit Score — Population Simulator (Extended)")
st.caption("Includes explicit default rule and PD* estimation (MC frequency + logistic).")

# Sidebar: presets and parameters
preset = st.sidebar.selectbox("Scenario preset", ["Default", "High stress", "Low trust", "Strict scoring"])
vals = {}

for group_name, keys in UI_GROUPS.items():
    st.sidebar.subheader(group_name)
    for k in keys:
        sec, typ, default, desc = SCHEMA[k]
        v0 = default
        if preset == "High stress" and k == "stress_level":
            v0 = 0.7
        if preset == "Low trust" and k == "p_ontime_mean":
            v0 = 0.6
        if preset == "Strict scoring" and k in ("a_al", "a_ls", "a_slip"):
            v0 = default * 1.3

        if typ is int:
            vals[k] = st.sidebar.number_input(k, value=v0, step=1, help=desc)
        elif typ is float:
            vals[k] = st.sidebar.number_input(k, value=v0, step=0.01, format="%.4f", help=desc)
        else:
            vals[k] = st.sidebar.text_input(k, value=str(v0), help=desc)

# Main controls
col1, col2, col3 = st.columns([1,1,1])
with col1:
    run_btn = st.button("Run simulation")
with col2:
    run_defaults_btn = st.button("Run simulation (with defaults)")
with col3:
    run_mc_btn = st.button("Run MC PD*")

# Storage in session state
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_mc" not in st.session_state:
    st.session_state["last_mc"] = None
if "last_logit" not in st.session_state:
    st.session_state["last_logit"] = None

# -------------------------------
# Run simulation (score only)
# -------------------------------
if run_btn:
    pop, macro, params = build_configs(vals)
    with st.spinner("Simulating population and computing scores…"):
        result = generate_population(pop, macro, params, seed=int(vals["seed"]))
    st.session_state["last_result"] = result
    st.success("Simulation complete — scores computed.")

# -------------------------------
# Run simulation with defaults (single run)
# -------------------------------
if run_defaults_btn:
    pop, macro, params = build_configs(vals)
    streak = int(vals.get("streak_threshold", 3))
    with st.spinner("Simulating population and detecting defaults…"):
        result = generate_population_with_defaults(pop, macro, params, seed=int(vals["seed"]), streak_threshold=streak)
    st.session_state["last_result"] = result
    st.session_state["last_stacked_single_run"] = result.member_df.copy()
    st.success("Simulation complete — defaults attached to member table.")

# -------------------------------
# Run Monte Carlo PD*
# -------------------------------
if run_mc_btn:
    pop, macro, params = build_configs(vals)
    n_runs = int(vals.get("mc_runs", 200))
    streak = int(vals.get("streak_threshold", 3))
    with st.spinner(f"Running MC PD* ({n_runs} runs) — this may take a while…"):
        mc_df = compute_pd_star_mc(pop, macro, params, n_runs=n_runs, base_seed=int(vals["seed"]), K_min=6, streak_threshold=streak)
    st.session_state["last_mc"] = mc_df
    st.success(f"MC PD* complete — {len(mc_df)} members, mean PD*={mc_df['pd_star'].mean():.4f}")

# -------------------------------
# Fit logistic PD* (button)
# -------------------------------
if st.sidebar.button("Fit logistic PD* (on stacked runs)"):
    # prefer stacked single-run if available, else require MC
    if "last_stacked_single_run" in st.session_state and st.session_state["last_stacked_single_run"] is not None:
        stacked = st.session_state["last_stacked_single_run"]
        if "default" not in stacked.columns:
            st.error("Single-run member table does not contain 'default'. Run 'Run simulation (with defaults)' first.")
        else:
            try:
                model, df_out = fit_logistic_pd_star(stacked)
                st.session_state["last_logit"] = df_out
                st.success("Logistic PD* fitted on single-run defaults.")
            except Exception as e:
                st.error(f"Logistic fit failed: {e}")
    elif st.session_state.get("last_mc") is not None:
        mc = st.session_state["last_mc"]
        # approximate stacked dataset by sampling per-member binary outcomes (small sample)
        rows = []
        n_runs = int(vals.get("mc_runs", 200))
        for _, r in mc.iterrows():
            p = r["pd_star"]
            draws = min(200, n_runs)
            sampled = np.random.binomial(1, p, size=draws)
            for s in sampled:
                rows.append({"mid": r["mid"], "p_ontime_raw": r["p_ontime_raw"], "true_pd": r["true_pd"], "default": int(s)})
        stacked = pd.DataFrame(rows)
        try:
            model, df_out = fit_logistic_pd_star(stacked)
            st.session_state["last_logit"] = df_out
            st.success("Logistic PD* fitted on sampled MC stacked dataset.")
        except Exception as e:
            st.error(f"Logistic fit failed: {e}")
    else:
        st.error("No data available to fit logistic. Run 'Run simulation (with defaults)' or 'Run MC PD*' first.")

# -------------------------------
# Display results panels
# -------------------------------
result = st.session_state.get("last_result")
mc_df = st.session_state.get("last_mc")
logit_df = st.session_state.get("last_logit")

if result is not None:
    st.header("Validation metrics and score overview")
    v = result.validation
    col1, col2 = st.columns([1,2])
    with col1:
        st.metric("Spearman ρ", f"{v['spearman_rho']:+.4f}")
        st.metric("Score separation", f"{v['score_separation']:+.2f}")
        st.metric("Mean score", f"{v['score_mean']:.2f}")
        st.metric("Mean true PD", f"{v['true_pd_mean']:.2%}")
    with col2:
        st.subheader("Score by true-PD quintile")
        qt = v["score_by_pd_quintile"].reset_index().rename(columns={"_pd_decile":"pd_decile"})
        st.dataframe(qt)

    st.subheader("Score distribution")
    st.altair_chart(hist_chart(result.member_df, "score", bins=30, title="Score histogram"), use_container_width=True)

    st.subheader("Pillar utilisation (mean)")
    pillars = ["s_pdis","s_ordr","s_gov","s_liq","s_soc"]
    pil_df = result.member_df[pillars].mean().reset_index()
    pil_df.columns = ["pillar","mean"]
    bar = alt.Chart(pil_df).mark_bar().encode(x="pillar:N", y="mean:Q", tooltip=["pillar","mean"]).properties(height=200)
    st.altair_chart(bar, use_container_width=True)

    st.subheader("Top 10 by score")
    st.dataframe(result.member_df.sort_values("score", ascending=False).head(10))

    st.subheader("Member-level table (first 200 rows)")
    st.dataframe(result.member_df.head(200))

# -------------------------------
# Display MC PD* results
# -------------------------------
if mc_df is not None:
    st.header("Monte Carlo PD* (frequency) results")
    st.metric("Mean PD*", f"{mc_df['pd_star'].mean():.4f}")
    st.subheader("PD* distribution")
    st.altair_chart(hist_chart(mc_df, "pd_star", bins=30, title="PD* histogram"), use_container_width=True)

    st.subheader("Top 20 by PD*")
    st.dataframe(mc_df.sort_values("pd_star", ascending=False).head(20))

# -------------------------------
# Display logistic PD* results
# -------------------------------
if logit_df is not None:
    st.header("Logistic PD* predictions (fitted probabilities)")
    st.subheader("Predicted PD* (sample)")
    st.dataframe(logit_df[["pd_star_logit"]].head(200))

    if SKLEARN_AVAILABLE:
        # evaluation: if we have a base result with scores, map predictions to members and compute metrics
        if result is not None:
            map_probs = logit_df.set_index("mid")["pd_star_logit"].to_dict()
            member_df = result.member_df.copy()
            member_df["pd_star_logit"] = member_df["mid"].map(map_probs).fillna(0.0)
            # normalize score to [0,1]
            smin, smax = member_df["score"].min(), member_df["score"].max()
            if smax > smin:
                member_df["score_norm"] = (member_df["score"] - smin) / (smax - smin)
            else:
                member_df["score_norm"] = 0.0
            try:
                auc = roc_auc_score(member_df["pd_star_logit"], member_df["score_norm"])
            except Exception:
                auc = float("nan")
            try:
                brier = brier_score_loss(member_df["pd_star_logit"], member_df["score_norm"])
            except Exception:
                brier = float("nan")
            st.metric("AUC (score → PD*)", f"{auc:.4f}")
            st.metric("Brier (score norm)", f"{brier:.4f}")

            st.subheader("Calibration plot (PD* logistic)")
            cal = calibration_plot(member_df["pd_star_logit"], member_df["pd_star_logit"])
            if cal is not None:
                st.altair_chart(cal, use_container_width=True)
    else:
        st.info("scikit-learn not available: AUC / calibration require sklearn.")

# -------------------------------
# Additional diagnostics and quick checks
# -------------------------------
st.header("Quick diagnostics")
diag_cols = st.columns(3)
with diag_cols[0]:
    st.write("MC runs")
    st.write(int(vals.get("mc_runs", 200)))
with diag_cols[1]:
    st.write("Default streak threshold")
    st.write(int(vals.get("streak_threshold", 3)))
with diag_cols[2]:
    st.write("Last simulation seed")
    st.write(int(vals.get("seed", 42)))

with st.expander("Parameter glossary"):
    st.markdown("""
**p_ontime_mean** — typical on-time payment rate (0–1).  
**p_ontime_conc** — concentration: higher = members cluster near the mean.  
**stress_level** — macro stress; higher reduces effective on-time probabilities.  
**a_al, a_ls, a_slip** — scoring penalties for lateness, streaks, and post-payout slip.  
**streak_threshold** — number of consecutive missed meetings after allocation that triggers a default.  
**mc_runs** — number of Monte Carlo re-simulations used to estimate PD* frequency.
""")

# Closing question to advance the user's goal
st.markdown("---")
st.write("Which PD* output would you like to explore next: **MC frequency table**, **logistic predictions**, or **calibration / ROC** for a chosen preset?")

