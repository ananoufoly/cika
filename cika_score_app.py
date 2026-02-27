import streamlit as st
import pandas as pd

from rosca_score_engine import (
    MacroEnvironment,
    PopulationParams,
    ScoreParams,
    generate_population,
)

# -------------------------------
# Parameter schema
# -------------------------------
SCHEMA = {
    "n_groups": ("pop", int, 20, "Number of groups"),
    "group_size_min": ("pop", int, 6, "Min members per group"),
    "group_size_max": ("pop", int, 20, "Max members per group"),
    "rtype_bidding_prob": ("pop", float, 0.50, "Prob group uses bidding [0-1]"),
    "rules_prob": ("pop", float, 0.75, "Prob group has formal rules [0-1]"),
    "p_ontime_mean": ("pop", float, 0.80, "Population avg on-time rate [0-1]"),
    "p_ontime_conc": ("pop", float, 9.0, "p_ontime concentration"),
    "post_slip_mean": ("pop", float, 0.08, "Post-payout slip tendency"),
    "bid_agg_mean": ("pop", float, 0.22, "Bid aggressiveness mean"),
    "p_rep": ("pop", float, 0.45, "P(repeat participation)"),
    "p_cent": ("pop", float, 0.30, "P(network centrality)"),
    "p_endf": ("pop", float, 0.25, "P(foreman endorsement)"),

    "stress_level": ("macro", float, 0.0, "Systemic stress [0-1]"),
    "within_group_corr": ("macro", float, 0.20, "Within-group shock correlation"),

    "a": ("score", float, 0.80, "Time-decay factor"),
    "c_otr": ("score", float, 0.85, "On-time sigmoid center"),
    "k_otr": ("score", float, 12.0, "On-time sigmoid slope"),
    "a_al": ("score", float, 0.70, "Avg-lateness penalty"),
    "a_ls": ("score", float, 0.60, "Late-streak penalty"),
    "a_slip": ("score", float, 0.80, "Post-payout slip enforcement"),
    "k_rules": ("score", float, 12.0, "Rules sigmoid slope"),
    "a_san": ("score", float, 0.60, "Sanction decay"),
    "q0": ("score", float, 0.50, "Bid centering"),
    "k_q": ("score", float, 10.0, "Bid slope"),
    "a_v": ("score", float, 0.80, "Bid volatility penalty"),
    "w_rep": ("score", float, 5.0, "Weight: repeat"),
    "w_cent": ("score", float, 4.0, "Weight: centrality"),
    "w_endf": ("score", float, 3.0, "Weight: foreman endorsement"),
    "w_ends": ("score", float, 3.0, "Weight: senior endorsement"),

    "seed": ("global", int, 42, "Random seed"),
}

DEFAULTS = {k: v[2] for k, v in SCHEMA.items()}

# -------------------------------
# Build config objects
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

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("ROSCA Credit Score Simulator")
st.caption("Population generator + scoring engine")

st.sidebar.header("Parameters")

vals = {}
sections = ["pop", "macro", "score", "global"]
section_titles = {
    "pop": "Population",
    "macro": "Macro Environment",
    "score": "Score Parameters",
    "global": "Global Settings",
}

for sec in sections:
    st.sidebar.subheader(section_titles[sec])
    for k, (s, typ, default, desc) in SCHEMA.items():
        if s != sec:
            continue
        if typ is int:
            vals[k] = st.sidebar.number_input(k, value=default, step=1)
        elif typ is float:
            vals[k] = st.sidebar.number_input(k, value=default, step=0.01, format="%.4f")
        else:
            vals[k] = st.sidebar.text_input(k, value=str(default))

# -------------------------------
# Run simulation
# -------------------------------
if st.button("Run simulation"):
    pop, macro, params = build_configs(vals)
    result = generate_population(pop, macro, params, seed=int(vals["seed"]))

    st.subheader("Validation Metrics")
    st.json(result.validation)

    st.subheader("Member-Level Data")
    st.dataframe(result.member_df)

# -------------------------------
# Sweep section
# -------------------------------
st.header("Parameter Sweep")

param = st.selectbox("Parameter", list(SCHEMA.keys()))
values = st.text_input("Values (comma-separated)", "0.2,0.5,0.8")

if st.button("Run sweep"):
    try:
        sweep_vals = [float(v.strip()) for v in values.split(",")]
    except:
        st.error("Invalid values")
        st.stop()

    rows = []
    for v in sweep_vals:
        test_vals = {**vals, param: v}
        pop, macro, params = build_configs(test_vals)
        result = generate_population(pop, macro, params, seed=int(vals["seed"]))
        val = result.validation
        rows.append({
            "value": v,
            "rho": val["spearman_rho"],
            "sep": val["score_separation"],
            "score_mean": val["score_mean"],
            "score_std": val["score_std"],
            "pd_mean": val["true_pd_mean"],
        })

    st.dataframe(pd.DataFrame(rows))
