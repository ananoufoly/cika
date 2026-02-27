import streamlit as st
import pandas as pd

from rosca_score_engine import (
    MacroEnvironment,
    PopulationParams,
    ScoreParams,
    generate_population,
)

# -------------------------------
# Parameter schema (with explanations)
# -------------------------------
SCHEMA = {
    # Population
    "n_groups": ("pop", int, 20, "Number of ROSCA groups in the simulated population."),
    "group_size_min": ("pop", int, 6, "Minimum number of members per group."),
    "group_size_max": ("pop", int, 20, "Maximum number of members per group."),
    "rtype_bidding_prob": ("pop", float, 0.50, "Probability a group uses bidding rather than fixed order."),
    "rules_prob": ("pop", float, 0.75, "Probability a group has formal written rules."),
    "p_ontime_mean": ("pop", float, 0.80, "Average on-time payment rate across members (0–1)."),
    "p_ontime_conc": ("pop", float, 9.0, "Concentration of on-time behavior (higher = less dispersion)."),
    "post_slip_mean": ("pop", float, 0.08, "Tendency to slip after receiving the pot (0–1)."),
    "bid_agg_mean": ("pop", float, 0.22, "Average aggressiveness in bidding for early pots (0–1)."),
    "p_rep": ("pop", float, 0.45, "Probability a member repeats participation in future cycles."),
    "p_cent": ("pop", float, 0.30, "Probability a member is network-central (well connected)."),
    "p_endf": ("pop", float, 0.25, "Probability a member has foreman endorsement."),

    # Macro
    "stress_level": ("macro", float, 0.0, "Systemic stress level (0 = calm, 1 = severe crisis)."),
    "within_group_corr": ("macro", float, 0.20, "Correlation of shocks within a group (0–1)."),

    # Score
    "a": ("score", float, 0.80, "Time-decay factor: how fast old behavior loses weight."),
    "c_otr": ("score", float, 0.85, "Center of the on-time sigmoid (typical on-time rate)."),
    "k_otr": ("score", float, 12.0, "Slope of the on-time sigmoid (steepness of response)."),
    "a_al": ("score", float, 0.70, "Penalty strength for average lateness."),
    "a_ls": ("score", float, 0.60, "Penalty strength for late streaks."),
    "a_slip": ("score", float, 0.80, "Penalty strength for post-payout slips."),
    "k_rules": ("score", float, 12.0, "Slope of the rules sigmoid (impact of formal rules)."),
    "a_san": ("score", float, 0.60, "Decay of sanction effects over time."),
    "q0": ("score", float, 0.50, "Bid centering: typical bid level as share of pot."),
    "k_q": ("score", float, 10.0, "Slope of bid response (how sharply bids affect score)."),
    "a_v": ("score", float, 0.80, "Penalty for volatile bidding behavior."),
    "w_rep": ("score", float, 5.0, "Weight of repeat participation in the score."),
    "w_cent": ("score", float, 4.0, "Weight of network centrality in the score."),
    "w_endf": ("score", float, 3.0, "Weight of foreman endorsement."),
    "w_ends": ("score", float, 3.0, "Weight of senior endorsement."),

    # Global
    "seed": ("global", int, 42, "Random seed for reproducibility."),
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
    "Macro environment": [
        "stress_level", "within_group_corr",
    ],
    "Scoring logic": [
        "a", "c_otr", "k_otr",
        "a_al", "a_ls", "a_slip",
        "k_rules", "a_san",
        "q0", "k_q", "a_v",
        "w_rep", "w_cent", "w_endf", "w_ends",
    ],
    "Randomness": [
        "seed",
    ],
}

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

st.set_page_config(page_title="ROSCA Score Simulator", layout="wide")
st.title("ROSCA Credit Score — Population Simulator")
st.caption("Simulate ROSCA populations and score behavior under different environments and score settings.")

# Presets
preset = st.sidebar.selectbox(
    "Scenario preset",
    ["Default", "High stress environment", "Low trust population", "Strict scoring"]
)

vals = {}

# Sidebar parameters
for group_name, keys in UI_GROUPS.items():
    st.sidebar.subheader(group_name)
    for k in keys:
        sec, typ, default, desc = SCHEMA[k]
        v0 = default
        # simple preset tweaks
        if preset == "High stress environment":
            if k == "stress_level":
                v0 = 0.7
        if preset == "Low trust population":
            if k == "p_ontime_mean":
                v0 = 0.6
        if preset == "Strict scoring":
            if k in ("a_al", "a_ls", "a_slip"):
                v0 = default * 1.3

        if typ is int:
            vals[k] = st.sidebar.number_input(k, value=v0, step=1, help=desc)
        elif typ is float:
            vals[k] = st.sidebar.number_input(k, value=v0, step=0.01, format="%.4f", help=desc)
        else:
            vals[k] = st.sidebar.text_input(k, value=str(v0), help=desc)

# Run simulation
if st.button("Run simulation"):
    pop, macro, params = build_configs(vals)
    result = generate_population(pop, macro, params, seed=int(vals["seed"]))

    st.subheader("Validation metrics")
    st.json(result.validation)

    st.subheader("Score distribution")
    st.histogram(result.member_df["score"])

    st.subheader("Member-level data")
    st.dataframe(result.member_df)

with st.expander("Parameter glossary"):
    st.markdown("""
**Population parameters**  
- **p_ontime_mean**: typical on-time payment rate. 0.8 means members pay on time 80% of the time on average.  
- **p_ontime_conc**: how concentrated that behavior is. Higher = most members close to the mean.

**Macro environment**  
- **stress_level**: captures macro shocks; higher values increase the chance of payment problems.  
- **within_group_corr**: how synchronized shocks are within a group.

**Scoring logic**  
- **a_al, a_ls, a_slip**: control how strongly lateness and slips reduce the score.  
- **w_rep, w_cent, w_endf, w_ends**: weights for relationship and reputation pillars.

**Randomness**  
- **seed**: change this to explore different random draws with the same parameters.
""")

