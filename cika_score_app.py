import streamlit as st
import pandas as pd
import altair as alt
from rosca_score_engine import (
    MacroEnvironment, PopulationParams, ScoreParams, generate_population
)

# Configuration Schema
SCHEMA = {
    "n_groups": ("pop", int, 20, "Number of groups."), "p_ontime_mean": ("pop", float, 0.80, "Avg on-time rate."),
    "stress_level": ("macro", float, 0.0, "Systemic stress."), "a": ("score", float, 0.80, "Time decay."),
    "seed": ("global", int, 42, "Seed.")
}
UI_GROUPS = {
    "Population": ["n_groups", "p_ontime_mean"], "Macro": ["stress_level"],
    "Scoring": ["a"], "Settings": ["seed"]
}

st.set_page_config(page_title="ROSCA Score App", layout="wide")
st.title("ROSCA Credit Score Dashboard — Sequence 2")

# Sidebar
vals = {}
for group_name, keys in UI_GROUPS.items():
    st.sidebar.subheader(group_name)
    for k in keys:
        sec, typ, default, desc = SCHEMA[k]
        vals[k] = st.sidebar.number_input(k, value=default, help=desc)

if st.sidebar.button("Run Simulation", type="primary"):
    pop = PopulationParams(n_groups=int(vals["n_groups"]), p_ontime_mean=vals["p_ontime_mean"])
    macro = MacroEnvironment(stress_level=vals["stress_level"])
    params = ScoreParams(a=vals["a"])
    
    result = generate_population(pop, macro, params, seed=int(vals["seed"]))
    df = result.member_df

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Hard Defaults", int(df["is_defaulter"].sum()))
    c2.metric("Spearman ρ*", f"{result.rho_star:.3f}")
    c3.metric("Avg Score", f"{df['score'].mean():.1f}")

    # Calibration Chart
    st.subheader("Pillar Importance (ML Calibrated)")
    st.bar_chart(pd.DataFrame(result.calibrated_weights.items(), columns=["Pillar", "Weight"]).set_index("Pillar"))

    # Scatter Plot
    st.subheader("Score vs Calibrated PD*")
    chart = alt.Chart(df).mark_circle().encode(
        x='score', y='true_pd_star', color='is_defaulter:N', tooltip=['score', 'true_pd_star']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    st.dataframe(df)