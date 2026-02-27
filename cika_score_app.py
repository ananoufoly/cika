import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

from rosca_score_engine import (
    MacroEnvironment,
    PopulationParams,
    ScoreParams,
    generate_population,
)

# ... (SCHEMA, DEFAULTS, and build_configs remain exactly as you provided) ...

st.set_page_config(page_title="ROSCA Score Simulator v2", layout="wide")
st.title("ROSCA Credit Score — Sequence 2 Simulator")
st.markdown("### Simulation with Hard Defaults & ML Calibration")

# --- Sidebar Configuration (using your existing logic) ---
st.sidebar.header("Scenario Configuration")
preset = st.sidebar.selectbox(
    "Scenario preset",
    ["Default", "High stress", "Low trust", "Strict scoring"]
)

vals = {}
for group_name, keys in UI_GROUPS.items():
    st.sidebar.subheader(group_name)
    for k in keys:
        sec, typ, default, desc = SCHEMA[k]
        v0 = default
        # (Your existing preset override logic)
        if preset == "High stress" and k == "stress_level": v0 = 0.7
        if preset == "Low trust" and k == "p_ontime_mean": v0 = 0.6
        if preset == "Strict scoring" and k in ("a_al", "a_ls", "a_slip"): v0 = default * 1.3

        if typ is int:
            vals[k] = st.sidebar.number_input(k, value=int(v0), step=1, help=desc)
        else:
            vals[k] = st.sidebar.number_input(k, value=float(v0), step=0.01, format="%.4f", help=desc)

# --- Execution ---
if st.button("Run Simulation & Sequence 2 Calibration", type="primary"):
    pop, macro, params = build_configs(vals)
    
    # Run the updated engine
    result = generate_population(pop, macro, params, seed=int(vals["seed"]))
    df = result.member_df
    weights = result.calibrated_weights
    
    # -------------------------------
    # 1. Top-Level Metrics
    # -------------------------------
    col1, col2, col3, col4 = st.columns(4)
    def_count = df["is_defaulter"].sum()
    def_rate = (def_count / len(df))
    
    col1.metric("Population Size", len(df))
    col2.metric("Hard Defaults", int(def_count), delta=f"{def_rate:.1%}", delta_color="inverse")
    col3.metric("Spearman ρ*", f"{result.rho_star:.3f}", help="Score vs 1-PD* (Calibrated)")
    col4.metric("Avg Score", f"{df['score'].mean():.1f}")

    st.divider()

    # -------------------------------
    # 2. Sequence 2: ML Calibration (Weights)
    # -------------------------------
    st.subheader("Pillar Importance (ML Calibration)")
    st.info("Large negative weights mean a higher pillar score significantly reduces the probability of default.")
    
    w_df = pd.DataFrame({
        "Pillar": list(weights.keys()),
        "Weight (Beta)": list(weights.values())
    })
    
    weight_chart = alt.Chart(w_df).mark_bar().encode(
        x=alt.X("Weight (Beta):Q"),
        y=alt.Y("Pillar:N", sort='x'),
        color=alt.condition(
            alt.datum["Weight (Beta)"] < 0,
            alt.value("#2ecc71"), # Green for risk reducers
            alt.value("#e74c3c")  # Red for risk indicators
        )
    ).properties(height=300)
    
    st.altair_chart(weight_chart, use_container_width=True)

    # -------------------------------
    # 3. Model Comparison: Oracle vs ML PD*
    # -------------------------------
    st.subheader("Model Comparison: Oracle vs. Calibrated ML")
    
    scatter = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X('true_pd_oracle:Q', title='Oracle PD (Expert Guess)'),
        y=alt.X('true_pd_star:Q', title='Calibrated PD* (Actual Defaults)'),
        color=alt.Color('is_defaulter:N', scale=alt.Scale(domain=[0, 1], range=['#3498db', '#e74c3c'])),
        tooltip=['mid', 'score', 'is_defaulter']
    ).interactive()
    
    st.altair_chart(scatter, use_container_width=True)

    # -------------------------------
    # 4. Distribution and Raw Data
    # -------------------------------
    st.subheader("Score Distribution (Post-Penalty)")
    
    # Color bars based on default status
    hist = alt.Chart(df).mark_bar().encode(
        alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Credit Score"),
        y='count()',
        color=alt.Color('is_defaulter:N', title="Defaulted", scale=alt.Scale(range=['#27ae60', '#c0392b']))
    )
    st.altair_chart(hist, use_container_width=True)

    st.subheader("Member-Level Inspection")
    # Styling defaulters in the table
    st.dataframe(df.style.background_gradient(subset=['score'], cmap='RdYlGn')
                         .background_gradient(subset=['true_pd_star'], cmap='YlOrRd'))

# --- Glossary update ---
with st.expander("Sequence 2 Glossary"):
    st.markdown("""
### The Hard Default Cliff
A member is flagged as `is_defaulter` if they miss **3 consecutive payments** after receiving the payout. Their score is immediately zeroed out.

### Calibrated PD* vs Oracle PD
- **Oracle PD**: The hidden "truth" defined by initial parameters.
- **Calibrated PD***: The probability of default calculated by a Logistic Regression trained on the *actual* default events that happened during this specific simulation run.
- **Pillar Weights**: The ML model's interpretation of which pillar is the best "snitch" for predicting a 3-month default.
""")