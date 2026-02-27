import streamlit as st
import pandas as pd
import altair as alt

from clcs_simulator import CLCSParams, CLCSSimulator, DeterministicScenario
from clcs_interactive_runner import (
    parse_cashrun_plan,
    parse_general_shocks,
    parse_member_shocks,
    parse_p_by_member,
)

SCHEMA = {
    # Group
    "N": ("group", int, 10, "Number of members."),
    "num_cycles": ("group", int, 2, "Number of cycles."),
    "vesting_lag": ("group", int, 5, "Periods before deferred payouts vest."),

    # Cash flow
    "c": ("flow", float, 100.0, "Contribution per period."),
    "gamma": ("flow", float, 0.75, "Immediate payout fraction γ."),
    "delta": ("flow", float, 0.10, "Deferred payout fraction δ."),
    "Rb_annual": ("flow", float, 0.042, "Buffer interest rate."),
    "Re_annual": ("flow", float, 0.035, "Escrow interest rate."),
    "periods_per_year": ("flow", int, 12, "Periods per year."),

    # Rules
    "strict_cashrun": ("rules", bool, True, "Stop immediately if buffer < 0."),
    "enable_replacement": ("rules", bool, False, "Allow member replacement."),
    "replacement_delay": ("rules", int, 0, "Delay before replacement."),
    "probation_q": ("rules", int, 2, "Probation quarters."),
    "phi": ("rules", float, 0.0, "Platform fee rate."),
    "shrink_cap": ("rules", float, 2.0, "Max arrears allowed."),
    "init_t0_first_cycle": ("rules", bool, True, "Pre-fund at t=0."),

    # Simulation
    "payment_mode": ("sim", str, "mc_probpay", "Payment mode."),
    "p_base": ("sim", float, 1.0, "Base payment probability."),
    "seed": ("sim", int, 42, "Random seed."),

    # Shocks
    "p_by_member": ("shocks", str, "", "Override per-member payment probabilities."),
    "general_shocks": ("shocks", str, "", "Global shocks over time."),
    "member_shocks": ("shocks", str, "", "Member-specific shocks."),
    "cashrun_plan": ("shocks", str, "", "Planned cashrun events."),
}

DEFAULTS = {k: v[2] for k, v in SCHEMA.items()}

UI_GROUPS = {
    "Group structure": ["N", "num_cycles", "vesting_lag"],
    "Cash flow mechanics": ["c", "gamma", "delta", "Rb_annual", "Re_annual", "periods_per_year"],
    "Rules & discipline": [
        "strict_cashrun", "enable_replacement", "replacement_delay",
        "probation_q", "phi", "shrink_cap", "init_t0_first_cycle",
    ],
    "Simulation mode": ["payment_mode", "p_base", "seed"],
    "Shocks & overrides": ["p_by_member", "general_shocks", "member_shocks", "cashrun_plan"],
}

def build_params(vals):
    return CLCSParams(
        N=int(vals["N"]),
        c=float(vals["c"]),
        gamma=float(vals["gamma"]),
        delta=float(vals["delta"]),
        num_cycles=int(vals["num_cycles"]),
        vesting_lag=int(vals["vesting_lag"]),
        Rb_annual=float(vals["Rb_annual"]),
        Re_annual=float(vals["Re_annual"]),
        periods_per_year=int(vals["periods_per_year"]),
        phi=float(vals["phi"]),
        shrink_cap=float(vals["shrink_cap"]),
        enable_replacement=bool(vals["enable_replacement"]),
        replacement_delay=int(vals["replacement_delay"]),
        probation_q=int(vals["probation_q"]),
        strict_cashrun=bool(vals["strict_cashrun"]),
        init_t0_first_cycle=bool(vals["init_t0_first_cycle"]),
    )

def parse_shocks(vals):
    pbm = parse_p_by_member(vals["p_by_member"]) if vals["p_by_member"].strip() else None
    gs = parse_general_shocks(vals["general_shocks"]) if vals["general_shocks"].strip() else None
    ms = parse_member_shocks(vals["member_shocks"]) if vals["member_shocks"].strip() else None
    cp = parse_cashrun_plan(vals["cashrun_plan"]) if vals["cashrun_plan"].strip() else None
    return pbm, gs, ms, cp

st.set_page_config(page_title="CLCS Simulator", layout="wide")
st.title("CLCS Interactive Simulator")

preset = st.sidebar.selectbox(
    "Scenario preset",
    ["Default", "High stress", "Loose discipline", "Strict discipline"]
)

vals = {}

for group_name, keys in UI_GROUPS.items():
    st.sidebar.subheader(group_name)
    for k in keys:
        sec, typ, default, desc = SCHEMA[k]
        v0 = default
        if preset == "High stress" and k == "p_base":
            v0 = 0.85
        if preset == "Loose discipline" and k == "strict_cashrun":
            v0 = False
        if preset == "Strict discipline" and k == "shrink_cap":
            v0 = min(default, 1.0)

        if typ is int:
            vals[k] = st.sidebar.number_input(k, value=v0, step=1, help=desc)
        elif typ is float:
            vals[k] = st.sidebar.number_input(k, value=v0, step=0.01, format="%.4f", help=desc)
        elif typ is bool:
            vals[k] = st.sidebar.checkbox(k, value=v0, help=desc)
        else:
            vals[k] = st.sidebar.text_input(k, value=v0, help=desc)

if st.button("Run simulation"):
    p = build_params(vals)
    pbm, gs, ms, cp = parse_shocks(vals)

    N = p.N
    total_turns = N * p.num_cycles
    scen = DeterministicScenario(A_sched=[N] * total_turns)

    sim = CLCSSimulator(p)
    result = sim.run_path(
        scen,
        payment_mode=vals["payment_mode"],
        seed=int(vals["seed"]),
        p_base=float(vals["p_base"]),
        p_by_member=pbm,
        general_shocks=gs,
        member_shocks=ms,
        cashrun_plan=cp,
    )

    st.subheader("Key performance indicators")
    st.json(result.kpi)

    st.subheader("Buffer trajectory")
    st.line_chart(result.period_df["B_end"])

    st.subheader("Period data (last 50 rows)")
    st.dataframe(result.period_df.tail(50))

    st.subheader("Member data")
    st.dataframe(result.member_df)

with st.expander("Parameter glossary"):
    st.markdown("""
**Group structure**  
- **N**: number of members.  
- **num_cycles**: number of full rotations.  

**Cash flow mechanics**  
- **γ**: immediate payout fraction.  
- **δ**: deferred payout fraction.  

**Rules & discipline**  
- **strict_cashrun**: stop immediately if buffer < 0.  
- **shrink_cap**: maximum arrears allowed.  

**Simulation mode**  
- **p_base**: baseline payment probability.  

**Shocks**  
- **p_by_member**: override payment probabilities.  
- **general_shocks**: global shocks.  
- **member_shocks**: member-specific shocks.  
- **cashrun_plan**: planned cashrun events.
""")
