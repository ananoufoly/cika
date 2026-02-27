import streamlit as st
import pandas as pd

from clcs_simulator import CLCSParams, CLCSSimulator, DeterministicScenario, pretty_params
from clcs_interactive_runner import (
    parse_cashrun_plan,
    parse_general_shocks,
    parse_member_shocks,
    parse_p_by_member,
)

# -------------------------------
# Parameter schema
# -------------------------------
SCHEMA = {
    "N": ("group", int, 10, "Number of members"),
    "num_cycles": ("group", int, 2, "Number of cycles"),
    "vesting_lag": ("group", int, 5, "Vesting lag K"),

    "c": ("flow", float, 100.0, "Contribution per period"),
    "gamma": ("flow", float, 0.75, "Immediate payout fraction γ"),
    "delta": ("flow", float, 0.10, "Deferred payout fraction δ"),
    "Rb_annual": ("flow", float, 0.042, "Buffer annual interest rate"),
    "Re_annual": ("flow", float, 0.035, "Escrow annual interest rate"),
    "periods_per_year": ("flow", int, 12, "Periods per year"),

    "strict_cashrun": ("rules", bool, True, "Strict cashrun"),
    "enable_replacement": ("rules", bool, False, "Enable replacement"),
    "replacement_delay": ("rules", int, 0, "Replacement delay"),
    "probation_q": ("rules", int, 2, "Probation quarters"),
    "phi": ("rules", float, 0.0, "Platform fee rate φ"),
    "shrink_cap": ("rules", float, 2.0, "Arrears shrink cap"),
    "init_t0_first_cycle": ("rules", bool, True, "Pre-fund at t=0"),

    "payment_mode": ("sim", str, "mc_probpay", "Payment mode"),
    "p_base": ("sim", float, 1.0, "Base payment probability"),
    "seed": ("sim", int, 42, "Random seed"),

    "p_by_member": ("shocks", str, "", "id:p, id:p"),
    "general_shocks": ("shocks", str, "", "t0-t1:mult"),
    "member_shocks": ("shocks", str, "", "id:t0-t1:p"),
    "cashrun_plan": ("shocks", str, "", "id:cycle;id:cycle"),
}

DEFAULTS = {k: v[2] for k, v in SCHEMA.items()}

# -------------------------------
# Build CLCSParams
# -------------------------------
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

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("CLCS Interactive Simulator")
st.caption("Group cash-flow engine with shocks and overrides")

vals = {}

st.sidebar.header("Parameters")

sections = ["group", "flow", "rules", "sim", "shocks"]
section_titles = {
    "group": "Group Setup",
    "flow": "Cash Flow",
    "rules": "Rules",
    "sim": "Simulation",
    "shocks": "Shocks & Overrides",
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
        elif typ is bool:
            vals[k] = st.sidebar.checkbox(k, value=default)
        else:
            vals[k] = st.sidebar.text_input(k, value=default)

# -------------------------------
# Run simulation
# -------------------------------
if st.button("Run simulation"):
    try:
        p = build_params(vals)
    except Exception as e:
        st.error(f"Invalid parameters: {e}")
        st.stop()

    try:
        pbm, gs, ms, cp = parse_shocks(vals)
    except Exception as e:
        st.error(f"Shock parse error: {e}")
        st.stop()

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

    st.subheader("KPI")
    st.json(result.kpi)

    st.subheader("Period Data (last 50 rows)")
    st.dataframe(result.period_df.tail(50))

    st.subheader("Member Data")
    st.dataframe(result.member_df)

    st.subheader("Buffer Trajectory")
    st.line_chart(result.period_df["B_end"])
