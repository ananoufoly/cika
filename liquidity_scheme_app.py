"""liquidity_scheme_app.py — CLCS Streamlit application (redesigned).

Two tabs:
  • Simulate  – run a single scenario, see KPI cards + Plotly charts
  • Optimize  – three Monte Carlo workflows to find optimal group design
  • Glossary  – plain-English definitions

Requires: streamlit, pandas, numpy, plotly, clcs_simulator, clcs_interactive_runner,
          clcs_design_optimizer
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from clcs_simulator import CLCSParams, CLCSSimulator, DeterministicScenario
from clcs_interactive_runner import (
    parse_cashrun_plan,
    parse_general_shocks,
    parse_member_shocks,
    parse_p_by_member,
)
from clcs_design_optimizer import mc_safety, gamma_max_bisect, compute_K

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CLCS — Rotating Savings Simulator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Defaults & presets
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "N": 10,
    "num_cycles": 2,
    "vesting_lag": 5,
    "c": 100.0,
    "gamma": 0.75,
    "delta": 0.10,
    "Rb_annual": 0.042,
    "Re_annual": 0.035,
    "periods_per_year": 12,
    "strict_cashrun": True,
    "enable_replacement": False,
    "replacement_delay": 0,
    "probation_q": 2,
    "phi": 0.0,
    "shrink_cap": 2.0,
    "init_t0_first_cycle": True,
    "payment_mode": "mc_probpay",
    "p_base": 1.0,
    "seed": 42,
    "p_by_member": "",
    "general_shocks": "",
    "member_shocks": "",
    "cashrun_plan": "",
}

PRESETS: dict[str, dict] = {
    "Default": {},
    "High stress — unreliable payers (p=82%)": {"p_base": 0.82},
    "Loose discipline — no cashrun halt": {"strict_cashrun": False, "shrink_cap": 3.0},
    "Strict discipline — tight rules": {
        "strict_cashrun": True, "shrink_cap": 1.0, "probation_q": 4,
    },
    "Large group — 20 members, 2 cycles": {"N": 20, "num_cycles": 2},
}

PPY_OPTIONS = [12, 26, 52, 24, 4, 1]
PPY_LABELS  = {12: "Monthly (12)", 26: "Bi-weekly (26)", 52: "Weekly (52)",
               24: "Twice-monthly (24)", 4: "Quarterly (4)", 1: "Annual (1)"}
PM_OPTIONS  = ["mc_probpay", "deterministic", "mc_fixedA"]
PM_LABELS   = {
    "mc_probpay":   "Probabilistic (each member pays with probability p_base)",
    "deterministic": "Deterministic (all active members always pay)",
    "mc_fixedA":    "Fixed count (exactly A members pay, chosen randomly)",
}


def get_preset(name: str) -> dict:
    v = dict(DEFAULTS)
    v.update(PRESETS.get(name, {}))
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — all simulation parameters
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Simulation Parameters")

    preset_name = st.selectbox(
        "Quick-load a preset scenario",
        list(PRESETS.keys()),
        help="Presets apply default values for common situations. You can still adjust any parameter below.",
    )
    pv = get_preset(preset_name)

    st.divider()

    # ── Group & duration ────────────────────────────────────────────────────
    with st.expander("👥 Group & Duration", expanded=True):
        N = st.number_input(
            "Number of members (N)",
            min_value=2, max_value=200, value=int(pv["N"]), step=1,
            help="Total members in the group. Each member receives a payout once per cycle.",
        )
        num_cycles = st.number_input(
            "Number of full rotations (cycles)",
            min_value=1, max_value=10, value=int(pv["num_cycles"]), step=1,
            help=(
                "A cycle is N periods. With 2 cycles each member receives the pot twice. "
                "Total simulation length = N × cycles periods."
            ),
        )
        vesting_lag = st.number_input(
            "Vesting delay — K (periods)",
            min_value=0, max_value=100, value=int(pv["vesting_lag"]), step=1,
            help=(
                "After receiving a turn, the member's deferred (escrow) portion is locked for K periods "
                "before it is released. Longer delay = more buffer stability."
            ),
        )

    # ── Contributions & payouts ─────────────────────────────────────────────
    with st.expander("💵 Contributions & Payouts", expanded=True):
        c = st.number_input(
            "Contribution per period (c)",
            min_value=1.0, value=float(pv["c"]), step=10.0, format="%.2f",
            help="Amount each member contributes every period (e.g. monthly). Pool = N × c.",
        )
        gamma_pct = st.slider(
            "Immediate payout share — γ (%)",
            min_value=5, max_value=90, value=int(round(pv["gamma"] * 100)), step=1,
            help=(
                "Fraction of the pool paid out immediately to the current beneficiary. "
                "Higher γ = bigger upfront payout, less buffer."
            ),
        )
        delta_pct = st.slider(
            "Deferred payout share — δ (%)",
            min_value=1, max_value=50, value=int(round(pv["delta"] * 100)), step=1,
            help=(
                "Fraction credited to the beneficiary's escrow account, "
                "paid out after the vesting delay K. Acts like a forced savings bonus."
            ),
        )

        gamma = gamma_pct / 100.0
        delta = delta_pct / 100.0
        buffer_share = 1.0 - gamma - delta
        pool = N * c

        if buffer_share < 0:
            st.error(f"⚠ γ ({gamma_pct}%) + δ ({delta_pct}%) exceeds 100%. Reduce one of them.")
        else:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Pool / period", f"{pool:,.0f}")
            col_b.metric("Buffer share S", f"{buffer_share*100:.0f}%")
            col_c.metric("Immediate payout", f"{gamma * pool:,.0f}")

    # ── Interest rates ───────────────────────────────────────────────────────
    with st.expander("📈 Interest Rates"):
        Rb_annual = st.number_input(
            "Buffer interest rate — annual (%)",
            min_value=0.0, max_value=0.5, value=float(pv["Rb_annual"]),
            step=0.001, format="%.3f",
            help="Annual return earned on the shared buffer balance. Compounds each period.",
        )
        Re_annual = st.number_input(
            "Escrow interest rate — annual (%)",
            min_value=0.0, max_value=0.5, value=float(pv["Re_annual"]),
            step=0.001, format="%.3f",
            help="Annual return earned on members' individual escrow balances.",
        )
        ppy_idx = PPY_OPTIONS.index(pv["periods_per_year"]) if pv["periods_per_year"] in PPY_OPTIONS else 0
        periods_per_year = st.selectbox(
            "Period frequency",
            PPY_OPTIONS,
            index=ppy_idx,
            format_func=lambda x: PPY_LABELS[x],
            help="Sets how many periods occur per year, used to convert annual rates to per-period rates.",
        )

    # ── Group rules & discipline ─────────────────────────────────────────────
    with st.expander("🔒 Group Rules & Discipline"):
        strict_cashrun = st.checkbox(
            "Halt group if buffer goes negative (strict cashrun)",
            value=bool(pv["strict_cashrun"]),
            help=(
                "If enabled, the simulation stops the moment the shared buffer drops below zero. "
                "Reflects a real group that cannot cover a payout."
            ),
        )
        init_t0_first_cycle = st.checkbox(
            "Collect contributions before first payout (pre-fund)",
            value=bool(pv["init_t0_first_cycle"]),
            help="At t=0, contributions are collected to seed the buffer before any member receives.",
        )
        shrink_cap = st.number_input(
            "Max arrears deducted from payout (× c)",
            min_value=0.0, max_value=10.0, value=float(pv["shrink_cap"]),
            step=0.5, format="%.1f",
            help=(
                "When a member receives the pot, any missed contributions (arrears) are deducted. "
                "This cap limits how much can be deducted. E.g. 2.0 = at most 2 × c is withheld."
            ),
        )
        phi = st.number_input(
            "Platform fee (fraction of interest income)",
            min_value=0.0, max_value=0.5, value=float(pv["phi"]),
            step=0.01, format="%.3f",
            help="Fraction of interest income taken as a platform/operator fee each period.",
        )
        probation_q = st.number_input(
            "New-member probation (periods)",
            min_value=0, max_value=20, value=int(pv["probation_q"]), step=1,
            help=(
                "A new or replacement member must pay for this many periods "
                "before they become eligible for a payout turn."
            ),
        )
        enable_replacement = st.checkbox(
            "Allow member replacement after exclusion",
            value=bool(pv["enable_replacement"]),
            help="If a member is expelled (cashrun/miss streak), a new member can fill the slot.",
        )
        replacement_delay = 0
        if enable_replacement:
            replacement_delay = st.number_input(
                "Replacement delay (periods)",
                min_value=0, max_value=50, value=int(pv["replacement_delay"]), step=1,
                help="How many periods pass between a member's exclusion and the replacement joining.",
            )

    # ── Simulation mode ──────────────────────────────────────────────────────
    with st.expander("🎲 Payment Simulation Mode"):
        pm_idx = PM_OPTIONS.index(pv["payment_mode"]) if pv["payment_mode"] in PM_OPTIONS else 0
        payment_mode = st.selectbox(
            "How member payments are simulated",
            PM_OPTIONS,
            index=pm_idx,
            format_func=lambda x: PM_LABELS[x],
        )
        p_base = st.slider(
            "Baseline payment reliability (p_base)",
            min_value=0.50, max_value=1.0, value=float(pv["p_base"]), step=0.01,
            help=(
                "In probabilistic mode, each member independently pays with this probability each period. "
                "1.0 = everyone always pays. 0.9 = 10% chance of missing each period."
            ),
        )
        seed = st.number_input(
            "Random seed",
            min_value=0, value=int(pv["seed"]), step=1,
            help="Fix this for reproducible results. Change it to explore different random outcomes.",
        )

    # ── Advanced: stress scenarios ───────────────────────────────────────────
    with st.expander("⚡ Stress Scenarios (Advanced)"):
        st.caption("Leave blank to skip. Hover the ⓘ for format.")
        p_by_member = st.text_input(
            "Per-member payment probability overrides",
            value=str(pv["p_by_member"]),
            help='Format: "id:prob, id:prob"  →  e.g. "3:0.7, 7:0.5" (member 3 pays 70% of the time).',
        )
        general_shocks = st.text_input(
            "Global payment shocks (all members, time window)",
            value=str(pv["general_shocks"]),
            help='Format: "t0-t1:mult, ..."  →  e.g. "10-20:0.8" reduces all payment probs by 20% from period 10 to 20.',
        )
        member_shocks = st.text_input(
            "Member-specific payment shocks",
            value=str(pv["member_shocks"]),
            help='Format: "id:t0-t1:prob, ..."  →  e.g. "7:15-18:0" makes member 7 miss all payments from period 15 to 18.',
        )
        cashrun_plan = st.text_input(
            "Forced cashrun events",
            value=str(pv["cashrun_plan"]),
            help='Format: "id:cycle,cycle; ..."  →  force cashrun events for specific members in specific cycles.',
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_params(v: dict) -> CLCSParams:
    return CLCSParams(
        N=int(v["N"]), c=float(v["c"]),
        gamma=float(v["gamma"]), delta=float(v["delta"]),
        num_cycles=int(v["num_cycles"]), vesting_lag=int(v["vesting_lag"]),
        Rb_annual=float(v["Rb_annual"]), Re_annual=float(v["Re_annual"]),
        periods_per_year=int(v["periods_per_year"]),
        phi=float(v["phi"]), shrink_cap=float(v["shrink_cap"]),
        enable_replacement=bool(v["enable_replacement"]),
        replacement_delay=int(v["replacement_delay"]),
        probation_q=int(v["probation_q"]),
        strict_cashrun=bool(v["strict_cashrun"]),
        init_t0_first_cycle=bool(v["init_t0_first_cycle"]),
    )


def parse_shocks(v: dict):
    pbm = parse_p_by_member(v["p_by_member"]) if v["p_by_member"].strip() else None
    gs  = parse_general_shocks(v["general_shocks"]) if v["general_shocks"].strip() else None
    ms  = parse_member_shocks(v["member_shocks"]) if v["member_shocks"].strip() else None
    cp  = parse_cashrun_plan(v["cashrun_plan"]) if v["cashrun_plan"].strip() else None
    return pbm, gs, ms, cp


def fmt_float_cols(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    d = df.copy()
    for col in d.select_dtypes("float").columns:
        d[col] = d[col].round(decimals)
    return d


# Collect current vals for simulation
vals = dict(
    N=N, num_cycles=num_cycles, vesting_lag=vesting_lag,
    c=c, gamma=gamma, delta=delta,
    Rb_annual=Rb_annual, Re_annual=Re_annual, periods_per_year=periods_per_year,
    strict_cashrun=strict_cashrun, enable_replacement=enable_replacement,
    replacement_delay=replacement_delay, probation_q=probation_q,
    phi=phi, shrink_cap=shrink_cap, init_t0_first_cycle=init_t0_first_cycle,
    payment_mode=payment_mode, p_base=p_base, seed=seed,
    p_by_member=p_by_member, general_shocks=general_shocks,
    member_shocks=member_shocks, cashrun_plan=cashrun_plan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

st.title("💰 Rotating Savings Scheme — CLCS Simulator")
st.caption(
    "A **Collective Liquidity & Credit Scheme (CLCS)** is an enhanced ROSCA: each period "
    "every member contributes **c**, and the pool **P = N × c** is split between an immediate "
    "payout (γ), a vesting escrow (δ), and a shared safety buffer (1 − γ − δ). "
    "Use the sidebar to configure, then simulate or optimize below."
)

tab_sim, tab_opt, tab_glossary = st.tabs(
    ["🔬 Simulate", "🎯 Optimize Design", "📖 Glossary"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Simulate
# ─────────────────────────────────────────────────────────────────────────────

with tab_sim:
    run_clicked = st.button(
        "▶ Run Simulation",
        type="primary",
        use_container_width=True,
        disabled=(buffer_share < 0),
    )

    if buffer_share < 0:
        st.error("Fix γ + δ in the sidebar before running: their sum exceeds 100%.")

    if run_clicked and buffer_share >= 0:
        try:
            p = build_params(vals)
        except Exception as e:
            st.error(f"Invalid parameters: {e}")
            st.stop()

        pbm, gs, ms, cp = parse_shocks(vals)
        total_turns = p.N * p.num_cycles
        scen = DeterministicScenario(A_sched=[p.N] * total_turns)
        sim  = CLCSSimulator(p)

        with st.spinner("Running simulation…"):
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

        kpi = result.kpi
        df  = result.period_df
        mdf = result.member_df

        # ── KPI cards ─────────────────────────────────────────────────────
        st.subheader("Key Indicators")
        k1, k2, k3, k4 = st.columns(4)

        min_b = kpi["min_B_end"]
        k1.metric(
            "Lowest buffer level",
            f"{min_b:,.1f}",
            delta="Never negative ✓" if min_b >= 0 else "Went negative ⚠",
            delta_color="normal" if min_b >= 0 else "inverse",
            help="Minimum buffer balance across all periods. Must stay ≥ 0 for the group to survive.",
        )

        active_end = int(kpi["disciplined_n_end"])
        k2.metric(
            "Members still active at end",
            active_end,
            delta=f"{active_end - p.N:+d} vs start",
            delta_color="inverse" if active_end < p.N else "off",
            help="Members who remained disciplined (never expelled) through the end of all cycles.",
        )

        k3.metric(
            "Cashrun events (total)",
            int(kpi["cashrun_out_total"]),
            help="Number of members expelled because the buffer could not cover their payout.",
        )

        k4.metric(
            "Avg total payout per member",
            f"{kpi['avg_total_received_disciplined']:,.1f}",
            help="Average of (immediate + vesting + end-bonus) received by disciplined members.",
        )

        # ── Buffer & escrow chart ──────────────────────────────────────────
        st.divider()
        st.subheader("Buffer & Escrow Balance Over Time")

        if HAS_PLOTLY and not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["t"], y=df["B_end"],
                name="Shared buffer (B)",
                line=dict(color="#2563eb", width=2.5),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
            ))
            fig.add_trace(go.Scatter(
                x=df["t"], y=df["E_end"],
                name="Escrow pool (E)",
                line=dict(color="#16a34a", width=2, dash="dash"),
            ))
            fig.add_hline(
                y=0, line_color="red", line_dash="dot",
                annotation_text="Buffer = 0  (danger zone)",
                annotation_font_color="red",
            )
            # Mark cycle boundaries
            for cyc_t in sorted(df["cycle"].unique()):
                first_t = df[df["cycle"] == cyc_t]["t"].min()
                if first_t > 0:
                    fig.add_vline(
                        x=first_t, line_dash="longdash",
                        line_color="gray", opacity=0.4,
                        annotation_text=f"Cycle {cyc_t}",
                        annotation_font_size=11,
                    )
            fig.update_layout(
                xaxis_title="Period",
                yaxis_title="Balance",
                legend=dict(orientation="h", y=1.05, x=0),
                height=360, margin=dict(t=50, b=40),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif not df.empty:
            st.line_chart(df.set_index("t")[["B_end", "E_end"]])

        # ── Contributors per period ────────────────────────────────────────
        st.subheader("Contributors Per Period")
        if HAS_PLOTLY and not df.empty:
            fig2 = px.bar(
                df, x="t", y="C_t",
                color=df["cycle"].astype(str),
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"t": "Period", "C_t": "Members who paid", "color": "Cycle"},
                height=260,
            )
            fig2.update_layout(
                margin=dict(t=20, b=30),
                legend_title="Cycle",
                hovermode="x unified",
            )
            st.plotly_chart(fig2, use_container_width=True)
        elif not df.empty:
            st.bar_chart(df.set_index("t")["C_t"])

        # ── Payout breakdown per turn ──────────────────────────────────────
        turn_df = df[df["phase"] != "init_t0"].copy() if not df.empty else pd.DataFrame()
        if HAS_PLOTLY and not turn_df.empty and "Gimm_eff" in turn_df.columns:
            st.subheader("Payout Breakdown Per Turn")
            pf = go.Figure()
            pf.add_trace(go.Bar(
                x=turn_df["t"], y=turn_df["Gimm_eff"],
                name="Immediate payout (γ × pool)",
                marker_color="#2563eb",
            ))
            pf.add_trace(go.Bar(
                x=turn_df["t"], y=turn_df["vesting_paid"],
                name="Escrow released (δ + interest)",
                marker_color="#16a34a",
            ))
            pf.add_trace(go.Bar(
                x=turn_df["t"], y=turn_df["bonus_paid"],
                name="End-of-cycle bonus",
                marker_color="#f59e0b",
            ))
            pf.update_layout(
                barmode="stack",
                xaxis_title="Period", yaxis_title="Amount paid out",
                height=300, margin=dict(t=20, b=30),
                legend=dict(orientation="h", y=1.05),
                hovermode="x unified",
            )
            st.plotly_chart(pf, use_container_width=True)

        # ── Member summary table ───────────────────────────────────────────
        st.subheader("Member Summary")
        if not mdf.empty:
            display_mdf = fmt_float_cols(mdf)

            def _row_style(row):
                if row.get("out", False):
                    return ["background-color: #fee2e2; color: #991b1b"] * len(row)
                return [""] * len(row)

            st.dataframe(
                display_mdf.style.apply(_row_style, axis=1),
                use_container_width=True,
                height=min(40 + 35 * len(display_mdf), 400),
            )
            st.caption("Rows highlighted in red = members who were expelled during the simulation.")

        # ── Full period log ────────────────────────────────────────────────
        with st.expander("Full period log (last 50 rows)"):
            if not df.empty:
                st.dataframe(fmt_float_cols(df.tail(50)), use_container_width=True)

    elif not run_clicked:
        st.info(
            "👈 Configure the group parameters in the sidebar, then click **▶ Run Simulation** above."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Optimizer
# ─────────────────────────────────────────────────────────────────────────────

with tab_opt:
    st.subheader("Find the Optimal Group Design")
    st.markdown(
        "The optimizer runs **Monte Carlo simulations** across many group configurations "
        "to find the one that best meets your goal. Pick a workflow, set the search range, "
        "and click **Run Optimizer**."
    )

    workflow = st.radio(
        "What do you want to optimize?",
        [
            "1 — Maximize the immediate payout share (γ) given group size & safety target",
            "2 — Find feasible group sizes for a fixed payout split (γ & δ)",
            "3 — Compute the required contribution (c) to hit a target payout amount",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    # ── Shared optimizer controls ──────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Safety & Monte Carlo settings**")
        alpha = st.number_input(
            "Max acceptable failure probability — α",
            min_value=0.001, max_value=0.20, value=0.05, step=0.005, format="%.3f",
            help=(
                "The optimizer only accepts configurations where the group survives "
                "with probability ≥ 1 − α. α = 0.05 means the group must stay solvent "
                "in at least 95% of simulated scenarios."
            ),
        )
        opt_p_base = st.slider(
            "Member payment reliability during optimization",
            min_value=0.50, max_value=1.0, value=0.95, step=0.01,
            help="Baseline payment probability used in Monte Carlo trials. Should reflect realistic expectations.",
        )
        n_sims = st.number_input(
            "Monte Carlo trials per configuration",
            min_value=100, max_value=10000, value=500, step=100,
            help=(
                "More trials → more accurate estimates but slower. "
                "500 is good for exploration; 2 000+ for reliable results."
            ),
        )
        opt_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        opt_strict = st.checkbox(
            "Halt group if buffer goes negative",
            value=True,
            help="Match the cashrun rule you plan to use in practice.",
        )

    with col_r:
        st.markdown("**Group configuration search range**")
        opt_N_lo = st.number_input("Minimum group size (N_min)", min_value=2, max_value=50, value=5, step=1)
        opt_N_hi = st.number_input("Maximum group size (N_max)", min_value=2, max_value=100, value=20, step=1)
        opt_cyc_lo = st.number_input("Minimum cycles", min_value=1, max_value=5, value=1, step=1)
        opt_cyc_hi = st.number_input("Maximum cycles", min_value=1, max_value=10, value=3, step=1)
        opt_K_fixed = st.number_input(
            "Vesting delay K (periods)",
            min_value=0, max_value=50, value=5, step=1,
            help="Fixed vesting lag used for all configurations in the search.",
        )

        n_configs = (max(opt_N_hi, opt_N_lo) - min(opt_N_hi, opt_N_lo) + 1) * \
                    (max(opt_cyc_hi, opt_cyc_lo) - min(opt_cyc_hi, opt_cyc_lo) + 1)
        st.info(f"Search grid: **{n_configs} configurations** to evaluate.")

    st.divider()

    # ── Workflow-specific params ───────────────────────────────────────────
    if workflow.startswith("1"):
        st.markdown("**Parameters for Workflow 1 — Maximize γ**")
        wf1_col1, wf1_col2 = st.columns(2)
        with wf1_col1:
            opt_delta_1 = st.slider("Deferred share δ (%)", 1, 40, 10, step=1,
                                    help="Fixed deferred payout fraction during the search.") / 100.0
            c_opt = st.number_input("Contribution per period (c)", min_value=1.0, value=100.0, step=10.0)
        with wf1_col2:
            gamma_lo_pct = st.slider("Minimum γ to test (%)", 10, 80, 30, step=5,
                                     help="Lower bound of the bisection search for γ.")
            gamma_hi_pct = st.slider("Maximum γ to test (%)", 20, 95, 90, step=5,
                                     help="Upper bound. Must be < 1 − δ.")
            tol = st.number_input("Bisection precision (tolerance)", min_value=0.001,
                                  max_value=0.05, value=0.01, step=0.001, format="%.3f",
                                  help="Stop bisection when the interval width is below this value.")
        st.caption(
            f"Workflow 1 runs ~10 bisection steps per configuration → "
            f"**~{n_configs * 10} × {n_sims:,} = {n_configs * 10 * int(n_sims):,} MC runs total**. "
            "Keep n_sims low for fast exploration."
        )

    elif workflow.startswith("2"):
        st.markdown("**Parameters for Workflow 2 — Find feasible group sizes**")
        wf2_col1, wf2_col2 = st.columns(2)
        with wf2_col1:
            opt_gamma_2 = st.slider("Immediate payout share γ (%)", 10, 90, 65, step=1,
                                    help="Fixed γ you want to offer members.") / 100.0
            opt_delta_2 = st.slider("Deferred payout share δ (%)", 1, 40, 10, step=1,
                                    help="Fixed δ (escrow fraction).") / 100.0
            if opt_gamma_2 + opt_delta_2 >= 1.0:
                st.error("γ + δ ≥ 100%. Reduce one.")
        with wf2_col2:
            c_opt = st.number_input("Contribution per period (c)", min_value=1.0, value=100.0, step=10.0)
        st.caption(
            f"Workflow 2 runs 1 MC batch per configuration → "
            f"**{n_configs} × {n_sims:,} = {n_configs * int(n_sims):,} MC runs total**."
        )

    else:
        st.markdown("**Parameters for Workflow 3 — Compute required contribution**")
        wf3_col1, wf3_col2 = st.columns(2)
        with wf3_col1:
            opt_gamma_3 = st.slider("Immediate payout share γ (%)", 10, 90, 65, step=1,
                                    help="Fraction of the pool paid immediately to the beneficiary.") / 100.0
            opt_delta_3 = st.slider("Deferred payout share δ (%)", 1, 40, 10, step=1,
                                    help="Fraction credited to escrow.") / 100.0
            if opt_gamma_3 + opt_delta_3 >= 1.0:
                st.error("γ + δ ≥ 100%. Reduce one.")
        with wf3_col2:
            target_payout = st.number_input(
                "Target immediate payout per turn",
                min_value=1.0, value=5000.0, step=100.0,
                help=(
                    "The required contribution is derived as c = target / (γ × N). "
                    "Smaller N requires a higher c to hit the same target."
                ),
            )
        st.caption(
            f"Workflow 3 runs 1 MC batch per configuration → "
            f"**{n_configs} × {n_sims:,} = {n_configs * int(n_sims):,} MC runs total**."
        )

    run_opt = st.button("🔍 Run Optimizer", type="primary", use_container_width=True)

    if run_opt:
        if opt_N_hi < opt_N_lo or opt_cyc_hi < opt_cyc_lo:
            st.error("Invalid search range: max < min.")
        else:
            N_range   = range(int(opt_N_lo), int(opt_N_hi) + 1)
            cyc_range = range(int(opt_cyc_lo), int(opt_cyc_hi) + 1)
            total_configs = len(list(N_range)) * len(list(cyc_range))
            progress_bar = st.progress(0, text="Starting optimizer…")
            results: list[dict] = []
            done = 0

            # ── Workflow 1: maximize γ ────────────────────────────────────
            if workflow.startswith("1"):
                g_lo = gamma_lo_pct / 100.0
                g_hi = gamma_hi_pct / 100.0
                for N_opt in N_range:
                    for cyc in cyc_range:
                        done += 1
                        progress_bar.progress(
                            done / total_configs,
                            text=f"Bisecting γ for N={N_opt}, cycles={cyc}…",
                        )
                        if g_hi + opt_delta_1 >= 1.0:
                            continue
                        out = gamma_max_bisect(
                            N=N_opt, cycles=cyc, delta=opt_delta_1,
                            c=float(c_opt), alpha=float(alpha),
                            p_base=float(opt_p_base), K=int(opt_K_fixed),
                            strict_cashrun=bool(opt_strict),
                            enable_replacement=False, replacement_delay=0, probation_q=2,
                            n_sims=int(n_sims), seed=int(opt_seed),
                            gamma_lo=g_lo, gamma_hi=g_hi, tol=float(tol),
                        )
                        if out:
                            results.append({
                                "N": N_opt,
                                "Cycles": cyc,
                                "K (vesting)": int(opt_K_fixed),
                                "δ (deferred %)": f"{opt_delta_1*100:.0f}%",
                                "Max γ (%)": round(out["gamma"] * 100, 1),
                                "Immediate payout": round(out["gamma"] * N_opt * float(c_opt), 1),
                                "Safety": f"{out['safe_prob']*100:.1f}%",
                                "Min buffer p1%": round(out["minB_p01"], 1),
                                "Active members (avg)": round(out["disc_end_mean"], 1),
                                "Cashrun events (avg)": round(out["cashrun_out_mean"], 2),
                                "_gamma": out["gamma"],
                            })

                progress_bar.empty()
                if not results:
                    st.warning(
                        "No feasible design found. Try: lower γ_max, smaller group range, "
                        "higher α (more tolerant), or fewer cycles."
                    )
                else:
                    results.sort(key=lambda r: r["_gamma"], reverse=True)
                    best = results[0]
                    st.success(
                        f"Best design: **N={best['N']}, {best['Cycles']} cycle(s)** — "
                        f"maximum γ = **{best['Max γ (%)']:.1f}%** "
                        f"(immediate payout ≈ {best['Immediate payout']:,.0f} per turn) "
                        f"with {best['Safety']} safety."
                    )
                    rdf = pd.DataFrame(results).drop(columns=["_gamma"])
                    st.dataframe(rdf.head(15), use_container_width=True)

                    if HAS_PLOTLY and len(results) > 1:
                        scatter = px.scatter(
                            rdf.head(30), x="N", y="Max γ (%)",
                            color="Cycles", size="Safety",
                            labels={"N": "Group size (N)", "Max γ (%)": "Max immediate payout share (%)"},
                            title="Feasibility frontier: group size vs max γ",
                            height=350,
                        )
                        st.plotly_chart(scatter, use_container_width=True)

            # ── Workflow 2: find feasible N/cycles ────────────────────────
            elif workflow.startswith("2"):
                if opt_gamma_2 + opt_delta_2 >= 1.0:
                    st.error("γ + δ ≥ 100%. Reduce one.")
                else:
                    target_safe = 1.0 - float(alpha)
                    for N_opt in N_range:
                        for cyc in cyc_range:
                            done += 1
                            progress_bar.progress(
                                done / total_configs,
                                text=f"Testing N={N_opt}, cycles={cyc}…",
                            )
                            r = mc_safety(
                                N=N_opt, cycles=cyc,
                                gamma=opt_gamma_2, delta=opt_delta_2,
                                c=float(c_opt), alpha=float(alpha),
                                p_base=float(opt_p_base), K=int(opt_K_fixed),
                                strict_cashrun=bool(opt_strict),
                                enable_replacement=False, replacement_delay=0, probation_q=2,
                                n_sims=int(n_sims), seed=int(opt_seed),
                            )
                            if r["safe_prob"] >= target_safe:
                                results.append({
                                    "N": N_opt,
                                    "Cycles": cyc,
                                    "K (vesting)": int(opt_K_fixed),
                                    "Pool per period": round(N_opt * float(c_opt), 1),
                                    "Safety": f"{r['safe_prob']*100:.1f}%",
                                    "Min buffer p1%": round(r["minB_p01"], 1),
                                    "Min buffer p5%": round(r["minB_p05"], 1),
                                    "Active members (avg)": round(r["disc_end_mean"], 1),
                                    "Cashrun events (avg)": round(r["cashrun_out_mean"], 2),
                                    "_safe_prob": r["safe_prob"],
                                })

                    progress_bar.empty()
                    if not results:
                        st.warning(
                            f"No feasible (N, cycles) found for γ={opt_gamma_2:.0%}, δ={opt_delta_2:.0%}. "
                            "Try a larger group size range, fewer cycles, or a higher α."
                        )
                    else:
                        results.sort(key=lambda r: (r["_safe_prob"], -r["N"]), reverse=True)
                        st.success(
                            f"Found **{len(results)} feasible configurations** that meet the "
                            f"≥ {(1-float(alpha))*100:.0f}% safety target."
                        )
                        rdf = pd.DataFrame(results).drop(columns=["_safe_prob"])
                        st.dataframe(rdf, use_container_width=True)

                        if HAS_PLOTLY and len(results) > 1:
                            heat_data = pd.DataFrame([
                                {"N": r["N"], "Cycles": str(r["Cycles"]),
                                 "Safety (%)": float(r["Safety"].replace("%", ""))}
                                for r in results
                            ])
                            heat_fig = px.scatter(
                                heat_data, x="N", y="Cycles",
                                color="Safety (%)", color_continuous_scale="RdYlGn",
                                size="Safety (%)",
                                title=f"Safety map for γ={opt_gamma_2:.0%}, δ={opt_delta_2:.0%}",
                                labels={"N": "Group size (N)", "Cycles": "Number of cycles"},
                                height=350,
                            )
                            st.plotly_chart(heat_fig, use_container_width=True)

            # ── Workflow 3: derive c from target payout ───────────────────
            else:
                if opt_gamma_3 + opt_delta_3 >= 1.0:
                    st.error("γ + δ ≥ 100%. Reduce one.")
                else:
                    target_safe = 1.0 - float(alpha)
                    for N_opt in N_range:
                        for cyc in cyc_range:
                            done += 1
                            progress_bar.progress(
                                done / total_configs,
                                text=f"Testing N={N_opt}, cycles={cyc}…",
                            )
                            r = mc_safety(
                                N=N_opt, cycles=cyc,
                                gamma=opt_gamma_3, delta=opt_delta_3,
                                c=1.0,  # scale-free feasibility check
                                alpha=float(alpha), p_base=float(opt_p_base), K=int(opt_K_fixed),
                                strict_cashrun=bool(opt_strict),
                                enable_replacement=False, replacement_delay=0, probation_q=2,
                                n_sims=int(n_sims), seed=int(opt_seed),
                            )
                            if r["safe_prob"] >= target_safe:
                                c_needed = float(target_payout) / (opt_gamma_3 * N_opt)
                                results.append({
                                    "N": N_opt,
                                    "Cycles": cyc,
                                    "K (vesting)": int(opt_K_fixed),
                                    "Required contribution (c)": round(c_needed, 2),
                                    "Total member commitment": round(c_needed * N_opt * cyc, 0),
                                    "Safety": f"{r['safe_prob']*100:.1f}%",
                                    "Min buffer p1%": round(r["minB_p01"] * c_needed, 1),
                                    "Active members (avg)": round(r["disc_end_mean"], 1),
                                    "_c_needed": c_needed,
                                    "_safe_prob": r["safe_prob"],
                                })

                    progress_bar.empty()
                    if not results:
                        st.warning(
                            "No feasible design found. Try a larger N range or adjust γ/δ."
                        )
                    else:
                        results.sort(key=lambda r: (r["_c_needed"], -r["_safe_prob"]))
                        best = results[0]
                        st.success(
                            f"Most affordable option: **N={best['N']}, {best['Cycles']} cycle(s)** — "
                            f"each member contributes **{best['Required contribution (c)']:,.2f}** per period "
                            f"to deliver an immediate payout of **{target_payout:,.0f}** "
                            f"(safety: {best['Safety']})."
                        )
                        rdf = pd.DataFrame(results).drop(columns=["_c_needed", "_safe_prob"])
                        st.dataframe(rdf.head(15), use_container_width=True)

                        if HAS_PLOTLY and len(results) > 1:
                            bar_fig = px.bar(
                                rdf.head(15), x="N", y="Required contribution (c)",
                                color="Cycles", barmode="group",
                                labels={
                                    "N": "Group size (N)",
                                    "Required contribution (c)": "Required c per period",
                                },
                                title=f"Required contribution to achieve payout = {target_payout:,.0f}",
                                height=350,
                            )
                            st.plotly_chart(bar_fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Glossary
# ─────────────────────────────────────────────────────────────────────────────

with tab_glossary:
    st.markdown("""
### How a CLCS works

In each period every active member contributes **c**. The **pool P = N × c** is split:

| Component | Symbol | What it does |
|---|---|---|
| Immediate payout | **γ** | Paid directly to the current beneficiary at the time of their turn |
| Deferred payout | **δ** | Credited to escrow; released to the member after K periods (with interest) |
| Safety buffer | **1 − γ − δ** | Retained as a shared reserve; earns interest and covers shortfalls |

---

### Parameter reference

| Parameter | Symbol | Plain-English meaning |
|---|---|---|
| Members | **N** | How many people are in the group |
| Cycles | — | How many full rotations (each member gets a turn once per cycle) |
| Vesting delay | **K** | Periods before escrow credits are released |
| Contribution | **c** | What each member pays every period |
| Immediate share | **γ** | Fraction of the pool paid out right away |
| Deferred share | **δ** | Fraction held in escrow (released with interest after K periods) |
| Buffer rate | **R_b** | Annual interest on the shared buffer |
| Escrow rate | **R_e** | Annual interest on individual escrow balances |
| Failure tolerance | **α** | Max acceptable probability of the buffer going negative (optimizer) |
| Strict cashrun | — | Halt the group the moment the buffer goes negative |
| Shrink cap | — | Maximum arrears that can be withheld from a member's payout (in multiples of c) |
| Probation | — | Periods a new member must pay before being eligible for a payout turn |
| p_base | — | Probability any member pays in a given period (Monte Carlo mode) |

---

### Reading the optimizer results

| Column | Meaning |
|---|---|
| **Safety (%)** | Fraction of Monte Carlo trials where the buffer never went negative — higher is better |
| **Min buffer p1%** | 1st percentile of the minimum buffer — near-worst-case scenario |
| **Min buffer p5%** | 5th percentile of the minimum buffer |
| **Active members (avg)** | Average number of members who finished all cycles without being expelled |
| **Cashrun events (avg)** | Average number of members expelled because the buffer couldn't cover their payout |
| **Max γ (%)** | Highest immediate payout share that still meets the safety target |
| **Required c** | Contribution per member per period needed to hit the target payout amount |

---

### Tips

- **Start with the optimizer** to find a safe design, then use the Simulator to explore edge cases with shocks.
- A **higher γ** means bigger upfront payouts but a thinner buffer — the group is riskier.
- **Longer vesting (K)** locks more money in escrow, which stabilises the buffer.
- **More members (N)** increases the pool and generally improves safety, but administrative complexity grows too.
- Use **p_base < 1.0** in the simulator to model real-world payment unreliability.
""")
