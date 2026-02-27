"""clcs_interactive_runner.py
Interactive single-run CLI for the CLCS simulator.

Replaces interative_usage.py (fixing the typo in the filename).
Prompt helpers imported from cli_prompts — no duplication with the optimizer.

Input formats
-------------
  p_by_member   : "id:p, id:p"             e.g. "3:0.95, 7:0.7"
  general_shocks: "t0-t1:mult, ..."         e.g. "10-20:0.8"
  member_shocks : "id:t0-t1:p, ..."         e.g. "7:15-18:0, 12:30-60:0.5"
  cashrun_plan  : "id:cycle,cycle; id:cycle" e.g. "7:2;12:1,3"
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from cli_prompts import ask_bool, ask_choice, ask_float, ask_int, ask_str
from clcs_simulator import CLCSParams, CLCSSimulator, DeterministicScenario, pretty_params


# ---------------------------------------------------------------------------
# Input parsers
# ---------------------------------------------------------------------------

def parse_p_by_member(s: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for part in s.strip().split(","):
        part = part.strip()
        if not part:
            continue
        mid_s, p_s = [x.strip() for x in part.split(":")]
        out[int(mid_s)] = float(p_s)
    return out


def parse_general_shocks(s: str) -> List[Tuple[int, int, float]]:
    shocks = []
    for part in s.strip().split(","):
        part = part.strip()
        if not part:
            continue
        span, mult_s = [x.strip() for x in part.split(":")]
        t0_s, t1_s = [x.strip() for x in span.split("-")]
        shocks.append((int(t0_s), int(t1_s), float(mult_s)))
    return shocks


def parse_member_shocks(s: str) -> Dict[int, List[Tuple[int, int, float]]]:
    out: Dict[int, List[Tuple[int, int, float]]] = {}
    for part in s.strip().split(","):
        part = part.strip()
        if not part:
            continue
        left, p_s = part.rsplit(":", 1)
        mid_s, span = left.split(":", 1)
        t0_s, t1_s = [x.strip() for x in span.split("-")]
        mid = int(mid_s)
        out.setdefault(mid, []).append((int(t0_s), int(t1_s), float(p_s)))
    return out


def parse_cashrun_plan(s: str) -> Dict[int, List[int]]:
    """Parse "id:cycle,cycle; id:cycle" into {member_id: [cycles]}."""
    out: Dict[int, List[int]] = {}
    for entry in s.strip().split(";"):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        mid_s, cycles_s = entry.split(":", 1)
        mid = int(mid_s.strip())
        cycles = [int(c.strip()) for c in cycles_s.split(",") if c.strip()]
        if cycles:
            out[mid] = sorted(set(cycles))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("CLCS Interactive Runner (v4.8 — per-member p + shocks + cashrun_plan)\n")

    N = ask_int("N (target group size)", 30, min_val=2)
    c = ask_float("c (contribution per period)", 10.0, min_val=0.01)
    gamma = ask_float("gamma (immediate share)", 0.65, min_val=0.01, max_val=0.99)
    delta = ask_float("delta (deferred share)", 0.10, min_val=0.01, max_val=0.99)
    if gamma + delta >= 1:
        print("gamma+delta must be < 1; reducing delta.")
        delta = min(delta, 0.99 - gamma)

    num_cycles = ask_int("num_cycles", 2, min_val=1)
    K = ask_int("vesting_lag K", 5, min_val=0)

    strict_cashrun = ask_choice("strict_cashrun", ["yes", "no"], "yes") == "yes"
    enable_repl = ask_choice("enable_replacement", ["no", "yes"], "yes") == "yes"
    repl_delay = ask_int("replacement_delay (turns)", 0, min_val=0)
    probation_q = ask_int("probation_q (payments to become eligible)", 2, min_val=0)

    params = CLCSParams(
        N=N, c=c, gamma=gamma, delta=delta,
        num_cycles=num_cycles, vesting_lag=K,
        strict_cashrun=strict_cashrun,
        enable_replacement=enable_repl,
        replacement_delay=repl_delay,
        probation_q=probation_q,
    )
    print("\nComputed parameters:")
    print(pretty_params(params))

    sim_mode = ask_choice("Simulation mode", ["deterministic", "mc_fixedA", "mc_probpay"], "mc_probpay")
    p_base = ask_float("p_base (baseline payment probability)", 1.0, min_val=0.0, max_val=1.0)

    p_by_member_raw = ask_str("p_by_member (id:p, comma separated)", "", help_text="Leave empty for none.")
    p_by_member = parse_p_by_member(p_by_member_raw) if p_by_member_raw.strip() else {}

    general_shocks_raw = ask_str("general shocks (t0-t1:mult, comma separated)", "", help_text="Leave empty for none.")
    general_shocks = parse_general_shocks(general_shocks_raw) if general_shocks_raw.strip() else []

    member_shocks_raw = ask_str("member shocks (id:t0-t1:p, comma separated)", "", help_text="Leave empty for none.")
    member_shocks = parse_member_shocks(member_shocks_raw) if member_shocks_raw.strip() else {}

    cashrun_plan_raw = ask_str("cashrun_plan (id:cycle,cycle; id:cycle)", "", help_text="Leave empty for none.")
    cashrun_plan = parse_cashrun_plan(cashrun_plan_raw) if cashrun_plan_raw.strip() else {}

    scen = DeterministicScenario(A_sched=[N] * (N * num_cycles))
    sim = CLCSSimulator(params)

    res = sim.run_path(
        scen,
        payment_mode=sim_mode,
        seed=42,
        p_base=p_base,
        p_by_member=p_by_member or None,
        general_shocks=general_shocks or None,
        member_shocks=member_shocks or None,
        cashrun_plan=cashrun_plan or None,
    )

    print("\nKPI")
    print(pd.Series(res.kpi))

    show = ask_int("How many period rows to print?", 25, min_val=0)
    if show > 0:
        print("\nPeriods (head)")
        print(res.period_df.head(show).to_string(index=False))

    print("\nMembers (sorted by id)")
    print(res.member_df.to_string(index=False))


if __name__ == "__main__":
    main()
