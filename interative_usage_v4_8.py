"""interactive_usage_v4_8.py
Interactive runner for clcs_sim_v4_8.py.
Adds per-member payment probabilities + shock schedules.

Update (Jan 2026):
- p_base defaults to 1.0 (deterministic "everyone pays" baseline).
- Removed q_cashrun.
- Added cashrun_plan (member_id -> cycles). If a member is beneficiary in one of the
  planned cycles, they are forced to not pay at t+1, which triggers strict cashrun.

Run:
 python interactive_usage_v4_8.py

Input formats
-------------
- General shock: "t0-t1:mult" (e.g., "10-20:0.8")
- Member shock: "id:t0-t1:p" (e.g., "7:15-18:0" or "12:30-60:0.5")
- p_by_member: "id:p" comma separated (e.g., "3:0.95, 7:0.7")
- cashrun_plan: "id:cycle,cycle; id:cycle" (e.g., "7:2;12:1,3")
  (cycle 1 = turns 1..N, cycle 2 = N+1..2N, ...)

You can leave them empty.
"""

from __future__ import annotations

import ast
from typing import List, Dict, Tuple

import pandas as pd

from clcs_sim_v4_8 import CLCSParams, DeterministicScenario, CLCSSimulator, pretty_params


def _prompt(msg: str) -> str:
    return input(msg).strip()


def ask_int(label: str, default: int, min_val=None, max_val=None) -> int:
    while True:
        raw = _prompt(f"{label} [{default}]: ")
        if raw == "":
            val = default
        else:
            try:
                val = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue
        if min_val is not None and val < min_val:
            print(f"Must be >= {min_val}.")
            continue
        if max_val is not None and val > max_val:
            print(f"Must be <= {max_val}.")
            continue
        return val


def ask_float(label: str, default: float, min_val=None, max_val=None) -> float:
    while True:
        raw = _prompt(f"{label} [{default}]: ")
        if raw == "":
            val = default
        else:
            try:
                val = float(raw)
            except ValueError:
                print("Please enter a number.")
                continue
        if min_val is not None and val < min_val:
            print(f"Must be >= {min_val}.")
            continue
        if max_val is not None and val > max_val:
            print(f"Must be <= {max_val}.")
            continue
        return val


def ask_choice(label: str, choices: List[str], default: str) -> str:
    opts = ", ".join(choices)
    while True:
        raw = _prompt(f"{label} ({opts}) [default={default}]: ")
        if raw == "":
            return default
        if raw in choices:
            return raw
        print("Invalid choice.")


def parse_p_by_member(s: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    s = s.strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for part in parts:
        mid_s, p_s = [x.strip() for x in part.split(":")]
        out[int(mid_s)] = float(p_s)
    return out


def parse_general_shocks(s: str) -> List[Tuple[int, int, float]]:
    s = s.strip()
    if not s:
        return []
    shocks = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for part in parts:
        span, mult_s = [x.strip() for x in part.split(":")]
        t0_s, t1_s = [x.strip() for x in span.split("-")]
        shocks.append((int(t0_s), int(t1_s), float(mult_s)))
    return shocks


def parse_member_shocks(s: str) -> Dict[int, List[Tuple[int, int, float]]]:
    s = s.strip()
    if not s:
        return {}
    out: Dict[int, List[Tuple[int, int, float]]] = {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for part in parts:
        # id:t0-t1:p
        left, p_s = part.rsplit(":", 1)
        mid_s, span = left.split(":", 1)
        t0_s, t1_s = [x.strip() for x in span.split("-")]
        mid = int(mid_s)
        out.setdefault(mid, []).append((int(t0_s), int(t1_s), float(p_s)))
    return out


def parse_cashrun_plan(s: str) -> Dict[int, List[int]]:
    """Parse cashrun plan input.

    Format: "id:cycle,cycle; id:cycle" e.g. "7:2;12:1,3".
    Member entries separated by ';'. Cycle lists separated by ','.
    """
    s = s.strip()
    if not s:
        return {}

    out: Dict[int, List[int]] = {}
    entries = [e.strip() for e in s.split(";") if e.strip()]
    for e in entries:
        if ":" not in e:
            continue
        mid_s, cycles_s = e.split(":", 1)
        mid = int(mid_s.strip())
        cycles = []
        for cs in cycles_s.split(","):
            cs = cs.strip()
            if not cs:
                continue
            cycles.append(int(cs))
        if cycles:
            out[mid] = sorted(list(set(cycles)))
    return out


def main():
    print("CLCS interactive (v4.8 - per-member p + shocks + cashrun_plan)"
          )

    N = ask_int("N (target group size)", 30, min_val=2)
    c = ask_float("c (contribution per period)", 10.0, min_val=0.01)
    gamma = ask_float("gamma (immediate share)", 0.65, min_val=0.01, max_val=0.99)
    delta = ask_float("delta (deferred share)", 0.10, min_val=0.01, max_val=0.99)
    if gamma + delta >= 1:
        print("gamma+delta must be < 1; reducing delta.")
        delta = min(delta, 0.99 - gamma)

    num_cycles = ask_int("num_cycles", 2, min_val=1)
    K = ask_int("vesting_lag K", 5, min_val=0)

    strict_cashrun = ask_choice(
        "strict_cashrun (beneficiary must pay at t+1)", ["yes", "no"], "yes"
    ) == "yes"

    enable_repl = ask_choice("enable_replacement", ["no", "yes"], "yes") == "yes"
    repl_delay = ask_int("replacement_delay (turns)", 0, min_val=0)
    probation_q = ask_int("probation_q (payments to become eligible)", 2, min_val=0)

    p = CLCSParams(
        N=N,
        c=c,
        gamma=gamma,
        delta=delta,
        num_cycles=num_cycles,
        vesting_lag=K,
        strict_cashrun=strict_cashrun,
        enable_replacement=enable_repl,
        replacement_delay=repl_delay,
        probation_q=probation_q,
    )

    print("Computed parameters:")
    print(pretty_params(p))

    base_len = N * num_cycles
    A_sched = [N] * base_len

    sim_mode = ask_choice(
        "Simulation mode", ["deterministic", "mc_fixedA", "mc_probpay"], "mc_probpay"
    )

    # Payment probability configuration (used in mc_probpay)
    p_base = ask_float(
        "p_base (baseline payment probability; set 1.0 for fully deterministic payments)",
        1.0,
        min_val=0.0,
        max_val=1.0,
    )

    p_by_member_raw = _prompt("p_by_member (id:p, comma separated) [empty=none]: ")
    p_by_member = parse_p_by_member(p_by_member_raw)

    general_shocks_raw = _prompt("general shocks (t0-t1:mult, comma separated) [empty=none]: ")
    general_shocks = parse_general_shocks(general_shocks_raw)

    member_shocks_raw = _prompt("member shocks (id:t0-t1:p, comma separated) [empty=none]: ")
    member_shocks = parse_member_shocks(member_shocks_raw)

    cashrun_plan_raw = _prompt("cashrun_plan (id:cycle,cycle; id:cycle) [empty=none]: ")
    cashrun_plan = parse_cashrun_plan(cashrun_plan_raw)

    scen = DeterministicScenario(A_sched=A_sched)
    sim = CLCSSimulator(p)

    res = sim.run_path(
        scen,
        payment_mode=sim_mode,
        seed=42,
        p_base=p_base,
        p_by_member=p_by_member if p_by_member else None,
        general_shocks=general_shocks if general_shocks else None,
        member_shocks=member_shocks if member_shocks else None,
        cashrun_plan=cashrun_plan if cashrun_plan else None,
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