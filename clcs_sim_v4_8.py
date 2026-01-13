"""clcs_sim_v4_8.py
CLCS simulator (deterministic + Monte Carlo) with:
- Strict cashrun rule: if member is beneficiary at t and does NOT pay at t+1 => OUT.
- Optional replacement: any OUT schedules a replacement join (delay + probation).
- Monte Carlo payments with PER-MEMBER payment probabilities + shock schedules.

Update (Jan 2026): simplified cashrun modeling
---------------------------------------------
- Removed stochastic cashrun trigger q_cashrun.
- Added deterministic cashrun_plan based on (member_id, cycle):
    cashrun_plan = {member_id: [cycle_index, ...]}
  where cycle_index is 1..num_cycles with cycle 1 = turns 1..N, cycle 2 = N+1..2N, etc.
  If member_id is selected as beneficiary during a planned cycle, then that member is
  forced to NOT pay at the next turn (t+1), which triggers the strict cashrun rule.

Payment modes:
* deterministic: contributors are first A active members in queue
* mc_fixedA: pick A contributors uniformly among active
* mc_probpay: each active pays with prob p(mid,t) computed from policy:
    - p_base
    - p_by_member overrides
    - general_shocks multiplicative windows
    - member_shocks overrides
* contributors_sched override

Horizon fixed to N * num_cycles turns.
Replacement members get ids N+1, N+2, ... and start with paid_since_join=0.

Dependencies: numpy, pandas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class CLCSParams:
    N: int = 10
    c: float = 100.0
    gamma: float = 0.75
    delta: float = 0.10
    num_cycles: int = 2
    vesting_lag: int = 5

    Rb_annual: float = 0.042
    Re_annual: float = 0.035
    periods_per_year: int = 12

    phi: float = 0.0
    shrink_cap: float = 2.0

    # Replacement / probation
    enable_replacement: bool = False
    replacement_delay: int = 0
    probation_q: int = 2

    # Strict cashrun rule switch
    strict_cashrun: bool = True

    # Whether to inject initial contributions at t=0 (init_t0 row)
    init_t0_first_cycle: bool = True

    def __post_init__(self):
        assert self.N >= 2
        assert self.c > 0
        assert 0 < self.gamma < 1
        assert 0 < self.delta < 1
        assert self.gamma + self.delta < 1
        assert self.num_cycles >= 1
        assert self.vesting_lag >= 0
        assert self.periods_per_year in (12, 52, 26, 24, 4, 1)
        assert 0 <= self.phi <= 1
        assert self.shrink_cap >= 0
        assert self.replacement_delay >= 0
        assert self.probation_q >= 0

    @property
    def P(self) -> float:
        return self.N * self.c

    @property
    def Gimm(self) -> float:
        return self.gamma * self.P

    @property
    def Gdiff(self) -> float:
        return self.delta * self.P

    @property
    def S(self) -> float:
        return (1.0 - (self.gamma + self.delta)) * self.P

    @property
    def r_b(self) -> float:
        return (1.0 + self.Rb_annual) ** (1.0 / self.periods_per_year) - 1.0

    @property
    def r_e(self) -> float:
        return (1.0 + self.Re_annual) ** (1.0 / self.periods_per_year) - 1.0


@dataclass
class MemberState:
    member_id: int
    out: bool = False
    eligible_bonus: bool = True
    eligible_vesting: bool = True
    arrears: float = 0.0
    join_t: int = 0
    paid_since_join: int = 0
    escrow_credits: List[Tuple[int, float]] = field(default_factory=list)
    missed_streak: int = 0


@dataclass
class DeterministicScenario:
    A_sched: List[int]
    force_out: Dict[int, int] = field(default_factory=dict)
    contributors_sched: Optional[List[List[int]]] = None


@dataclass
class PathResult:
    period_df: pd.DataFrame
    member_df: pd.DataFrame
    kpi: Dict[str, Any]


class CLCSSimulator:
    def __init__(self, params: CLCSParams):
        self.p = params

    # -------------------------
    # Core helpers
    # -------------------------
    def _init_members(self) -> Dict[int, MemberState]:
        # Original members eligible immediately (paid_since_join set to probation_q)
        return {
            i: MemberState(member_id=i, join_t=0, paid_since_join=self.p.probation_q)
            for i in range(1, self.p.N + 1)
        }

    def _active_ids(self, members: Dict[int, MemberState]) -> List[int]:
        return [mid for mid, m in members.items() if not m.out]

    def _apply_interest_and_fee(self, B: float, E: float) -> Tuple[float, float, float, float]:
        I = self.p.r_b * B + self.p.r_e * E
        fee = self.p.phi * max(I, 0.0)
        B = B * (1.0 + self.p.r_b) - fee
        E = E * (1.0 + self.p.r_e)
        return B, E, I, fee

    def _is_eligible_for_turn(self, m: MemberState) -> bool:
        if m.out:
            return False
        if self.p.probation_q > 0 and m.paid_since_join < self.p.probation_q:
            return False
        return True

    def _pick_beneficiary(self, queue: List[int], members: Dict[int, MemberState]) -> Tuple[Optional[int], int]:
        if not queue:
            return None, 0
        skips = 0
        for _ in range(len(queue)):
            cand = queue[0]
            m = members.get(cand)
            if m is None or not self._is_eligible_for_turn(m):
                queue.append(queue.pop(0))
                skips += 1
            else:
                return cand, skips
        return None, skips

    def _apply_shrink(self, m: MemberState) -> Tuple[float, float]:
        shrink = min(self.p.shrink_cap * self.p.c, m.arrears)
        Gimm_eff = max(0.0, self.p.Gimm - shrink)
        m.arrears -= shrink
        if m.arrears < 1e-9:
            m.arrears = 0.0
        return Gimm_eff, shrink

    def _credit_escrow(self, m: MemberState, t: int, x: float):
        if x > 0:
            m.escrow_credits.append((t, x))

    def _settle_vesting(self, members: Dict[int, MemberState], B: float, E: float, t: int, settle_all: bool):
        K = self.p.vesting_lag
        r_e = self.p.r_e
        paid_total = 0.0
        forfeit_total = 0.0
        n_credits = 0

        for m in members.values():
            if not m.escrow_credits:
                continue
            keep: List[Tuple[int, float]] = []
            for t_credit, principal in m.escrow_credits:
                mature = (t >= t_credit + K)
                if settle_all or mature:
                    val = principal * ((1.0 + r_e) ** (t - t_credit))
                    if (not m.out) and m.eligible_vesting:
                        paid_total += val
                    else:
                        forfeit_total += val
                    n_credits += 1
                else:
                    keep.append((t_credit, principal))
            m.escrow_credits = keep

        total = paid_total + forfeit_total
        if total > 0:
            E = max(0.0, E - total)
            B = B + forfeit_total
        return B, E, paid_total, forfeit_total, n_credits

    def _pay_bonus_final(self, members: Dict[int, MemberState], B: float):
        disciplined = [mid for mid, m in members.items() if (not m.out) and m.eligible_bonus]
        n_disc = len(disciplined)
        bonus_per = B / n_disc if n_disc else 0.0
        bonus_paid_total = B
        B_after = 0.0
        return B_after, bonus_per, n_disc, bonus_paid_total

    # -------------------------
    # Payment policies
    # -------------------------
    def _contributors_deterministic(self, queue: List[int], active_ids: List[int], A: int) -> List[int]:
        q = [mid for mid in queue if mid in active_ids]
        return q[: max(0, min(A, len(q)))]

    def _contributors_mc_fixedA(self, rng: np.random.Generator, active_ids: List[int], A: int) -> List[int]:
        A = max(0, min(A, len(active_ids)))
        if A == 0:
            return []
        return list(rng.choice(active_ids, size=A, replace=False))

    def _p_effective(
        self,
        mid: int,
        t: int,
        p_base: float,
        p_by_member: Optional[Dict[int, float]],
        general_shocks: Optional[List[Tuple[int, int, float]]],
        member_shocks: Optional[Dict[int, List[Tuple[int, int, float]]]],
    ) -> float:
        # baseline
        p = float(p_by_member.get(mid, p_base)) if p_by_member else float(p_base)
        # general shocks: multiply
        if general_shocks:
            for t0, t1, mult in general_shocks:
                if t0 <= t <= t1:
                    p *= float(mult)
        # member shocks: override (last match wins)
        if member_shocks and mid in member_shocks:
            for t0, t1, p_override in member_shocks[mid]:
                if t0 <= t <= t1:
                    p = float(p_override)
        # clip
        if p < 0.0:
            p = 0.0
        if p > 1.0:
            p = 1.0
        return p

    def _contributors_mc_probpay(
        self,
        rng: np.random.Generator,
        active_ids: List[int],
        t: int,
        p_base: float,
        p_by_member: Optional[Dict[int, float]],
        general_shocks: Optional[List[Tuple[int, int, float]]],
        member_shocks: Optional[Dict[int, List[Tuple[int, int, float]]]],
        cashrun_due_member: Optional[int],
        cashrun_due_time: Optional[int],
        cashrun_force_due: bool,
    ) -> List[int]:
        contributors: List[int] = []
        for mid in active_ids:
            p_eff = self._p_effective(mid, t, p_base, p_by_member, general_shocks, member_shocks)

            # deterministic cashrun forcing: if beneficiary at t-1 is due now (t)
            # and the plan requested a cashrun, force p=0 at the due time.
            if (
                cashrun_force_due
                and cashrun_due_member is not None
                and cashrun_due_time == t
                and mid == cashrun_due_member
            ):
                p_eff = 0.0

            if rng.random() < p_eff:
                contributors.append(mid)
        return contributors

    # -------------------------
    # Replacement
    # -------------------------
    def _schedule_replacement(self, pending: List[Tuple[int, int]], next_id: int, t: int) -> int:
        if not self.p.enable_replacement:
            return next_id
        join_t = t + self.p.replacement_delay
        pending.append((join_t, next_id))
        return next_id + 1

    def _process_pending_replacements(
        self,
        pending: List[Tuple[int, int]],
        members: Dict[int, MemberState],
        queue: List[int],
        t: int,
    ):
        still: List[Tuple[int, int]] = []
        for join_t, mid in pending:
            if join_t <= t:
                if len(self._active_ids(members)) < self.p.N:
                    members[mid] = MemberState(member_id=mid, join_t=t, paid_since_join=0)
                    queue.append(mid)
                else:
                    still.append((join_t, mid))
            else:
                still.append((join_t, mid))
        return still

    # -------------------------
    # OUT helper
    # -------------------------
    def _set_out(self, members: Dict[int, MemberState], mid: int):
        m = members.get(mid)
        if m is None or m.out:
            return
        m.out = True
        m.eligible_bonus = False
        m.eligible_vesting = False

    # -------------------------
    # Main simulation
    # -------------------------
    def run_path(
        self,
        scen: DeterministicScenario,
        payment_mode: str = "mc_probpay",
        seed: int = 42,
        # payment params
        p_base: float = 1.0,
        p_by_member: Optional[Dict[int, float]] = None,
        general_shocks: Optional[List[Tuple[int, int, float]]] = None,
        member_shocks: Optional[Dict[int, List[Tuple[int, int, float]]]] = None,
        # deterministic cashrun plan
        cashrun_plan: Optional[Dict[int, List[int]]] = None,
    ) -> PathResult:
        p = self.p
        rng = np.random.default_rng(seed)

        # normalize cashrun_plan cycles to set() for O(1) checks
        cashrun_plan_sets: Optional[Dict[int, set]] = None
        if cashrun_plan:
            cashrun_plan_sets = {mid: set(cycles) for mid, cycles in cashrun_plan.items()}

        members = self._init_members()
        next_member_id = p.N + 1
        pending_replacements: List[Tuple[int, int]] = []

        B = 0.0
        E = 0.0
        platform_rev = 0.0
        rows: List[Dict[str, Any]] = []

        # event logs
        cashrun_forced_events: List[Dict[str, Any]] = []  # when plan triggers forcing at t+1
        cashrun_out_events: List[Dict[str, Any]] = []     # when strict cashrun causes OUT

        if p.init_t0_first_cycle:
            B = p.N * p.c
            rows.append(
                {
                    "t": 0,
                    "cycle": 0,
                    "phase": "init_t0",
                    "A_t": p.N,
                    "C_t": p.N * p.c,
                    "beneficiary": None,
                    "B_end": B,
                    "E_end": E,
                    "members_active": len(self._active_ids(members)),
                    "pending_repl": len(pending_replacements),
                    "cashrun_out": 0,
                    "miss3_out": 0,
                    "force_out": 0,
                    "cashrun_forced": 0,
                }
            )

        queue = self._active_ids(members)
        total_turns = p.N * p.num_cycles
        a_idx = 0

        # strict cashrun tracking (beneficiary at t must pay at t+1)
        cashrun_due_member: Optional[int] = None
        cashrun_due_time: Optional[int] = None
        cashrun_force_due: bool = False
        cashrun_due_cycle: Optional[int] = None
        cashrun_due_ben_t: Optional[int] = None

        cashrun_out_count = 0
        miss3_out_count = 0
        force_out_count = 0

        for t in range(1, total_turns + 1):
            cycle_t = (t - 1) // p.N + 1

            pending_replacements = self._process_pending_replacements(pending_replacements, members, queue, t)

            # force_out
            for mid, t_out in scen.force_out.items():
                if t >= t_out and mid in members and (not members[mid].out):
                    self._set_out(members, mid)
                    force_out_count += 1
                    next_member_id = self._schedule_replacement(pending_replacements, next_member_id, t)

            # purge queue
            queue = [mid for mid in queue if mid in members and (not members[mid].out)]

            # interest
            B, E, I, fee = self._apply_interest_and_fee(B, E)
            platform_rev += fee
            B_after_interest = B
            E_after_interest = E

            # A_sched (for deterministic / mc_fixedA; placeholder for mc_probpay)
            if a_idx >= len(scen.A_sched):
                raise ValueError(f"A_sched too short at turn #{a_idx + 1}.")
            A_sched_val = int(scen.A_sched[a_idx])
            a_idx += 1

            active_ids = self._active_ids(members)
            A = max(0, min(A_sched_val, len(active_ids)))

            # contributors
            if scen.contributors_sched is not None:
                if (t - 1) >= len(scen.contributors_sched):
                    raise ValueError(f"contributors_sched too short at turn t={t}.")
                contributors = [mid for mid in scen.contributors_sched[t - 1] if mid in active_ids]
                A = len(contributors)
            else:
                if payment_mode == "deterministic":
                    contributors = self._contributors_deterministic(queue, active_ids, A)
                elif payment_mode == "mc_fixedA":
                    contributors = self._contributors_mc_fixedA(rng, active_ids, A)
                elif payment_mode == "mc_probpay":
                    contributors = self._contributors_mc_probpay(
                        rng,
                        active_ids,
                        t,
                        p_base,
                        p_by_member,
                        general_shocks,
                        member_shocks,
                        cashrun_due_member,
                        cashrun_due_time,
                        cashrun_force_due,
                    )
                    A = len(contributors)
                else:
                    raise ValueError(f"Unknown payment_mode={payment_mode}")

            contrib_set = set(contributors)

            # strict cashrun check at t: last beneficiary must pay now
            if p.strict_cashrun and cashrun_due_member is not None and cashrun_due_time == t:
                mid = cashrun_due_member
                if mid in members and (not members[mid].out):
                    if mid not in contrib_set:
                        self._set_out(members, mid)
                        cashrun_out_count += 1
                        cashrun_out_events.append(
                            {
                                "t_out": t,
                                "member_id": mid,
                                "cycle": int(cashrun_due_cycle or 0),
                                "t_beneficiary": int(cashrun_due_ben_t or 0),
                                "forced": bool(cashrun_force_due),
                            }
                        )
                        next_member_id = self._schedule_replacement(pending_replacements, next_member_id, t)
                        queue = [q for q in queue if q != mid]

                # reset due
                cashrun_due_member = None
                cashrun_due_time = None
                cashrun_force_due = False
                cashrun_due_cycle = None
                cashrun_due_ben_t = None

            # update missed streaks for active members
            active_ids = self._active_ids(members)
            newly_out: List[int] = []
            for mid in list(active_ids):
                m = members[mid]
                if mid in contrib_set:
                    m.missed_streak = 0
                    m.paid_since_join += 1
                else:
                    m.missed_streak += 1
                    m.arrears += p.c
                    if m.missed_streak >= 3:
                        self._set_out(members, mid)
                        newly_out.append(mid)

            for mid in newly_out:
                miss3_out_count += 1
                next_member_id = self._schedule_replacement(pending_replacements, next_member_id, t)
                queue = [q for q in queue if q != mid]

            # purge queue
            queue = [mid for mid in queue if mid in members and (not members[mid].out)]

            # recompute A after removals
            A = len([mid for mid in contributors if mid in members and (not members[mid].out)])
            C = A * p.c

            # beneficiary
            beneficiary, skips = self._pick_beneficiary(queue, members)

            shrink = 0.0
            Gimm_eff = 0.0
            shortfall = 0.0
            x = 0.0
            y_base = 0.0
            surplus_to_buffer = 0.0
            y_add = 0.0

            cashrun_forced_flag = 0

            if beneficiary is not None:
                m_ben = members[beneficiary]
                Gimm_eff, shrink = self._apply_shrink(m_ben)
                shortfall = max(0.0, Gimm_eff - C)
                B -= shortfall
                residual = max(0.0, C - Gimm_eff)
                x = min(p.Gdiff, residual)
                residual2 = residual - x
                y_base = min(p.S, max(0.0, residual2))
                surplus_to_buffer = max(0.0, residual2 - y_base)
                y_add = y_base + surplus_to_buffer
                E += x
                B += y_add
                self._credit_escrow(m_ben, t, x)

                if queue:
                    queue.append(queue.pop(0))

                # set strict cashrun due for next turn
                if p.strict_cashrun:
                    cashrun_due_member = beneficiary
                    cashrun_due_time = t + 1
                    cashrun_due_cycle = cycle_t
                    cashrun_due_ben_t = t

                    # deterministic cashrun forcing based on plan (member_id, cycle)
                    if cashrun_plan_sets and beneficiary in cashrun_plan_sets and cycle_t in cashrun_plan_sets[beneficiary]:
                        cashrun_force_due = True
                        cashrun_forced_flag = 1
                        cashrun_forced_events.append(
                            {
                                "member_id": beneficiary,
                                "cycle": cycle_t,
                                "t_beneficiary": t,
                                "t_due": t + 1,
                            }
                        )
                    else:
                        cashrun_force_due = False

            else:
                # no beneficiary: clear due
                cashrun_due_member = None
                cashrun_due_time = None
                cashrun_force_due = False
                cashrun_due_cycle = None
                cashrun_due_ben_t = None

            E_after_credit = E

            is_final_turn = (t == total_turns)
            B, E, vest_paid, vest_forfeit, vest_n = self._settle_vesting(members, B, E, t, settle_all=is_final_turn)

            bonus_per = 0.0
            bonus_paid = 0.0
            disciplined_count = 0
            if is_final_turn:
                B, bonus_per, disciplined_count, bonus_paid = self._pay_bonus_final(members, B)

            rows.append(
                {
                    "t": t,
                    "cycle": cycle_t,
                    "phase": "turn" if not is_final_turn else "final_turn_settlement",
                    "A_t": A,
                    "C_t": C,
                    "beneficiary": beneficiary,
                    "Gimm": p.Gimm,
                    "Gimm_eff": Gimm_eff,
                    "shortfall": shortfall,
                    "x_escrow_credit": x,
                    "y_buffer_add": y_add,
                    "y_buffer_base": y_base,
                    "surplus_to_buffer": surplus_to_buffer,
                    "vesting_paid": vest_paid,
                    "vesting_forfeit": vest_forfeit,
                    "bonus_per": bonus_per,
                    "bonus_paid": bonus_paid,
                    "B_after_interest": B_after_interest,
                    "E_after_interest": E_after_interest,
                    "E_after_credit": E_after_credit,
                    "B_end": B,
                    "E_end": E,
                    "interest_income_I": I,
                    "fee": fee,
                    "platform_rev_cum": platform_rev,
                    "skips": skips,
                    "queue_len": len(queue),
                    "members_active": len(self._active_ids(members)),
                    "pending_repl": len(pending_replacements),
                    "cashrun_out": cashrun_out_count,
                    "miss3_out": miss3_out_count,
                    "force_out": force_out_count,
                    "cashrun_forced": cashrun_forced_flag,
                }
            )

        period_df = pd.DataFrame(rows)
        member_df = pd.DataFrame(
            [
                {
                    "member_id": mid,
                    "out": m.out,
                    "join_t": m.join_t,
                    "paid_since_join": m.paid_since_join,
                    "missed_streak": m.missed_streak,
                    "arrears": m.arrears,
                    "eligible_bonus": m.eligible_bonus,
                    "eligible_vesting": m.eligible_vesting,
                    "remaining_escrow_credits": len(m.escrow_credits),
                }
                for mid, m in members.items()
            ]
        ).sort_values("member_id")

        turn_mask = period_df["phase"] != "init_t0"
        total_immediate = float(period_df.loc[turn_mask, "Gimm_eff"].sum()) if not period_df.empty else 0.0
        total_vesting = float(period_df.loc[turn_mask, "vesting_paid"].sum()) if not period_df.empty else 0.0
        total_bonus = float(period_df.loc[turn_mask, "bonus_paid"].sum()) if not period_df.empty else 0.0
        disciplined_n_end = int((member_df["out"] == False).sum())

        kpi = {
            "scenario": "v4_8",
            "payment_mode": payment_mode,
            "strict_cashrun": p.strict_cashrun,
            "enable_replacement": p.enable_replacement,
            "replacement_delay": p.replacement_delay,
            "probation_q": p.probation_q,
            "num_cycles": p.num_cycles,
            "vesting_lag": p.vesting_lag,
            "platform_rev_total": float(period_df["fee"].sum()) if not period_df.empty else 0.0,
            "disciplined_n_end": disciplined_n_end,
            "avg_total_received_disciplined": ((total_immediate + total_vesting + total_bonus) / disciplined_n_end)
            if disciplined_n_end > 0
            else 0.0,
            "min_B_end": float(period_df["B_end"].min()) if not period_df.empty else 0.0,
            "cashrun_out_total": cashrun_out_count,
            "cashrun_out_events": cashrun_out_events,
            "cashrun_forced_events": cashrun_forced_events,
            "miss3_out_total": miss3_out_count,
            "force_out_total": force_out_count,
            "p_base": p_base,
        }

        return PathResult(period_df=period_df, member_df=member_df, kpi=kpi)


def pretty_params(p: CLCSParams) -> str:
    return (
        f"N={p.N}, c={p.c:,.2f}, gamma={p.gamma:.2f}, delta={p.delta:.2f}, P={p.P:,.2f}, "
        f"Gimm={p.Gimm:,.2f}, Gdiff={p.Gdiff:,.2f}, S={p.S:,.2f}, cycles={p.num_cycles}, K={p.vesting_lag}, "
        f"strict_cashrun={p.strict_cashrun}, repl={p.enable_replacement}, delay={p.replacement_delay}, probation={p.probation_q}"
    )