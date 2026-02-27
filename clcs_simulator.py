"""clcs_simulator.py
CLCS core simulation engine — deterministic + Monte Carlo.

Replaces clcs_sim.py with these optimizations:
- active_set (set[int]) maintained throughout run_path for O(1) active-member
  count and lookup, replacing 3-4 O(N) _active_ids() calls per turn.
- _set_out() updates active_set in-place so it never goes stale.
- _process_pending_replacements() uses len(active_set) instead of rebuilding
  the full active list.
- Two redundant queue-cleanup passes merged into one per logical removal event.
- _contributors_deterministic uses set lookup (O(1)) instead of list scan (O(N)).

Payment modes:
  deterministic   — first A active members in queue order
  mc_fixedA       — A members sampled uniformly from active set
  mc_probpay      — each active member pays with per-member probability
                    (p_base, per-member overrides, shock windows)

Cashrun rules, vesting escrow, replacement and probation are identical to
clcs_sim.py (v4.8 semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

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

    enable_replacement: bool = False
    replacement_delay: int = 0
    probation_q: int = 2

    strict_cashrun: bool = True
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


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class CLCSSimulator:
    def __init__(self, params: CLCSParams):
        self.p = params

    # -----------------------------------------------------------------------
    # Core helpers
    # -----------------------------------------------------------------------

    def _init_members(self) -> Dict[int, MemberState]:
        return {
            i: MemberState(member_id=i, join_t=0, paid_since_join=self.p.probation_q)
            for i in range(1, self.p.N + 1)
        }

    def _active_ids(self, members: Dict[int, MemberState]) -> List[int]:
        """Public helper — returns sorted list of active member IDs.
        Not called in the hot path (run_path uses active_set directly).
        """
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

    def _pick_beneficiary(
        self, queue: List[int], members: Dict[int, MemberState]
    ) -> Tuple[Optional[int], int]:
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

    def _credit_escrow(self, m: MemberState, t: int, x: float) -> None:
        if x > 0:
            m.escrow_credits.append((t, x))

    def _settle_vesting(
        self,
        members: Dict[int, MemberState],
        B: float,
        E: float,
        t: int,
        settle_all: bool,
    ) -> Tuple[float, float, float, float, int]:
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
                if settle_all or (t >= t_credit + K):
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

    def _pay_bonus_final(
        self, members: Dict[int, MemberState], B: float
    ) -> Tuple[float, float, int, float]:
        disciplined = [mid for mid, m in members.items() if (not m.out) and m.eligible_bonus]
        n_disc = len(disciplined)
        bonus_per = B / n_disc if n_disc else 0.0
        bonus_paid_total = B
        return 0.0, bonus_per, n_disc, bonus_paid_total

    # -----------------------------------------------------------------------
    # Member removal — updates active_set in O(1)
    # -----------------------------------------------------------------------

    def _set_out(
        self, members: Dict[int, MemberState], mid: int, active_set: Set[int]
    ) -> None:
        m = members.get(mid)
        if m is None or m.out:
            return
        m.out = True
        m.eligible_bonus = False
        m.eligible_vesting = False
        active_set.discard(mid)

    # -----------------------------------------------------------------------
    # Payment policies
    # -----------------------------------------------------------------------

    def _contributors_deterministic(
        self, queue: List[int], active_set: Set[int], A: int
    ) -> List[int]:
        # O(1) set lookup per member — faster than the original O(N) list scan
        q = [mid for mid in queue if mid in active_set]
        return q[: max(0, min(A, len(q)))]

    def _contributors_mc_fixedA(
        self, rng: np.random.Generator, active_list: List[int], A: int
    ) -> List[int]:
        A = max(0, min(A, len(active_list)))
        if A == 0:
            return []
        return list(rng.choice(active_list, size=A, replace=False))

    def _p_effective(
        self,
        mid: int,
        t: int,
        p_base: float,
        p_by_member: Optional[Dict[int, float]],
        general_shocks: Optional[List[Tuple[int, int, float]]],
        member_shocks: Optional[Dict[int, List[Tuple[int, int, float]]]],
    ) -> float:
        p = float(p_by_member.get(mid, p_base)) if p_by_member else float(p_base)
        if general_shocks:
            for t0, t1, mult in general_shocks:
                if t0 <= t <= t1:
                    p *= float(mult)
        if member_shocks and mid in member_shocks:
            for t0, t1, p_override in member_shocks[mid]:
                if t0 <= t <= t1:
                    p = float(p_override)
        return float(np.clip(p, 0.0, 1.0))

    def _contributors_mc_probpay(
        self,
        rng: np.random.Generator,
        active_list: List[int],
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
        for mid in active_list:
            p_eff = self._p_effective(mid, t, p_base, p_by_member, general_shocks, member_shocks)
            if cashrun_force_due and cashrun_due_member == mid and cashrun_due_time == t:
                p_eff = 0.0
            if rng.random() < p_eff:
                contributors.append(mid)
        return contributors

    # -----------------------------------------------------------------------
    # Replacement management
    # -----------------------------------------------------------------------

    def _schedule_replacement(
        self, pending: List[Tuple[int, int]], next_id: int, t: int
    ) -> int:
        if not self.p.enable_replacement:
            return next_id
        pending.append((t + self.p.replacement_delay, next_id))
        return next_id + 1

    def _process_pending_replacements(
        self,
        pending: List[Tuple[int, int]],
        members: Dict[int, MemberState],
        queue: List[int],
        t: int,
        active_set: Set[int],
    ) -> List[Tuple[int, int]]:
        still: List[Tuple[int, int]] = []
        for join_t, mid in pending:
            if join_t <= t:
                # O(1) count check — was O(N) call to _active_ids()
                if len(active_set) < self.p.N:
                    members[mid] = MemberState(member_id=mid, join_t=t, paid_since_join=0)
                    active_set.add(mid)
                    queue.append(mid)
                else:
                    still.append((join_t, mid))
            else:
                still.append((join_t, mid))
        return still

    # -----------------------------------------------------------------------
    # Main simulation
    # -----------------------------------------------------------------------

    def run_path(
        self,
        scen: DeterministicScenario,
        payment_mode: str = "mc_probpay",
        seed: int = 42,
        p_base: float = 1.0,
        p_by_member: Optional[Dict[int, float]] = None,
        general_shocks: Optional[List[Tuple[int, int, float]]] = None,
        member_shocks: Optional[Dict[int, List[Tuple[int, int, float]]]] = None,
        cashrun_plan: Optional[Dict[int, List[int]]] = None,
    ) -> PathResult:
        p = self.p
        rng = np.random.default_rng(seed)

        cashrun_plan_sets: Optional[Dict[int, set]] = (
            {mid: set(cycles) for mid, cycles in cashrun_plan.items()}
            if cashrun_plan
            else None
        )

        members = self._init_members()
        # OPTIMIZATION: active_set gives O(1) count + lookup throughout the loop.
        active_set: Set[int] = set(members.keys())
        next_member_id = p.N + 1
        pending_replacements: List[Tuple[int, int]] = []

        B = 0.0
        E = 0.0
        platform_rev = 0.0
        rows: List[Dict[str, Any]] = []

        cashrun_forced_events: List[Dict[str, Any]] = []
        cashrun_out_events: List[Dict[str, Any]] = []

        if p.init_t0_first_cycle:
            B = p.N * p.c
            rows.append({
                "t": 0, "cycle": 0, "phase": "init_t0",
                "A_t": p.N, "C_t": p.N * p.c, "beneficiary": None,
                "B_end": B, "E_end": E,
                "members_active": len(active_set),
                "pending_repl": len(pending_replacements),
                "cashrun_out": 0, "miss3_out": 0, "force_out": 0, "cashrun_forced": 0,
            })

        # Queue starts in member-id order
        queue: List[int] = sorted(active_set)
        total_turns = p.N * p.num_cycles
        a_idx = 0

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

            pending_replacements = self._process_pending_replacements(
                pending_replacements, members, queue, t, active_set
            )

            # Scenario-forced exits
            for mid, t_out in scen.force_out.items():
                if t >= t_out and mid in members and not members[mid].out:
                    self._set_out(members, mid, active_set)
                    force_out_count += 1
                    next_member_id = self._schedule_replacement(pending_replacements, next_member_id, t)

            # OPTIMIZATION: single queue cleanup per turn (was two passes)
            queue = [mid for mid in queue if mid in active_set]

            # Interest
            B, E, I, fee = self._apply_interest_and_fee(B, E)
            platform_rev += fee
            B_after_interest = B
            E_after_interest = E

            # A_sched
            if a_idx >= len(scen.A_sched):
                raise ValueError(f"A_sched too short at turn #{a_idx + 1}.")
            A_sched_val = int(scen.A_sched[a_idx])
            a_idx += 1
            A = max(0, min(A_sched_val, len(active_set)))

            # Contributors
            if scen.contributors_sched is not None:
                if (t - 1) >= len(scen.contributors_sched):
                    raise ValueError(f"contributors_sched too short at turn t={t}.")
                contributors = [mid for mid in scen.contributors_sched[t - 1] if mid in active_set]
                A = len(contributors)
            else:
                # Snapshot active list once for mc modes (order irrelevant)
                active_list = list(active_set)
                if payment_mode == "deterministic":
                    contributors = self._contributors_deterministic(queue, active_set, A)
                elif payment_mode == "mc_fixedA":
                    contributors = self._contributors_mc_fixedA(rng, active_list, A)
                elif payment_mode == "mc_probpay":
                    contributors = self._contributors_mc_probpay(
                        rng, active_list, t, p_base,
                        p_by_member, general_shocks, member_shocks,
                        cashrun_due_member, cashrun_due_time, cashrun_force_due,
                    )
                    A = len(contributors)
                else:
                    raise ValueError(f"Unknown payment_mode={payment_mode!r}")

            contrib_set = set(contributors)

            # Strict cashrun check: last beneficiary must have paid at t
            if p.strict_cashrun and cashrun_due_member is not None and cashrun_due_time == t:
                mid = cashrun_due_member
                if mid in active_set:
                    if mid not in contrib_set:
                        self._set_out(members, mid, active_set)
                        cashrun_out_count += 1
                        cashrun_out_events.append({
                            "t_out": t, "member_id": mid,
                            "cycle": int(cashrun_due_cycle or 0),
                            "t_beneficiary": int(cashrun_due_ben_t or 0),
                            "forced": bool(cashrun_force_due),
                        })
                        next_member_id = self._schedule_replacement(pending_replacements, next_member_id, t)
                        queue = [q for q in queue if q != mid]
                cashrun_due_member = None
                cashrun_due_time = None
                cashrun_force_due = False
                cashrun_due_cycle = None
                cashrun_due_ben_t = None

            # Missed-streak updates (iterate snapshot; active_set may shrink)
            newly_out: List[int] = []
            for mid in list(active_set):
                m = members[mid]
                if mid in contrib_set:
                    m.missed_streak = 0
                    m.paid_since_join += 1
                else:
                    m.missed_streak += 1
                    m.arrears += p.c
                    if m.missed_streak >= 3:
                        self._set_out(members, mid, active_set)
                        newly_out.append(mid)

            if newly_out:
                miss3_out_count += len(newly_out)
                newly_out_set = set(newly_out)
                for _ in newly_out:
                    next_member_id = self._schedule_replacement(pending_replacements, next_member_id, t)
                # OPTIMIZATION: single filtered rebuild (was a second full-pass cleanup)
                queue = [q for q in queue if q not in newly_out_set]

            # Recompute A after removals (some contributors may have been removed)
            A = sum(1 for mid in contributors if mid in active_set)
            C = A * p.c

            # Beneficiary selection
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

                if p.strict_cashrun:
                    cashrun_due_member = beneficiary
                    cashrun_due_time = t + 1
                    cashrun_due_cycle = cycle_t
                    cashrun_due_ben_t = t
                    if (
                        cashrun_plan_sets
                        and beneficiary in cashrun_plan_sets
                        and cycle_t in cashrun_plan_sets[beneficiary]
                    ):
                        cashrun_force_due = True
                        cashrun_forced_flag = 1
                        cashrun_forced_events.append({
                            "member_id": beneficiary, "cycle": cycle_t,
                            "t_beneficiary": t, "t_due": t + 1,
                        })
                    else:
                        cashrun_force_due = False
            else:
                cashrun_due_member = None
                cashrun_due_time = None
                cashrun_force_due = False
                cashrun_due_cycle = None
                cashrun_due_ben_t = None

            E_after_credit = E

            is_final_turn = (t == total_turns)
            B, E, vest_paid, vest_forfeit, vest_n = self._settle_vesting(
                members, B, E, t, settle_all=is_final_turn
            )

            bonus_per = 0.0
            bonus_paid = 0.0
            disciplined_count = 0
            if is_final_turn:
                B, bonus_per, disciplined_count, bonus_paid = self._pay_bonus_final(members, B)

            rows.append({
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
                "members_active": len(active_set),
                "pending_repl": len(pending_replacements),
                "cashrun_out": cashrun_out_count,
                "miss3_out": miss3_out_count,
                "force_out": force_out_count,
                "cashrun_forced": cashrun_forced_flag,
            })

        # ------------------------------------------------------------------
        # Build output DataFrames
        # ------------------------------------------------------------------
        period_df = pd.DataFrame(rows)
        member_df = pd.DataFrame([
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
        ]).sort_values("member_id")

        turn_mask = period_df["phase"] != "init_t0"
        total_immediate = float(period_df.loc[turn_mask, "Gimm_eff"].sum()) if not period_df.empty else 0.0
        total_vesting = float(period_df.loc[turn_mask, "vesting_paid"].sum()) if not period_df.empty else 0.0
        total_bonus = float(period_df.loc[turn_mask, "bonus_paid"].sum()) if not period_df.empty else 0.0
        disciplined_n_end = int((member_df["out"] == False).sum())

        kpi: Dict[str, Any] = {
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
            "avg_total_received_disciplined": (
                (total_immediate + total_vesting + total_bonus) / disciplined_n_end
                if disciplined_n_end > 0 else 0.0
            ),
            "min_B_end": float(period_df["B_end"].min()) if not period_df.empty else 0.0,
            "cashrun_out_total": cashrun_out_count,
            "cashrun_out_events": cashrun_out_events,
            "cashrun_forced_events": cashrun_forced_events,
            "miss3_out_total": miss3_out_count,
            "force_out_total": force_out_count,
            "p_base": p_base,
        }

        return PathResult(period_df=period_df, member_df=member_df, kpi=kpi)


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def pretty_params(p: CLCSParams) -> str:
    return (
        f"N={p.N}, c={p.c:,.2f}, gamma={p.gamma:.2f}, delta={p.delta:.2f}, "
        f"P={p.P:,.2f}, Gimm={p.Gimm:,.2f}, Gdiff={p.Gdiff:,.2f}, S={p.S:,.2f}, "
        f"cycles={p.num_cycles}, K={p.vesting_lag}, "
        f"strict_cashrun={p.strict_cashrun}, repl={p.enable_replacement}, "
        f"delay={p.replacement_delay}, probation={p.probation_q}"
    )
