"""bidrosca_simulator.py
Bid-funded ROSCA — Version B (bank-backed, no t=0 buffer).

This is a re-architecture of the CLCS liquidity scheme that removes the
pre-funded t=0 buffer and the investment-return reward mechanism, and replaces
them with:

  1. A *discount auction* per turn.  The pot P = N*c is awarded to the member
     who bids the largest discount d_t (fraction of the pot they forgo to take
     it now).  Winner receives  payout = P - d_t*P.  The discount d_t*P is the
     only source of "income" in the scheme.

  2. A three-way split of every discount:
        fee_t      = fee_rate      * discount      -> fintech revenue
        retain_t   = retain_theta  * discount      -> solvency reserve R
        dividend_t = (1 - fee_rate - retain_theta) * discount
                                                    -> shared to active members
     A late-slot member who bids ~0 collects accumulated dividends; that is
     their effective return for waiting (the classic chit-fund economics).

  3. A solvency stack, drawn in order when contributions C < winner payout:
        reserve R  (first loss, member-funded via retained discounts)
        bank line L (senior, capped, repaid from future retained discounts)
     If both are exhausted, an insolvency event is recorded.

The discipline machinery (strict cashrun, 3-consecutive-miss exit, escrow with
vesting lag K and forfeiture-on-exit) is preserved from clcs_simulator.py.  The
only structural changes are: no init_t0 buffer, payout sourced from collected
contributions + reserve + bank line, and rewards sourced from auction discounts.

Two versions share this engine:
  Version A (pure mutual)  : set bank_line = 0.0
  Version B (bank-backed)  : set bank_line > 0.0

Payment modes match clcs_simulator: deterministic / mc_fixedA / mc_probpay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class BidROSCAParams:
    # --- group structure ---
    N: int = 10
    c: float = 100.0
    num_cycles: int = 1
    vesting_lag: int = 5          # K: escrow release lag (periods)

    # --- discount auction ---
    # The winning discount is modelled as a fraction of the pot P.  Bids rise
    # for early (scarce) slots and fall toward the end of a cycle.  d_max is the
    # discount a member would bid for the very first slot; d_min for the last.
    d_max: float = 0.30           # max discount fraction (first slot)
    d_min: float = 0.02           # min discount fraction (last slot)
    bid_noise: float = 0.03       # std of idiosyncratic bid noise

    # --- discount split (must sum to <= 1; remainder is dividend) ---
    fee_rate: float = 0.20        # fintech share of each discount
    retain_theta: float = 0.50    # share of each discount retained into reserve R

    # --- solvency stack ---
    reserve0: float = 0.0         # reserve seeded at 0 (no t=0 buffer)
    reserve_target: float = 0.0   # R*: target first-loss reserve carried across cycles.
                                  # At each cycle-end, excess (R - R*) is returned to
                                  # disciplined survivors as a final dividend; min(R, R*)
                                  # rolls forward. 0 = return everything each cycle.
    bank_line: float = 0.0        # committed senior liquidity line L (cap on draws outstanding)
    bank_commitment_bps: float = 50.0   # annual commitment fee on the FULL line, bps
    bank_draw_bps: float = 800.0        # annual interest on DRAWN balance, bps
    periods_per_year: int = 12

    # --- investment sleeve (the reserve is NOT idle: excess is invested for yield) ---
    # Strategy: "liquid floor + invest the excess".  Each period, the reserve keeps a
    # liquid floor = invest_liquid_floor (covers normal shortfalls instantly); the
    # balance ABOVE the floor is invested and earns invest_yield_annual (net of the
    # bank's management fee, which the bank decides).  The gross yield is split:
    #   bank management fee  -> invest_bank_mgmt_bps (annual, on invested balance)
    #   net yield then splits: member_yield_share to members, rest to fintech.
    # If a shortfall exceeds the liquid floor, the invested sleeve is redeemed (at a
    # small break cost invest_break_cost) to cover it before touching the bank line.
    invest_enabled: bool = False
    invest_yield_annual: float = 0.06   # GROSS annual yield the bank generates on the sleeve
    invest_bank_mgmt_bps: float = 100.0 # bank management fee, annual bps on invested balance
    invest_liquid_floor: float = 0.0    # min reserve kept liquid (absolute FCFA)
    member_yield_share: float = 0.75    # share of NET yield distributed to members
    invest_break_cost: float = 0.0      # fractional cost to redeem the sleeve early (0..1)

    # --- SPLIT MODEL: fixed-immediate half, bid eats the deferred half ---
    # When split_fixed_half is on, every winner takes the SAME immediate slice
    # (theta_immediate * P, e.g. half the pot) regardless of slot. The other
    # (1-theta)*P is DEFERRED and unlocks at cycle-end. The winner's bid (discount)
    # is subtracted from their own deferred half and flows to the income pool
    # (reserve/dividend/fee). Because deferred forfeits on exit, early-exit nets
    # exactly 0 for any bid -> the asking price is a pure revenue lever, not a gate.
    split_fixed_half: bool = False
    theta_immediate: float = 0.50       # immediate share of the pot (rest deferred)

    # --- winner's deferred fraction (legacy model, used when split_fixed_half is off) ---
    # The winner is entitled to (1 - discount_frac) of the pot.  Of THAT entitlement,
    # `winner_defer_share` is escrowed (vested after a lag, forfeited on exit) and the
    # rest is paid immediately.  0 = winner takes their whole entitlement now.
    winner_defer_share: float = 0.30

    # --- POSITION-DEPENDENT escrow & vesting (incentive to wait) ---
    # When enabled, the escrow share AND the vesting lag scale with the winner's slot
    # position in the cycle: the EARLIEST winner has the most held back for the longest
    # (they are a borrower who must prove discipline); the LAST winner has 0% escrow and
    # 0 lag (they already proved discipline by waiting).  This makes patience pay and
    # makes win-then-quit structurally unprofitable.
    position_escrow: bool = False
    pos_escrow_max: float = 0.50      # escrow share for the FIRST slot (slot 0)
    pos_escrow_min: float = 0.00      # escrow share for the LAST slot
    pos_lag_max: int = 10             # vesting lag (periods) for the FIRST slot
    pos_lag_min: int = 0              # vesting lag for the LAST slot
    pos_lag_mode: str = "linear"      # "linear" | "remaining" (lag = turns left in cycle)
    pos_curve: float = 1.0            # shape exponent: 1=linear, >1 convex, <1 concave

    # --- auto-solved per-slot asking price (the gate that makes early-exit unprofitable) ---
    # When on, each slot has a MINIMUM discount (asking price) derived so that a winner
    # who takes the pot then exits nets <= 0.  Bidders bid at or above the asking price.
    # This makes "stay to the end" the dominant strategy by construction.
    auto_ask_price: bool = False
    bid_above_ask: float = 0.04       # mean extra discount bid above the asking price (competition)

    # --- OPEN time-value auction (members set the price; we set a reserve floor) ---
    # When on, the bid is NOT a fixed schedule.  Each member's willingness to bid for a
    # slot = rho_monthly * months_saved * immediate_sum, i.e. the time-value of receiving
    # the lump sum that many months earlier than waiting to the last slot, plus a private
    # liquidity-need multiplier and noise.  The highest bidder wins at their bid (the price
    # EMERGES from competition).  reserve_price_frac is OUR minimum acceptable bid (floor);
    # below it the slot still clears but is flagged sub-reserve (we'd rely on the bank).
    open_auction: bool = False
    rho_monthly: float = 0.02         # member monthly cost-of-money (time value of early cash)
    bid_need_sigma: float = 0.25      # spread of private liquidity-need multipliers (lognormal-ish)
    reserve_price_frac: float = 0.0   # our minimum acceptable bid as a fraction of the pot

    # --- escrow on the dividend (optional deferral, mirrors CLCS delta) ---
    # Fraction of each member's per-turn dividend that is escrowed (vested after
    # K periods, forfeited on exit).  0 = pay dividends immediately.
    dividend_escrow_share: float = 0.0

    # --- member protection ---
    # No member may end with a net position worse than -(loss_cap_months * c).
    # If their settled net would be below that floor, the reserve tops them up.
    # 0 = disabled (no protection).  This is funded from R; if R is insufficient
    # the floor is honoured as far as R allows and the gap is recorded.
    loss_cap_months: float = 0.0

    # --- discipline rules (same semantics as CLCS) ---
    strict_cashrun: bool = True
    enable_replacement: bool = False
    replacement_delay: int = 0
    probation_q: int = 2
    miss_streak_out: int = 3
    shrink_cap: float = 2.0       # arrears shrink cap (x c) applied to payout

    def __post_init__(self):
        assert self.N >= 2
        assert self.c > 0
        assert self.num_cycles >= 1
        assert self.vesting_lag >= 0
        assert 0.0 <= self.d_min <= self.d_max < 1.0
        assert self.fee_rate >= 0 and self.retain_theta >= 0
        assert self.fee_rate + self.retain_theta <= 1.0 + 1e-9, "fee + retain must be <= 1"
        assert self.bank_line >= 0
        assert self.periods_per_year in (12, 52, 26, 24, 4, 1)
        assert 0.0 <= self.dividend_escrow_share <= 1.0
        assert 0.0 <= self.winner_defer_share <= 1.0
        assert 0.0 <= self.member_yield_share <= 1.0
        assert 0.0 <= self.invest_break_cost < 1.0
        assert self.invest_liquid_floor >= 0.0

    @property
    def P(self) -> float:
        return self.N * self.c

    @property
    def dividend_share(self) -> float:
        return max(0.0, 1.0 - self.fee_rate - self.retain_theta)

    @property
    def r_commit(self) -> float:
        return (self.bank_commitment_bps / 1e4) / self.periods_per_year

    @property
    def r_draw(self) -> float:
        return (1.0 + self.bank_draw_bps / 1e4) ** (1.0 / self.periods_per_year) - 1.0

    @property
    def r_invest_gross(self) -> float:
        """Per-period gross investment yield on the invested sleeve."""
        return (1.0 + self.invest_yield_annual) ** (1.0 / self.periods_per_year) - 1.0

    @property
    def r_invest_mgmt(self) -> float:
        """Per-period bank management fee rate on the invested balance."""
        return (self.invest_bank_mgmt_bps / 1e4) / self.periods_per_year


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class MemberState:
    member_id: int
    out: bool = False
    eligible_dividend: bool = True
    eligible_vesting: bool = True
    arrears: float = 0.0
    join_t: int = 0
    paid_since_join: int = 0
    won_turn: Optional[int] = None        # turn at which this member won the pot (this cycle)
    escrow_credits: List[Tuple[int, float]] = field(default_factory=list)
    missed_streak: int = 0
    # economics tracking
    total_contributed: float = 0.0
    total_received: float = 0.0           # pot payouts received
    total_dividends: float = 0.0          # dividends paid out (cash, post-vesting)
    total_bid_paid: float = 0.0           # discount forgone when winning
    loss_topup: float = 0.0               # reserve top-up received via loss cap
    total_yield: float = 0.0              # investment yield distributed to this member
    defector_clawback: float = 0.0        # gains clawed back on exit (defection penalty)


@dataclass
class DeterministicScenario:
    A_sched: List[int]
    force_out: Dict[int, int] = field(default_factory=dict)
    contributors_sched: Optional[List[List[int]]] = None
    win_order: Optional[List[int]] = None   # explicit winner per turn (overrides auction)


@dataclass
class PathResult:
    period_df: pd.DataFrame
    member_df: pd.DataFrame
    kpi: Dict[str, Any]


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class BidROSCASimulator:
    def __init__(self, params: BidROSCAParams):
        self.p = params

    # ---- helpers -----------------------------------------------------------

    def _init_members(self) -> Dict[int, MemberState]:
        return {
            i: MemberState(member_id=i, join_t=0, paid_since_join=self.p.probation_q)
            for i in range(1, self.p.N + 1)
        }

    def _is_eligible_for_turn(self, m: MemberState) -> bool:
        if m.out:
            return False
        if self.p.probation_q > 0 and m.paid_since_join < self.p.probation_q:
            return False
        if m.won_turn is not None:          # already won this cycle -> can't win again
            return False
        return True

    def _set_out(self, members, mid, active_set: Set[int]) -> None:
        m = members.get(mid)
        if m is None or m.out:
            return
        m.out = True
        m.eligible_dividend = False
        m.eligible_vesting = False
        active_set.discard(mid)

    def _apply_shrink(self, m: MemberState, gross_payout: float) -> Tuple[float, float]:
        """Reduce a member's payout by their accumulated arrears (capped)."""
        shrink = min(self.p.shrink_cap * self.p.c, m.arrears)
        payout_eff = max(0.0, gross_payout - shrink)
        m.arrears -= shrink
        if m.arrears < 1e-9:
            m.arrears = 0.0
        return payout_eff, shrink

    # ---- contributors ------------------------------------------------------

    def _p_effective(self, mid, t, p_base, p_by_member, general_shocks, member_shocks) -> float:
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

    def _contributors(self, rng, active_list, t, payment_mode, A,
                      p_base, p_by_member, general_shocks, member_shocks,
                      cashrun_due_member, cashrun_due_time, cashrun_force_due) -> List[int]:
        if payment_mode == "deterministic":
            return active_list[: max(0, min(A, len(active_list)))]
        if payment_mode == "mc_fixedA":
            A = max(0, min(A, len(active_list)))
            return list(rng.choice(active_list, size=A, replace=False)) if A else []
        # mc_probpay
        contributors = []
        for mid in active_list:
            p_eff = self._p_effective(mid, t, p_base, p_by_member, general_shocks, member_shocks)
            if cashrun_force_due and cashrun_due_member == mid and cashrun_due_time == t:
                p_eff = 0.0
            if rng.random() < p_eff:
                contributors.append(mid)
        return contributors

    # ---- auction -----------------------------------------------------------

    def _slot_discount_baseline(self, slot_idx_in_cycle: int) -> float:
        """Baseline discount fraction for the s-th slot awarded this cycle.

        Linearly decays from d_max (first slot, slot_idx 0) to d_min (last slot).
        slot_idx_in_cycle is 0-based: 0 = first winner of the cycle.
        """
        N = self.p.N
        if N <= 1:
            return self.p.d_min
        frac = slot_idx_in_cycle / (N - 1)          # 0 .. 1
        return self.p.d_max + (self.p.d_min - self.p.d_max) * frac

    def _run_auction(self, rng, eligible: List[int], members, slot_idx_in_cycle: int,
                     bid_need: Optional[Dict[int, float]]) -> Tuple[Optional[int], float]:
        """Return (winner_id, winning_discount_fraction).

        Each eligible member draws a bid = baseline(slot) * need_multiplier + noise.
        Highest bid wins.  bid_need lets callers raise specific members' urgency.
        """
        if not eligible:
            return None, 0.0
        if self.p.open_auction:
            # Open time-value auction: each member's willingness = time-value of getting
            # the immediate sum `months_saved` months early, scaled by a private need.
            N = self.p.N
            months_saved = max(0, (N - 1) - slot_idx_in_cycle)
            base_frac = self.p.rho_monthly * months_saved * self.p.theta_immediate
            best_mid, best_bid = None, -1.0
            for mid in eligible:
                need = float(bid_need.get(mid, 1.0)) if bid_need else 1.0
                # private need multiplier ~ lognormal around 1.0
                priv = float(np.exp(rng.normal(0.0, self.p.bid_need_sigma)))
                wtp = base_frac * need * priv
                if wtp > best_bid:
                    best_bid, best_mid = wtp, mid
            # winner pays their willingness-to-pay (clipped); price emerges from demand
            best_bid = float(np.clip(best_bid, 0.0, 1.0 - self.p.theta_immediate))
            return best_mid, best_bid
        if self.p.auto_ask_price:
            # Bids start at the auto-solved asking price (the gate) and compete above it.
            ask = self._ask_price_for_slot(slot_idx_in_cycle)
            best_mid, best_bid = None, -1.0
            for mid in eligible:
                need = float(bid_need.get(mid, 1.0)) if bid_need else 1.0
                premium = max(0.0, self.p.bid_above_ask * need + rng.normal(0.0, self.p.bid_noise))
                bid = float(np.clip(ask + premium, ask, 0.99))
                if bid > best_bid:
                    best_bid, best_mid = bid, mid
            return best_mid, best_bid
        base = self._slot_discount_baseline(slot_idx_in_cycle)
        best_mid, best_bid = None, -1.0
        for mid in eligible:
            need = float(bid_need.get(mid, 1.0)) if bid_need else 1.0
            bid = base * need + rng.normal(0.0, self.p.bid_noise)
            bid = float(np.clip(bid, 0.0, 0.99))
            if bid > best_bid:
                best_bid, best_mid = bid, mid
        return best_mid, best_bid

    # ---- position-dependent escrow & lag -----------------------------------

    def _pos_frac(self, slot_idx_in_cycle: int) -> float:
        """Position weight in [0,1]: 1.0 for the FIRST slot, 0.0 for the LAST.
        Shaped by pos_curve (1=linear, >1 convex, <1 concave)."""
        N = self.p.N
        if N <= 1:
            return 0.0
        lin = 1.0 - slot_idx_in_cycle / (N - 1)      # 1 at slot 0 -> 0 at slot N-1
        lin = float(np.clip(lin, 0.0, 1.0))
        return lin ** self.p.pos_curve

    def _escrow_share_for_slot(self, slot_idx_in_cycle: int) -> float:
        if not self.p.position_escrow:
            return self.p.winner_defer_share
        f = self._pos_frac(slot_idx_in_cycle)
        return self.p.pos_escrow_min + (self.p.pos_escrow_max - self.p.pos_escrow_min) * f

    def _ask_price_for_slot(self, slot_idx_in_cycle: int) -> float:
        """Auto-solved MINIMUM discount fraction (asking price) for a slot, set so an
        early-taker who exits nets <= 0:  immediate_cash - discount - forfeit <= 0.

        With escrow share e at this slot, immediate = (1-e)(P-dP), forfeit-on-exit =
        e(P-dP), discount = dP.  Requiring (1-e)(P-dP) - dP - e(P-dP) <= 0 gives
            d >= (1 - 2e) / (2 - 2e).
        If e >= 0.5 the immediate slice is already <= the forfeit, so no premium is
        needed and the asking price is just d_min.  Late slots (e->0) ask ~0.5 only
        if nothing is deferred; in practice e and d_min interact, so we floor at d_min.
        """
        if not self.p.auto_ask_price:
            return self.p.d_min
        e = self._escrow_share_for_slot(slot_idx_in_cycle)
        if e >= 0.5:
            return self.p.d_min
        d_req = (1.0 - 2.0 * e) / (2.0 - 2.0 * e)
        return float(np.clip(max(self.p.d_min, d_req), 0.0, 0.99))

    def _lag_for_slot(self, slot_idx_in_cycle: int, turns_left_in_cycle: int) -> int:
        if not self.p.position_escrow:
            return self.p.vesting_lag
        if self.p.pos_lag_mode == "remaining":
            return int(max(0, turns_left_in_cycle))
        f = self._pos_frac(slot_idx_in_cycle)
        return int(round(self.p.pos_lag_min + (self.p.pos_lag_max - self.p.pos_lag_min) * f))

    # ---- escrow / vesting (dividends only here) ----------------------------

    def _credit_escrow(self, m, t, x, lag=None):
        if x > 0:
            m.escrow_credits.append((t, x, self.p.vesting_lag if lag is None else int(lag)))

    def _settle_vesting(self, members, t, settle_all: bool) -> Tuple[float, float, int]:
        """Release escrowed dividends whose per-credit lag has elapsed.  Returns
        (paid_to_active, forfeited_to_reserve, n_credits)."""
        paid_total = 0.0
        forfeit_total = 0.0
        n_credits = 0
        for m in members.values():
            if not m.escrow_credits:
                continue
            keep = []
            for t_credit, principal, lag in m.escrow_credits:
                if settle_all or (t >= t_credit + lag):
                    if (not m.out) and m.eligible_vesting:
                        paid_total += principal
                        m.total_dividends += principal
                    else:
                        forfeit_total += principal
                    n_credits += 1
                else:
                    keep.append((t_credit, principal, lag))
            m.escrow_credits = keep
        return paid_total, forfeit_total, n_credits

    # ---- replacement -------------------------------------------------------

    def _schedule_replacement(self, pending, next_id, t):
        if not self.p.enable_replacement:
            return next_id
        pending.append((t + self.p.replacement_delay, next_id))
        return next_id + 1

    def _process_pending_replacements(self, pending, members, queue, t, active_set):
        still = []
        for join_t, mid in pending:
            if join_t <= t and len(active_set) < self.p.N:
                members[mid] = MemberState(member_id=mid, join_t=t, paid_since_join=0)
                active_set.add(mid)
                queue.append(mid)
            else:
                still.append((join_t, mid))
        return still

    # ---- main loop ---------------------------------------------------------

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
        bid_need: Optional[Dict[int, float]] = None,
    ) -> PathResult:
        p = self.p
        rng = np.random.default_rng(seed)

        cashrun_plan_sets = ({mid: set(cy) for mid, cy in cashrun_plan.items()}
                             if cashrun_plan else None)

        members = self._init_members()
        active_set: Set[int] = set(members.keys())
        next_member_id = p.N + 1
        pending_replacements: List[Tuple[int, int]] = []

        # Solvency stack — NO t=0 buffer.
        R = p.reserve0          # reserve (first loss)
        L_drawn = 0.0           # outstanding bank line draw
        fintech_rev = 0.0       # cumulative fintech fee income
        bank_fee_paid = 0.0     # cumulative bank commitment+draw interest (a fintech cost)

        rows: List[Dict[str, Any]] = []
        insolvency_events = 0
        loss_cap_unmet = 0.0    # total loss-cap top-up the reserve could not fund
        invest_yield_cum = 0.0          # cumulative yield distributed (member + fintech)
        invest_bank_fee_cum = 0.0       # cumulative bank management fee on the sleeve
        invest_break_cum = 0.0          # cumulative early-redemption break cost

        queue: List[int] = sorted(active_set)
        total_turns = p.N * p.num_cycles

        cashrun_due_member = cashrun_due_time = cashrun_due_cycle = cashrun_due_ben_t = None
        cashrun_force_due = False
        cashrun_out_count = miss_out_count = force_out_count = 0
        a_idx = 0
        slot_in_cycle = 0

        for t in range(1, total_turns + 1):
            cycle_t = (t - 1) // p.N + 1
            if (t - 1) % p.N == 0:
                slot_in_cycle = 0
                # new cycle: members may win again
                for m in members.values():
                    m.won_turn = None

            pending_replacements = self._process_pending_replacements(
                pending_replacements, members, queue, t, active_set)

            # scenario-forced exits
            for mid, t_out in scen.force_out.items():
                if t >= t_out and mid in members and not members[mid].out:
                    self._set_out(members, mid, active_set)
                    force_out_count += 1
                    next_member_id = self._schedule_replacement(pending_replacements, next_member_id, t)

            queue = [mid for mid in queue if mid in active_set]

            # --- bank carrying cost (commitment on full line + interest on draw) ---
            commit_fee = p.r_commit * p.bank_line
            draw_interest = p.r_draw * L_drawn
            bank_period_cost = commit_fee + draw_interest
            bank_fee_paid += bank_period_cost
            L_drawn += draw_interest        # interest capitalises onto the draw

            # --- investment yield on the reserve (the reserve is NOT idle) ---
            # The balance ABOVE the liquid floor is invested.  It earns a gross yield;
            # the bank takes a management fee; the NET yield is split member/fintech.
            # The member share is paid out as a yield dividend; the fintech share is
            # booked to revenue.  Principal stays in R (yield is the only thing
            # distributed) so solvency capacity is unchanged.
            invest_yield_member = invest_yield_fintech = invest_bank_fee = 0.0
            invested_balance = 0.0
            if p.invest_enabled and R > p.invest_liquid_floor:
                invested_balance = R - p.invest_liquid_floor
                gross = p.r_invest_gross * invested_balance
                invest_bank_fee = p.r_invest_mgmt * invested_balance
                net = max(0.0, gross - invest_bank_fee)
                invest_yield_member = p.member_yield_share * net
                invest_yield_fintech = net - invest_yield_member
                fintech_rev += invest_yield_fintech
                # distribute member yield to active eligible members as a dividend
                recip = [mm for mm in active_set if members[mm].eligible_dividend]
                if recip and invest_yield_member > 0:
                    per = invest_yield_member / len(recip)
                    for mm in recip:
                        members[mm].total_dividends += per
                        members[mm].total_yield += per
                # the yield is real cash earned by the sleeve; principal R is unchanged,
                # the distributed member+fintech yield is funded by the gross return, and
                # the bank fee is netted out — so no draw on R principal occurs.
                invest_yield_cum += (invest_yield_member + invest_yield_fintech)
                invest_bank_fee_cum += invest_bank_fee

            # --- A schedule ---
            if a_idx >= len(scen.A_sched):
                raise ValueError(f"A_sched too short at turn #{a_idx + 1}.")
            A_sched_val = int(scen.A_sched[a_idx]); a_idx += 1
            A = max(0, min(A_sched_val, len(active_set)))

            # --- contributors ---
            if scen.contributors_sched is not None:
                if (t - 1) >= len(scen.contributors_sched):
                    raise ValueError(f"contributors_sched too short at t={t}.")
                contributors = [mid for mid in scen.contributors_sched[t - 1] if mid in active_set]
            else:
                active_list = list(active_set)
                contributors = self._contributors(
                    rng, active_list, t, payment_mode, A,
                    p_base, p_by_member, general_shocks, member_shocks,
                    cashrun_due_member, cashrun_due_time, cashrun_force_due)
            contrib_set = set(contributors)

            # --- strict cashrun check on previous winner ---
            if p.strict_cashrun and cashrun_due_member is not None and cashrun_due_time == t:
                mid = cashrun_due_member
                if mid in active_set and mid not in contrib_set:
                    self._set_out(members, mid, active_set)
                    cashrun_out_count += 1
                    next_member_id = self._schedule_replacement(pending_replacements, next_member_id, t)
                    queue = [q for q in queue if q != mid]
                cashrun_due_member = cashrun_due_time = None
                cashrun_force_due = False

            # --- missed-streak / arrears / contribution tally ---
            newly_out = []
            for mid in list(active_set):
                m = members[mid]
                if mid in contrib_set:
                    m.missed_streak = 0
                    m.paid_since_join += 1
                    m.total_contributed += p.c
                else:
                    m.missed_streak += 1
                    m.arrears += p.c
                    if m.missed_streak >= p.miss_streak_out:
                        self._set_out(members, mid, active_set)
                        newly_out.append(mid)
            if newly_out:
                miss_out_count += len(newly_out)
                s = set(newly_out)
                for _ in newly_out:
                    next_member_id = self._schedule_replacement(pending_replacements, next_member_id, t)
                queue = [q for q in queue if q not in s]

            A_eff = sum(1 for mid in contributors if mid in active_set)
            C = A_eff * p.c

            # --- auction: who wins this slot ---
            eligible = [mid for mid in queue if self._is_eligible_for_turn(members[mid])]
            if scen.win_order is not None and (t - 1) < len(scen.win_order):
                forced_w = scen.win_order[t - 1]
                winner = forced_w if (forced_w in active_set and members[forced_w].won_turn is None) else None
                disc_frac = self._slot_discount_baseline(slot_in_cycle) if winner is not None else 0.0
            else:
                winner, disc_frac = self._run_auction(rng, eligible, members, slot_in_cycle, bid_need)

            discount = disc_frac * p.P
            fee = dividend_pool = retain = 0.0
            winner_entitlement = winner_immediate = winner_deferred = 0.0
            payout_eff = shrink = shortfall = 0.0
            draw_from_reserve = draw_from_bank = repay_bank = 0.0
            insolvent_flag = 0

            # ── MONEY CONSERVATION ─────────────────────────────────────────────
            # The pot is P.  It is partitioned every turn as:
            #   discount (disc_frac * P)  -> income pool, split fee+dividend+retain
            #   winner entitlement (1-disc_frac)*P  -> split immediate + deferred(escrow)
            # The discount is NOT new money: it is the slice of the pot the winner
            # forgoes.  Real cash available to fund the payout is C (contributions)
            # plus reserve/bank for any shortfall.  The income pool (fee+dividend+
            # retain) is funded from the same C — it is the part of P the winner
            # did NOT take immediately.  Total cash out (winner_immediate + fee +
            # dividend_cash + retain) == cash in (C [+reserve/bank draw]) by
            # construction, so the closed-system identity holds exactly.
            if winner is not None:
                winner_slot = slot_in_cycle          # 0-based slot index of THIS winner
                slot_in_cycle += 1
                m_w = members[winner]
                m_w.won_turn = t

                winner_entitlement = p.P - discount
                turns_left = p.N - 1 - winner_slot   # turns remaining in this cycle

                if p.split_fixed_half:
                    # Fixed-immediate half; bid is subtracted from the deferred half.
                    # immediate = theta*P (same for everyone); deferred = (1-theta)*P - bid.
                    esc_lag = max(0, turns_left)     # deferred unlocks at cycle-end
                    winner_immediate_gross = p.theta_immediate * p.P
                    winner_deferred = max(0.0, (1.0 - p.theta_immediate) * p.P - discount)
                else:
                    # legacy: position-dependent escrow on the (P-discount) entitlement
                    esc_share = self._escrow_share_for_slot(winner_slot)
                    esc_lag = self._lag_for_slot(winner_slot, turns_left)
                    winner_deferred = esc_share * winner_entitlement
                    winner_immediate_gross = winner_entitlement - winner_deferred

                # arrears shrink applies to the immediate cash leg only
                winner_immediate, shrink = self._apply_shrink(m_w, winner_immediate_gross)
                # any shrunk amount stays in the pot as extra income -> reserve
                shrink_to_pool = winner_immediate_gross - winner_immediate

                # split the discount (the income pool)
                fee = p.fee_rate * discount
                retain = p.retain_theta * discount + shrink_to_pool
                dividend_pool = p.dividend_share * discount
                fintech_rev += fee

                m_w.total_received += winner_immediate
                m_w.total_bid_paid += discount
                if winner_deferred > 0:
                    self._credit_escrow(m_w, t, winner_deferred, lag=esc_lag)   # position-dep lag

                # --- cash out this turn = winner_immediate + dividend_cash + fee ---
                # (retain stays in reserve, not paid out; deferred is escrowed, not cash yet)
                recipients = [mm for mm in active_set if members[mm].eligible_dividend]
                div_esc = p.dividend_escrow_share
                dividend_cash = dividend_pool * (1.0 - div_esc) if recipients else 0.0
                cash_out = winner_immediate + dividend_cash + fee

                # --- fund cash_out: contributions first, then reserve, then bank line ---
                # Net cash flowing into the reserve this turn = C - cash_out.  This
                # ALREADY contains the retained income and the winner-deferred slice
                # (both stayed in the pot rather than being paid out as cash), so we
                # must NOT add `retain` again — that would double-count it.
                net_to_reserve = C - cash_out
                if net_to_reserve >= 0:
                    R += net_to_reserve
                    if L_drawn > 0 and R > 0:
                        repay_bank = min(L_drawn, R)
                        L_drawn -= repay_bank
                        R -= repay_bank
                else:
                    shortfall = -net_to_reserve
                    draw_from_reserve = min(R, shortfall)
                    # break cost: portion of the draw that dips below the liquid floor
                    # redeems the invested sleeve early, at a fractional penalty.
                    if p.invest_enabled and p.invest_break_cost > 0:
                        invested_drawn = max(0.0, draw_from_reserve - max(0.0, R - invested_balance))
                        invested_drawn = min(invested_drawn, draw_from_reserve)
                        brk = p.invest_break_cost * invested_drawn
                        if brk > 0:
                            R = max(0.0, R - brk)
                            invest_break_cum += brk
                    R -= draw_from_reserve
                    rem = shortfall - draw_from_reserve
                    if rem > 0:
                        avail_line = max(0.0, p.bank_line - L_drawn)
                        draw_from_bank = min(avail_line, rem)
                        L_drawn += draw_from_bank
                        rem -= draw_from_bank
                    if rem > 1e-9:
                        insolvency_events += 1
                        insolvent_flag = 1

                # pay dividends (cash leg now, escrow leg vested later)
                if recipients and dividend_pool > 0:
                    per = dividend_pool / len(recipients)
                    for mm in recipients:
                        members[mm].total_dividends += per * (1.0 - div_esc)
                        if div_esc > 0:
                            self._credit_escrow(members[mm], t, per * div_esc)

                # advance queue
                if queue:
                    queue = [q for q in queue if q != winner] + [winner]

                # set up cashrun obligation for next turn
                if p.strict_cashrun:
                    cashrun_due_member = winner
                    cashrun_due_time = t + 1
                    cashrun_due_cycle = cycle_t
                    cashrun_due_ben_t = t
                    cashrun_force_due = bool(
                        cashrun_plan_sets and winner in cashrun_plan_sets
                        and cycle_t in cashrun_plan_sets[winner])
            else:
                # no winner this turn: surplus contributions top up reserve
                R += C
                if L_drawn > 0 and R > 0:
                    repay_bank = min(L_drawn, R)
                    L_drawn -= repay_bank
                    R -= repay_bank

            # --- vesting settlement (winner-deferred + escrowed dividends) ---
            # Escrowed principal was retained in the reserve when credited, so paying
            # it out now reduces R; if the reserve was depleted by earlier shortfalls,
            # the vesting payout draws the bank line (then flags insolvency).
            is_final = (t == total_turns)
            vest_paid, vest_forfeit, _ = self._settle_vesting(members, t, settle_all=is_final)
            # vest_forfeit already sits in R (never left); nothing to add.
            if vest_paid > 0:
                R -= vest_paid
                if R < 0:
                    need = -R
                    R = 0.0
                    avail_line = max(0.0, p.bank_line - L_drawn)
                    d = min(avail_line, need)
                    L_drawn += d
                    draw_from_bank += d
                    if need - d > 1e-9:
                        insolvency_events += 1
                        insolvent_flag = 1

            # --- cycle-end reserve settlement: return excess over target R* ---
            # Triggered on the last turn of every cycle (incl. the final turn).
            # First repay any outstanding bank draw from the reserve, then keep
            # min(R, R*) rolling forward and pay the excess to disciplined survivors.
            is_cycle_end = (t % p.N == 0)
            reserve_returned = 0.0
            if is_cycle_end:
                if L_drawn > 0 and R > 0:
                    extra_repay = min(L_drawn, R)
                    L_drawn -= extra_repay
                    R -= extra_repay
                    repay_bank += extra_repay
                # Outstanding escrow principal is a liability backed by the reserve;
                # it must NOT be returned as excess or the reserve will go negative
                # when that escrow later vests.
                escrow_liability = sum(
                    pr for m in members.values() for (_, pr, _lg) in m.escrow_credits
                )
                # --- defector clawback (final turn): exiting must never be profitable ---
                # Any member who exited (quit / cashrun-out / 3-miss-out / force-out)
                # forfeits ALL gains: their net is clawed back to at most 0 (they can
                # never end ahead by abandoning the circle).  The clawed-back surplus
                # returns to the reserve, which protects the disciplined members who
                # stayed.  This restores the CLCS property that only discipline wins.
                clawback_total = 0.0
                if is_final:
                    for m in members.values():
                        if not m.out:
                            continue
                        net = m.total_received + m.total_dividends - m.total_contributed
                        if net > 0:
                            # remove the surplus by treating it as a forfeited gain
                            m.total_dividends -= net   # drives this member's net to 0
                            m.defector_clawback += net
                            R += net
                            clawback_total += net

                # --- member loss-cap top-up (final turn only, before returning excess) ---
                # Top up any member whose settled net position is below the floor
                # -(loss_cap_months * c), funded from the reserve.
                if is_final and p.loss_cap_months > 0:
                    floor = -p.loss_cap_months * p.c
                    for m in members.values():
                        # Loss protection is a benefit of DISCIPLINE: it applies only to
                        # members still active at settlement.  Anyone who exited (quit,
                        # cashrun-out, 3-miss-out, force-out) forfeits the floor and bears
                        # their full net loss.  Defecting must never be protected.
                        if m.out:
                            continue
                        net = m.total_received + m.total_dividends - m.total_contributed
                        if net < floor:
                            need = floor - net          # > 0
                            topup = min(R, need)
                            m.loss_topup += topup
                            m.total_dividends += topup   # paid as a protection credit
                            R -= topup
                            if need - topup > 1e-9:
                                loss_cap_unmet += (need - topup)

                target = escrow_liability if is_final else (p.reserve_target + escrow_liability)
                excess = max(0.0, R - target)
                if excess > 0:
                    recipients = [mm for mm in active_set if members[mm].eligible_dividend]
                    if recipients:
                        per = excess / len(recipients)
                        for mm in recipients:
                            members[mm].total_dividends += per
                        R -= excess
                        reserve_returned = excess

            rows.append({
                "t": t, "cycle": cycle_t,
                "phase": "turn" if not is_final else "final_turn",
                "A_t": A_eff, "C_t": C,
                "winner": winner,
                "discount_frac": round(disc_frac, 4),
                "discount": discount,
                "winner_entitlement": winner_entitlement,
                "winner_immediate": winner_immediate,
                "winner_deferred": winner_deferred,
                "shrink": shrink,
                "fee": fee,
                "dividend_pool": dividend_pool,
                "retain_to_reserve": retain,
                "shortfall": shortfall,
                "draw_from_reserve": draw_from_reserve,
                "draw_from_bank": draw_from_bank,
                "repay_bank": repay_bank,
                "R_end": R,
                "L_drawn_end": L_drawn,
                "bank_period_cost": bank_period_cost,
                "vesting_paid": vest_paid,
                "vesting_forfeit": vest_forfeit,
                "reserve_returned": reserve_returned,
                "invested_balance": invested_balance,
                "invest_yield_member": invest_yield_member,
                "invest_yield_fintech": invest_yield_fintech,
                "invest_bank_fee": invest_bank_fee,
                "fintech_rev_cum": fintech_rev,
                "bank_fee_cum": bank_fee_paid,
                "insolvent": insolvent_flag,
                "members_active": len(active_set),
                "queue_len": len(queue),
            })

        # --- outputs ---
        period_df = pd.DataFrame(rows)
        member_df = pd.DataFrame([{
            "member_id": mid, "out": m.out, "join_t": m.join_t,
            "paid_since_join": m.paid_since_join, "missed_streak": m.missed_streak,
            "arrears": m.arrears, "won_turn": m.won_turn,
            "total_contributed": m.total_contributed,
            "total_received": m.total_received,
            "total_dividends": m.total_dividends,
            "total_bid_paid": m.total_bid_paid,
            "loss_topup": m.loss_topup,
            "total_yield": m.total_yield,
            "defector_clawback": m.defector_clawback,
            "net_position": m.total_received + m.total_dividends - m.total_contributed,
        } for mid, m in members.items()]).sort_values("member_id")

        fintech_net = fintech_rev - bank_fee_paid
        kpi = {
            "version": "B_bank_backed" if p.bank_line > 0 else "A_pure_mutual",
            "payment_mode": payment_mode,
            "N": p.N, "c": p.c, "num_cycles": p.num_cycles,
            "P": p.P,
            "fee_rate": p.fee_rate, "retain_theta": p.retain_theta,
            "dividend_share": p.dividend_share,
            "bank_line": p.bank_line,
            "fintech_rev_total": fintech_rev,
            "bank_fee_total": bank_fee_paid,
            "fintech_net_profit": fintech_net,
            "min_R_end": float(period_df["R_end"].min()) if not period_df.empty else 0.0,
            "max_L_drawn": float(period_df["L_drawn_end"].max()) if not period_df.empty else 0.0,
            "insolvency_events": insolvency_events,
            "solvent": insolvency_events == 0,
            "loss_cap_months": p.loss_cap_months,
            "loss_topup_total": float(member_df["loss_topup"].sum()) if not member_df.empty else 0.0,
            "loss_cap_unmet": loss_cap_unmet,
            "loss_cap_honoured": loss_cap_unmet < 1e-6,
            "worst_member_net": float(member_df["net_position"].min()) if not member_df.empty else 0.0,
            "invest_enabled": p.invest_enabled,
            "invest_yield_annual": p.invest_yield_annual,
            "invest_yield_member_total": float(member_df["total_yield"].sum()) if not member_df.empty else 0.0,
            "invest_yield_fintech_total": float(period_df["invest_yield_fintech"].sum()) if (not period_df.empty and "invest_yield_fintech" in period_df) else 0.0,
            "invest_bank_fee_total": invest_bank_fee_cum,
            "invest_break_total": invest_break_cum,
            "cashrun_out_total": cashrun_out_count,
            "miss_out_total": miss_out_count,
            "force_out_total": force_out_count,
            "disciplined_n_end": int((member_df["out"] == False).sum()),
            "p_base": p_base,
        }
        return PathResult(period_df=period_df, member_df=member_df, kpi=kpi)


def pretty_params(p: BidROSCAParams) -> str:
    return (f"N={p.N}, c={p.c:,.2f}, P={p.P:,.2f}, cycles={p.num_cycles}, K={p.vesting_lag} | "
            f"auction d=[{p.d_min:.2f},{p.d_max:.2f}] | split fee={p.fee_rate:.2f} "
            f"retain={p.retain_theta:.2f} div={p.dividend_share:.2f} | "
            f"bank_line={p.bank_line:,.0f} (commit {p.bank_commitment_bps:.0f}bps / "
            f"draw {p.bank_draw_bps:.0f}bps)")
