/* engine.js — faithful JavaScript port of bidrosca_simulator.py (Version B, yield-led).
 *
 * COMPLIANCE NOTE: user-facing copy in the page avoids "investment/savings/return/
 * interest/deposit". Internally the variables keep their engineering names; the UI
 * layer maps them to compliant terms (contribution, allocation, surplus pool,
 * distribution, treasury management fee).
 *
 * This is a closed-system, money-conserving simulator:
 *   pot P = N*c each turn is split into:
 *     - winner immediate slice (fraction of (1-discount) entitlement)
 *     - winner deferred slice  (escrow, vested after a position-dependent lag,
 *                               forfeited on exit)
 *     - the discount (auction price) -> surplus pool (reserve): fee + dividend + retain
 *   reserve absorbs payment shortfalls; excess above a liquid floor is placed with
 *   the bank treasury and earns a distribution (split member/fintech, net of bank fee).
 */

// ---- Mulberry32 seeded RNG (deterministic, like numpy default_rng for reproducibility) ----
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussian(rng) {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
function clip(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function quantile(sorted, q) {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos), rest = pos - base;
  if (sorted[base + 1] !== undefined) return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  return sorted[base];
}
function mean(a) { return a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0; }

// ---- default parameters (mirror BidROSCAParams) ----
export const DEFAULTS = {
  N: 12, c: 25000, num_cycles: 2,
  // auction / slot pricing
  d_min: 0.30, d_max: 0.45, bid_noise: 0.02, bid_above_ask: 0.03, auto_ask_price: false,
  // split model: every winner takes theta*P now, rest deferred to cycle-end, bid eats the deferred
  split_fixed_half: true, theta_immediate: 0.50,
  // discount split
  fee_rate: 0.0, retain_theta: 0.70,
  // position escrow + lag
  position_escrow: true, pos_escrow_max: 0.50, pos_escrow_min: 0.0,
  pos_lag_mode: 'remaining', pos_lag_max: 11, pos_lag_min: 0, pos_curve: 1.0,
  vesting_lag: 4, winner_defer_share: 0.30,
  // solvency stack
  reserve_target: 600000, bank_line: 300000,
  bank_commitment_bps: 50, bank_draw_bps: 900, periods_per_year: 12,
  // treasury / yield sleeve
  invest_enabled: true, invest_yield_annual: 0.08, invest_bank_mgmt_bps: 100,
  invest_liquid_floor: 80000, member_yield_share: 0.85,
  // member protection
  loss_cap_months: 2.0,
  // discipline
  strict_cashrun: true, enable_replacement: true, replacement_delay: 1,
  probation_q: 2, miss_streak_out: 3, shrink_cap: 2.0,
  // stochastic stress (UEMOA realistic)
  p_lo: 0.85, p_hi: 0.92,
  shock_windows: [[5, 7, 0.85], [17, 19, 0.85]],
};

// ---- derived helpers ----
export function potP(p) { return p.N * p.c; }
export function dividendShare(p) { return Math.max(0, 1 - p.fee_rate - p.retain_theta); }
function rCommit(p) { return (p.bank_commitment_bps / 1e4) / p.periods_per_year; }
function rDraw(p) { return Math.pow(1 + p.bank_draw_bps / 1e4, 1 / p.periods_per_year) - 1; }
function rInvestGross(p) { return Math.pow(1 + p.invest_yield_annual, 1 / p.periods_per_year) - 1; }
function rInvestMgmt(p) { return (p.invest_bank_mgmt_bps / 1e4) / p.periods_per_year; }

function posFrac(p, slot) {
  if (p.N <= 1) return 0;
  let lin = clip(1 - slot / (p.N - 1), 0, 1);
  return Math.pow(lin, p.pos_curve);
}
export function escrowShareForSlot(p, slot) {
  if (!p.position_escrow) return p.winner_defer_share;
  return p.pos_escrow_min + (p.pos_escrow_max - p.pos_escrow_min) * posFrac(p, slot);
}
export function lagForSlot(p, slot, turnsLeft) {
  if (!p.position_escrow) return p.vesting_lag;
  if (p.pos_lag_mode === 'remaining') return Math.max(0, turnsLeft);
  return Math.round(p.pos_lag_min + (p.pos_lag_max - p.pos_lag_min) * posFrac(p, slot));
}
// auto-solved minimum discount (asking price) so an early-taker who exits nets <= 0
export function askPriceForSlot(p, slot) {
  if (!p.auto_ask_price) return p.d_min;
  const e = escrowShareForSlot(p, slot);
  if (e >= 0.5) return p.d_min;
  const dReq = (1 - 2 * e) / (2 - 2 * e);
  return clip(Math.max(p.d_min, dReq), 0, 0.99);
}

// Build the per-slot pricing schedule for display
export function slotSchedule(p) {
  const P = potP(p);
  const rows = [];
  for (let slot = 0; slot < p.N; slot++) {
    if (p.split_fixed_half) {
      // Fixed-immediate model: immediate = theta*P (same for all). The asking price
      // (bid) is a slot-dependent schedule that scales from d_max (slot 1) to d_min
      // (last slot); it is subtracted from the deferred half and feeds the reserve.
      const frac = p.N > 1 ? slot / (p.N - 1) : 0;
      const d = p.d_max + (p.d_min - p.d_max) * frac;   // d_max at slot 0 -> d_min at last
      const disc = d * P;
      const immediate = p.theta_immediate * P;
      const deferred = Math.max(0, (1 - p.theta_immediate) * P - disc);
      const exitNet = immediate - disc - deferred; // = 0 by construction (deferred forfeits)
      rows.push({
        slot: slot + 1, escrow: (1 - p.theta_immediate), askFrac: d, discount: disc,
        immediate, deferred, lag: p.N - 1 - slot, exitNet,
        toReserve: disc, // the bid that flows to the reserve/pool
      });
    } else {
      const e = escrowShareForSlot(p, slot);
      const d = askPriceForSlot(p, slot);
      const disc = d * P;
      const immediate = (1 - e) * (P - disc);
      const deferred = e * (P - disc);
      const exitNet = immediate - disc - deferred;
      rows.push({
        slot: slot + 1, escrow: e, askFrac: d, discount: disc,
        immediate, deferred, lag: lagForSlot(p, slot, p.N - 1 - slot), exitNet, toReserve: disc,
      });
    }
  }
  return rows;
}

// ---- single path simulation ----
export function runPath(p, seed, pBase) {
  const P = potP(p);
  const rng = mulberry32(seed);
  const N = p.N, totalTurns = p.N * p.num_cycles;

  // members
  const M = [];
  for (let i = 1; i <= N; i++) M.push({
    id: i, out: false, eligDiv: true, eligVest: true, arrears: 0,
    joinT: 0, paidSince: p.probation_q, wonTurn: null, missed: 0,
    escrow: [], // [tCredit, principal, lag]
    contributed: 0, received: 0, dividends: 0, bidPaid: 0,
    lossTopup: 0, yield: 0, clawback: 0,
  });
  const byId = {}; M.forEach(m => byId[m.id] = m);
  let active = new Set(M.map(m => m.id));
  let nextId = N + 1;
  let pending = []; // [joinT, id]

  let R = 0, Ldrawn = 0, fintechRev = 0, bankFee = 0;
  let investYieldCum = 0, investBankFeeCum = 0;
  let insolvencies = 0, lossCapUnmet = 0;
  let cashrunOut = 0, missOut = 0;
  let queue = [...active].sort((a, b) => a - b);
  let slotInCycle = 0;
  let cashrunDueMember = null, cashrunDueTime = null, cashrunForceDue = false;
  const rows = [];

  for (let t = 1; t <= totalTurns; t++) {
    const cycle = Math.floor((t - 1) / N) + 1;
    if ((t - 1) % N === 0) { slotInCycle = 0; M.forEach(m => m.wonTurn = null); }

    // process replacements
    const stillP = [];
    for (const [jt, id] of pending) {
      if (jt <= t && active.size < N) {
        byId[id] = { id, out: false, eligDiv: true, eligVest: true, arrears: 0, joinT: t, paidSince: 0, wonTurn: null, missed: 0, escrow: [], contributed: 0, received: 0, dividends: 0, bidPaid: 0, lossTopup: 0, yield: 0, clawback: 0 };
        M.push(byId[id]); active.add(id); queue.push(id);
      } else stillP.push([jt, id]);
    }
    pending = stillP;

    queue = queue.filter(id => active.has(id));

    // bank carrying cost
    const commit = rCommit(p) * p.bank_line;
    const drawInt = rDraw(p) * Ldrawn;
    bankFee += commit + drawInt;
    Ldrawn += drawInt;

    // investment yield on reserve above liquid floor
    let yMember = 0, yFintech = 0, bankMgmt = 0, investedBal = 0;
    if (p.invest_enabled && R > p.invest_liquid_floor) {
      investedBal = R - p.invest_liquid_floor;
      const gross = rInvestGross(p) * investedBal;
      bankMgmt = rInvestMgmt(p) * investedBal;
      const net = Math.max(0, gross - bankMgmt);
      yMember = p.member_yield_share * net;
      yFintech = net - yMember;
      fintechRev += yFintech;
      const recip = [...active].filter(id => byId[id].eligDiv);
      if (recip.length && yMember > 0) {
        const per = yMember / recip.length;
        recip.forEach(id => { byId[id].dividends += per; byId[id].yield += per; });
      }
      investYieldCum += yMember + yFintech;
      investBankFeeCum += bankMgmt;
    }

    // contributors (mc_probpay)
    const activeList = [...active];
    let contributors = [];
    for (const id of activeList) {
      let pe = pBase;
      for (const [t0, t1, mult] of p.shock_windows) if (t0 <= t && t <= t1) pe *= mult;
      if (cashrunForceDue && cashrunDueMember === id && cashrunDueTime === t) pe = 0;
      if (rng() < clip(pe, 0, 1)) contributors.push(id);
    }
    const contribSet = new Set(contributors);

    // strict cashrun check on previous winner
    if (p.strict_cashrun && cashrunDueMember !== null && cashrunDueTime === t) {
      const id = cashrunDueMember;
      if (active.has(id) && !contribSet.has(id)) {
        setOut(byId[id], active); cashrunOut++;
        if (p.enable_replacement) { pending.push([t + p.replacement_delay, nextId++]); }
        queue = queue.filter(q => q !== id);
      }
      cashrunDueMember = null; cashrunDueTime = null; cashrunForceDue = false;
    }

    // missed streak / arrears / contributions
    const newlyOut = [];
    for (const id of [...active]) {
      const m = byId[id];
      if (contribSet.has(id)) { m.missed = 0; m.paidSince++; m.contributed += p.c; }
      else {
        m.missed++; m.arrears += p.c;
        if (m.missed >= p.miss_streak_out) { setOut(m, active); newlyOut.push(id); }
      }
    }
    if (newlyOut.length) {
      missOut += newlyOut.length;
      const s = new Set(newlyOut);
      newlyOut.forEach(() => { if (p.enable_replacement) pending.push([t + p.replacement_delay, nextId++]); });
      queue = queue.filter(q => !s.has(q));
    }

    let Aeff = contributors.filter(id => active.has(id)).length;
    const C = Aeff * p.c;

    // auction
    const eligible = queue.filter(id => isEligible(p, byId[id]));
    let winner = null, discFrac = 0;
    if (eligible.length) {
      if (p.auto_ask_price) {
        const ask = askPriceForSlot(p, slotInCycle);
        let best = -1;
        for (const id of eligible) {
          const prem = Math.max(0, p.bid_above_ask + gaussian(rng) * p.bid_noise);
          const bid = clip(ask + prem, ask, 0.99);
          if (bid > best) { best = bid; winner = id; }
        }
        discFrac = best;
      } else {
        const base = slotDiscountBaseline(p, slotInCycle);
        let best = -1;
        for (const id of eligible) {
          const bid = clip(base + gaussian(rng) * p.bid_noise, 0, 0.99);
          if (bid > best) { best = bid; winner = id; }
        }
        discFrac = best;
      }
    }

    const discount = discFrac * P;
    let fee = 0, divPool = 0, retain = 0, winnerImm = 0, winnerDef = 0, shortfall = 0;
    let drawRes = 0, drawBank = 0, repayBank = 0, insolvent = 0;

    if (winner !== null) {
      const winnerSlot = slotInCycle; slotInCycle++;
      const m = byId[winner]; m.wonTurn = t;
      const entitlement = P - discount;
      const turnsLeft = N - 1 - winnerSlot;
      let eLag, immGross;
      if (p.split_fixed_half) {
        // fixed-immediate half; bid eats the deferred half; deferred unlocks at cycle-end
        eLag = Math.max(0, turnsLeft);
        immGross = p.theta_immediate * P;
        winnerDef = Math.max(0, (1 - p.theta_immediate) * P - discount);
      } else {
        const eShare = escrowShareForSlot(p, winnerSlot);
        eLag = lagForSlot(p, winnerSlot, turnsLeft);
        winnerDef = eShare * entitlement;
        immGross = entitlement - winnerDef;
      }
      // shrink (arrears) on immediate leg
      const shrink = Math.min(p.shrink_cap * p.c, m.arrears);
      winnerImm = Math.max(0, immGross - shrink);
      m.arrears = Math.max(0, m.arrears - shrink);
      const shrinkToPool = immGross - winnerImm;

      fee = p.fee_rate * discount;
      retain = p.retain_theta * discount + shrinkToPool;
      divPool = dividendShare(p) * discount;
      fintechRev += fee;
      m.received += winnerImm; m.bidPaid += discount;
      if (winnerDef > 0) m.escrow.push([t, winnerDef, eLag]);

      const recipients = [...active].filter(id => byId[id].eligDiv);
      const dividendCash = recipients.length ? divPool : 0;
      const cashOut = winnerImm + dividendCash + fee;
      const netToReserve = C - cashOut;
      if (netToReserve >= 0) {
        R += netToReserve;
        if (Ldrawn > 0 && R > 0) { repayBank = Math.min(Ldrawn, R); Ldrawn -= repayBank; R -= repayBank; }
      } else {
        shortfall = -netToReserve;
        drawRes = Math.min(R, shortfall); R -= drawRes;
        let rem = shortfall - drawRes;
        if (rem > 0) {
          const avail = Math.max(0, p.bank_line - Ldrawn);
          drawBank = Math.min(avail, rem); Ldrawn += drawBank; rem -= drawBank;
        }
        if (rem > 1e-6) { insolvencies++; insolvent = 1; }
      }
      if (recipients.length && divPool > 0) {
        const per = divPool / recipients.length;
        recipients.forEach(id => byId[id].dividends += per);
      }
      queue = queue.filter(q => q !== winner); queue.push(winner);
      if (p.strict_cashrun) {
        cashrunDueMember = winner; cashrunDueTime = t + 1; cashrunForceDue = false;
      }
    } else {
      R += C;
      if (Ldrawn > 0 && R > 0) { repayBank = Math.min(Ldrawn, R); Ldrawn -= repayBank; R -= repayBank; }
    }

    // vesting
    const isFinal = (t === totalTurns);
    let vestPaid = 0;
    for (const m of M) {
      if (!m.escrow.length) continue;
      const keep = [];
      for (const [tc, pr, lg] of m.escrow) {
        if (isFinal || t >= tc + lg) {
          if (!m.out && m.eligVest) { vestPaid += pr; m.dividends += pr; }
          // forfeited principal already sits in R (it never left)
        } else keep.push([tc, pr, lg]);
      }
      m.escrow = keep;
    }
    if (vestPaid > 0) {
      R -= vestPaid;
      if (R < 0) {
        const need = -R; R = 0;
        const avail = Math.max(0, p.bank_line - Ldrawn);
        const d = Math.min(avail, need); Ldrawn += d; drawBank += d;
        if (need - d > 1e-6) { insolvencies++; insolvent = 1; }
      }
    }

    // cycle-end settlement
    let reserveReturned = 0;
    const isCycleEnd = (t % N === 0);
    if (isCycleEnd) {
      if (Ldrawn > 0 && R > 0) { const er = Math.min(Ldrawn, R); Ldrawn -= er; R -= er; repayBank += er; }
      const escrowLiab = M.reduce((s, m) => s + m.escrow.reduce((a, e) => a + e[1], 0), 0);
      // defector clawback (final): exiters can never end positive
      if (isFinal) {
        for (const m of M) {
          if (!m.out) continue;
          const net = m.received + m.dividends - m.contributed;
          if (net > 0) { m.dividends -= net; m.clawback += net; R += net; }
        }
      }
      // loss-cap top-up (final): disciplined members only
      if (isFinal && p.loss_cap_months > 0) {
        const floor = -p.loss_cap_months * p.c;
        for (const m of M) {
          if (m.out) continue;
          const net = m.received + m.dividends - m.contributed;
          if (net < floor) {
            const need = floor - net, top = Math.min(R, need);
            m.lossTopup += top; m.dividends += top; R -= top;
            if (need - top > 1e-6) lossCapUnmet += (need - top);
          }
        }
      }
      const target = isFinal ? escrowLiab : (p.reserve_target + escrowLiab);
      const excess = Math.max(0, R - target);
      if (excess > 0) {
        const recip = [...active].filter(id => byId[id].eligDiv);
        if (recip.length) { const per = excess / recip.length; recip.forEach(id => byId[id].dividends += per); R -= excess; reserveReturned = excess; }
      }
    }

    rows.push({
      t, cycle, A: Aeff, C, winner, discFrac, discount, winnerImm, winnerDef,
      fee, divPool, retain, shortfall, drawRes, drawBank, R, Ldrawn,
      investedBal, yMember, yFintech, bankMgmt, insolvent,
      activeCount: active.size, reserveReturned,
    });
  }

  // member summary
  const members = M.map(m => ({
    id: m.id, out: m.out, contributed: m.contributed, received: m.received,
    dividends: m.dividends, yield: m.yield, bidPaid: m.bidPaid, lossTopup: m.lossTopup,
    clawback: m.clawback, wonTurn: m.wonTurn,
    net: m.received + m.dividends - m.contributed,
    paidFully: !m.out && m.contributed >= 0.95 * p.c * N * p.num_cycles,
  }));

  const fintechNet = fintechRev - bankFee;
  const minR = Math.min(...rows.map(r => r.R));
  const maxL = Math.max(...rows.map(r => r.Ldrawn));
  return {
    rows, members,
    kpi: {
      solvent: insolvencies === 0, insolvencies,
      fintechRev, bankFee, fintechNet,
      investYieldMember: members.reduce((s, m) => s + m.yield, 0),
      investYieldFintech: investYieldCum - members.reduce((s, m) => s + m.yield, 0),
      bankMgmtTotal: investBankFeeCum,
      minR, maxL, lossCapUnmet,
      disciplinedEnd: members.filter(m => !m.out).length,
      avgInvestedBal: mean(rows.map(r => r.investedBal)),
      peakInvestedBal: Math.max(...rows.map(r => r.investedBal)),
      pBase,
    },
  };

  function setOut(m, act) { if (m.out) return; m.out = true; m.eligDiv = false; m.eligVest = false; act.delete(m.id); }
}

function isEligible(p, m) {
  if (m.out) return false;
  if (p.probation_q > 0 && m.paidSince < p.probation_q) return false;
  if (m.wonTurn !== null) return false;
  return true;
}
function slotDiscountBaseline(p, slot) {
  if (p.N <= 1) return p.d_min;
  const frac = slot / (p.N - 1);
  const hi = p.split_fixed_half ? p.d_max : 0.22;
  return hi + (p.d_min - hi) * frac; // d_max at slot 0 -> d_min at last slot
}

// ---- Monte Carlo across many paths ----
export function runMonteCarlo(p, nPaths, baseSeed) {
  const rng = mulberry32(baseSeed);
  let insolv = 0;
  const maxDraws = [], fintechNets = [], loyalNets = [], allStayNets = [], exitNets = [];
  const memberNets = [];
  let yMemberTot = 0, yFintechTot = 0, bankMgmtTot = 0, bankFeeTot = 0, nRunsAcc = 0;
  let avgInvested = 0;
  for (let i = 0; i < nPaths; i++) {
    const pBase = p.p_lo + (p.p_hi - p.p_lo) * rng();
    const seed = Math.floor(rng() * 2 ** 31);
    const res = runPath(p, seed, pBase);
    const k = res.kpi;
    if (!k.solvent) insolv++;
    maxDraws.push(k.maxL); fintechNets.push(k.fintechNet);
    yMemberTot += k.investYieldMember; yFintechTot += k.investYieldFintech;
    bankMgmtTot += k.bankMgmtTotal; bankFeeTot += k.bankFee; avgInvested += k.avgInvestedBal; nRunsAcc++;
    for (const m of res.members) {
      memberNets.push(m.net);
      if (m.out) exitNets.push(m.net);
      else { allStayNets.push(m.net); if (m.paidFully) loyalNets.push(m.net); }
    }
  }
  const sortNum = a => a.slice().sort((x, y) => x - y);
  const md = sortNum(maxDraws), ln = sortNum(loyalNets), sn = sortNum(allStayNets), en = sortNum(exitNets);
  return {
    nPaths, pInsolv: insolv / nPaths,
    drawP50: quantile(md, 0.5), drawP95: quantile(md, 0.95), drawP99: quantile(md, 0.99),
    fintechMean: mean(fintechNets),
    loyalMean: mean(loyalNets), loyalMedian: quantile(ln, 0.5),
    loyalPctPos: 100 * loyalNets.filter(x => x >= 0).length / Math.max(1, loyalNets.length),
    stayMedian: quantile(sn, 0.5),
    exitMedian: en.length ? quantile(en, 0.5) : 0,
    exitMax: en.length ? en[en.length - 1] : 0,
    yMemberPerCircle: yMemberTot / nRunsAcc,
    yFintechPerCircle: yFintechTot / nRunsAcc,
    bankMgmtPerCircle: bankMgmtTot / nRunsAcc,
    bankFeePerCircle: bankFeeTot / nRunsAcc,
    avgInvested: avgInvested / nRunsAcc,
    memberNets,
  };
}
