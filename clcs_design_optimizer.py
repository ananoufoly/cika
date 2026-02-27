"""clcs_design_optimizer.py
Interactive CLCS design optimizer — Monte Carlo bisection search.

Replaces design_solver.py with these improvements:
- inspect.signature() is called ONCE per mc_safety() invocation (before the
  n_sims loop), not 5 000× inside it. With n_sims=5000 and ~10 bisection
  steps this eliminates ~50 000 reflection calls per workflow run.
- Prompt helpers imported from cli_prompts (no duplication).
- Workflow logic and math are identical to the original.

Three workflows:
  1. Maximize gamma (forward design)
  2. Feasible (N, cycles) for given gamma & delta
  3. Feasible (N, cycles) + derive c from a monetary target
"""

from __future__ import annotations

import inspect
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from cli_prompts import (
    ask_bool, ask_choice, ask_float, ask_float_list,
    ask_int, ask_int_range,
)

if "/mnt/data" not in sys.path:
    sys.path.append("/mnt/data")

try:
    from clcs_simulator import CLCSParams, DeterministicScenario, CLCSSimulator
except ImportError as e:
    raise ImportError(
        "Cannot import clcs_simulator. Make sure clcs_simulator.py is in the same folder."
    ) from e


# ---------------------------------------------------------------------------
# Compatibility wrapper (resolves p_base kwarg across simulator versions)
# OPTIMIZATION: inspect.signature is called ONCE and the result cached.
# The original design_solver.py called it inside every MC iteration.
# ---------------------------------------------------------------------------

def _resolve_p_base_kwarg(sim: CLCSSimulator, p_base: float) -> Dict:
    """Inspect run_path signature once and return the correct kwargs dict."""
    sig = inspect.signature(sim.run_path)
    kwargs: Dict = {"payment_mode": "mc_probpay"}
    for name in ("p_base", "p_pay", "pay_prob", "p"):
        if name in sig.parameters:
            kwargs[name] = p_base
            break
    return kwargs


# ---------------------------------------------------------------------------
# K (vesting lag) helper
# ---------------------------------------------------------------------------

def compute_K(N: int, mode: str, fixed: Optional[int]) -> int:
    if mode == "fixed":
        return int(fixed or 0)
    return max(0, N // 3)


# ---------------------------------------------------------------------------
# Monte Carlo safety evaluation
# ---------------------------------------------------------------------------

def mc_safety(
    N: int,
    cycles: int,
    gamma: float,
    delta: float,
    c: float,
    alpha: float,
    p_base: float,
    K: int,
    strict_cashrun: bool,
    enable_replacement: bool,
    replacement_delay: int,
    probation_q: int,
    n_sims: int,
    seed: int,
) -> Dict[str, float]:
    params = CLCSParams(
        N=N, c=c, gamma=gamma, delta=delta,
        num_cycles=cycles, vesting_lag=K,
        strict_cashrun=strict_cashrun,
        enable_replacement=enable_replacement,
        replacement_delay=replacement_delay,
        probation_q=probation_q,
    )
    scen = DeterministicScenario(A_sched=[N] * (N * cycles))
    sim = CLCSSimulator(params)
    rng = np.random.default_rng(seed)

    # OPTIMIZATION: resolve kwargs once before the loop
    base_kwargs = _resolve_p_base_kwarg(sim, p_base)

    safe = 0
    minB = np.empty(n_sims)
    disc_end = np.empty(n_sims)
    cashrun_out = np.empty(n_sims)

    for i in range(n_sims):
        sub_seed = int(rng.integers(0, 2**32 - 1))
        res = sim.run_path(scen, seed=sub_seed, **base_kwargs)

        df = res.period_df
        minB[i] = float(df["B_end"].min()) if not df.empty else 0.0
        disc_end[i] = float(res.kpi.get("disciplined_n_end", np.nan))
        cashrun_out[i] = float(res.kpi.get("cashrun_out_total", 0.0))
        if minB[i] >= 0:
            safe += 1

    safe_prob = safe / n_sims
    return {
        "safe_prob": float(safe_prob),
        "target_safe": float(1 - alpha),
        "breach_prob": float(1 - safe_prob),
        "minB_p01": float(np.quantile(minB, 0.01)),
        "minB_p05": float(np.quantile(minB, 0.05)),
        "disc_end_mean": float(np.nanmean(disc_end)),
        "disc_end_p05": float(np.nanquantile(disc_end, 0.05)),
        "cashrun_out_mean": float(np.nanmean(cashrun_out)),
    }


# ---------------------------------------------------------------------------
# Bisection: find maximum feasible gamma
# ---------------------------------------------------------------------------

def gamma_max_bisect(
    N: int,
    cycles: int,
    delta: float,
    c: float,
    alpha: float,
    p_base: float,
    K: int,
    strict_cashrun: bool,
    enable_replacement: bool,
    replacement_delay: int,
    probation_q: int,
    n_sims: int,
    seed: int,
    gamma_lo: float = 0.10,
    gamma_hi: float = 0.95,
    tol: float = 0.01,
) -> Optional[Dict[str, float]]:
    gamma_hi = min(gamma_hi, 0.999 - delta)

    common = dict(
        N=N, cycles=cycles, delta=delta, c=c, alpha=alpha, p_base=p_base,
        K=K, strict_cashrun=strict_cashrun, enable_replacement=enable_replacement,
        replacement_delay=replacement_delay, probation_q=probation_q,
        n_sims=n_sims, seed=seed,
    )

    r_lo = mc_safety(gamma=gamma_lo, **common)
    if r_lo["safe_prob"] < r_lo["target_safe"]:
        return None

    lo, hi = gamma_lo, gamma_hi
    best = None
    while hi - lo > tol:
        mid = (hi + lo) / 2
        r = mc_safety(gamma=mid, **common)
        if r["safe_prob"] >= r["target_safe"]:
            best = {"gamma": mid, **r}
            lo = mid
        else:
            hi = mid
    return best


# ---------------------------------------------------------------------------
# Shared controls (common to all workflows)
# ---------------------------------------------------------------------------

def _common_controls() -> Tuple:
    c = ask_float("c (contribution)", 10.0, min_val=0.01)
    alpha = ask_float("alpha", 0.01, min_val=0.0001, max_val=0.2)
    p_base = ask_float("p_base (baseline payment probability)", 1.0, min_val=0.0, max_val=1.0)

    K_mode = ask_choice("vesting lag K mode", ["fixed", "N/3"], "fixed")
    K_fixed = ask_int("vesting lag K", 5, min_val=0, max_val=10_000) if K_mode == "fixed" else None

    strict_cashrun = ask_choice("strict_cashrun", ["yes", "no"], "yes") == "yes"
    enable_repl = ask_choice("enable_replacement", ["no", "yes"], "yes") == "yes"
    repl_delay = ask_int("replacement_delay", 0, min_val=0, max_val=10_000)
    probation_q = ask_int("probation_q", 2, min_val=0, max_val=10_000)
    n_sims = ask_int("Monte Carlo n_sims", 5000, min_val=200)
    seed = ask_int("seed", 42, min_val=0)

    return c, alpha, p_base, K_mode, K_fixed, strict_cashrun, enable_repl, repl_delay, probation_q, n_sims, seed


# ---------------------------------------------------------------------------
# Workflow 1: Maximize gamma
# ---------------------------------------------------------------------------

def workflow_maximize_gamma() -> None:
    print("\n=== Forward: maximize gamma (strict cashrun accounted) ===")

    c, alpha, p_base, K_mode, K_fixed, strict_cashrun, enable_repl, repl_delay, probation_q, n_sims, seed = _common_controls()

    N_lo, N_hi = ask_int_range("N range", 10, 30, min_val=2)
    cycles_lo, cycles_hi = ask_int_range("cycles range", 1, 3, min_val=1)

    delta_min = ask_float("delta_min", 0.10, min_val=0.01, max_val=0.9)
    delta_mode = ask_choice("delta mode", ["fixed", "grid"], "fixed")
    if delta_mode == "fixed":
        deltas = [delta_min]
    else:
        deltas = ask_float_list("delta grid", [delta_min, delta_min + 0.05, delta_min + 0.10])
    deltas = [d for d in deltas if d >= delta_min and d < 1.0] or [delta_min]

    gamma_lo = ask_float("gamma lower bound", 0.30, min_val=0.01, max_val=0.95)
    gamma_hi = ask_float("gamma upper bound", 0.95, min_val=0.05, max_val=0.99)
    tol = ask_float("bisection tolerance", 0.01, min_val=0.001, max_val=0.05)

    results: List[Dict] = []
    best_global: Optional[Dict] = None

    for N in range(N_lo, N_hi + 1):
        K = compute_K(N, K_mode, K_fixed)
        for cycles in range(cycles_lo, cycles_hi + 1):
            for delta in deltas:
                if gamma_hi + delta >= 1:
                    continue
                out = gamma_max_bisect(
                    N=N, cycles=cycles, delta=delta, c=c,
                    alpha=alpha, p_base=p_base, K=K,
                    strict_cashrun=strict_cashrun, enable_replacement=enable_repl,
                    replacement_delay=repl_delay, probation_q=probation_q,
                    n_sims=n_sims, seed=seed, gamma_lo=gamma_lo, gamma_hi=gamma_hi, tol=tol,
                )
                if out is None:
                    continue
                rec = {
                    "N": N, "cycles": cycles, "delta": delta, "K": K,
                    "gamma_max": out["gamma"], "safe_prob": out["safe_prob"],
                    "minB_p01": out["minB_p01"],
                    "disc_end_mean": out["disc_end_mean"],
                    "disc_end_p05": out["disc_end_p05"],
                    "cashrun_out_mean": out["cashrun_out_mean"],
                }
                results.append(rec)
                if best_global is None or rec["gamma_max"] > best_global["gamma_max"]:
                    best_global = rec

    if not results:
        print("No feasible design found. Try lowering gamma_hi, lowering cycles, increasing N, or lowering delta.")
        return

    results.sort(key=lambda r: (r["gamma_max"], r["safe_prob"]), reverse=True)

    print("\nTop 10 by gamma_max:")
    for r in results[:10]:
        print(
            f"gamma_max={r['gamma_max']:.4f}  N={r['N']}, cycles={r['cycles']}, "
            f"delta={r['delta']:.3f}, K={r['K']}  "
            f"safe={r['safe_prob']:.4f}  minB_p1={r['minB_p01']:.2f}  "
            f"disc_mean={r['disc_end_mean']:.2f}  disc_p05={r['disc_end_p05']:.2f}  "
            f"cashrun_out_mean={r['cashrun_out_mean']:.2f}"
        )
    print("\nBEST:", best_global)


# ---------------------------------------------------------------------------
# Workflow 2: Feasible (N, cycles) for given gamma & delta
# ---------------------------------------------------------------------------

def workflow_feasible_N_cycles_given_gamma_delta() -> None:
    print("\n=== Inverse: feasible (N, cycles) for given gamma, delta ===")

    gamma = ask_float("gamma", 0.65, min_val=0.01, max_val=0.99)
    delta = ask_float("delta", 0.10, min_val=0.01, max_val=0.99)
    if gamma + delta >= 1:
        print("gamma+delta must be < 1; reducing delta.")
        delta = min(delta, 0.999 - gamma)

    c, alpha, p_base, K_mode, K_fixed, strict_cashrun, enable_repl, repl_delay, probation_q, n_sims, seed = _common_controls()

    N_lo, N_hi = ask_int_range("N range", 10, 30, min_val=2)
    cycles_lo, cycles_hi = ask_int_range("cycles range", 1, 3, min_val=1)

    target_safe = 1 - alpha
    feasible: List[Dict] = []

    for N in range(N_lo, N_hi + 1):
        K = compute_K(N, K_mode, K_fixed)
        for cycles in range(cycles_lo, cycles_hi + 1):
            r = mc_safety(
                N=N, cycles=cycles, gamma=gamma, delta=delta, c=c,
                alpha=alpha, p_base=p_base, K=K,
                strict_cashrun=strict_cashrun, enable_replacement=enable_repl,
                replacement_delay=repl_delay, probation_q=probation_q,
                n_sims=n_sims, seed=seed,
            )
            if r["safe_prob"] >= target_safe:
                feasible.append({"N": N, "cycles": cycles, "K": K, **r})

    if not feasible:
        print("No feasible (N, cycles) found for the given gamma, delta.")
        return

    feasible.sort(key=lambda x: (x["safe_prob"], x["N"], -x["cycles"]), reverse=True)
    for f in feasible:
        print(
            f"N={f['N']}, cycles={f['cycles']}, K={f['K']}  "
            f"safe={f['safe_prob']:.4f}  minB_p1={f['minB_p01']:.2f}  "
            f"disc_mean={f['disc_end_mean']:.2f}  disc_p05={f['disc_end_p05']:.2f}  "
            f"cashrun_out_mean={f['cashrun_out_mean']:.2f}"
        )


# ---------------------------------------------------------------------------
# Workflow 3: Feasible (N, cycles) + derive c from monetary target
# ---------------------------------------------------------------------------

def workflow_feasible_N_cycles_and_derive_c() -> None:
    print("\n=== Inverse + scale: find feasible (N, cycles) then compute c from target ===")

    gamma = ask_float("gamma", 0.65, min_val=0.01, max_val=0.99)
    delta = ask_float("delta", 0.10, min_val=0.01, max_val=0.99)
    if gamma + delta >= 1:
        print("gamma+delta must be < 1; reducing delta.")
        delta = min(delta, 0.999 - gamma)

    alpha = ask_float("alpha", 0.01, min_val=0.0001, max_val=0.2)
    p_base = ask_float("p_base", 1.0, min_val=0.0, max_val=1.0)

    K_mode = ask_choice("vesting lag K mode", ["fixed", "N/3"], "fixed")
    K_fixed = ask_int("vesting lag K", 5, min_val=0) if K_mode == "fixed" else None

    strict_cashrun = ask_choice("strict_cashrun", ["yes", "no"], "yes") == "yes"
    enable_repl = ask_choice("enable_replacement", ["no", "yes"], "yes") == "yes"
    repl_delay = ask_int("replacement_delay", 0, min_val=0)
    probation_q = ask_int("probation_q", 2, min_val=0)

    N_lo, N_hi = ask_int_range("N range", 10, 30, min_val=2)
    cycles_lo, cycles_hi = ask_int_range("cycles range", 1, 3, min_val=1)
    n_sims = ask_int("Monte Carlo n_sims", 5000, min_val=200)
    seed = ask_int("seed", 42, min_val=0)

    _ = ask_choice("Scale target", ["immediate_payout"], "immediate_payout")
    Gimm_target = ask_float("Target immediate payout per beneficiary (Gimm*)", 300.0, min_val=1.0)

    candidates: List[Dict] = []

    for N in range(N_lo, N_hi + 1):
        K = compute_K(N, K_mode, K_fixed)
        for cycles in range(cycles_lo, cycles_hi + 1):
            r = mc_safety(
                N=N, cycles=cycles, gamma=gamma, delta=delta,
                c=1.0,  # scale-free feasibility test
                alpha=alpha, p_base=p_base, K=K,
                strict_cashrun=strict_cashrun, enable_replacement=enable_repl,
                replacement_delay=repl_delay, probation_q=probation_q,
                n_sims=n_sims, seed=seed,
            )
            if r["safe_prob"] >= (1 - alpha):
                c_needed = Gimm_target / (gamma * N)
                candidates.append({"N": N, "cycles": cycles, "K": K, "c_needed": c_needed, **r})

    if not candidates:
        print("No feasible (N, cycles) found for the given gamma, delta.")
        return

    candidates.sort(key=lambda x: (x["c_needed"], -x["safe_prob"]))
    for cand in candidates[:10]:
        print(
            f"c≈{cand['c_needed']:.2f}  N={cand['N']}, cycles={cand['cycles']}, K={cand['K']}  "
            f"safe={cand['safe_prob']:.4f}  minB_p1={cand['minB_p01']:.2f}  "
            f"disc_mean={cand['disc_end_mean']:.2f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("CLCS Interactive Design Optimizer (v2 — strict cashrun)\n")
    while True:
        print("Choose a workflow:")
        print("  1. Maximize gamma (forward design)")
        print("  2. Feasible (N, cycles) given gamma & delta")
        print("  3. Feasible (N, cycles) + compute c from monetary target")
        print("  0. Exit")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            workflow_maximize_gamma()
        elif choice == "2":
            workflow_feasible_N_cycles_given_gamma_delta()
        elif choice == "3":
            workflow_feasible_N_cycles_and_derive_c()
        elif choice == "0":
            print("Bye.")
            break
        else:
            print("Invalid choice.\n")


if __name__ == "__main__":
    main()
