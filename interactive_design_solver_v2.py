"""Interactive design solver for CLCS with STRICT CASHRUN and (optional) REPLACEMENT.

This solver searches feasible CLCS designs under a safety constraint using Monte Carlo.

Update (Jan 2026):
- Uses p_base as the baseline payment probability (default 1.0 for "deterministic" payments).
- Vesting lag K can be fixed or computed dynamically as floor(N/3).
- Compatible call helper for sim.run_path() to support simulator API differences.

Safety constraint:
 P(min_t B_end >= 0) >= 1 - alpha

Run:
 python interactive_design_solver_v2.py
"""

from __future__ import annotations

import sys
import inspect
from typing import Dict, List, Tuple, Optional

import numpy as np

# Ensure local directory is on the path
if "/mnt/data" not in sys.path:
    sys.path.append("/mnt/data")

try:
    from clcs_sim_v4_8 import CLCSParams, DeterministicScenario, CLCSSimulator
except Exception as e:
    raise ImportError(
        "Cannot import clcs_sim_v4_8. Make sure clcs_sim_v4_8.py is in the same folder."
    ) from e


# ---------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------

def _prompt(msg: str) -> str:
    return input(msg).strip()


def ask_choice(label: str, choices: List[str], default: str) -> str:
    opts = ", ".join(choices)
    while True:
        raw = _prompt(f"{label} ({opts}) [default={default}]: ")
        if raw == "":
            return default
        if raw in choices:
            return raw
        print("Invalid choice.")


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


def ask_int_range(
    label: str, default_lo: int, default_hi: int, min_val=None, max_val=None
) -> Tuple[int, int]:
    while True:
        raw = _prompt(f"{label} as lo,hi [{default_lo},{default_hi}]: ")
        if raw == "":
            lo, hi = default_lo, default_hi
        else:
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) != 2:
                print("Enter as lo,hi (e.g., 10,30)")
                continue
            try:
                lo, hi = int(parts[0]), int(parts[1])
            except ValueError:
                print("Integers only.")
                continue
        if lo > hi:
            lo, hi = hi, lo
        if min_val is not None and lo < min_val:
            print(f"lo must be >= {min_val}")
            continue
        if max_val is not None and hi > max_val:
            print(f"hi must be <= {max_val}")
            continue
        return lo, hi


def ask_float_list(label: str, default: List[float]) -> List[float]:
    raw = _prompt(f"{label} as comma-separated floats [default={default}]: ")
    if raw == "":
        return default
    out: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


# ---------------------------------------------------------------------
# Compatibility wrapper for run_path() across simulator versions
# ---------------------------------------------------------------------

def run_path_mc_probpay(sim: CLCSSimulator, scen: DeterministicScenario, seed: int, p_base: float):
    """Call sim.run_path() for mc_probpay in a version-robust way."""
    sig = inspect.signature(sim.run_path)
    kwargs = {"payment_mode": "mc_probpay", "seed": seed}

    # Preferred in v4.8+
    if "p_base" in sig.parameters:
        kwargs["p_base"] = p_base
    # Backward-compatible naming fallbacks
    elif "p_pay" in sig.parameters:
        kwargs["p_pay"] = p_base
    elif "pay_prob" in sig.parameters:
        kwargs["pay_prob"] = p_base
    elif "p" in sig.parameters:
        kwargs["p"] = p_base

    return sim.run_path(scen, **kwargs)


# ---------------------------------------------------------------------
# K (vesting lag) mode helper
# ---------------------------------------------------------------------

def compute_K_from_mode(N: int, K_mode: str, K_fixed: Optional[int]) -> int:
    """Compute vesting lag K.

    - fixed: return K_fixed
    - N/3 : return floor(N/3)
    """
    if K_mode == "fixed":
        return int(K_fixed or 0)
    return max(0, N // 3)


# ---------------------------------------------------------------------
# Monte Carlo evaluation
# ---------------------------------------------------------------------

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
        N=N,
        c=c,
        gamma=gamma,
        delta=delta,
        num_cycles=cycles,
        vesting_lag=K,
        strict_cashrun=strict_cashrun,
        enable_replacement=enable_replacement,
        replacement_delay=replacement_delay,
        probation_q=probation_q,
    )

    # Placeholder A_sched (ignored in mc_probpay except length check)
    A_sched = [N] * (N * cycles)
    scen = DeterministicScenario(A_sched=A_sched)

    sim = CLCSSimulator(params)
    rng = np.random.default_rng(seed)

    safe = 0
    minB = np.empty(n_sims)
    disc_end = np.empty(n_sims)
    cashrun_out = np.empty(n_sims)

    for i in range(n_sims):
        sub_seed = int(rng.integers(0, 2**32 - 1))
        res = run_path_mc_probpay(sim, scen, sub_seed, p_base)

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

    r_lo = mc_safety(
        N,
        cycles,
        gamma_lo,
        delta,
        c,
        alpha,
        p_base,
        K,
        strict_cashrun,
        enable_replacement,
        replacement_delay,
        probation_q,
        n_sims,
        seed,
    )
    if r_lo["safe_prob"] < r_lo["target_safe"]:
        return None

    lo, hi = gamma_lo, gamma_hi
    best = None
    while hi - lo > tol:
        mid = (hi + lo) / 2
        r = mc_safety(
            N,
            cycles,
            mid,
            delta,
            c,
            alpha,
            p_base,
            K,
            strict_cashrun,
            enable_replacement,
            replacement_delay,
            probation_q,
            n_sims,
            seed,
        )
        if r["safe_prob"] >= r["target_safe"]:
            best = {"gamma": mid, **r}
            lo = mid
        else:
            hi = mid
    return best


# ---------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------

def common_controls():
    c = ask_float("c (contribution)", 10.0, min_val=0.01)
    alpha = ask_float("alpha", 0.01, min_val=0.0001, max_val=0.2)

    # p_base defaults to 1.0 (deterministic "all pay")
    p_base = ask_float("p_base (baseline payment probability)", 1.0, min_val=0.0, max_val=1.0)

    # K mode (fixed vs N/3)
    K_mode = ask_choice("vesting lag K mode", ["fixed", "N/3"], "fixed")
    if K_mode == "fixed":
        K_fixed = ask_int("vesting lag K", 5, min_val=0, max_val=10_000)
    else:
        K_fixed = None

    strict_cashrun = ask_choice("strict_cashrun", ["yes", "no"], "yes") == "yes"
    enable_repl = ask_choice("enable_replacement", ["no", "yes"], "yes") == "yes"
    repl_delay = ask_int("replacement_delay", 0, min_val=0, max_val=10_000)
    probation_q = ask_int("probation_q", 2, min_val=0, max_val=10_000)
    n_sims = ask_int("Monte Carlo n_sims", 5000, min_val=200)
    seed = ask_int("seed", 42, min_val=0)

    return (
        c,
        alpha,
        p_base,
        K_mode,
        K_fixed,
        strict_cashrun,
        enable_repl,
        repl_delay,
        probation_q,
        n_sims,
        seed,
    )


def workflow_maximize_gamma():
    print("\n=== Forward: maximize gamma (strict cashrun accounted) ===")

    (
        c,
        alpha,
        p_base,
        K_mode,
        K_fixed,
        strict_cashrun,
        enable_repl,
        repl_delay,
        probation_q,
        n_sims,
        seed,
    ) = common_controls()

    N_lo, N_hi = ask_int_range("N range", 10, 30, min_val=2)
    cycles_lo, cycles_hi = ask_int_range("cycles range", 1, 3, min_val=1)

    delta_min = ask_float("delta_min", 0.10, min_val=0.01, max_val=0.9)
    delta_mode = ask_choice("delta mode", ["fixed", "grid"], "fixed")
    if delta_mode == "fixed":
        deltas = [delta_min]
    else:
        deltas = ask_float_list("delta grid", [delta_min, delta_min + 0.05, delta_min + 0.10])
    deltas = [d for d in deltas if d >= delta_min and d < 1.0]
    if not deltas:
        deltas = [delta_min]

    gamma_lo = ask_float("gamma lower bound", 0.30, min_val=0.01, max_val=0.95)
    gamma_hi = ask_float("gamma upper bound", 0.95, min_val=0.05, max_val=0.99)
    tol = ask_float("bisection tolerance", 0.01, min_val=0.001, max_val=0.05)

    best_global = None
    results = []

    for N in range(N_lo, N_hi + 1):
        K_used = compute_K_from_mode(N, K_mode, K_fixed)
        for cycles in range(cycles_lo, cycles_hi + 1):
            for delta in deltas:
                if gamma_hi + delta >= 1:
                    continue

                out = gamma_max_bisect(
                    N=N,
                    cycles=cycles,
                    delta=delta,
                    c=c,
                    alpha=alpha,
                    p_base=p_base,
                    K=K_used,
                    strict_cashrun=strict_cashrun,
                    enable_replacement=enable_repl,
                    replacement_delay=repl_delay,
                    probation_q=probation_q,
                    n_sims=n_sims,
                    seed=seed,
                    gamma_lo=gamma_lo,
                    gamma_hi=gamma_hi,
                    tol=tol,
                )
                if out is None:
                    continue

                rec = {
                    "N": N,
                    "cycles": cycles,
                    "delta": delta,
                    "K": K_used,
                    "gamma_max": out["gamma"],
                    "safe_prob": out["safe_prob"],
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

    results = sorted(results, key=lambda r: (r["gamma_max"], r["safe_prob"]), reverse=True)

    print("\nTop 10 by gamma_max:")
    for r in results[:10]:
        print(
            f"gamma_max={r['gamma_max']:.4f}\n"
            f" N={r['N']}, cycles={r['cycles']}, delta={r['delta']:.3f}, K={r['K']}\n"
            f" safe={r['safe_prob']:.4f}\n"
            f" minB p1={r['minB_p01']:.2f}\n"
            f" disc_mean={r['disc_end_mean']:.2f}\n"
            f" disc_p05={r['disc_end_p05']:.2f}\n"
            f" cashrun_out_mean={r['cashrun_out_mean']:.2f}"
        )

    print("\nBEST:")
    print(best_global)


def workflow_feasible_N_cycles_given_gamma_delta():
    print("\n=== Inverse: feasible (N, cycles) for given gamma, delta (strict cashrun accounted) ===")

    gamma = ask_float("gamma", 0.65, min_val=0.01, max_val=0.99)
    delta = ask_float("delta", 0.10, min_val=0.01, max_val=0.99)
    if gamma + delta >= 1:
        print("gamma+delta must be < 1; reducing delta.")
        delta = min(delta, 0.999 - gamma)

    (
        c,
        alpha,
        p_base,
        K_mode,
        K_fixed,
        strict_cashrun,
        enable_repl,
        repl_delay,
        probation_q,
        n_sims,
        seed,
    ) = common_controls()

    N_lo, N_hi = ask_int_range("N range", 10, 30, min_val=2)
    cycles_lo, cycles_hi = ask_int_range("cycles range", 1, 3, min_val=1)

    target_safe = 1 - alpha
    feasible = []

    for N in range(N_lo, N_hi + 1):
        K_used = compute_K_from_mode(N, K_mode, K_fixed)
        for cycles in range(cycles_lo, cycles_hi + 1):
            r = mc_safety(
                N,
                cycles,
                gamma,
                delta,
                c,
                alpha,
                p_base,
                K_used,
                strict_cashrun,
                enable_repl,
                repl_delay,
                probation_q,
                n_sims,
                seed,
            )
            if r["safe_prob"] >= target_safe:
                feasible.append({"N": N, "cycles": cycles, "K": K_used, **r})

    if not feasible:
        print("No feasible (N, cycles) found for the given gamma, delta.")
        return

    feasible = sorted(feasible, key=lambda x: (x["safe_prob"], x["N"], -x["cycles"]), reverse=True)

    for f in feasible:
        print(
            f"N={f['N']}, cycles={f['cycles']}, K={f['K']}\n"
            f" safe={f['safe_prob']:.4f}\n"
            f" minB p1={f['minB_p01']:.2f}\n"
            f" disc_mean={f['disc_end_mean']:.2f}\n"
            f" disc_p05={f['disc_end_p05']:.2f}\n"
            f" cashrun_out_mean={f['cashrun_out_mean']:.2f}"
        )


def workflow_find_c_and_feasible_N_cycles_given_gamma_delta():
    print("\n=== Inverse + scale: find feasible (N, cycles) then compute c from target ===")

    gamma = ask_float("gamma", 0.65, min_val=0.01, max_val=0.99)
    delta = ask_float("delta", 0.10, min_val=0.01, max_val=0.99)
    if gamma + delta >= 1:
        print("gamma+delta must be < 1; reducing delta.")
        delta = min(delta, 0.999 - gamma)

    alpha = ask_float("alpha", 0.01, min_val=0.0001, max_val=0.2)
    p_base = ask_float("p_base (baseline payment probability)", 1.0, min_val=0.0, max_val=1.0)

    # K mode here as well (since N varies)
    K_mode = ask_choice("vesting lag K mode", ["fixed", "N/3"], "fixed")
    if K_mode == "fixed":
        K_fixed = ask_int("vesting lag K", 5, min_val=0)
    else:
        K_fixed = None

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

    # scale-free feasibility test using c=1
    c_test = 1.0
    candidates = []

    for N in range(N_lo, N_hi + 1):
        K_used = compute_K_from_mode(N, K_mode, K_fixed)
        for cycles in range(cycles_lo, cycles_hi + 1):
            r = mc_safety(
                N,
                cycles,
                gamma,
                delta,
                c_test,
                alpha,
                p_base,
                K_used,
                strict_cashrun,
                enable_repl,
                repl_delay,
                probation_q,
                n_sims,
                seed,
            )
            if r["safe_prob"] >= (1 - alpha):
                c_needed = Gimm_target / (gamma * N)
                candidates.append({"N": N, "cycles": cycles, "K": K_used, "c_needed": c_needed, **r})

    if not candidates:
        print("No feasible (N, cycles) found for the given gamma, delta.")
        return

    candidates = sorted(candidates, key=lambda x: (x["c_needed"], -x["safe_prob"]))

    for cand in candidates[:10]:
        print(
            f"c≈{cand['c_needed']:.2f}\n"
            f" N={cand['N']}, cycles={cand['cycles']}, K={cand['K']}\n"
            f" safe={cand['safe_prob']:.4f}\n"
            f" minB p1={cand['minB_p01']:.2f}\n"
            f" disc_mean={cand['disc_end_mean']:.2f}"
        )


def main():
    print("CLCS Interactive Design Solver (v2 - strict cashrun)\n")
    while True:
        print("Choose a workflow:")
        print(" 1. Maximize gamma (forward design)")
        print(" 2. Feasible (N, cycles) given gamma & delta")
        print(" 3. Feasible (N, cycles) + compute c from monetary target")
        print(" 0. Exit")
        choice = _prompt("Enter choice: ")

        if choice == "1":
            workflow_maximize_gamma()
        elif choice == "2":
            workflow_feasible_N_cycles_given_gamma_delta()
        elif choice == "3":
            workflow_find_c_and_feasible_N_cycles_given_gamma_delta()
        elif choice == "0":
            print("Bye.")
            break
        else:
            print("Invalid choice.\n")


if __name__ == "__main__":
    main()
