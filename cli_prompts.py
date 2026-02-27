"""cli_prompts.py
Shared interactive CLI prompt helpers used by all CLCS and TRIBUS scripts.

Replaces duplicated ask_*/prompt_* helpers that were copy-pasted across
design_solver.py and interative_usage.py, and broken in tribus_check.py /
risk_analysis.py.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Text / bool prompts
# ---------------------------------------------------------------------------

def ask_str(label: str, default: str = "", help_text: Optional[str] = None) -> str:
    if help_text:
        print(f"  - {help_text}")
    raw = input(f"{label} [{default}]: ").strip()
    return raw if raw else default


def ask_bool(label: str, default: bool, help_text: Optional[str] = None) -> bool:
    if help_text:
        print(f"  - {help_text}")
    d = "y" if default else "n"
    raw = input(f"{label} [y/n, default={d}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true", "t")


def ask_choice(label: str, choices: List[str], default: str) -> str:
    opts = ", ".join(choices)
    while True:
        raw = input(f"{label} ({opts}) [default={default}]: ").strip()
        if raw == "":
            return default
        if raw in choices:
            return raw
        print("Invalid choice.")


# ---------------------------------------------------------------------------
# Numeric prompts
# ---------------------------------------------------------------------------

def ask_int(
    label: str,
    default: int,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    help_text: Optional[str] = None,
) -> int:
    if help_text:
        print(f"  - {help_text}")
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        val = default if raw == "" else None
        if val is None:
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


def ask_float(
    label: str,
    default: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    help_text: Optional[str] = None,
) -> float:
    if help_text:
        print(f"  - {help_text}")
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        val = default if raw == "" else None
        if val is None:
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


# ---------------------------------------------------------------------------
# Range / list prompts
# ---------------------------------------------------------------------------

def ask_int_range(
    label: str,
    default_lo: int,
    default_hi: int,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
) -> Tuple[int, int]:
    while True:
        raw = input(f"{label} as lo,hi [{default_lo},{default_hi}]: ").strip()
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
    raw = input(f"{label} as comma-separated floats [default={default}]: ").strip()
    if raw == "":
        return list(default)
    return [float(p.strip()) for p in raw.split(",") if p.strip()]


def parse_ints(csv: str, default: List[int]) -> List[int]:
    s = csv.strip()
    if not s:
        return list(default)
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_floats(csv: str, default: List[float]) -> List[float]:
    s = csv.strip()
    if not s:
        return list(default)
    return [float(x.strip()) for x in s.split(",") if x.strip()]
