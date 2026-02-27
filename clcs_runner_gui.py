"""clcs_runner_gui.py
Terminal interface for the CLCS interactive simulator.

Uses rich for display and prompt_toolkit for input with autocomplete.

Commands
--------
  run                    — run simulation with current params
  set <param> <value>    — change a parameter  (tab-complete names)
  show [section]         — display current params
  reset                  — restore all defaults
  help                   — command reference
  q / quit               — exit

Shock / override string formats (same as clcs_interactive_runner.py)
  p_by_member    "id:p, id:p"            e.g. 3:0.95, 7:0.70
  general_shocks "t0-t1:mult, ..."       e.g. 10-20:0.80
  member_shocks  "id:t0-t1:p, ..."       e.g. 7:15-18:0, 12:30-60:0.5
  cashrun_plan   "id:cycle,...; id:..."  e.g. 7:2;12:1,3

Run:
  python clcs_runner_gui.py
"""

from __future__ import annotations

import io
import sys
from typing import Any, Dict, List, Optional, Tuple

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from clcs_simulator import CLCSParams, CLCSSimulator, DeterministicScenario, pretty_params
from clcs_interactive_runner import (
    parse_cashrun_plan,
    parse_general_shocks,
    parse_member_shocks,
    parse_p_by_member,
)

console = Console()

# ---------------------------------------------------------------------------
# Parameter schema:  name → (section, type, default, description)
# ---------------------------------------------------------------------------

SCHEMA: Dict[str, Tuple[str, type, Any, str]] = {
    # ── group ────────────────────────────────────────────────────────────────
    "N":                  ("group",   int,   10,         "Number of members"),
    "num_cycles":         ("group",   int,   2,          "Number of cycles"),
    "vesting_lag":        ("group",   int,   5,          "Vesting lag K (periods)"),
    # ── cash flow ────────────────────────────────────────────────────────────
    "c":                  ("flow",    float, 100.0,      "Contribution per period"),
    "gamma":              ("flow",    float, 0.75,       "Immediate payout fraction γ"),
    "delta":              ("flow",    float, 0.10,       "Deferred payout fraction δ"),
    "Rb_annual":          ("flow",    float, 0.042,      "Buffer annual interest rate"),
    "Re_annual":          ("flow",    float, 0.035,      "Escrow annual interest rate"),
    "periods_per_year":   ("flow",    int,   12,         "Periods per year (12/52/4…)"),
    # ── rules ────────────────────────────────────────────────────────────────
    "strict_cashrun":     ("rules",   bool,  True,       "Enforce strict cashrun (bool)"),
    "enable_replacement": ("rules",   bool,  False,      "Enable member replacement (bool)"),
    "replacement_delay":  ("rules",   int,   0,          "Replacement delay (periods)"),
    "probation_q":        ("rules",   int,   2,          "Probation quarters"),
    "phi":                ("rules",   float, 0.0,        "Platform fee rate φ"),
    "shrink_cap":         ("rules",   float, 2.0,        "Arrears shrink cap (×c)"),
    "init_t0_first_cycle":("rules",   bool,  True,       "Pre-fund at t=0 (bool)"),
    # ── simulation ───────────────────────────────────────────────────────────
    "payment_mode":       ("sim",     str,   "mc_probpay","deterministic|mc_fixedA|mc_probpay"),
    "p_base":             ("sim",     float, 1.0,        "Base payment probability"),
    "seed":               ("sim",     int,   42,         "Random seed"),
    # ── shocks / overrides (string format) ───────────────────────────────────
    "p_by_member":        ("shocks",  str,   "",         "id:p, id:p  e.g. 3:0.95,7:0.7"),
    "general_shocks":     ("shocks",  str,   "",         "t0-t1:mult  e.g. 10-20:0.8"),
    "member_shocks":      ("shocks",  str,   "",         "id:t0-t1:p  e.g. 7:15-18:0"),
    "cashrun_plan":       ("shocks",  str,   "",         "id:cycle;id:cycle  e.g. 7:2;12:1,3"),
}

DEFAULTS = {k: v[2] for k, v in SCHEMA.items()}

BOOL_PARAMS = {k for k, (_, typ, _, _) in SCHEMA.items() if typ is bool}
SECTION_COLOURS = {
    "group": "cyan", "flow": "yellow", "rules": "magenta",
    "sim": "green", "shocks": "dim white",
}


# ---------------------------------------------------------------------------
# Build CLCSParams from current values
# ---------------------------------------------------------------------------

def _build_params(vals: Dict[str, Any]) -> CLCSParams:
    return CLCSParams(
        N=int(vals["N"]),
        c=float(vals["c"]),
        gamma=float(vals["gamma"]),
        delta=float(vals["delta"]),
        num_cycles=int(vals["num_cycles"]),
        vesting_lag=int(vals["vesting_lag"]),
        Rb_annual=float(vals["Rb_annual"]),
        Re_annual=float(vals["Re_annual"]),
        periods_per_year=int(vals["periods_per_year"]),
        phi=float(vals["phi"]),
        shrink_cap=float(vals["shrink_cap"]),
        enable_replacement=bool(vals["enable_replacement"]),
        replacement_delay=int(vals["replacement_delay"]),
        probation_q=int(vals["probation_q"]),
        strict_cashrun=bool(vals["strict_cashrun"]),
        init_t0_first_cycle=bool(vals["init_t0_first_cycle"]),
    )


def _parse_shocks(vals: Dict[str, Any]):
    pbm = parse_p_by_member(vals["p_by_member"])    if vals["p_by_member"].strip()    else None
    gs  = parse_general_shocks(vals["general_shocks"]) if vals["general_shocks"].strip() else None
    ms  = parse_member_shocks(vals["member_shocks"])   if vals["member_shocks"].strip()  else None
    cp  = parse_cashrun_plan(vals["cashrun_plan"])      if vals["cashrun_plan"].strip()   else None
    return pbm, gs, ms, cp


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _hbar(value: float, maximum: float, width: int = 24) -> str:
    filled = int(value / maximum * width) if maximum else 0
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


def show_params(vals: Dict[str, Any], section: Optional[str] = None):
    sections = (
        [section] if section
        else ["group", "flow", "rules", "sim", "shocks"]
    )
    titles = {
        "group": "Group Setup", "flow": "Cash Flow",
        "rules": "Rules", "sim": "Simulation", "shocks": "Shocks & Overrides",
    }
    for sec in sections:
        t = Table(box=box.SIMPLE, show_header=True, header_style="bold white",
                  show_edge=False, padding=(0, 1))
        t.add_column("param",   style="bold white", no_wrap=True, width=22)
        t.add_column("value",   no_wrap=True, width=18)
        t.add_column("default", style="dim white", no_wrap=True, width=14)
        t.add_column("note",    style="dim white")

        for k, (s, typ, default, desc) in SCHEMA.items():
            if s != sec:
                continue
            v = vals[k]
            changed = v != default
            col = SECTION_COLOURS.get(sec, "white")
            val_str = f"[bold {'magenta' if changed else 'yellow'}]{v}[/]"
            t.add_row(f"[{col}]{k}[/]", val_str,
                      f"[dim]{default}[/]", f"[dim]{desc}[/]")

        col = SECTION_COLOURS.get(sec, "white")
        console.print(Panel(t, title=f"[bold {col}]{titles[sec]}[/]",
                            border_style=col, padding=(0, 1)))


# ---------------------------------------------------------------------------
# Run and display results
# ---------------------------------------------------------------------------

def run_and_show(vals: Dict[str, Any]):
    try:
        p = _build_params(vals)
    except (AssertionError, ValueError) as e:
        console.print(f"[red]Invalid params: {e}[/]")
        return

    try:
        pbm, gs, ms, cp = _parse_shocks(vals)
    except Exception as e:
        console.print(f"[red]Shock parse error: {e}[/]")
        return

    N = p.N
    total_turns = N * p.num_cycles
    A_sched = [N] * total_turns
    scen = DeterministicScenario(A_sched=A_sched)

    sim = CLCSSimulator(p)
    mode = str(vals["payment_mode"])
    seed = int(vals["seed"])
    p_base = float(vals["p_base"])

    with console.status("[bold green]Running simulation…[/]", spinner="dots"):
        result = sim.run_path(
            scen, payment_mode=mode, seed=seed, p_base=p_base,
            p_by_member=pbm, general_shocks=gs, member_shocks=ms,
            cashrun_plan=cp,
        )

    kpi = result.kpi
    pdf = result.period_df
    mdf = result.member_df

    console.print(Rule(f"[bold white]{pretty_params(p)}[/]"))

    # ── KPI panel ──────────────────────────────────────────────────────────
    min_B      = kpi["min_B_end"]
    plat_rev   = kpi["platform_rev_total"]
    disc_n     = kpi["disciplined_n_end"]
    avg_rcv    = kpi["avg_total_received_disciplined"]
    co_total   = kpi["cashrun_out_total"]
    m3_total   = kpi["miss3_out_total"]
    fo_total   = kpi["force_out_total"]

    solvent    = min_B >= -1e-6
    sol_colour = "green" if solvent else "red"
    sol_label  = "SOLVENT" if solvent else "INSOLVENT"

    kpi_lines = [
        f"  [bold]Buffer solvency[/]  [{sol_colour}]{sol_label}[/]   min B = [{sol_colour}]{min_B:,.2f}[/]",
        f"  [bold]Platform revenue[/] [yellow]{plat_rev:,.4f}[/]",
        f"  [bold]Disciplined end[/]  [cyan]{disc_n}/{N}[/]   avg received = [cyan]{avg_rcv:,.2f}[/]",
        f"  [bold]Cashrun out[/]      [{'red' if co_total else 'dim'}]{co_total}[/]   "
        f"Miss-3 out [{'red' if m3_total else 'dim'}]{m3_total}[/]   "
        f"Forced out [{'red' if fo_total else 'dim'}]{fo_total}[/]",
    ]

    # Cashrun forced events
    if kpi["cashrun_forced_events"]:
        kpi_lines.append(f"\n  [bold]Cashrun forced events[/]")
        for ev in kpi["cashrun_forced_events"][:5]:
            kpi_lines.append(
                f"    member {ev['member_id']}  cycle {ev['cycle']}  "
                f"t_ben={ev['t_beneficiary']}  t_due={ev['t_due']}"
            )

    console.print(Panel("\n".join(kpi_lines), title="[bold]KPI[/]",
                        border_style=sol_colour, padding=(0, 1)))

    # ── Period table (last 20 turns) ────────────────────────────────────────
    turn_df = pdf[pdf["phase"].str.startswith("turn") | pdf["phase"].str.startswith("final")]
    show_df = turn_df.tail(20)

    pt = Table(box=box.SIMPLE, show_header=True, header_style="bold white",
               show_edge=False, padding=(0, 1))
    for col in ["t","cycle","A_t","C_t","beneficiary","B_end","E_end",
                "vesting_paid","cashrun_out","miss3_out"]:
        pt.add_column(col, no_wrap=True)

    for _, r in show_df.iterrows():
        b_col = "green" if r["B_end"] >= 0 else "red"
        pt.add_row(
            str(int(r["t"])), str(int(r["cycle"])),
            str(int(r["A_t"])), f"{r['C_t']:,.1f}",
            str(r["beneficiary"]) if r["beneficiary"] else "[dim]–[/]",
            f"[{b_col}]{r['B_end']:,.2f}[/]",
            f"{r['E_end']:,.2f}",
            f"{r['vesting_paid']:,.2f}",
            str(int(r["cashrun_out"])), str(int(r["miss3_out"])),
        )

    console.print(Panel(pt,
        title=f"[bold]Periods  [dim](last {len(show_df)} of {len(turn_df)})[/][/]",
        border_style="blue", padding=(0, 1),
    ))

    # ── Member table ─────────────────────────────────────────────────────────
    mt = Table(box=box.SIMPLE, show_header=True, header_style="bold white",
               show_edge=False, padding=(0, 1))
    for col in ["member_id","out","paid_since_join","missed_streak",
                "arrears","eligible_bonus"]:
        mt.add_column(col, no_wrap=True)

    for _, r in mdf.iterrows():
        out_col    = "red" if r["out"] else "green"
        bonus_col  = "green" if r["eligible_bonus"] else "dim"
        arr_col    = "red"  if r["arrears"] > 0 else "dim"
        mt.add_row(
            str(int(r["member_id"])),
            f"[{out_col}]{'yes' if r['out'] else 'no'}[/]",
            str(int(r["paid_since_join"])),
            f"[{'red' if r['missed_streak']>0 else 'dim'}]{int(r['missed_streak'])}[/]",
            f"[{arr_col}]{r['arrears']:,.2f}[/]",
            f"[{bonus_col}]{'✓' if r['eligible_bonus'] else '✗'}[/]",
        )

    console.print(Panel(mt, title="[bold]Members[/]",
                        border_style="magenta", padding=(0, 1)))

    # ── Buffer trajectory sparkline ──────────────────────────────────────────
    b_vals = turn_df["B_end"].values
    if len(b_vals) > 0:
        b_max   = float(max(b_vals.max(), 1.0))
        b_min   = float(b_vals.min())
        steps   = min(len(b_vals), 60)
        sampled = b_vals[::max(1, len(b_vals)//steps)]
        chars   = ["▁","▂","▃","▄","▅","▆","▇","█"]
        def _to_char(v):
            if b_max == b_min:
                return "▄"
            idx = int((v - b_min) / (b_max - b_min) * 7)
            return chars[max(0, min(idx, 7))]
        spark = "".join(_to_char(v) for v in sampled)
        sol_c = "green" if b_min >= 0 else "red"
        console.print(Panel(
            f"  [{sol_c}]{spark}[/]\n"
            f"  [dim]min {b_min:,.2f}  max {b_max:,.2f}[/]",
            title="[bold]Buffer B — trajectory[/]",
            border_style=sol_c, padding=(0, 1),
        ))


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

HELP = """
[bold cyan]Commands[/]
  [bold]run[/]                    run simulation, show KPI / Periods / Members
  [bold]set[/] [cyan]<param>[/] [yellow]<value>[/]    change a parameter (tab-complete names)
  [bold]show[/] [dim][group|flow|rules|sim|shocks][/]  display params
  [bold]reset[/]                  restore all defaults
  [bold]help[/]                   this message
  [bold]q[/] / [bold]quit[/]             exit

[bold cyan]Bool params[/]  set strict_cashrun true|false

[bold cyan]Shock formats[/]
  p_by_member    "3:0.95, 7:0.7"
  general_shocks "10-20:0.8"
  member_shocks  "7:15-18:0, 12:30-60:0.5"
  cashrun_plan   "7:2;12:1,3"

[bold cyan]Payment modes[/]  deterministic | mc_fixedA | mc_probpay
"""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    vals = dict(DEFAULTS)

    all_params = list(SCHEMA.keys())
    commands   = ["run","set","show","reset","help","quit","q"]
    sections   = ["group","flow","rules","sim","shocks"]
    completer  = WordCompleter(commands + all_params + sections, sentence=True)
    history    = InMemoryHistory()

    console.print(Panel(
        "[bold white]CLCS Interactive Simulator[/]\n"
        "[dim]type [bold]help[/] for commands · tab-complete params after [bold]set[/][/]",
        border_style="cyan", padding=(0, 2),
    ))
    show_params(vals)

    while True:
        try:
            raw = prompt("\n[clcs]> ", completer=completer, history=history)
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]bye.[/]")
            break

        parts = raw.strip().split(None, 2)
        if not parts:
            continue
        cmd, *args = parts

        # ── run ─────────────────────────────────────────────────────────────
        if cmd == "run":
            run_and_show(vals)

        # ── set ─────────────────────────────────────────────────────────────
        elif cmd == "set":
            if len(args) < 2:
                console.print("[red]Usage: set <param> <value>[/]")
                continue
            key, raw_val = args[0], args[1]
            if key not in SCHEMA:
                console.print(f"[red]Unknown param '{key}'. Try 'show'.[/]")
                continue
            _, typ, default, _ = SCHEMA[key]
            try:
                if typ is bool:
                    new_val = raw_val.lower() in ("true", "yes", "1", "on")
                elif typ is str:
                    # For string params (shocks), join remaining args
                    new_val = " ".join([raw_val] + (args[2:] if len(args) > 2 else []))
                else:
                    new_val = typ(raw_val)
                old_val = vals[key]
                vals[key] = new_val
                console.print(
                    f"  [cyan]{key}[/]  [dim]{old_val}[/] → [bold yellow]{new_val}[/]"
                )
            except ValueError:
                console.print(f"[red]Cannot convert '{raw_val}' to {typ.__name__}[/]")

        # ── show ─────────────────────────────────────────────────────────────
        elif cmd == "show":
            sec = args[0] if args else None
            show_params(vals, sec)

        # ── reset ─────────────────────────────────────────────────────────────
        elif cmd == "reset":
            vals = dict(DEFAULTS)
            console.print("[green]All parameters reset to defaults.[/]")
            show_params(vals)

        # ── help ─────────────────────────────────────────────────────────────
        elif cmd in ("help", "h"):
            console.print(HELP)

        # ── quit ─────────────────────────────────────────────────────────────
        elif cmd in ("q", "quit", "exit"):
            console.print("[dim]bye.[/]")
            break

        else:
            console.print(f"[red]Unknown command '{cmd}'. Type 'help'.[/]")


if __name__ == "__main__":
    main()
