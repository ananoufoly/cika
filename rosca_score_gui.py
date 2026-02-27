"""rosca_score_gui.py
Terminal interface for the ROSCA credit score population simulator.

Uses rich for display and prompt_toolkit for input with autocomplete.

Commands
--------
  run                      — simulate with current params, show results
  set <param> <value>      — change one parameter  (tab-complete names)
  sweep <param> v1 v2 ...  — sensitivity: run once per value, compare ρ
  show [pop|macro|score]   — print current params (all sections if no arg)
  reset                    — restore all defaults
  help                     — command reference
  q / quit                 — exit

Run:
  python rosca_score_gui.py
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from rosca_score_engine import (
    MacroEnvironment,
    PopulationParams,
    ScoreParams,
    generate_population,
)

console = Console()

# ---------------------------------------------------------------------------
# Parameter schema:  name → (section, type, default, description)
# ---------------------------------------------------------------------------

SCHEMA: Dict[str, Tuple[str, type, Any, str]] = {
    # ── population ──────────────────────────────────────────────────────────
    "n_groups":           ("pop",   int,   20,    "Number of groups"),
    "group_size_min":     ("pop",   int,   6,     "Min members per group"),
    "group_size_max":     ("pop",   int,   20,    "Max members per group"),
    "rtype_bidding_prob": ("pop",   float, 0.50,  "Prob group uses bidding [0-1]"),
    "rules_prob":         ("pop",   float, 0.75,  "Prob group has formal rules [0-1]"),
    "p_ontime_mean":      ("pop",   float, 0.80,  "Population avg on-time rate [0-1]"),
    "p_ontime_conc":      ("pop",   float, 9.0,   "p_ontime concentration (spread)"),
    "post_slip_mean":     ("pop",   float, 0.08,  "Post-payout slip tendency [0-1]"),
    "bid_agg_mean":       ("pop",   float, 0.22,  "Bid aggressiveness mean [0-1]"),
    "p_rep":              ("pop",   float, 0.45,  "P(repeat participation)"),
    "p_cent":             ("pop",   float, 0.30,  "P(network centrality)"),
    "p_endf":             ("pop",   float, 0.25,  "P(foreman endorsement)"),
    # ── macro ────────────────────────────────────────────────────────────────
    "stress_level":       ("macro", float, 0.0,   "Systemic stress [0-1]"),
    "within_group_corr":  ("macro", float, 0.20,  "Within-group shock correlation [0-1]"),
    # ── score ────────────────────────────────────────────────────────────────
    "a":       ("score", float, 0.80, "Time-decay factor"),
    "c_otr":   ("score", float, 0.85, "On-time sigmoid center"),
    "k_otr":   ("score", float, 12.0, "On-time sigmoid slope"),
    "a_al":    ("score", float, 0.70, "Avg-lateness penalty"),
    "a_ls":    ("score", float, 0.60, "Late-streak penalty"),
    "a_slip":  ("score", float, 0.80, "Post-payout slip enforcement"),
    "k_rules": ("score", float, 12.0, "Rules sigmoid slope"),
    "a_san":   ("score", float, 0.60, "Sanction decay"),
    "q0":      ("score", float, 0.50, "Bid centering"),
    "k_q":     ("score", float, 10.0, "Bid slope"),
    "a_v":     ("score", float, 0.80, "Bid volatility penalty"),
    "w_rep":   ("score", float, 5.0,  "Weight: repeat"),
    "w_cent":  ("score", float, 4.0,  "Weight: centrality"),
    "w_endf":  ("score", float, 3.0,  "Weight: foreman endorsement"),
    "w_ends":  ("score", float, 3.0,  "Weight: senior endorsement"),
    # ── global ───────────────────────────────────────────────────────────────
    "seed":    ("global", int,  42,   "Random seed"),
}

DEFAULTS = {k: v[2] for k, v in SCHEMA.items()}


# ---------------------------------------------------------------------------
# Build config objects from current values
# ---------------------------------------------------------------------------

def _build_configs(vals: Dict[str, Any]):
    pop = PopulationParams(
        n_groups=int(vals["n_groups"]),
        group_size_min=int(vals["group_size_min"]),
        group_size_max=max(int(vals["group_size_max"]), int(vals["group_size_min"]) + 1),
        rtype_bidding_prob=float(vals["rtype_bidding_prob"]),
        rules_prob=float(vals["rules_prob"]),
        p_ontime_mean=float(vals["p_ontime_mean"]),
        p_ontime_conc=float(vals["p_ontime_conc"]),
        post_slip_mean=float(vals["post_slip_mean"]),
        bid_agg_mean=float(vals["bid_agg_mean"]),
        p_rep=float(vals["p_rep"]),
        p_cent=float(vals["p_cent"]),
        p_endf=float(vals["p_endf"]),
    )
    macro = MacroEnvironment(
        stress_level=float(vals["stress_level"]),
        within_group_corr=float(vals["within_group_corr"]),
    )
    score_kwargs = {k: float(vals[k]) for k in SCHEMA if SCHEMA[k][0] == "score"}
    params = ScoreParams(**score_kwargs)
    return pop, macro, params


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

SECTION_COLOURS = {"pop": "cyan", "macro": "yellow", "score": "green", "global": "dim white"}


def _param_table(vals: Dict[str, Any], section: Optional[str] = None) -> Table:
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold white",
              show_edge=False, padding=(0, 1))
    t.add_column("param",   style="bold white",  no_wrap=True, width=20)
    t.add_column("value",   style="bold yellow",  no_wrap=True, width=10)
    t.add_column("default", style="dim white",    no_wrap=True, width=10)
    t.add_column("note",    style="dim white",    no_wrap=True)

    for k, (sec, typ, default, desc) in SCHEMA.items():
        if section and sec != section:
            continue
        v = vals[k]
        col = SECTION_COLOURS.get(sec, "white")
        changed = v != default
        val_str  = f"[bold {'magenta' if changed else 'yellow'}]{v}[/]"
        def_str  = f"[dim]{default}[/]"
        t.add_row(f"[{col}]{k}[/]", val_str, def_str, f"[dim]{desc}[/]")
    return t


def _hbar(value: float, maximum: float, width: int = 28) -> str:
    filled = int(value / maximum * width) if maximum else 0
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


def show_params(vals: Dict[str, Any], section: Optional[str] = None):
    sections = (
        [section] if section
        else ["pop", "macro", "score", "global"]
    )
    panels = []
    titles = {"pop": "Population", "macro": "Macro Env",
              "score": "Score Params", "global": "Global"}
    for sec in sections:
        t = _param_table(vals, sec)
        col = SECTION_COLOURS.get(sec, "white")
        panels.append(Panel(t, title=f"[bold {col}]{titles[sec]}[/]",
                            border_style=col, padding=(0, 1)))
    for p in panels:
        console.print(p)


def show_results(result, vals: Dict[str, Any]):
    v = result.validation
    df = result.member_df
    rho = v["spearman_rho"]

    # ── header ──────────────────────────────────────────────────────────────
    rho_colour = "green" if rho >= 0.35 else "yellow" if rho >= 0.15 else "red"
    console.print(Rule(
        f"[bold white]{v['n_members']} members · "
        f"{int(vals['n_groups'])} groups · "
        f"seed {int(vals['seed'])}[/]"
    ))

    # ── validation panel ────────────────────────────────────────────────────
    val_lines = [
        f"  [bold]Spearman ρ[/]  [{rho_colour}]{rho:+.4f}[/]   "
        f"[dim]{'strong' if rho>=0.35 else 'moderate' if rho>=0.15 else 'weak'}[/]",
        f"  [bold]Separation[/]  [cyan]{v['score_separation']:+.2f} pts[/]"
        f"  [dim](low-PD − high-PD score)[/]",
        f"  [bold]Score[/]       [white]{v['score_mean']:.1f} ± {v['score_std']:.1f}[/]",
        f"  [bold]True PD[/]     [white]{v['true_pd_mean']:.2%} ± {v['true_pd_std']:.2%}[/]",
    ]
    console.print(Panel("\n".join(val_lines), title="[bold]Validation[/]",
                        border_style=rho_colour, padding=(0, 1)))

    # ── score by PD quintile bar chart ──────────────────────────────────────
    qt = v["score_by_pd_quintile"]
    means = qt["mean"].values
    max_m = float(max(means)) or 1.0

    qt_lines = []
    for i, (idx, row) in enumerate(qt.iterrows()):
        bar = _hbar(row["mean"], 100, 30)
        colour = ["green", "green", "yellow", "red", "red"][i]
        qt_lines.append(
            f"  [{colour}]{str(idx):<6}[/] [dim white]│[/] "
            f"[{colour}]{bar}[/] "
            f"[bold white]{row['mean']:5.1f}[/] [dim]n={int(row['count'])}[/]"
        )
    console.print(Panel(
        "\n".join(qt_lines),
        title="[bold]Score by True-PD Quintile  [dim](Q1=lowest risk)[/][/]",
        border_style="blue", padding=(0, 1),
    ))

    # ── pillar utilisation ───────────────────────────────────────────────────
    pillars = [("s_pdis",35),("s_ordr",15),("s_gov",20),("s_liq",15),("s_soc",15)]
    pil_lines = []
    for col, mx in pillars:
        m = df[col].mean()
        bar = _hbar(m, mx, 20)
        pct = m / mx * 100
        colour = "green" if pct >= 65 else "yellow" if pct >= 40 else "red"
        pil_lines.append(
            f"  [bold white]{col:<8}[/] [{colour}]{bar}[/] "
            f"[{colour}]{m:5.2f}[/][dim]/{mx}  {pct:.0f}%[/]"
        )
    console.print(Panel(
        "\n".join(pil_lines),
        title="[bold]Pillar Utilisation  [dim](mean across population)[/][/]",
        border_style="magenta", padding=(0, 1),
    ))

    # ── top / bottom members ────────────────────────────────────────────────
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold white",
              show_edge=False, padding=(0, 1))
    for col in ["mid","gid","true_pd","p_ontime","score","pdis","ordr","gov","liq","soc"]:
        t.add_column(col, no_wrap=True)

    top5    = df.nlargest(5,  "score")
    bottom5 = df.nsmallest(5, "score")

    def _add_rows(subset, colour):
        for _, r in subset.iterrows():
            pd_col = "green" if r["true_pd"] < 0.05 else "yellow" if r["true_pd"] < 0.15 else "red"
            sc_col = "green" if r["score"] >= 60   else "yellow" if r["score"] >= 35   else "red"
            t.add_row(
                f"[dim]{r['mid']}[/]",
                f"[dim]{r['gid']}[/]",
                f"[{pd_col}]{r['true_pd']:.3f}[/]",
                f"[dim]{r['p_ontime_raw']:.3f}[/]",
                f"[bold {sc_col}]{r['score']:.1f}[/]",
                f"{r['s_pdis']:.1f}",f"{r['s_ordr']:.1f}",
                f"{r['s_gov']:.1f}", f"{r['s_liq']:.1f}",f"{r['s_soc']:.1f}",
            )

    _add_rows(top5, "green")
    t.add_row(*["[dim]─────[/]"]*10)
    _add_rows(bottom5, "red")

    console.print(Panel(t, title="[bold]Top 5 / Bottom 5 by Score[/]",
                        border_style="white", padding=(0, 1)))


def show_sweep(param: str, values: List[float], vals: Dict[str, Any]):
    console.print(Rule(f"[bold]Sweep: [cyan]{param}[/] over {values}[/]"))

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold white",
              show_edge=False, padding=(0, 1))
    t.add_column(param,     style="bold cyan",   width=14)
    t.add_column("ρ",       style="bold yellow", width=8)
    t.add_column("sep",     style="bold green",  width=8)
    t.add_column("mean",    width=8)
    t.add_column("std",     width=8)
    t.add_column("PD mean", width=9)
    t.add_column("visual",  width=32)

    rhos = []
    for v_val in values:
        test_vals = {**vals, param: v_val}
        pop, macro, params = _build_configs(test_vals)
        with console.status(f"[dim]  {param}={v_val}…[/]", spinner="dots"):
            result = generate_population(pop, macro, params, seed=int(vals["seed"]))
        vv = result.validation
        rho = vv["spearman_rho"]
        rhos.append(rho)
        rho_col = "green" if rho >= 0.35 else "yellow" if rho >= 0.15 else "red"
        bar = _hbar(max(rho, 0), 0.7, 20)
        t.add_row(
            str(v_val),
            f"[{rho_col}]{rho:+.4f}[/]",
            f"{vv['score_separation']:+.2f}",
            f"{vv['score_mean']:.2f}",
            f"{vv['score_std']:.2f}",
            f"{vv['true_pd_mean']:.2%}",
            f"[{rho_col}]{bar}[/]",
        )

    console.print(t)
    delta = max(rhos) - min(rhos)
    colour = "green" if delta < 0.05 else "yellow" if delta < 0.15 else "red"
    console.print(f"  [dim]ρ range:[/] [{colour}]{min(rhos):+.4f} → {max(rhos):+.4f}  Δ={delta:.4f}[/]")
    if delta < 0.05:
        console.print("  [green]✓  Score is robust to this parameter.[/]")
    elif delta < 0.15:
        console.print("  [yellow]⚠  Moderate sensitivity — monitor during calibration.[/]")
    else:
        console.print("  [red]✗  High sensitivity — review parameter range.[/]")


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

HELP = """
[bold cyan]Commands[/]
  [bold]run[/]                      run simulation, display results
  [bold]set[/] [cyan]<param>[/] [yellow]<value>[/]      change a parameter
  [bold]sweep[/] [cyan]<param>[/] [yellow]v1 v2 ...[/]  sensitivity: run once per value
  [bold]show[/] [dim][pop|macro|score][/]  display current params (all if no arg)
  [bold]reset[/]                    restore all defaults
  [bold]help[/]                     this message
  [bold]q[/] / [bold]quit[/]               exit

[bold cyan]Examples[/]
  set stress_level 0.40
  set n_groups 30
  sweep a_al 0.4 0.7 1.2
  sweep stress_level 0.0 0.2 0.5 0.8
  show macro
"""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    vals = dict(DEFAULTS)

    all_params = list(SCHEMA.keys())
    commands   = ["run","set","sweep","show","reset","help","quit","q"]
    sections   = ["pop","macro","score","global"]
    completer  = WordCompleter(commands + all_params + sections, sentence=True)
    history    = InMemoryHistory()

    console.print(Panel(
        "[bold white]ROSCA Credit Score — Population Simulator[/]\n"
        "[dim]type [bold]help[/] for commands · tab-complete params after [bold]set[/][/]",
        border_style="cyan", padding=(0, 2),
    ))
    show_params(vals)

    while True:
        try:
            raw = prompt("\n[rosca]> ", completer=completer, history=history)
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]bye.[/]")
            break

        parts = raw.strip().split()
        if not parts:
            continue
        cmd, *args = parts

        # ── run ─────────────────────────────────────────────────────────────
        if cmd == "run":
            pop, macro, params = _build_configs(vals)
            with console.status("[bold green]Simulating…[/]", spinner="dots"):
                result = generate_population(pop, macro, params, seed=int(vals["seed"]))
            show_results(result, vals)

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
                new_val = typ(raw_val)
                old_val = vals[key]
                vals[key] = new_val
                console.print(
                    f"  [cyan]{key}[/]  "
                    f"[dim]{old_val}[/] → [bold yellow]{new_val}[/]"
                )
            except ValueError:
                console.print(f"[red]Cannot convert '{raw_val}' to {typ.__name__}[/]")

        # ── sweep ────────────────────────────────────────────────────────────
        elif cmd == "sweep":
            if len(args) < 2:
                console.print("[red]Usage: sweep <param> v1 v2 ...[/]")
                continue
            key = args[0]
            if key not in SCHEMA:
                console.print(f"[red]Unknown param '{key}'.[/]")
                continue
            _, typ, _, _ = SCHEMA[key]
            try:
                sweep_vals = [typ(v) for v in args[1:]]
            except ValueError:
                console.print("[red]All sweep values must be numeric.[/]")
                continue
            show_sweep(key, sweep_vals, vals)

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