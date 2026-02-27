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

"""
rosca_score_gui.py
Updated for Sequence 2 Engine (Hard Defaults & ML Calibration).
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

# Ensure the updated engine is imported
from rosca_score_engine import (
    MacroEnvironment,
    PopulationParams,
    ScoreParams,
    generate_population,
)

console = Console()

# ... (SCHEMA and DEFAULTS remain the same as your provided code) ...

# ---------------------------------------------------------------------------
# UPDATED: Display results with Sequence 2 Logic
# ---------------------------------------------------------------------------

def show_results(result, vals: Dict[str, Any]):
    df = result.member_df
    weights = result.calibrated_weights
    rho_star = result.rho_star
    
    # Calculate default rate
    def_count = df["is_defaulter"].sum()
    def_pct = (def_count / len(df)) * 100

    # ── Header ──────────────────────────────────────────────────────────────
    console.print(Rule(
        f"[bold white]{len(df)} members · "
        f"{int(vals['n_groups'])} groups · "
        f"seed {int(vals['seed'])}[/]"
    ))

    # ── Validation Panel (Enhanced) ─────────────────────────────────────────
    rho_colour = "green" if rho_star >= 0.40 else "yellow" if rho_star >= 0.20 else "red"
    
    val_lines = [
        f"  [bold]Calibrated ρ*[/]  [{rho_colour}]{rho_star:+.4f}[/] [dim](Score vs 1-PD*)[/]",
        f"  [bold]Hard Defaults[/] [bold red]{int(def_count)}[/] [dim]({def_pct:.1f}% of population)[/]",
        f"  [bold]Score Mean[/]    [white]{df['score'].mean():.1f} ± {df['score'].std():.1f}[/]",
        f"  [bold]Oracle PD[/]    [white]{df['true_pd_oracle'].mean():.2%} ± {df['true_pd_oracle'].std():.2%}[/]",
    ]
    console.print(Panel("\n".join(val_lines), title="[bold]Sequence 2: Validation[/]", 
                        border_style=rho_colour, padding=(0, 1)))

    # ── Calibrated Weights (New) ────────────────────────────────────────────
    weight_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    weight_table.add_column("Pillar", style="cyan")
    weight_table.add_column("Weight (Beta)", justify="right")
    weight_table.add_column("Influence")

    for pillar, val in weights.items():
        # High negative weight means the pillar is a strong "Default Preventer"
        inf_bar = "█" * int(abs(val) * 2)
        weight_table.add_row(pillar, f"{val:+.3f}", f"[magenta]{inf_bar}[/]")
    
    console.print(Panel(weight_table, title="[bold magenta]ML Calibration: Feature Importance[/]", 
                        border_style="magenta", padding=(0, 1)))

    # ── Score by PD* Quintile ───────────────────────────────────────────────
    # (Grouping by the calibrated PD* instead of just the oracle)
    df['_pd_q'] = pd.qcut(df['true_pd_star'], q=5, labels=["Q1","Q2","Q3","Q4","Q5"])
    qt = df.groupby('_pd_q')['score'].mean()
    
    qt_lines = []
    for i, (idx, mean_val) in enumerate(qt.items()):
        bar = _hbar(mean_val, 100, 30)
        colour = ["green", "green", "yellow", "red", "red"][i]
        qt_lines.append(f"  [{colour}]{str(idx):<6}[/] [dim white]│[/] [{colour}]{bar}[/] [bold white]{mean_val:5.1f}[/]")
    
    console.print(Panel("\n".join(qt_lines), title="[bold]Score by Calibrated PD* Quintile[/]", border_style="blue"))

    # ── Member Table (Added PD* and Default status) ────────────────────────
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold white")
    for col in ["mid", "score", "PD (Oracle)", "PD* (ML)", "Def", "Pdis", "Liq"]:
        t.add_column(col, no_wrap=True)

    subset = pd.concat([df.nlargest(5, "score"), df.nsmallest(5, "score")])
    for _, r in subset.iterrows():
        def_flag = "[bold red]YES[/]" if r["is_defaulter"] else "[dim]no[/]"
        t.add_row(
            f"[dim]{r['mid']}[/]",
            f"[bold]{r['score']:.1f}[/]",
            f"{r['true_pd_oracle']:.3f}",
            f"[cyan]{r['true_pd_star']:.3f}[/]",
            def_flag,
            f"{r['s_pdis']:.1f}", f"{r['s_liq']:.1f}"
        )
    console.print(Panel(t, title="[bold]Top/Bottom Members[/]"))

# ---------------------------------------------------------------------------
# UPDATED: Sweep with Sequence 2 Return Signature
# ---------------------------------------------------------------------------

def show_sweep(param: str, values: List[float], vals: Dict[str, Any]):
    console.print(Rule(f"[bold]Sweep: [cyan]{param}[/] over {values}[/]"))
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold white")
    t.add_column(param, style="bold cyan")
    t.add_column("ρ* (Calib)", style="bold yellow")
    t.add_column("Default %", style="bold red")
    t.add_column("Mean Score")
    t.add_column("Visual ρ*")

    for v_val in values:
        test_vals = {**vals, param: v_val}
        pop, macro, params = _build_configs(test_vals)
        
        with console.status(f"[dim]  {param}={v_val}…[/]", spinner="dots"):
            result = generate_population(pop, macro, params, seed=int(vals["seed"]))
        
        rho = result.rho_star
        def_pct = (result.member_df["is_defaulter"].sum() / len(result.member_df)) * 100
        rho_col = "green" if rho >= 0.40 else "yellow" if rho >= 0.20 else "red"
        
        t.add_row(
            str(v_val),
            f"[{rho_col}]{rho:+.4f}[/]",
            f"{def_pct:.1f}%",
            f"{result.member_df['score'].mean():.1f}",
            f"[{rho_col}]{_hbar(max(rho,0), 0.7, 20)}[/]"
        )
    console.print(t)

# ---------------------------------------------------------------------------
# Main Loop Update
# ---------------------------------------------------------------------------

# In the main() loop, the 'run' command needs to handle the SimulationResult object
# instead of a tuple (if you used the SimulationResult dataclass I provided).

# ... inside while True:
# if cmd == "run":
#     pop, macro, params = _build_configs(vals)
#     with console.status("[bold green]Running Sequence 1 & 2...[/]", spinner="dots"):
#         result = generate_population(pop, macro, params, seed=int(vals["seed"]))
#     show_results(result, vals)