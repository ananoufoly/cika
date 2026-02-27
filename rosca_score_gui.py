#!/usr/bin/env python3
# rosca_score_gui.py  (fixed)
"""
Terminal interface for the ROSCA credit score population simulator.

Features:
 - simulate population and compute scores
 - detect defaults (miss N meetings after allocation)
 - Monte Carlo PD* estimation and logistic PD* fitting
 - interactive CLI with tab completion
"""

from __future__ import annotations

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

from rosca_score_engine import (
    MacroEnvironment,
    PopulationParams,
    ScoreParams,
    generate_population,
    generate_population_with_defaults,
    compute_pd_star_mc,
    fit_logistic_pd_star,
    compute_pd_star_validation,
)

console = Console()

# ---------------------------------------------------------------------------
# Parameter schema:  name → (section, type, default, description)
# ---------------------------------------------------------------------------

SCHEMA: Dict[str, Tuple[str, type, Any, str]] = {
    # population
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
    # macro
    "stress_level":       ("macro", float, 0.0,   "Systemic stress [0-1]"),
    "within_group_corr":  ("macro", float, 0.20,  "Within-group shock correlation [0-1]"),
    # score
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
    # defaults & PD*
    "streak_threshold":   ("pdstar", int,   3,    "Missed meetings after allocation → default threshold"),
    "mc_runs":            ("pdstar", int, 200,    "Monte Carlo runs for PD* estimation"),
    # global
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

SECTION_COLOURS = {"pop": "cyan", "macro": "yellow", "score": "green", "global": "dim white", "pdstar": "magenta"}

def _param_table(vals: Dict[str, Any], section: Optional[str] = None) -> Table:
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold white",
              show_edge=False, padding=(0, 1))
    t.add_column("param",   style="bold white",  no_wrap=True, width=20)
    t.add_column("value",   style="bold yellow",  no_wrap=True, width=12)
    t.add_column("default", style="dim white",    no_wrap=True, width=10)
    t.add_column("note",    style="dim white",    no_wrap=True)

    for k, (sec, typ, default, desc) in SCHEMA.items():
        if section and sec != section:
            continue
        v = vals.get(k, default)
        col = SECTION_COLOURS.get(sec, "white")
        changed = v != default
        val_str  = f"[bold {'magenta' if changed else 'yellow'}]{v}[/]"
        def_str  = f"[dim]{default}[/]"
        t.add_row(f"[{col}]{k}[/]", val_str, def_str, f"[dim]{desc}[/]")
    return t

def show_params(vals: Dict[str, Any], section: Optional[str] = None):
    sections = ([section] if section else ["pop", "macro", "score", "pdstar", "global"])
    panels = []
    titles = {"pop": "Population", "macro": "Macro Env", "score": "Score Params", "pdstar": "PD* / Defaults", "global": "Global"}
    for sec in sections:
        t = _param_table(vals, sec)
        col = SECTION_COLOURS.get(sec, "white")
        panels.append(Panel(t, title=f"[bold {col}]{titles[sec]}[/]", border_style=col, padding=(0, 1)))
    for p in panels:
        console.print(p)

def _hbar(value: float, maximum: float, width: int = 28) -> str:
    filled = int(value / maximum * width) if maximum else 0
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)

# ---------------------------------------------------------------------------
# Results display (with defaults/PD* awareness)
# ---------------------------------------------------------------------------

def show_results(result, vals: Dict[str, Any]):
    v = result.validation
    df = result.member_df
    rho = v["spearman_rho"]

    console.print(Rule(f"[bold white]{v['n_members']} members · {int(vals['n_groups'])} groups · seed {int(vals['seed'])}[/]"))

    val_lines = [
        f"  [bold]Spearman ρ[/]  [{'green' if rho>=0.35 else 'yellow' if rho>=0.15 else 'red'}]{rho:+.4f}[/]",
        f"  [bold]Separation[/]  [cyan]{v['score_separation']:+.2f} pts[/]",
        f"  [bold]Score[/]       [white]{v['score_mean']:.1f} ± {v['score_std']:.1f}[/]",
        f"  [bold]True PD[/]     [white]{v['true_pd_mean']:.2%} ± {v['true_pd_std']:.2%}[/]",
    ]
    console.print(Panel("\n".join(val_lines), title="[bold]Validation[/]", border_style="blue", padding=(0,1)))

    # Score by PD quintile
    qt = v["score_by_pd_quintile"]
    qt_lines = []
    for i, (idx, row) in enumerate(qt.iterrows()):
        bar = _hbar(row["mean"], 100, 30)
        colour = ["green", "green", "yellow", "red", "red"][i]
        qt_lines.append(f"  [{colour}]{str(idx):<6}[/] [dim white]│[/] [{colour}]{bar}[/] [bold white]{row['mean']:5.1f}[/] [dim]n={int(row['count'])}[/]")
    console.print(Panel("\n".join(qt_lines), title="[bold]Score by True-PD Quintile[/]", border_style="magenta", padding=(0,1)))

    # Defaulter panel (shown when default zeroing was applied)
    if "defaulted" in df.columns:
        n_def = int(df["defaulted"].sum())
        def_rate = df["defaulted"].mean()
        sc_def  = df.loc[df["defaulted"], "score"].mean() if n_def > 0 else 0.0
        sc_ndef = df.loc[~df["defaulted"], "score"].mean() if n_def < len(df) else 0.0
        console.print(Panel(
            f"  [bold]Defaulters (score → 0)[/]  [red]{n_def}[/] / {len(df)}  ({def_rate:.2%})\n"
            f"  [bold]Mean score — defaulted[/]  [red]{sc_def:.1f}[/]    [bold]non-defaulted[/]  [green]{sc_ndef:.1f}[/]",
            title="[bold]Default Rule[/]", border_style="red", padding=(0, 1),
        ))

    # Top / bottom by score
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold white", show_edge=False, padding=(0,1))
    for col in ["mid","gid","true_pd","p_ontime_raw","score","s_pdis","s_ordr","s_gov","s_liq","s_soc"]:
        t.add_column(col, no_wrap=True)
    top5 = df.nlargest(5, "score")
    bottom5 = df.nsmallest(5, "score")
    def _add_rows(subset):
        for _, r in subset.iterrows():
            t.add_row(str(r["mid"]), str(r["gid"]), f"{r['true_pd']:.3f}", f"{r['p_ontime_raw']:.3f}", f"{r['score']:.1f}",
                      f"{r['s_pdis']:.1f}", f"{r['s_ordr']:.1f}", f"{r['s_gov']:.1f}", f"{r['s_liq']:.1f}", f"{r['s_soc']:.1f}")
    _add_rows(top5)
    t.add_row(*["[dim]─────[/]"]*10)
    _add_rows(bottom5)
    console.print(Panel(t, title="[bold]Top / Bottom by Score[/]", border_style="white", padding=(0,1)))

# ---------------------------------------------------------------------------
# PD* helpers for CLI: run MC, fit logistic, evaluate
# ---------------------------------------------------------------------------

def run_mc_pd(pop, macro, params, n_runs: int, seed: int, streak_threshold: int):
    console.print(f"[dim]Running Monte Carlo PD* estimation ({n_runs} runs)…[/]")
    df = compute_pd_star_mc(pop, macro, params, n_runs=n_runs, base_seed=seed, K_min=6, streak_threshold=streak_threshold)
    console.print(Panel(f"[bold]MC PD* complete[/]\nMembers: {len(df)}\nMean PD*: {df['pd_star'].mean():.4f}", title="[bold]PD* (MC) Summary[/]", border_style="cyan"))
    return df

def run_fit_logit(stacked_runs_df, feature_cols: Optional[List[str]] = None):
    console.print("[dim]Fitting logistic PD* model…[/]")
    try:
        model, df_out = fit_logistic_pd_star(stacked_runs_df, feature_cols=feature_cols)
    except Exception as e:
        console.print(f"[red]Logistic fit failed: {e}[/]")
        return None, None
    console.print(Panel(f"[bold]Logistic PD* fitted[/]\nRows used: {len(df_out)}", title="[bold]PD* (Logistic) Summary[/]", border_style="green"))
    return model, df_out

# ---------------------------------------------------------------------------
# Help text (extended with PD* commands)
# ---------------------------------------------------------------------------

HELP = """
Commands
  run_defaults             — simulate with default-zeroing rule, fit logistic PD*, show results
  eval_pdstar              — compare score vs PD* (primary validation)
  fit_logit                — (re-)fit logistic PD* on last run_defaults result
  run                      — simulate without default-zeroing (baseline)
  mc_pd                    — optional: run Monte Carlo PD* estimation (mc_runs simulations, slower)
  set <param> <value>      — change one parameter
  sweep <param> v1 v2 ...  — sensitivity: run once per value, compare ρ
  show [pop|macro|score|pdstar] — print current params
  reset                    — restore all defaults
  help                     — command reference
  q / quit                 — exit
"""

# ---------------------------------------------------------------------------
# Main loop (extended)
# ---------------------------------------------------------------------------

def main():
    vals = dict(DEFAULTS)

    all_params = list(SCHEMA.keys())
    commands   = ["run","run_defaults","mc_pd","fit_logit","eval_pdstar","set","sweep","show","reset","help","quit","q"]
    sections   = ["pop","macro","score","pdstar","global"]
    completer  = WordCompleter(commands + all_params + sections, sentence=True)
    history    = InMemoryHistory()

    console.print(Panel("[bold white]ROSCA Credit Score — Population Simulator (CLI)[/]\n[dim]type 'help' for commands", border_style="cyan", padding=(0,2)))
    show_params(vals)

    # storage for PD* artifacts
    last_mc_df = None        # DataFrame from compute_pd_star_mc
    last_stacked_runs = None # optional stacked runs DataFrame for logistic fitting
    last_logit_df = None     # DataFrame with pd_star_logit

    while True:
        try:
            raw = prompt("\n[rosca]> ", completer=completer, history=history)
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]bye.[/]")
            break

        parts = raw.strip().split(None, 2)
        if not parts:
            continue
        cmd, *args = parts

        if cmd == "run":
            pop, macro, params = _build_configs(vals)
            with console.status("[bold green]Simulating…[/]", spinner="dots"):
                result = generate_population(pop, macro, params, seed=int(vals["seed"]))
            show_results(result, vals)

        elif cmd == "run_defaults":
            pop, macro, params = _build_configs(vals)
            with console.status("[bold green]Simulating (with defaults)…[/]", spinner="dots"):
                result = generate_population_with_defaults(pop, macro, params, seed=int(vals["seed"]), streak_threshold=int(vals["streak_threshold"]))
            show_results(result, vals)
            last_stacked_runs = result.member_df.copy()
            console.print("[dim]Stored last single-run member table (with 'default') for logistic fitting.[/]")

        elif cmd == "mc_pd":
            pop, macro, params = _build_configs(vals)
            n_runs = int(vals.get("mc_runs", 200))
            streak = int(vals.get("streak_threshold", 3))
            with console.status(f"[bold green]Running MC PD* ({n_runs} runs)…[/]", spinner="dots"):
                mc_df = run_mc_pd(pop, macro, params, n_runs=n_runs, seed=int(vals["seed"]), streak_threshold=streak)
            last_mc_df = mc_df
            console.print("[green]MC PD* finished and stored as last_mc_df.[/]")

        elif cmd == "fit_logit":
            if last_mc_df is None and last_stacked_runs is None:
                console.print("[red]No MC results or single-run with defaults available. Run 'mc_pd' or 'run_defaults' first.[/]")
                continue
            if last_stacked_runs is not None:
                console.print("[dim]Fitting logistic on last single-run member table (requires 'default' column).[/]")
                try:
                    model, df_out = fit_logistic_pd_star(last_stacked_runs)
                except Exception as e:
                    console.print(f"[red]Logistic fit failed: {e}[/]")
                    continue
            else:
                console.print("[yellow]No stacked runs available. Using MC frequency table to create an approximate stacked dataset (sampling).[/]")
                mc = last_mc_df.copy()
                rows = []
                n_runs = int(vals.get("mc_runs", 200))
                for _, r in mc.iterrows():
                    p = r["pd_star"]
                    draws = min(100, n_runs)
                    import numpy as _np
                    sampled = _np.random.binomial(1, p, size=draws)
                    for s in sampled:
                        rows.append({
                            "mid": r["mid"],
                            "p_ontime_raw": r["p_ontime_raw"],
                            "true_pd": r["true_pd"],
                            "default": int(s),
                        })
                stacked = __import__("pandas").DataFrame(rows)
                try:
                    model, df_out = fit_logistic_pd_star(stacked)
                except Exception as e:
                    console.print(f"[red]Logistic fit failed: {e}[/]")
                    continue
            last_logit_df = df_out
            console.print("[green]Logistic PD* model fitted and results stored in last_logit_df.[/]")

        elif cmd == "eval_pdstar":
            # Determine PD* source
            if last_mc_df is not None:
                pd_source_df = last_mc_df[["mid", "pd_star"]].copy()
                source = "mc_freq"
            elif last_logit_df is not None:
                pd_source_df = last_logit_df[["mid", "pd_star_logit"]].rename(
                    columns={"pd_star_logit": "pd_star"})
                source = "logistic"
            elif last_stacked_runs is not None and "default" in last_stacked_runs.columns:
                pd_source_df = last_stacked_runs[["mid"]].copy()
                pd_source_df["pd_star"] = last_stacked_runs["default"].astype(float).values
                source = "single_run_default"
            else:
                console.print("[red]No PD* source available. Run 'mc_pd', 'run_defaults', or 'fit_logit' first.[/]")
                continue

            # Merge PD* into current member_df (re-use last result if available, else re-run)
            if last_stacked_runs is not None and "score" in last_stacked_runs.columns:
                member_df = last_stacked_runs.copy()
            else:
                pop, macro, params = _build_configs(vals)
                streak = int(vals.get("streak_threshold", 3))
                with console.status("[bold green]Simulating for evaluation…[/]", spinner="dots"):
                    base_result = generate_population_with_defaults(
                        pop, macro, params, seed=int(vals["seed"]),
                        streak_threshold=streak,
                    )
                member_df = base_result.member_df.copy()

            pd_map = pd_source_df.set_index("mid")["pd_star"].to_dict()
            member_df["pd_star"] = member_df["mid"].map(pd_map)
            member_df = member_df.dropna(subset=["pd_star"])

            if member_df.empty:
                console.print("[red]No members matched between simulation and PD* source.[/]")
                continue

            # Primary validation via engine function
            pv = compute_pd_star_validation(member_df)
            rho = pv["spearman_rho_pdstar"]
            sep = pv["score_separation_pdstar"]

            rho_colour = "green" if rho >= 0.40 else ("yellow" if rho >= 0.20 else "red")
            lines = [
                f"  [bold]Source[/]                    [dim]{source}[/]",
                f"  [bold]Members evaluated[/]         {pv['n_members']}",
                f"  [bold]Spearman ρ (score, 1−PD*)[/] [{rho_colour}]{rho:+.4f}[/]  ← primary metric",
                f"  [bold]Score separation[/]          [cyan]{sep:+.2f} pts[/]  (low-PD* minus high-PD*)",
                f"  [bold]Mean PD*[/]                  {pv['pd_star_mean']:.4f} ± {pv['pd_star_std']:.4f}",
            ]
            if "n_defaulted" in pv:
                lines += [
                    f"  [bold]Defaulters[/]                [red]{pv['n_defaulted']}[/] ({pv['default_rate']:.2%})",
                    f"  [bold]Score: defaulted[/]          [red]{pv['score_mean_defaulted']:.1f}[/]    "
                    f"[bold]non-defaulted[/]  [green]{pv['score_mean_non_defaulted']:.1f}[/]",
                ]
            console.print(Panel("\n".join(lines), title="[bold]PD* Validation[/]", border_style="cyan"))

            # AUC if sklearn available and binary default column exists
            try:
                from sklearn.metrics import roc_auc_score, brier_score_loss
                if "defaulted" in member_df.columns:
                    y_true = member_df["defaulted"].astype(int).values
                    y_score = member_df["score"].values.astype(float)
                    smin, smax = y_score.min(), y_score.max()
                    y_norm = (y_score - smin) / (smax - smin + 1e-9)
                    auc = roc_auc_score(y_true, y_norm) if y_true.sum() > 0 else float("nan")
                    brier = brier_score_loss(y_true, y_norm)
                    auc_c = "green" if auc >= 0.70 else "yellow"
                    console.print(Panel(
                        f"  [bold]AUC  (score → default)[/]   [{auc_c}]{auc:.4f}[/]\n"
                        f"  [bold]Brier score[/]              {brier:.4f}",
                        title="[bold]Binary Default Metrics[/]", border_style="green",
                    ))
            except Exception:
                pass

            # Quintile table
            qt = pv["score_by_pdstar_quintile"]
            console.print(Panel(
                qt.to_string(),
                title="[bold]Score by PD* Quintile  (should decrease Q1 → Q5)[/]",
                border_style="magenta",
            ))

            # Top 10 by PD*
            top = member_df.sort_values("pd_star", ascending=False).head(10)
            show_cols = ["mid", "gid", "pd_star", "score", "true_pd"] + \
                        (["defaulted"] if "defaulted" in top.columns else [])
            console.print(Panel(top[show_cols].to_string(index=False),
                                title="[bold]Top 10 by PD*[/]", border_style="red"))

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
                old_val = vals.get(key, default)
                vals[key] = new_val
                console.print(f"  [cyan]{key}[/]  [dim]{old_val}[/] → [bold yellow]{new_val}[/]")
            except ValueError:
                console.print(f"[red]Cannot convert '{raw_val}' to {typ.__name__}[/]")

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
            try:
                show_sweep(key, sweep_vals, vals)
            except NameError:
                console.print("[red]Sweep helper not available.[/]")

        elif cmd == "show":
            sec = args[0] if args else None
            show_params(vals, sec)

        elif cmd == "reset":
            vals = dict(DEFAULTS)
            console.print("[green]All parameters reset to defaults.[/]")
            show_params(vals)

        elif cmd in ("help", "h"):
            console.print(HELP)

        elif cmd in ("q", "quit", "exit"):
            console.print("[dim]bye.[/]")
            break

        else:
            console.print(f"[red]Unknown command '{cmd}'. Type 'help'.[/]")

if __name__ == "__main__":
    main()
