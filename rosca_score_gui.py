import sys
import pandas as pd
from rich.console import Console
from rich.table import Table
from rosca_score_engine import (
    MacroEnvironment, PopulationParams, ScoreParams, generate_population
)

console = Console()

def main():
    pop = PopulationParams(n_groups=20)
    macro = MacroEnvironment(stress_level=0.1)
    params = ScoreParams()
    
    console.print("[bold green]Running Terminal Simulation...[/]")
    result = generate_population(pop, macro, params, seed=42)
    df = result.member_df

    # Results Table
    t = Table(title="Simulation Summary")
    t.add_column("MID"); t.add_column("Score"); t.add_column("Default?"); t.add_column("PD*")
    for _, r in df.head(10).iterrows():
        t.add_row(f"M{_}", f"{r['score']:.1f}", "YES" if r['is_defaulter'] else "no", f"{r['true_pd_star']:.3f}")
    console.print(t)
    
    console.print(f"\n[bold yellow]Spearman ρ* (Calibrated): {result.rho_star:.4f}[/]")
    console.print(f"[bold magenta]Pillar Weights:[/] {result.calibrated_weights}")

if __name__ == "__main__":
    main()