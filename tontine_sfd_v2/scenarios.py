"""scenarios.py — 5 scénarios + Monte Carlo (Sorties A + B agrégées).

Scénarios : nominal / comportemental / macro / combiné / bids_faibles.
Monte Carlo ≥ 1000 runs par scénario ; moyennes + P5/P95.
"""

from __future__ import annotations
import copy
from typing import Callable, Dict, List

import numpy as np

from config import Config, config_par_defaut
from moteur import simuler_run
from pnl import calculer_pnl_operateur
from promesse import agreger_promesse, dimensionner_capital_p99


def _nominal(c): return c
def _comportemental(c):
    c.stress.comportemental_actif = True
    c.stress.choc_fuite = 0.06          # +6 pts de proba de fuite
    c.stress.bascule_urgents = 0.30
    return c
def _macro(c):
    c.stress.macro_actif = True
    c.stress.z_choc = -2.5
    c.stress.z_persistance = 4
    return c
def _combine(c):
    c = _comportemental(c); c = _macro(c); return c
def _bids_faibles(c):
    # population d'épargnants, faible valeur-temps : peu de bideurs, beaucoup de tours gratuits
    c.preferences.part_urgent = 0.05
    c.preferences.part_modere = 0.25
    c.preferences.part_epargnant = 0.70
    c.enchere.rho_mensuel = 0.01
    return c

SCENARIOS: Dict[str, Callable[[Config], Config]] = {
    "nominal": _nominal,
    "comportemental": _comportemental,
    "macro": _macro,
    "combine": _combine,
    "bids_faibles": _bids_faibles,
}


def simuler_scenario(nom: str, cfg_base: Config = None, n_runs: int = None) -> dict:
    cfg_base = cfg_base or config_par_defaut()
    n_runs = n_runs or cfg_base.monte_carlo.n_runs
    graine0 = cfg_base.monte_carlo.graine_base
    transform = SCENARIOS[nom]

    runs = []
    pnl_nets = []
    for i in range(n_runs):
        c = transform(copy.deepcopy(cfg_base))
        r = simuler_run(c, graine=graine0 + i)
        runs.append(r)
        pnl_nets.append(calculer_pnl_operateur(r, c).resultat_net)

    promesse = agreger_promesse(runs)
    capital = dimensionner_capital_p99(runs)
    pnl = np.array(pnl_nets)
    couts_membre = np.array([r.cout_membre_tour1 for r in runs])
    pot = (cfg_base.structure.m_membres - 1) * cfg_base.structure.c

    return {
        "scenario": nom, "n_runs": n_runs,
        # Sortie A — P&L Opérateur
        "pnl_operateur_moyen": float(pnl.mean()),
        "pnl_operateur_p5": float(np.percentile(pnl, 5)),
        "pnl_operateur_p95": float(np.percentile(pnl, 95)),
        # Sortie B — promesse
        "taux_continuite": promesse["taux_continuite"],
        "p_promesse_cassee": promesse["p_promesse_cassee"],
        "exposition_sfd_max": promesse["exposition_max_moyenne"],
        "freq_perte_sfd": promesse["freq_perte_sfd"],
        "perte_sfd_moyenne": promesse["perte_sfd_moyenne"],
        "perte_sfd_p95": promesse["perte_sfd_p95"],
        "fuites_moyenne": promesse["fuites_moyenne"],
        # dimensionnement
        "besoin_couverture_p99": capital["besoin_couverture_p99"],
        "ratio_couverture_p99": capital["ratio_couverture_p99"],
        # secondaires
        "cout_membre_tour1_moyen": float(couts_membre.mean()),
        "cout_membre_tour1_pct_pot": float(couts_membre.mean() / pot),
        "_promesse_detail": promesse,
        "_runs": runs,  # gardé pour les graphiques
    }


def lancer_tous(cfg_base: Config = None, n_runs: int = None) -> Dict[str, dict]:
    cfg_base = cfg_base or config_par_defaut()
    return {nom: simuler_scenario(nom, cfg_base, n_runs) for nom in SCENARIOS}


def afficher_recap(resultats: Dict[str, dict]):
    noms = list(resultats.keys())
    n_runs = resultats[noms[0]]["n_runs"]
    print("=" * 104)
    print(f"RÉCAPITULATIF — {n_runs} runs Monte Carlo par scénario")
    print("=" * 104)
    print(f"  {'KPI':<34}" + "".join(f"{n:>14}" for n in noms))
    print("  " + "-" * 100)

    def ligne(label, cle, fmt="M", pct=False):
        cells = []
        for n in noms:
            v = resultats[n][cle]
            if pct:
                cells.append(f"{v*100:.1f}%")
            elif fmt == "M":
                cells.append(f"{v/1e6:,.1f}M")
            elif fmt == "x":
                cells.append(f"{v:.1f}x")
            elif fmt == "k":
                cells.append(f"{v/1e3:,.0f}k")
            else:
                cells.append(f"{v:,.0f}")
        print(f"  {label:<34}" + "".join(f"{c:>14}" for c in cells))

    print("  — SORTIE A : P&L Opérateur —")
    ligne("P&L Opérateur (moyen)", "pnl_operateur_moyen", "M")
    print("  — SORTIE B : Promesse —")
    ligne("Taux de continuité", "taux_continuite", pct=True)
    ligne("P[promesse cassée]", "p_promesse_cassee", pct=True)
    ligne("Exposition SFD max", "exposition_sfd_max", "M")
    ligne("Fréq. perte SFD", "freq_perte_sfd", pct=True)
    ligne("Perte SFD moyenne", "perte_sfd_moyenne", "M")
    ligne("Perte SFD P95", "perte_sfd_p95", "M")
    print("  — Dimensionnement & conformité —")
    ligne("Besoin couverture P99", "besoin_couverture_p99", "M")
    ligne("Coût membre tour 1", "cout_membre_tour1_moyen", "k")
    ligne("Coût membre tour 1 (% pot)", "cout_membre_tour1_pct_pot", pct=True)
    ligne("Fuites (moyenne)", "fuites_moyenne", "n")
    print("=" * 104)
