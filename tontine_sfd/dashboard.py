"""dashboard.py — Graphiques du simulateur (matplotlib).

Trois graphiques principaux :
  1. Distribution des pertes SFD (par scénario) — Monte Carlo
  2. P&L Opérateur vs nombre de pools (courbe de break-even)
  3. Activation de la cascade par niveau (par scénario)

Plus un tableau récapitulatif texte (depuis scenarios.afficher_recap).

Usage :
  python dashboard.py            # lance le MC si besoin et produit les figures PNG
  python dashboard.py --rapide   # 200 runs au lieu de 1000 (itération rapide)
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")  # backend sans affichage (sauvegarde PNG)
import matplotlib.pyplot as plt

from config import config_par_defaut
from orchestrateur import Orchestrateur
from pnl import calculer_pnl_operateur
from scenarios import (SCENARIOS, simuler_scenario_mc, lancer_tous_scenarios,
                       afficher_recap, config_par_defaut_depuis, _kpis_run)


# Palette neutre et sobre
COULEURS = {
    "nominal": "#0f4c4a",
    "comportemental": "#b18a3a",
    "macro": "#b91c1c",
    "combine": "#5b6b87",
    "bids_faibles": "#14706c",
}
LABELS = {
    "nominal": "Nominal",
    "comportemental": "Stress comportemental",
    "macro": "Stress macro",
    "combine": "Combiné",
    "bids_faibles": "Bids faibles",
}


# ---------------------------------------------------------------------------
# Collecte des distributions brutes (pour les histogrammes)
# ---------------------------------------------------------------------------

def collecter_distributions(cfg_base=None, n_runs=1000):
    """Relance les scénarios en gardant les valeurs PAR RUN (pas seulement agrégées),
    pour tracer les distributions."""
    cfg_base = cfg_base or config_par_defaut()
    distributions = {}
    for nom in SCENARIOS:
        transform = SCENARIOS[nom]
        pertes, pnl_op, pnl_sfd = [], [], []
        for i in range(n_runs):
            cfg = transform(config_par_defaut_depuis(cfg_base))
            res = Orchestrateur(cfg).simuler(graine=cfg_base.monte_carlo.graine_base + i,
                                             mode_leger=True)
            k = _kpis_run(res, cfg)
            pertes.append(k["perte_sfd"])
            pnl_op.append(k["pnl_operateur"])
            pnl_sfd.append(k["pnl_sfd"])
        distributions[nom] = {"perte_sfd": np.array(pertes),
                              "pnl_operateur": np.array(pnl_op),
                              "pnl_sfd": np.array(pnl_sfd)}
    return distributions


# ---------------------------------------------------------------------------
# Graphique 1 : distribution des pertes SFD
# ---------------------------------------------------------------------------

def graphique_pertes_sfd(distributions, fichier="fig_pertes_sfd.png"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for nom in SCENARIOS:
        pertes = distributions[nom]["perte_sfd"] / 1000.0  # en milliers
        ax.hist(pertes, bins=40, alpha=0.55, label=LABELS[nom], color=COULEURS[nom])
    ax.set_xlabel("Perte SFD (milliers XOF)")
    ax.set_ylabel("Fréquence (runs)")
    ax.set_title("Distribution des pertes SFD par scénario\n"
                 "(la perte est bornée par la ligne contingente — argument du pitch partenaire)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(fichier, dpi=130)
    plt.close(fig)
    return fichier


# ---------------------------------------------------------------------------
# Graphique 2 : P&L Opérateur vs nombre de pools (break-even)
# ---------------------------------------------------------------------------

def graphique_break_even(cfg_base=None, n_pools_max=1200, fichier="fig_break_even.png"):
    """Trace le P&L Opérateur en fonction du nombre de pools (scénario nominal),
    en extrapolant à partir de la marge contributive par pool et des coûts fixes."""
    cfg_base = cfg_base or config_par_defaut()
    cfg = config_par_defaut_depuis(cfg_base)
    res = Orchestrateur(cfg).simuler(graine=cfg.monte_carlo.graine_base, mode_leger=True)
    op = calculer_pnl_operateur(res, cfg)
    n_ref = op.n_pools
    marge_par_pool = (op.commissions - op.cout_acquisition - op.cout_ops) / n_ref
    fixes_total = op.couts_fixes

    pools = np.arange(0, n_pools_max + 1, 10)
    pnl = marge_par_pool * pools - fixes_total
    break_even = fixes_total / marge_par_pool if marge_par_pool > 0 else np.inf

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(pools, pnl / 1e6, color="#0f4c4a", lw=2.2, label="P&L Opérateur")
    ax.axhline(0, color="#5b6b87", ls="--", lw=1)
    if np.isfinite(break_even):
        ax.axvline(break_even, color="#b18a3a", ls="--", lw=1.5,
                   label=f"Break-even ≈ {break_even:.0f} pools")
    ax.set_xlabel("Nombre de pools")
    ax.set_ylabel("P&L Opérateur (millions XOF)")
    ax.set_title("P&L Opérateur vs nombre de pools — courbe de break-even\n"
                 f"(marge contributive ≈ {marge_par_pool:,.0f} XOF/pool ; "
                 f"coûts fixes {fixes_total/1e6:.1f} M/horizon)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(fichier, dpi=130)
    plt.close(fig)
    return fichier, break_even


# ---------------------------------------------------------------------------
# Graphique 3 : activation de la cascade par niveau
# ---------------------------------------------------------------------------

def graphique_cascade(resultats_mc, fichier="fig_cascade.png"):
    """Barres empilées : fréquence moyenne d'activation de chaque niveau, par scénario."""
    noms = list(SCENARIOS.keys())
    niveaux = ["freq_niv1", "freq_niv2", "freq_niv3", "freq_niv4"]
    labels_niv = ["Niv 1 — Réserve pool", "Niv 2 — Méta-réserve",
                  "Niv 3 — Ligne SFD", "Niv 4 — Report"]
    couleurs_niv = ["#0f4c4a", "#14706c", "#b18a3a", "#b91c1c"]

    data = np.array([[resultats_mc[nom]["kpis"][niv]["moyenne"] for nom in noms]
                     for niv in niveaux])  # (4 niveaux, 5 scénarios)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bas = np.zeros(len(noms))
    x = np.arange(len(noms))
    for i, niv in enumerate(niveaux):
        ax.bar(x, data[i], bottom=bas, label=labels_niv[i], color=couleurs_niv[i])
        bas += data[i]
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in noms], rotation=15, ha="right")
    ax.set_ylabel("Activations moyennes (nb de tours × pools)")
    ax.set_title("Activation de la cascade de garanties par niveau et par scénario\n"
                 "(en nominal la cascade s'arrête bas ; sous stress elle monte vers la ligne SFD)")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(fichier, dpi=130)
    plt.close(fig)
    return fichier


# ---------------------------------------------------------------------------
# Pilotage
# ---------------------------------------------------------------------------

def main():
    rapide = "--rapide" in sys.argv
    n_runs = 200 if rapide else 1000
    cfg = config_par_defaut()

    # charger les résultats MC agrégés si disponibles, sinon les calculer
    if os.path.exists("resultats_mc.pkl") and not rapide:
        print("Chargement de resultats_mc.pkl…")
        with open("resultats_mc.pkl", "rb") as f:
            resultats_mc = pickle.load(f)
    else:
        print(f"Lancement Monte Carlo ({n_runs} runs × {len(SCENARIOS)} scénarios)…")
        resultats_mc = lancer_tous_scenarios(cfg, n_runs=n_runs)

    afficher_recap(resultats_mc)

    print("\nGénération des graphiques…")
    print("  Collecte des distributions (pertes SFD, P&L)…")
    distributions = collecter_distributions(cfg, n_runs=n_runs)

    f1 = graphique_pertes_sfd(distributions)
    print(f"  ✓ {f1}")
    f2, be = graphique_break_even(cfg)
    print(f"  ✓ {f2}  (break-even ≈ {be:.0f} pools)")
    f3 = graphique_cascade(resultats_mc)
    print(f"  ✓ {f3}")
    print("\nTerminé.")


if __name__ == "__main__":
    main()
