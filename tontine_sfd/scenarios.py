"""scenarios.py — Scénarios de stress + simulation Monte Carlo.

Cinq scénarios :
  1. nominal        — conjoncture normale, aucun stress
  2. comportemental — choc sur les PD individuelles + bascule vers plus d'urgents
  3. macro          — choc systémique Z négatif (défauts corrélés via les charges sectorielles)
  4. combine        — comportemental + macro simultanés
  5. bids_faibles   — peu de membres enchérissent (réserve endogène réduite) — TEST DE VIABILITÉ

Pour chaque scénario : N runs Monte Carlo (défaut 1000), on rapporte moyennes et
percentiles 5/95 de chaque KPI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List

import numpy as np

from config import Config, config_par_defaut
from orchestrateur import Orchestrateur
from pnl import calculer_pnl_operateur, calculer_pnl_sfd


# ---------------------------------------------------------------------------
# Définition des scénarios (chacun = une transformation de la config)
# ---------------------------------------------------------------------------

def _scenario_nominal(cfg: Config) -> Config:
    return cfg


def _scenario_comportemental(cfg: Config) -> Config:
    cfg.stress.comportemental_actif = True
    cfg.stress.choc_pd = 0.03           # +3 pts de PD mensuelle
    cfg.stress.bascule_urgents = 0.30   # 30% des modérés/épargnants deviennent urgents
    return cfg


def _scenario_macro(cfg: Config) -> Config:
    cfg.stress.macro_actif = True
    cfg.stress.z_choc = -2.5            # choc systémique sévère
    cfg.stress.z_persistance = 4        # persiste 4 mois
    return cfg


def _scenario_combine(cfg: Config) -> Config:
    cfg = _scenario_comportemental(cfg)
    cfg = _scenario_macro(cfg)
    return cfg


def _scenario_bids_faibles(cfg: Config) -> Config:
    """Peu de membres enchérissent : on bascule la population vers les épargnants
    (faible urgence d'enchère) et on réduit rho. La réserve endogène (alimentée par
    les bids) est donc minimale — test de viabilité clé de la cascade."""
    cfg.preferences.part_urgent = 0.05
    cfg.preferences.part_modere = 0.25
    cfg.preferences.part_epargnant = 0.70
    cfg.mecanisme.rho_mensuel = 0.01    # valeur-temps faible -> bids faibles
    return cfg


SCENARIOS: Dict[str, Callable[[Config], Config]] = {
    "nominal": _scenario_nominal,
    "comportemental": _scenario_comportemental,
    "macro": _scenario_macro,
    "combine": _scenario_combine,
    "bids_faibles": _scenario_bids_faibles,
}


# ---------------------------------------------------------------------------
# Extraction des KPIs d'un run
# ---------------------------------------------------------------------------

def _kpis_run(res, cfg) -> Dict[str, float]:
    t = res.tracker
    m = res._meta
    n_pools = m["n_pools_eff"]
    n_membres = sum(len(p) for p in res.resultats_pools)

    op = calculer_pnl_operateur(res, cfg)
    sfd = calculer_pnl_sfd(res, cfg)

    # taux de complétion des cycles : part des membres NON en défaut à la fin
    n_actifs_fin = sum(1 for p in res.resultats_pools for mb in p if not mb.en_defaut)
    taux_completion = n_actifs_fin / n_membres if n_membres else 0.0

    # rétention cycle 1 -> cycle 2 : part des membres sans défaut au cycle 1 qui
    # restent actifs au cycle 2 (ici : membres dont le défaut, s'il a lieu, survient
    # au cycle 2 ; approximé par : actifs fin / actifs fin cycle 1)
    tours_c1 = cfg.structure.m_membres
    # un membre est "actif fin cycle 1" si son tour de défaut > tours_c1 (ou jamais)
    actifs_c1 = sum(1 for p in res.resultats_pools for mb in p
                    if (mb.tour_defaut is None or mb.tour_defaut > tours_c1))
    retention = (n_actifs_fin / actifs_c1) if actifs_c1 else 1.0

    # bonus moyen versé aux membres disciplinés
    elig = [mb for p in res.resultats_pools for mb in p
            if (not mb.en_defaut and mb.cycles_sans_defaut >= cfg.bonus.cycles_requis)]
    bonus_moyen = (sum(mb.bonus_recu for mb in elig) / len(elig)) if elig else 0.0

    return {
        "taux_completion": taux_completion,
        "retention_c1_c2": retention,
        "n_defauts": float(m["n_defauts_total"]),
        "taux_respect_k": res.infos_composition["taux_respect_k"],
        # cascade : fréquence d'activation par niveau
        "freq_niv1": float(t.freq_niv1),
        "freq_niv2": float(t.freq_niv2),
        "freq_niv3": float(t.freq_niv3),
        "freq_niv4": float(t.freq_niv4),
        "montant_niv1": t.niv1_reserve_pool,
        "montant_niv2": t.niv2_meta_reserve,
        "montant_niv3": t.niv3_ligne_sfd,
        "montant_niv4": t.niv4_prorata_report,
        # perte SFD
        "perte_sfd": t.perte_sfd,
        # P&L
        "pnl_operateur": op.resultat_net,
        "commissions_op": op.commissions,
        "break_even_pools": op.break_even_pools if np.isfinite(op.break_even_pools) else np.nan,
        "pnl_sfd": sfd.resultat_net,
        "rendement_sfd": sfd.rendement_net,
        # bonus
        "bonus_moyen": bonus_moyen,
    }


# ---------------------------------------------------------------------------
# Monte Carlo d'un scénario
# ---------------------------------------------------------------------------

def simuler_scenario_mc(nom_scenario: str, cfg_base: Config = None,
                        n_runs: int = None, graine_base: int = None) -> Dict:
    """Lance N runs Monte Carlo d'un scénario et agrège (moyenne, P5, P95) chaque KPI."""
    cfg_base = cfg_base or config_par_defaut()
    n_runs = n_runs or cfg_base.monte_carlo.n_runs
    graine_base = graine_base if graine_base is not None else cfg_base.monte_carlo.graine_base

    transform = SCENARIOS[nom_scenario]

    lignes: List[Dict[str, float]] = []
    for i in range(n_runs):
        # config fraîche par run (sinon le stress s'accumulerait)
        cfg = transform(config_par_defaut_depuis(cfg_base))
        res = Orchestrateur(cfg).simuler(graine=graine_base + i, mode_leger=True)
        lignes.append(_kpis_run(res, cfg))

    # agrégation
    agg: Dict[str, Dict[str, float]] = {}
    cles = lignes[0].keys()
    for k in cles:
        vals = np.array([l[k] for l in lignes], dtype=float)
        vals_valides = vals[~np.isnan(vals)]
        if len(vals_valides) == 0:
            agg[k] = {"moyenne": np.nan, "p5": np.nan, "p95": np.nan}
        else:
            agg[k] = {
                "moyenne": float(np.mean(vals_valides)),
                "p5": float(np.percentile(vals_valides, 5)),
                "p95": float(np.percentile(vals_valides, 95)),
            }
    return {"scenario": nom_scenario, "n_runs": n_runs, "kpis": agg}


def config_par_defaut_depuis(cfg_base: Config) -> Config:
    """Copie profonde d'une config (pour repartir d'une base propre à chaque run)."""
    import copy
    return copy.deepcopy(cfg_base)


# ---------------------------------------------------------------------------
# Lancer tous les scénarios
# ---------------------------------------------------------------------------

def lancer_tous_scenarios(cfg_base: Config = None, n_runs: int = None) -> Dict[str, Dict]:
    cfg_base = cfg_base or config_par_defaut()
    resultats = {}
    for nom in SCENARIOS:
        resultats[nom] = simuler_scenario_mc(nom, cfg_base, n_runs=n_runs)
    return resultats


# ---------------------------------------------------------------------------
# Affichage tableau récapitulatif
# ---------------------------------------------------------------------------

def afficher_recap(resultats: Dict[str, Dict]) -> None:
    def fmt(v, suffixe="", k=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        if k:
            return f"{v/1000:,.0f}k{suffixe}"
        return f"{v:,.0f}{suffixe}"

    noms = list(resultats.keys())
    print("=" * 100)
    print(f"RÉCAPITULATIF — {resultats[noms[0]]['n_runs']} runs Monte Carlo par scénario")
    print("=" * 100)

    def ligne(label, cle, suffixe="", pct=False, k=False):
        cells = []
        for nom in noms:
            kp = resultats[nom]["kpis"][cle]
            moy = kp["moyenne"]
            if pct:
                cells.append(f"{moy*100:.0f}%")
            elif k:
                cells.append(fmt(moy, suffixe, k=True))
            else:
                cells.append(fmt(moy, suffixe))
        print(f"  {label:<32}" + "".join(f"{c:>16}" for c in cells))

    print(f"  {'KPI (moyenne)':<32}" + "".join(f"{n:>16}" for n in noms))
    print("  " + "-" * 96)
    ligne("Taux complétion cycles", "taux_completion", pct=True)
    ligne("Rétention cycle 1->2", "retention_c1_c2", pct=True)
    ligne("Défauts (total)", "n_defauts")
    ligne("Taux respect contrainte K", "taux_respect_k", pct=True)
    print("  " + "-" * 96)
    ligne("Activation niv1 (réserve)", "freq_niv1")
    ligne("Activation niv2 (méta)", "freq_niv2")
    ligne("Activation niv3 (ligne SFD)", "freq_niv3")
    ligne("Activation niv4 (report)", "freq_niv4")
    print("  " + "-" * 96)
    ligne("Perte SFD", "perte_sfd", k=True)
    ligne("P&L Opérateur", "pnl_operateur", k=True)
    ligne("Break-even (pools)", "break_even_pools")
    ligne("P&L SFD", "pnl_sfd", k=True)
    ligne("Rendement net SFD", "rendement_sfd", pct=True)
    ligne("Bonus moyen / discipliné", "bonus_moyen")
    print("=" * 100)
