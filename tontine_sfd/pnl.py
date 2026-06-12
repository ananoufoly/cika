"""pnl.py — Modules P&L Opérateur et P&L SFD.

Calculés à partir d'un ResultatPortefeuille (sortie de l'orchestrateur).

P&L OPÉRATEUR (la fintech) :
  Revenus : commissions fermes sur bids (alpha × total bids), rétrocession SFD optionnelle
  Coûts   : acquisition par membre, opérationnel par pool actif par mois, fixes mensuels
  Sorties : break-even en nombre de pools, sensibilité au taux de bid moyen et à alpha

P&L SFD (le partenaire dépositaire) :
  Revenus : intérêts sur lignes de crédit (décaissements immédiats), spread sur les dépôts
            (séquestre + consignations rémunérés à r_epargne mais replacés à r_replacement)
  Coûts   : coût du risque réalisé via la ligne contingente (niveau 3 de la cascade),
            rémunération de l'épargne
  Sortie clé : rendement net SFD par scénario + fréquence d'activation de la ligne contingente
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# P&L Opérateur
# ---------------------------------------------------------------------------

@dataclass
class PnLOperateur:
    # revenus
    commissions: float = 0.0
    retrocession_sfd: float = 0.0
    revenus_total: float = 0.0
    # coûts
    cout_acquisition: float = 0.0
    cout_ops: float = 0.0
    couts_fixes: float = 0.0
    couts_total: float = 0.0
    # résultat
    resultat_net: float = 0.0
    # contexte
    n_pools: int = 0
    n_membres: int = 0
    mois: int = 0
    break_even_pools: float = 0.0   # nb de pools pour atteindre l'équilibre


def calculer_pnl_operateur(res, cfg) -> PnLOperateur:
    m = res._meta
    n_pools = m["n_pools_eff"]
    n_membres = n_pools * cfg.structure.m_membres
    mois = cfg.structure.total_tours
    po = cfg.pnl_operateur

    # --- revenus ---
    commissions = m["commission_op_total"]
    # rétrocession : % des intérêts de crédit générés par les décaissements immédiats
    interets_credit = _interets_credit_sfd(res, cfg)
    retro = po.retrocession_sfd * interets_credit
    revenus = commissions + retro

    # --- coûts ---
    cout_acq = po.cout_acquisition_membre * n_membres
    # coût ops : décomposé (mobile money + notifications + support), par pool ACTIF par mois
    pools_actifs_mois = _pools_actifs_mois(res, cfg)
    cout_ops_unitaire = po.cout_ops_pool_mois(
        cfg.structure.pool_mensuel, cfg.structure.m_membres, cfg.mecanisme.part_immediate)
    cout_ops = cout_ops_unitaire * pools_actifs_mois
    couts_fixes = po.couts_fixes_mensuels * mois
    couts = cout_acq + cout_ops + couts_fixes

    net = revenus - couts

    # --- break-even en nombre de pools ---
    # On exprime revenus et coûts variables PAR POOL, et on résout :
    #   net(P) = (rev_par_pool - cout_var_par_pool) * P - couts_fixes_total = 0
    rev_par_pool = revenus / n_pools if n_pools else 0.0
    cout_var_par_pool = (cout_acq + cout_ops) / n_pools if n_pools else 0.0
    marge_par_pool = rev_par_pool - cout_var_par_pool
    break_even = (couts_fixes / marge_par_pool) if marge_par_pool > 1e-9 else float("inf")

    return PnLOperateur(
        commissions=commissions, retrocession_sfd=retro, revenus_total=revenus,
        cout_acquisition=cout_acq, cout_ops=cout_ops, couts_fixes=couts_fixes,
        couts_total=couts, resultat_net=net,
        n_pools=n_pools, n_membres=n_membres, mois=mois, break_even_pools=break_even,
    )


# ---------------------------------------------------------------------------
# P&L SFD
# ---------------------------------------------------------------------------

@dataclass
class PnLSFD:
    # revenus
    interets_credit: float = 0.0     # intérêts sur les décaissements immédiats (lignes de crédit)
    spread_depots: float = 0.0       # (r_replacement - r_epargne) sur les dépôts moyens
    revenus_total: float = 0.0
    # coûts
    cout_risque: float = 0.0         # tirages non remboursés sur la ligne contingente (niv3)
    remuneration_epargne: float = 0.0
    couts_total: float = 0.0
    # résultat
    resultat_net: float = 0.0
    rendement_net: float = 0.0       # résultat net / dépôts moyens
    # risque
    freq_activation_ligne: int = 0   # nb de (pool,tour) ayant tiré sur la ligne contingente
    perte_ligne: float = 0.0         # = cout_risque
    depots_moyens: float = 0.0


def _interets_credit_sfd(res, cfg) -> float:
    """Intérêts perçus par la SFD sur les décaissements immédiats (lignes de crédit).
    Pré-calculé à la volée par l'orchestrateur (champ _meta)."""
    return float(res._meta.get("interets_credit_total", 0.0))


def _depots_moyens(res, cfg) -> float:
    """Dépôts moyens à la SFD (réserve moyenne + séquestre/collecte moyenne).
    Pré-calculé à la volée par l'orchestrateur."""
    return float(res._meta.get("depots_moyens", 0.0))


def _pools_actifs_mois(res, cfg) -> float:
    """Somme sur tous les mois du nombre de pools actifs. Pré-calculé à la volée."""
    return float(res._meta.get("pools_actifs_mois", 0.0))


def calculer_pnl_sfd(res, cfg) -> PnLSFD:
    t = res.tracker

    interets = _interets_credit_sfd(res, cfg)
    depots_moy = _depots_moyens(res, cfg)
    mois = cfg.structure.total_tours

    # spread sur les dépôts : la SFD rémunère l'épargne à r_epargne mais replace à r_replacement
    spread_mensuel = cfg.sfd.r_replacement - cfg.sfd.r_epargne
    spread = depots_moy * spread_mensuel * mois

    revenus = interets + spread

    # coût du risque = tirages sur la ligne contingente (niv3) non remboursés
    cout_risque = t.perte_sfd
    # rémunération de l'épargne versée aux membres (coût pour la SFD)
    remuneration = depots_moy * cfg.sfd.r_epargne * mois

    couts = cout_risque + remuneration
    net = revenus - couts
    rendement = (net / depots_moy) if depots_moy > 1e-9 else 0.0

    return PnLSFD(
        interets_credit=interets, spread_depots=spread, revenus_total=revenus,
        cout_risque=cout_risque, remuneration_epargne=remuneration, couts_total=couts,
        resultat_net=net, rendement_net=rendement,
        freq_activation_ligne=t.freq_niv3, perte_ligne=t.perte_sfd, depots_moyens=depots_moy,
    )


# ---------------------------------------------------------------------------
# Sensibilités Opérateur (alpha, taux de bid moyen)
# ---------------------------------------------------------------------------

def sensibilite_operateur(res, cfg, alphas: List[float], facteurs_bid: List[float]) -> List[dict]:
    """Sensibilité du résultat net Opérateur à alpha et au taux de bid moyen.

    On réutilise le total des bids observé et on rescale :
      - alpha agit linéairement sur la commission,
      - facteur_bid rescale le volume total des bids (proxy du taux d'enchère moyen).
    Les coûts (acquisition, ops, fixes) sont inchangés.
    """
    base = calculer_pnl_operateur(res, cfg)
    total_bids = res._meta["total_bids"]
    couts = base.couts_total
    retro_base = base.retrocession_sfd  # ne dépend pas d'alpha/bid ici (conservateur)
    out = []
    for a in alphas:
        for fb in facteurs_bid:
            commissions = a * total_bids * fb
            revenus = commissions + retro_base
            net = revenus - couts
            out.append({
                "alpha": a, "facteur_bid": fb,
                "commissions": commissions, "resultat_net": net,
            })
    return out
