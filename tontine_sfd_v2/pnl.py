"""pnl.py — SORTIE A : P&L Opérateur + taux de fuite de break-even.

Revenus Opérateur : marge Opérateur (prime_operateur_taux × pot par encaissement)
                    + surplus d'enchère (bids au-dessus du coût). La prime de GARANTIE
                    n'est PAS un revenu Opérateur : elle alimente le FGE.
Pertes Opérateur  : l'Opérateur n'apporte pas de capital → ses pertes directes sont nulles
                    (le risque est porté par FGE puis tranche SFD). On suit néanmoins le
                    coût du capital théorique du FGE immobilisé (pour un ROE honnête).
Coûts             : acquisition/membre, opérations/pool/mois, fixes mensuels.

LE chiffre du pitch : TAUX DE FUITE DE BREAK-EVEN = niveau de fuite au-delà duquel le FGE
ne couvre plus les trous (la tranche SFD est alors sollicitée). Calculé par bissection.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from moteur import simuler_run


@dataclass
class PnLOperateur:
    revenus: float = 0.0           # marge Opérateur + surplus d'enchère
    marge_operateur: float = 0.0
    surplus_enchere: float = 0.0
    cout_acquisition: float = 0.0
    cout_ops: float = 0.0
    couts_fixes: float = 0.0
    resultat_net: float = 0.0
    cout_capital_fge: float = 0.0  # rémunération théorique du FGE immobilisé
    resultat_net_apres_capital: float = 0.0
    n_pools: int = 0
    break_even_pools: float = 0.0
    marge_par_pool: float = 0.0


def calculer_pnl_operateur(res, cfg) -> PnLOperateur:
    po = cfg.pnl_operateur
    n_pools = res.n_pools
    n_membres = n_pools * cfg.structure.m_membres
    mois = cfg.structure.total_tours

    marge_op = res.primes               # marge Opérateur (prime_operateur_taux)
    surplus = res.surplus_enchere
    revenus = marge_op + surplus

    cout_acq = po.cout_acquisition_membre * n_membres
    cout_ops = po.cout_ops_pool_mois * n_pools * mois
    couts_fixes = po.couts_fixes_mensuels * mois
    net = revenus - (cout_acq + cout_ops + couts_fixes)

    # coût du capital : le FGE immobilisé a un coût d'opportunité (même endogène)
    fge_immobilise = res.fge_provisions + res.fge_saisies
    cout_cap = fge_immobilise * (po.cout_capital_annuel * mois / 12.0)
    net_apres_cap = net - cout_cap

    marge_pool = (revenus - cout_acq - cout_ops) / n_pools if n_pools else 0.0
    # sans coûts fixes (cas isolé) : rentable dès le 1er pool si la marge par pool est positive
    if couts_fixes <= 0:
        break_even = 0.0 if marge_pool > 1e-9 else float("inf")
    else:
        break_even = (couts_fixes / marge_pool) if marge_pool > 1e-9 else float("inf")

    return PnLOperateur(
        revenus=revenus, marge_operateur=marge_op, surplus_enchere=surplus,
        cout_acquisition=cout_acq, cout_ops=cout_ops, couts_fixes=couts_fixes,
        resultat_net=net, cout_capital_fge=cout_cap, resultat_net_apres_capital=net_apres_cap,
        n_pools=n_pools, break_even_pools=break_even, marge_par_pool=marge_pool,
    )


def taux_fuite_break_even(cfg, graines=8, lo=0.0, hi=0.60, tol=0.01) -> float:
    """Taux de fuite (p_fuite_base) au-delà duquel le FGE ne couvre plus les trous, i.e.
    la tranche SFD commence à être sollicitée de façon significative. Bissection sur
    p_fuite_base : on cherche le seuil où la perte SFD moyenne devient > 0 de façon notable.

    C'est LE chiffre du pitch : 'le FGE absorbe les fuites tant que le taux reste sous X%'.
    """
    import copy

    def perte_sfd_moy(p):
        pertes = []
        for g in range(graines):
            c = copy.deepcopy(cfg)
            c.fuite.p_fuite_base = p
            r = simuler_run(c, graine=g)
            pertes.append(r.perte_sfd)
        return float(np.mean(pertes))

    # seuil : perte SFD dépasse 1% des avances (sollicitation notable de la tranche)
    seuil_rel = 0.01
    def depasse(p):
        c = copy.deepcopy(cfg); c.fuite.p_fuite_base = p
        r = simuler_run(c, graine=0)
        avance = max(1.0, r.avance_cumulee)
        return perte_sfd_moy(p) > seuil_rel * avance

    if not depasse(hi):
        return hi  # même au taux max, le FGE tient
    if depasse(lo):
        return lo
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if depasse(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def sensibilite_pnl(cfg, axe: str, valeurs, graines=6):
    """Sensibilité du P&L Opérateur à un axe (ex. 'rho_mensuel', 'part_urgent', 'n_pools')."""
    import copy
    out = []
    for v in valeurs:
        nets = []
        for g in range(graines):
            c = copy.deepcopy(cfg)
            if axe == "rho_mensuel": c.enchere.rho_mensuel = v
            elif axe == "part_urgent":
                c.preferences.part_urgent = v
                c.preferences.part_modere = max(0.0, 1 - v - c.preferences.part_epargnant)
            elif axe == "n_pools": c.structure.n_pools = int(v)
            r = simuler_run(c, graine=g)
            nets.append(calculer_pnl_operateur(r, c).resultat_net)
        out.append({"valeur": v, "pnl_moyen": float(np.mean(nets))})
    return out
