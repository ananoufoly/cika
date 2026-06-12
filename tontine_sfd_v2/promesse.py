"""promesse.py — SORTIE B : tenue de la promesse + exposition / perte SFD.

La promesse côté membre : « votre tour arrive, vous recevez vos 900K, même si un autre
membre disparaît ». Dans ce modèle, la SFD avance toujours → le membre est servi tant que
la cascade (FGE → tranche SFD) couvre les trous. La promesse CASSE seulement si le résiduel
non couvert > 0 (FGE + tranche SFD épuisés).

KPIs (par scénario, agrégés en Monte Carlo dans scenarios.py) :
  - TAUX DE CONTINUITÉ : % de runs où tous les tours sont servis (résiduel = 0)
  - P[promesse cassée]  : fréquence de résiduel > 0 (cible < 1/1000 en combiné avec mitigations)
  - quand ça casse       : montant résiduel, ampleur
  - EXPOSITION SFD        : profil mois par mois (avances en cours) + max
  - FRÉQUENCE PERTE SFD   : part des runs où la tranche SFD est sollicitée (doit être faible)
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class KPIPromesse:
    continuite: bool                # tous les tours servis (résiduel = 0)
    residuel: float                 # montant non couvert (0 si promesse tenue)
    perte_sfd: bool                 # la tranche SFD a-t-elle été sollicitée
    perte_sfd_montant: float
    exposition_max: float
    n_fuites: int


def kpi_promesse_run(res) -> KPIPromesse:
    return KPIPromesse(
        continuite=(res.residuel_non_couvert <= 1e-6),
        residuel=res.residuel_non_couvert,
        perte_sfd=(res.perte_sfd > 1e-6),
        perte_sfd_montant=res.perte_sfd,
        exposition_max=res.exposition_max,
        n_fuites=res.n_fuites,
    )


def agreger_promesse(resultats_runs) -> dict:
    """Agrège les KPIs de promesse sur N runs Monte Carlo."""
    kpis = [kpi_promesse_run(r) for r in resultats_runs]
    n = len(kpis)
    continuites = np.array([k.continuite for k in kpis], dtype=float)
    residuels = np.array([k.residuel for k in kpis])
    perte_flags = np.array([k.perte_sfd for k in kpis], dtype=float)
    pertes = np.array([k.perte_sfd_montant for k in kpis])
    expo = np.array([k.exposition_max for k in kpis])
    fuites = np.array([k.n_fuites for k in kpis])

    # profil d'exposition moyen mois par mois
    expo_profil = None
    if resultats_runs and resultats_runs[0].exposition_par_mois:
        L = len(resultats_runs[0].exposition_par_mois)
        mat = np.array([r.exposition_par_mois for r in resultats_runs if len(r.exposition_par_mois) == L])
        if len(mat):
            expo_profil = {
                "moyenne": mat.mean(axis=0).tolist(),
                "p95": np.percentile(mat, 95, axis=0).tolist(),
            }

    n_casses = int((residuels > 1e-6).sum())
    return {
        "n_runs": n,
        "taux_continuite": float(continuites.mean()),
        "p_promesse_cassee": n_casses / n if n else 0.0,
        "p_promesse_cassee_texte": f"{n_casses}/{n}",
        "residuel_moyen": float(residuels.mean()),
        "residuel_p95": float(np.percentile(residuels, 95)),
        "residuel_max": float(residuels.max()),
        # perte SFD
        "freq_perte_sfd": float(perte_flags.mean()),
        "perte_sfd_moyenne": float(pertes.mean()),
        "perte_sfd_p95": float(np.percentile(pertes, 95)),
        "perte_sfd_max": float(pertes.max()),
        # exposition
        "exposition_max_moyenne": float(expo.mean()),
        "exposition_max_p95": float(np.percentile(expo, 95)),
        "exposition_profil": expo_profil,
        # fuites
        "fuites_moyenne": float(fuites.mean()),
    }


def dimensionner_capital_p99(resultats_runs) -> dict:
    """Dimensionnement du coussin nécessaire pour tenir la promesse au P99.

    Combien le FGE (ou la tranche SFD) doit-il pouvoir absorber pour que la promesse tienne
    dans 99% des cas ? = P99 de la somme des trous couverts (FGE + tranche SFD) par run.
    C'est le collatéral que la SFD exigera / le niveau-cible du FGE.
    """
    trous_couverts = np.array([r.couvert_fge + r.couvert_tranche_sfd + r.residuel_non_couvert
                               for r in resultats_runs])
    fge_alimente = np.array([r.fge_provisions + r.fge_saisies for r in resultats_runs])
    return {
        "besoin_couverture_p50": float(np.percentile(trous_couverts, 50)),
        "besoin_couverture_p95": float(np.percentile(trous_couverts, 95)),
        "besoin_couverture_p99": float(np.percentile(trous_couverts, 99)),
        "fge_alimente_moyen": float(fge_alimente.mean()),
        "fge_alimente_p05": float(np.percentile(fge_alimente, 5)),
        # le FGE suffit-il au P99 ? (le pitch : ratio de couverture)
        "ratio_couverture_p99": float(np.percentile(fge_alimente, 5) /
                                      max(1.0, np.percentile(trous_couverts, 99))),
    }
