"""risque.py — Risque de crédit corrélé : facteur systémique unique + secteurs.

Modèle de corrélation (style Vasicek mono-facteur) :
  - Un unique facteur systémique Z ~ N(0,1) représente la conjoncture commune.
  - Chaque membre appartient à un SECTEUR ; le secteur fixe sa CHARGE rho sur Z.
    Charge élevée => forte corrélation (les membres du secteur défaillent ensemble
    quand Z est mauvais). Charge faible => secteur défensif.
  - La PD conditionnelle d'un membre, étant donné Z, est :
        PD_i(Z) = Phi( (Phi^{-1}(PD_i) - sqrt(rho_i) * Z) / sqrt(1 - rho_i) )
    avec la convention Z>0 = bonne conjoncture (PD baisse), Z<0 = mauvaise (PD monte).

La contrainte de composition K agit sur les secteurs : on limite à K le nombre de
membres d'un même secteur dans un pool, ce qui borne la corrélation intra-pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# --- Normale standard sans scipy ------------------------------------------------

def _phi(x: np.ndarray) -> np.ndarray:
    """CDF normale standard via erf."""
    from math import sqrt
    # np.vectorize-free : utilise la relation avec erf
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _erf(x: np.ndarray) -> np.ndarray:
    """Approximation d'Abramowitz & Stegun 7.1.26 (erreur < 1.5e-7)."""
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return sign * y


def _phi_inv(p: np.ndarray) -> np.ndarray:
    """Quantile normal standard (Acklam), sans scipy."""
    p = np.asarray(np.clip(p, 1e-12, 1 - 1e-12), dtype=float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.zeros_like(p)
    # queue basse
    lo = p < plow
    if np.any(lo):
        q = np.sqrt(-2 * np.log(p[lo]))
        out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                  ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    # centre
    mid = (~lo) & (p <= phigh)
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    # queue haute
    hi = p > phigh
    if np.any(hi):
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        out[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    return out


# --- Profil de risque d'un membre ----------------------------------------------

@dataclass
class ProfilRisque:
    secteur: str
    rho: float              # charge sur le facteur systémique Z
    pd_base_mensuelle: float  # PD mensuelle inconditionnelle de base
    pd_seuil: float         # Phi^{-1}(pd_base_mensuelle), pré-calculé


def tirer_profils_secteurs(rng, n: int, params_risque, periodes_par_an: int) -> List[ProfilRisque]:
    """Tire pour n membres un secteur (selon les parts) et une PD de base individuelle."""
    secteurs = params_risque.secteurs
    noms = [s[0] for s in secteurs]
    parts = np.array([s[1] for s in secteurs], dtype=float)
    parts = parts / parts.sum()
    charges = {s[0]: s[2] for s in secteurs}

    idx = rng.choice(len(noms), size=n, p=parts)
    # PD annuelle individuelle ~ tronquée autour de la base
    pd_ann = np.clip(rng.normal(params_risque.pd_base_annuel, params_risque.pd_base_sigma, n),
                     0.005, 0.60)
    pd_mens = 1.0 - (1.0 - pd_ann) ** (1.0 / periodes_par_an)
    seuils = _phi_inv(pd_mens)

    profils = []
    for i in range(n):
        sec = noms[idx[i]]
        profils.append(ProfilRisque(
            secteur=sec, rho=float(charges[sec]),
            pd_base_mensuelle=float(pd_mens[i]), pd_seuil=float(seuils[i]),
        ))
    return profils


def pd_conditionnelle(profil: ProfilRisque, z: float) -> float:
    """PD mensuelle conditionnelle au facteur systémique Z (Vasicek mono-facteur).
    Convention : Z>0 = bonne conjoncture (PD baisse), Z<0 = mauvaise (PD monte).
    """
    rho = profil.rho
    num = profil.pd_seuil - np.sqrt(rho) * z
    den = np.sqrt(max(1e-9, 1.0 - rho))
    return float(_phi(np.array([num / den]))[0])


def pd_conditionnelle_vec(seuils: np.ndarray, rhos: np.ndarray, z: float) -> np.ndarray:
    """Version VECTORISÉE : PD conditionnelles de plusieurs membres en un seul appel.
    seuils, rhos : tableaux numpy (un par membre). Renvoie un tableau de PD."""
    num = seuils - np.sqrt(rhos) * z
    den = np.sqrt(np.maximum(1e-9, 1.0 - rhos))
    return _phi(num / den)


def k_faisable_minimal(profils: List[ProfilRisque], m: int) -> int:
    """K minimal théoriquement faisable étant donné les parts sectorielles.
    Un secteur de part s nécessite au moins ceil(s_membres / n_pools) places par pool.
    """
    from collections import Counter
    n = len(profils)
    n_pools = max(1, n // m)
    cnt = Counter(p.secteur for p in profils)
    return max(int(np.ceil(v / n_pools)) for v in cnt.values()) if cnt else 1


def composer_pools_contrainte_k(rng, profils: List[ProfilRisque], m: int, k_max: int):
    """Répartit les profils en pools de taille m en visant : au plus k_max membres d'un
    même secteur par pool (contrainte de non-corrélation).

    Hypothèse réaliste : la contrainte est parfois mathématiquement infaisable (un secteur
    sur-représenté dépasse k_max × n_pools / m places). Dans ce cas, on place AU MIEUX et
    on MESURE le résiduel plutôt que de déformer la population. C'est l'argument honnête du
    pitch : « voici le taux de diversification réellement atteignable ».

    Renvoie (pools, infos) où infos contient :
      - violations : nb de placements ayant dû dépasser k_max
      - taux_respect_k : part des pools respectant strictement k_max
      - k_min_faisable : K minimal théoriquement atteignable pour cette population
      - secteurs_satures : secteurs ayant causé des violations
    """
    from collections import Counter
    n = len(profils)
    n_pools = n // m
    pools = [[] for _ in range(n_pools)]
    compte_secteur = [dict() for _ in range(n_pools)]
    violations = 0
    secteurs_satures: dict = {}

    # ordre de placement : secteurs les plus CONTRAINTS d'abord —
    # à la fois les plus sur-représentés (rares à caser) et à charge élevée.
    parts = Counter(p.secteur for p in profils)
    def priorite(i):
        sec = profils[i].secteur
        return (-parts[sec], -profils[i].rho)   # sur-représentés + corrélés d'abord
    ordre = sorted(range(n), key=priorite)

    for i in ordre:
        sec = profils[i].secteur
        ouverts = [p for p in range(n_pools) if len(pools[p]) < m]
        if not ouverts:
            continue  # plus de place (membres résiduels — n non divisible par m)
        # pools respectant k_max pour ce secteur
        conformes = [p for p in ouverts if compte_secteur[p].get(sec, 0) < k_max]
        if conformes:
            # équilibrer : pool conforme le moins rempli
            p = min(conformes, key=lambda p: (len(pools[p]), compte_secteur[p].get(sec, 0)))
        else:
            # infaisable : minimiser le dépassement -> pool avec le moins de ce secteur
            p = min(ouverts, key=lambda p: compte_secteur[p].get(sec, 0))
            violations += 1
            secteurs_satures[sec] = secteurs_satures.get(sec, 0) + 1
        pools[p].append(i)
        compte_secteur[p][sec] = compte_secteur[p].get(sec, 0) + 1

    # taux de respect : pools où aucun secteur ne dépasse k_max
    respectants = sum(
        1 for pool in pools
        if pool and max(Counter(profils[i].secteur for i in pool).values()) <= k_max
    )
    pools_non_vides = sum(1 for pool in pools if pool)
    infos = {
        "violations": violations,
        "taux_respect_k": (respectants / pools_non_vides) if pools_non_vides else 1.0,
        "k_min_faisable": k_faisable_minimal(profils, m),
        "secteurs_satures": secteurs_satures,
    }
    return pools, infos


def tirer_z(rng, params_stress, mois: int, etat) -> float:
    """Tire le facteur systémique Z pour un mois donné.
    En conjoncture nominale, Z ~ N(0,1). Sous stress macro, Z est décalé vers z_choc
    (négatif = mauvaise conjoncture), avec persistance optionnelle.
    """
    if not params_stress.macro_actif or params_stress.z_choc == 0.0:
        return float(rng.normal(0.0, 1.0))
    # choc persistant : on garde le même Z décalé pendant z_persistance mois
    if params_stress.z_persistance > 0:
        if etat.get("z_actif_restant", 0) > 0:
            etat["z_actif_restant"] -= 1
            return etat["z_courant"]
        # nouveau régime de choc avec proba liée à l'intensité (ici : toujours actif au début)
    z = float(rng.normal(params_stress.z_choc, 0.6))
    if params_stress.z_persistance > 0:
        etat["z_courant"] = z
        etat["z_actif_restant"] = params_stress.z_persistance - 1
    return z
