"""fuite.py — Modèles de défaillance : fuite délibérée (post-encaissement) + friction.

DEUX mécanismes distincts (validés) :

1. FUITE DÉLIBÉRÉE (le cas critique) — uniquement APRÈS avoir encaissé son tour.
   Le membre a reçu son pot puis cesse de cotiser → trou pour le compte commun.
   Probabilité conditionnelle :
     p_fuite(t', z) = p_fuite_base
                      × mult_tour_pris(t_encaissement)   (plus on a pris tôt, plus tentant)
                      × facteur_macro(z)                  (corrélation systémique via Z)
                      + choc_comportemental
   Modélisée par mois écoulé depuis l'encaissement.

2. ÉCHEC DE FRICTION (non critique) — prélèvement automatique qui échoue temporairement.
   Récupérable le mois suivant (taux de récupération). N'est PAS une fuite.

REMPLACEMENT (cas non critique) : un membre qui n'a PAS encore encaissé et s'arrête est
remplacé (on lui rend sa mise moins pénalité). Le remplaçant cotise pour SON tour à venir,
il ne comble PAS le trou d'un fuyard — c'est le rôle du FGE.
"""

from __future__ import annotations
import numpy as np

from risque import pd_conditionnelle  # réutilise la mécanique Vasicek (charge sectorielle sur Z)


def proba_fuite(p_total: float, t_encaissement: int, m: int, z: float, mois_restants: int,
                charge_z: float, mult_tour_precoce: float, choc: float) -> float:
    """Probabilité MENSUELLE de fuite, dérivée d'une probabilité TOTALE p_total de fuir
    sur toute la durée post-encaissement.

    p_total est la proba qu'un bénéficiaire fuie un jour (interprétable : « 6% des
    bénéficiaires fuient »). On la convertit en proba mensuelle conditionnelle :
        p_mens = 1 - (1 - p_total_ajustée)^(1/mois_restants)

    Ajustements appliqués à p_total AVANT conversion :
    - mult_tour_precoce : plein au tour 1, décroît vers 1 au dernier (prendre tôt = plus tentant)
    - facteur macro (charge_z, Z<0) : mauvaise conjoncture augmente la fuite
    - choc comportemental additif
    """
    if m > 1:
        frac = (t_encaissement - 1) / (m - 1)
    else:
        frac = 0.0
    mult = mult_tour_precoce + (1.0 - mult_tour_precoce) * frac
    p_tot = p_total * mult

    if charge_z > 0:
        p_tot = float(np.clip(p_tot, 1e-6, 1 - 1e-6))
        logit = np.log(p_tot / (1 - p_tot)) - charge_z * z
        p_tot = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    p_tot = float(np.clip(p_tot + choc, 0.0, 1.0))

    d = max(1, mois_restants)
    p_mens = 1.0 - (1.0 - p_tot) ** (1.0 / d)
    return float(np.clip(p_mens, 0.0, 1.0))


def echec_friction(rng, taux_echec: float, efficacite_prelevement_auto: float,
                   mitigation_active: bool) -> bool:
    """Tire si la cotisation d'un membre échoue ce mois (friction temporaire).
    Le prélèvement automatique réduit le taux d'échec selon son efficacité."""
    taux = taux_echec
    if mitigation_active and efficacite_prelevement_auto > 0:
        taux *= (1.0 - efficacite_prelevement_auto)
    return rng.random() < taux
