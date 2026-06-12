"""compte.py — Le compte tontine chez la SFD : modèle CRÉDIT-RELAIS.

Mécanisme exact (validé), M membres, cotisation c, pot collecté par tour = (M-1)*c
(le gagnant du tour ne cotise pas pour lui-même) :

  Tour t — un gagnant remporte l'enchère avec un bid B_t :
    - Le compte reçoit les cotisations des membres encore actifs et non-encore-gagnants.
    - La SFD DÉCAISSE au gagnant :  net_t = pot_collecte_t - B_t.
    - C'est un PRÊT : la SFD le récupère linéairement sur la durée restante d = (M - t)
      tours :  mensualite_principal = net_t / d.
    - Le bid B_t = INTÉRÊTS SFD + PRIME Opérateur, payés D'AVANCE (déjà retenus : le
      gagnant a reçu net = pot - B_t). Le bid est donc le PRIX du crédit-relais.

  Chaque mois suivant, la SFD prélève sur le compte les mensualités de principal des
  prêts en cours. Tant que le gagnant « rembourse » (ses cotisations continuent d'alimenter
  le compte / il ne fuit pas), l'avance se récupère intégralement.

  FUITE au tour t' > t : le gagnant cesse de contribuer au remboursement → le principal
  restant de SON prêt devient une AVANCE NON RÉCUPÉRÉE (le trou). La SFD avance toujours
  pour servir les tours suivants ; le trou est absorbé par le FGE puis, s'il est épuisé,
  par la SFD (perte).

Décomposition du bid :
    interets_sfd_t = net_t * r_sfd_mensuel * d        (prix du temps, va à la SFD)
    prime_t        = prime_taux * pot                  (va à l'Opérateur / FGE)
    B_t            = interets_sfd_t + prime_t + marge_min_fixe*pot   (plancher) + surplus enchère
Le surplus d'enchère (au-dessus du plancher) est un revenu Opérateur supplémentaire.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Pret:
    """Un prêt-relais accordé à un gagnant, à récupérer linéairement.

    Deux vues du « trou » en cas de fuite (gardées pour comparaison) :
      - vue PROMESSE/compte commun : cotisations futures manquantes du fuyard
        = (tours_restants_dans_la_vie_du_produit) * c
      - vue PRÊT individuel        : principal restant de l'avance qui lui a été faite
    """
    membre_idx: int
    tour_origine: int
    principal_initial: float        # net décaissé = pot - bid
    principal_restant: float
    mensualite: float               # principal_initial / duree
    duree: int                      # M - tour_origine
    cotisation_mensuelle: float     # c (sa cotisation due au compte commun)
    tours_restants_vie: int         # tours restants jusqu'à fin du produit (pour la vue promesse)
    actif: bool = True              # False si le membre a fui (remboursement stoppé)


@dataclass
class EtatCompte:
    m: int
    c: float
    r_sfd_mensuel: float

    prets: List[Pret] = field(default_factory=list)
    # cumuls
    decaisse_cumule: float = 0.0        # total net décaissé aux gagnants
    principal_recupere: float = 0.0     # principal effectivement remboursé
    interets_sfd_cumules: float = 0.0   # intérêts SFD (dans les bids), prix du crédit
    primes_cumulees: float = 0.0        # primes Opérateur (dans les bids)
    provisions_fuite_cumulees: float = 0.0  # provisions-fuite (dans les bids) -> alimentent le FGE
    surplus_enchere_cumule: float = 0.0 # marge d'enchère au-dessus du plancher (revenu Opérateur)
    # deux vues du trou cumulé (pour comparaison)
    trou_principal_cumule: float = 0.0   # vue prêt : principal restant des fuyards
    trou_cotisations_cumule: float = 0.0 # vue promesse : cotisations futures manquantes des fuyards
    avance_non_recuperee: float = 0.0    # = trou_principal_cumule (compat)
    exposition_par_tour: List[float] = field(default_factory=list)  # principal restant total / tour
    journal: List[dict] = field(default_factory=list)


def plancher_bid(net_estime: float, pot: float, t: int, m: int, r_sfd_mensuel: float,
                 prime_taux: float, marge_min_fixe: float) -> float:
    """Plancher d'enchère = intérêts SFD (sur la durée restante) + prime + marge min.
    net_estime : estimation du net décaissé (pot - bid) ; on approxime par le pot pour
    le plancher (le bid est petit devant le pot)."""
    duree = max(1, m - t)
    interets = net_estime * r_sfd_mensuel * duree
    prime = prime_taux * pot
    return interets + prime + marge_min_fixe * pot


def decompose_bid(bid: float, net: float, pot: float, t: int, m: int,
                  r_sfd_mensuel: float, prime_taux: float, marge_min_fixe: float,
                  provision_fuite: float = 0.0):
    """Décompose un bid en (intérêts SFD, prime Opérateur, provision-fuite, surplus).

    Ordre de priorité du plancher : intérêts SFD, puis provision-fuite (FGE), puis prime
    Opérateur, puis marge min. Le surplus au-dessus du plancher est un revenu Opérateur.
    """
    duree = max(1, m - t)
    interets = net * r_sfd_mensuel * duree
    prime = prime_taux * pot
    plancher = interets + provision_fuite + prime + marge_min_fixe * pot
    surplus = max(0.0, bid - plancher)
    # bornage si bid < plancher (ne devrait pas arriver)
    reste = bid
    interets_eff = min(interets, reste); reste -= interets_eff
    provision_eff = min(provision_fuite, max(0.0, reste)); reste -= provision_eff
    prime_eff = min(prime, max(0.0, reste))
    return interets_eff, prime_eff, provision_eff, surplus


def accorder_pret(etat: EtatCompte, membre_idx: int, t: int, pot_collecte: float, bid: float,
                  prime_taux: float, marge_min_fixe: float, provision_fuite: float,
                  tours_restants_vie: int) -> dict:
    """Le gagnant du tour t reçoit net = pot_collecte - bid ; on enregistre le prêt-relais."""
    m = etat.m
    duree = max(1, m - t)
    net = max(0.0, pot_collecte - bid)
    interets, prime, provision, surplus = decompose_bid(
        bid, net, pot_collecte, t, m, etat.r_sfd_mensuel, prime_taux, marge_min_fixe, provision_fuite)
    pret = Pret(membre_idx=membre_idx, tour_origine=t, principal_initial=net,
                principal_restant=net, mensualite=net / duree, duree=duree,
                cotisation_mensuelle=etat.c, tours_restants_vie=tours_restants_vie)
    etat.prets.append(pret)
    etat.decaisse_cumule += net
    etat.interets_sfd_cumules += interets
    etat.primes_cumulees += prime
    etat.provisions_fuite_cumulees += provision
    etat.surplus_enchere_cumule += surplus
    ligne = dict(tour=t, membre=membre_idx, pot_collecte=pot_collecte, bid=bid, net=net,
                 interets=interets, prime=prime, provision_fuite=provision, surplus=surplus,
                 mensualite=pret.mensualite, duree=duree)
    etat.journal.append(ligne)
    return ligne


def recuperer_mensualites(etat: EtatCompte, t: int) -> float:
    """Prélève les mensualités de principal des prêts actifs. Renvoie le total récupéré."""
    total = 0.0
    for pret in etat.prets:
        if pret.actif and pret.principal_restant > 1e-9:
            paie = min(pret.mensualite, pret.principal_restant)
            pret.principal_restant -= paie
            etat.principal_recupere += paie
            total += paie
    return total


def marquer_fuite(etat: EtatCompte, membre_idx: int, tour_fuite: int):
    """Le membre (qui a encaissé) fuit : remboursement stoppé. On enregistre les DEUX vues
    du trou : principal restant (vue prêt) et cotisations futures manquantes (vue promesse).
    Renvoie (trou_principal, trou_cotisations) pour ce membre."""
    trou_p = 0.0
    trou_c = 0.0
    for pret in etat.prets:
        if pret.membre_idx == membre_idx and pret.actif and pret.principal_restant > 1e-9:
            pret.actif = False
            trou_p += pret.principal_restant
            # vue promesse : cotisations futures manquantes = (tours restants de vie) * c
            mois_restants = max(0, pret.tours_restants_vie - (tour_fuite - pret.tour_origine))
            trou_c += mois_restants * pret.cotisation_mensuelle
    etat.trou_principal_cumule += trou_p
    etat.trou_cotisations_cumule += trou_c
    etat.avance_non_recuperee = etat.trou_principal_cumule
    return trou_p, trou_c


def exposition_courante(etat: EtatCompte) -> float:
    """Principal restant total des prêts ENCORE ACTIFS = exposition récupérable de la SFD."""
    return sum(p.principal_restant for p in etat.prets if p.actif)


def provision_fuite_actuarielle(net: float, duree: int, p_fuite: float, duree_max: int = None) -> float:
    """Provision actuariellement juste, basée sur le PRINCIPAL exposé ET sa DURÉE d'exposition.

    Le trou attendu dépend de (a) combien la SFD a avancé (net) et (b) combien de temps ce
    capital reste exposé au risque de fuite (durée de remboursement). Un prêt long expose le
    capital plus longtemps. On approxime l'exposition par l'aire du profil d'amortissement,
    proportionnelle à la durée :
        E[trou] ≈ p_fuite × (net / 2) × (duree / duree_max)
    Conséquence voulue : tour précoce (longue durée) = provision élevée ; DERNIER tour du
    cycle (durée 1) = provision quasi nulle — « celui qui prend le dernier tour ne paie pas
    d'assurance ».
    """
    if duree_max is None:
        duree_max = duree
    facteur_duree = duree / max(1, duree_max)
    principal_moyen_expose = net / 2.0
    return p_fuite * principal_moyen_expose * facteur_duree


def provision_fuite_p99(net: float) -> float:
    """Borne pire cas par bénéficiaire : fuite immédiate après encaissement = tout le
    principal exposé (net). Sert de référence (le dimensionnement P99 réel est agrégé)."""
    return net


def prime_garantie(avance: float, duree: int, p_fuite: float, duree_max: int,
                   facteur_prudence: float = 1.0) -> float:
    """Prime de garantie OBLIGATOIRE, payée par tout encaisseur, proportionnelle à son
    AVANCE (= pot reçu − épargne déjà versée) et à la durée d'exposition.

    avance         : montant reçu au-delà de ce que le membre a déjà cotisé (le « crédit »).
    duree          : durée de remboursement (tours restants dans le cycle).
    p_fuite        : probabilité totale de fuite.
    duree_max      : durée du tour le plus précoce (pour normaliser le facteur durée).
    facteur_prudence : 1.0 = actuariellement juste ; >1 = majorée (robuste au stress).

    Un encaisseur SANS avance (dernier tour, a déjà tout épargné) paie ~0 → l'épargnant pur
    ne paie pas. Le tour 1 (grosse avance, longue durée) paie le plus.
    """
    if avance <= 0:
        return 0.0
    facteur_duree = duree / max(1, duree_max)
    avance_moyenne_exposee = avance / 2.0
    return facteur_prudence * p_fuite * avance_moyenne_exposee * facteur_duree
