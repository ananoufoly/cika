"""moteur.py — Moteur de portefeuille : N pools, mécanique mensuelle, cascade de couverture.

Assemble :
  - composition des pools (contrainte K, réutilisée de risque.py)
  - facteur systémique Z commun (corrélation des fuites)
  - par tour : collecte (avec friction), enchère (plancher = intérêts SFD + provision-fuite
    + prime ; surplus endogène), crédit-relais (accorder_pret), récupération des mensualités,
    fuite conditionnelle post-encaissement, remplacement des non-encaisseurs qui s'arrêtent
  - couverture en cascade des trous : FGE endogène -> tranche SFD junior -> résiduel

Sorties brutes par run (agrégées ensuite par pnl.py / promesse.py) :
  revenus Opérateur, FGE, trous, recours tranche SFD, résiduel, exposition SFD par mois,
  continuité (tous les tours servis), coût membre par tour, etc.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

from risque import (tirer_profils_secteurs, composer_pools_contrainte_k, tirer_z)
from compte import (EtatCompte, accorder_pret, recuperer_mensualites, marquer_fuite,
                    exposition_courante, provision_fuite_actuarielle, decompose_bid,
                    prime_garantie)
from fuite import proba_fuite, echec_friction


@dataclass
class Membre:
    idx: int
    seuil: float; rho: float          # paramètres Vasicek (charge sectorielle sur Z)
    type_pref: str; urgence: float
    a_historique: bool                # ≥1 cycle d'historique (accès séquencé)
    a_encaisse: bool = False
    tour_encaissement: int = None
    a_fui: bool = False
    consignation: float = 0.0         # garantie d'enchère consignée (saisie si fuite)
    cotise_total: float = 0.0
    recu: float = 0.0
    bid_paye: float = 0.0


@dataclass
class ResultatRun:
    n_pools: int
    # revenus Opérateur
    primes: float = 0.0
    surplus_enchere: float = 0.0
    # FGE
    fge_provisions: float = 0.0       # provisions-fuite accumulées (dans les bids)
    fge_saisies: float = 0.0          # garanties d'enchère saisies sur fuyards
    # trous & couverture
    trou_total_promesse: float = 0.0  # somme des trous (vue cotisations manquantes)
    trou_total_pret: float = 0.0      # somme des trous (vue principal restant)
    couvert_fge: float = 0.0
    couvert_tranche_sfd: float = 0.0
    residuel_non_couvert: float = 0.0 # ce qui dépasse FGE + tranche SFD (menace la promesse)
    # SFD
    interets_sfd: float = 0.0
    avance_cumulee: float = 0.0
    exposition_max: float = 0.0
    exposition_par_mois: List[float] = field(default_factory=list)
    perte_sfd: float = 0.0            # = recours à la tranche SFD (sa peau dans le jeu)
    # promesse
    n_fuites: int = 0
    continuite_ok: bool = True        # tous les tours servis intégralement
    cout_membre_tour1: float = 0.0    # bid du tour 1 (proxy coût d'accès précoce)
    # divers
    n_remplacements: int = 0
    n_tours_gratuits: int = 0         # tours attribués sans bideur (bid=0, non rémunérés)


def simuler_run(cfg, graine: int) -> ResultatRun:
    rng = np.random.default_rng(graine)
    S, Mc, E, Pr, Fu, Co, Ri, St = (cfg.structure, cfg.compte, cfg.enchere, cfg.preferences,
                                    cfg.fuite, cfg.couverture, cfg.risque, cfg.stress)
    Pd = cfg.produit
    m = S.m_membres
    total_tours = S.total_tours
    vie = total_tours
    pot = (m - 1) * S.c                      # pot collecté par tour (gagnant ne cotise pas pour lui)
    r_sfd = Mc.r_sfd_mensuel

    # profils + pools
    profils = tirer_profils_secteurs(rng, S.n_pools * m, Ri, Mc.periodes_par_an)
    pools_idx, _infos = composer_pools_contrainte_k(rng, profils, m, S.k_max_meme_secteur)
    pools_idx = [p for p in pools_idx if len(p) == m]
    n_pools = len(pools_idx)

    # Z commun par mois
    etat_z = {}; z_mois = [tirer_z(rng, St, t, etat_z) for t in range(total_tours)]

    res = ResultatRun(n_pools=n_pools)
    fge = 0.0                                # fonds de garantie endogène (mutualisé portefeuille)
    tranche_sfd_utilisee = 0.0

    # construire les membres par pool
    def pref(rng):
        r = rng.random()
        if r < Pr.part_urgent: return "urgent", Pr.urgence_urgent
        if r < Pr.part_urgent + Pr.part_modere: return "modere", Pr.urgence_modere
        return "epargnant", Pr.urgence_epargnant

    pools: List[List[Membre]] = []
    comptes: List[EtatCompte] = []
    for pid in range(n_pools):
        membres = []
        for i, gi in enumerate(pools_idx[pid]):
            pr = profils[gi]; tp, urg = pref(rng)
            # stress comportemental : bascule vers urgent
            if St.comportemental_actif and tp in ("modere", "epargnant") and rng.random() < St.bascule_urgents:
                tp, urg = "urgent", Pr.urgence_urgent
            a_hist = rng.random() < Co.part_avec_historique
            membres.append(Membre(idx=i, seuil=pr.pd_seuil, rho=pr.rho, type_pref=tp,
                                  urgence=urg, a_historique=a_hist))
        pools.append(membres)
        comptes.append(EtatCompte(m=m, c=S.c, r_sfd_mensuel=r_sfd))

    exposition_mois = [0.0] * total_tours

    for t in range(1, total_tours + 1):
        z = z_mois[t - 1]
        slot_cycle = (t - 1) % m
        # nouvelle "vague" de gagnants éligibles à chaque cycle
        if slot_cycle == 0:
            for membres in pools:
                for mb in membres:
                    if not mb.a_fui:
                        mb.a_encaisse = False  # ré-éligible au pot dans le nouveau cycle

        for pid in range(n_pools):
            membres = pools[pid]; cpt = comptes[pid]

            # 1. FUITE conditionnelle (uniquement membres ayant encaissé, pas encore fui)
            for mb in membres:
                if mb.a_encaisse and not mb.a_fui:
                    mois_restants = max(1, vie - mb.tour_encaissement)
                    p = proba_fuite(Fu.p_fuite_base, mb.tour_encaissement, m, z, mois_restants,
                                    Fu.charge_z_fuite, Fu.fuite_mult_tour_precoce,
                                    St.choc_fuite if St.comportemental_actif else 0.0)
                    if rng.random() < p:
                        mb.a_fui = True
                        res.n_fuites += 1
                        tp, tc = marquer_fuite(cpt, mb.idx, t)
                        res.trou_total_pret += tp
                        res.trou_total_promesse += tc
                        # saisie de la garantie d'enchère du fuyard -> FGE
                        if Co.garantie_enchere_active and mb.consignation > 0:
                            fge += mb.consignation; res.fge_saisies += mb.consignation
                            mb.consignation = 0.0
                        # COUVERTURE EN CASCADE du trou (vue prêt = ce que la SFD a réellement avancé)
                        trou = tp
                        pris_fge = min(fge, trou) if Co.fge_actif else 0.0
                        fge -= pris_fge; trou -= pris_fge; res.couvert_fge += pris_fge
                        if trou > 1e-9 and Co.tranche_sfd_active:
                            plafond = Co.plafond_tranche_sfd_frac * max(cpt.decaisse_cumule, 1.0) * n_pools
                            dispo = max(0.0, plafond - tranche_sfd_utilisee)
                            pris_sfd = min(dispo, trou)
                            tranche_sfd_utilisee += pris_sfd; trou -= pris_sfd
                            res.couvert_tranche_sfd += pris_sfd; res.perte_sfd += pris_sfd
                        if trou > 1e-9:
                            res.residuel_non_couvert += trou
                            res.continuite_ok = False   # la promesse est menacée

            # 2. COLLECTE des cotisations (membres actifs non-fuis), avec friction
            actifs = [mb for mb in membres if not mb.a_fui]
            pot_collecte = 0.0
            for mb in actifs:
                if not mb.a_encaisse:   # ceux qui n'ont pas encore encaissé cotisent au pot du tour
                    if echec_friction(rng, Fu.taux_echec_friction, Co.prelevement_auto_efficacite,
                                      Co.mitigation_active):
                        # échec temporaire : récupéré au taux de récupération (approché)
                        pot_collecte += S.c * Fu.taux_recuperation_friction
                    else:
                        pot_collecte += S.c
                    mb.cotise_total += S.c

            # 3. ATTRIBUTION DU TOUR.
            #    - En mode GARANTIE : tout encaisseur paie une PRIME DE GARANTIE obligatoire
            #      (∝ son avance), bideur ou non. Le BID est optionnel et ne sert qu'à CHOISIR
            #      sa position (passer devant) : c'est une couche au-dessus de la prime.
            #    - En mode NUE : pas de prime, pas de bid, ordre pur. Le risque reste au groupe.
            eligibles = [mb for mb in actifs if not mb.a_encaisse]
            if Pd.mode == "garantie" and Co.mitigation_active and Co.acces_sequence_active and slot_cycle < Co.t_restreint:
                eligibles_bid = [mb for mb in eligibles if mb.a_historique]
            else:
                eligibles_bid = eligibles

            duree_pret = max(1, m - (slot_cycle + 1))

            gagnant = None; bid_surplus = 0.0; bideur = False
            if Pd.mode == "garantie":
                # surplus de bid = valeur-temps au-dessus de 0 (choix de position), plafonné
                plafond_surplus = E.bid_plafond_frac_pot * pot
                meilleur = None; meilleure_wtp = -1.0
                for mb in eligibles_bid:
                    mois_gagnes = max(0, (m - 1) - slot_cycle)
                    wtp = E.rho_mensuel * mois_gagnes * pot * mb.urgence * np.exp(rng.normal(0, E.bid_bruit_sigma))
                    if wtp > meilleure_wtp:
                        meilleure_wtp = wtp; meilleur = mb
                # un membre "bide pour la position" si sa valeur-temps est significative
                if meilleur is not None and meilleure_wtp > 0.01 * pot:
                    gagnant = meilleur; bideur = True
                    bid_surplus = min(meilleure_wtp, plafond_surplus)
                    if (Co.mitigation_active and Co.garantie_enchere_active
                            and slot_cycle < Co.t_restreint and gagnant.consignation == 0):
                        gagnant.consignation = Co.g_cotisations_consignees * S.c
                elif eligibles:
                    gagnant = min(eligibles, key=lambda mb: mb.idx); bideur = False
            else:
                # mode NUE : ordre pur, aucun coût
                if eligibles:
                    gagnant = min(eligibles, key=lambda mb: mb.idx); bideur = False

            # 4. DÉCAISSEMENT. La SFD avance TOUJOURS (mode garantie). Le coût total du gagnant :
            #      intérêts SFD (∝ durée) + prime de garantie (∝ avance, OBLIGATOIRE) + surplus bid + marge Op.
            if gagnant is not None:
                tours_restants_vie = vie - t
                # avance = ce qu'il reçoit au-delà de ce qu'il a DÉJÀ épargné
                avance = max(0.0, pot - gagnant.cotise_total)
                if Pd.mode == "garantie":
                    prime_gar = prime_garantie(avance, duree_pret, Fu.p_fuite_base, m - 1,
                                               Pd.prime_facteur_prudence) if Co.prime_active else 0.0
                    interets = pot * r_sfd * duree_pret
                    marge_op = E.prime_operateur_taux * pot
                    cout_obligatoire = interets + prime_gar + marge_op
                    # le SURPLUS = ce que le bideur paie EN PLUS du coût obligatoire (pour la
                    # position), borné par sa valeur-temps et le plafond de compétitivité.
                    surplus = 0.0
                    if bideur:
                        surplus = max(0.0, min(bid_surplus - cout_obligatoire,
                                               E.bid_plafond_frac_pot * pot))
                    bid = cout_obligatoire + surplus   # coût total payé
                    net = max(0.0, pot - bid)
                    # enregistrer le prêt (le bid est retenu à la source)
                    pretln = accorder_pret(cpt, gagnant.idx, slot_cycle + 1, pot, bid,
                                           0.0, 0.0, 0.0, tours_restants_vie)
                    # ventilation explicite (override de la décompo interne)
                    res.interets_sfd += interets
                    res.primes += marge_op
                    res.surplus_enchere += surplus
                    res.fge_provisions += prime_gar
                    fge += prime_gar
                    res.avance_cumulee += net
                    gagnant.recu += net; gagnant.bid_paye += bid
                else:
                    # mode NUE : le gagnant reçoit le pot facial, aucun coût, mais c'est une avance
                    bid = 0.0
                    pretln = accorder_pret(cpt, gagnant.idx, slot_cycle + 1, pot, 0.0,
                                           0.0, 0.0, 0.0, tours_restants_vie)
                    res.avance_cumulee += pot
                    gagnant.recu += pot
                gagnant.a_encaisse = True
                gagnant.tour_encaissement = t
                if not bideur:
                    res.n_tours_gratuits += 1
                if t == 1 and res.cout_membre_tour1 == 0.0:
                    res.cout_membre_tour1 = bid

            # 5. RÉCUPÉRATION des mensualités des prêts actifs (le compte se rembourse)
            recuperer_mensualites(cpt, slot_cycle + 1)

            exposition_mois[t - 1] += exposition_courante(cpt)

    res.exposition_par_mois = exposition_mois
    res.exposition_max = max(exposition_mois) if exposition_mois else 0.0
    res.fge_provisions = res.fge_provisions  # (déjà cumulé)
    return res
