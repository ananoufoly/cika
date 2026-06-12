"""orchestrateur.py — N pools en parallèle + cascade de garanties complète.

Les pools tournent de façon SYNCHRONISÉE tour par tour (même mois pour tous), ce qui
permet la mutualisation inter-pools : quand un pool a un besoin résiduel à un tour, la
méta-réserve puise dans les réserves disponibles des AUTRES pools au même instant.

Cascade de garanties (ordre strict) :
  Niveau 1 — Réserve du pool        (gérée dans le moteur : enchères + confiscations du pool)
  Niveau 2 — Méta-réserve inter-pools (mutualisation des réserves de tous les pools, au prorata)
  Niveau 3 — Ligne contingente SFD   (plafonnée à X défauts/pool, pré-provisionnée)
  Niveau 4 — Pro-rata + report        (versement partiel + report du solde au cycle suivant)

Le facteur systémique Z est COMMUN à tous les pools chaque mois (un seul choc macro frappe
toute l'économie) — c'est ce qui crée la corrélation inter-pools que la méta-réserve doit
absorber, et qui teste la contrainte de composition K.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from risque import (ProfilRisque, pd_conditionnelle, tirer_profils_secteurs,
                    composer_pools_contrainte_k, tirer_z)
from moteur import MoteurPool, EtatMembre


# ---------------------------------------------------------------------------
# Suivi de la cascade
# ---------------------------------------------------------------------------

@dataclass
class TrackerCascade:
    # montants couverts par niveau, cumulés
    niv1_reserve_pool: float = 0.0
    niv2_meta_reserve: float = 0.0
    niv3_ligne_sfd: float = 0.0
    niv4_prorata_report: float = 0.0
    # fréquence d'activation (nb de (pool, tour) ayant sollicité chaque niveau)
    freq_niv1: int = 0
    freq_niv2: int = 0
    freq_niv3: int = 0
    freq_niv4: int = 0
    # perte SFD = tirages sur la ligne contingente non remboursés
    perte_sfd: float = 0.0
    # report : solde non couvert reporté
    report_total: float = 0.0


# ---------------------------------------------------------------------------
# Résultat global
# ---------------------------------------------------------------------------

@dataclass
class ResultatPortefeuille:
    cfg: object
    resultats_pools: List = field(default_factory=list)
    tracker: TrackerCascade = field(default_factory=TrackerCascade)
    infos_composition: dict = field(default_factory=dict)
    journal_tours: pd.DataFrame = None
    # agrégats inter-pools
    fonds_bonus: float = 0.0
    bonus_distribue: float = 0.0
    n_membres_eligibles_bonus: int = 0


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------

class Orchestrateur:
    def __init__(self, cfg):
        self.cfg = cfg

    def simuler(self, graine: int = 12345, mode_leger: bool = False) -> ResultatPortefeuille:
        """mode_leger : pour le Monte Carlo. N'accumule pas le journal pandas détaillé ;
        calcule à la volée les agrégats nécessaires aux P&L (intérêts crédit, dépôts
        moyens, pools actifs/mois). ~2-3× plus rapide."""
        cfg = self.cfg
        rng = np.random.default_rng(graine)

        n_pools = cfg.structure.n_pools
        m = cfg.structure.m_membres
        total_tours = cfg.structure.total_tours

        # --- 1. tirage des profils + composition contrainte K ---
        n_membres = n_pools * m
        profils = tirer_profils_secteurs(rng, n_membres, cfg.risque, cfg.sfd.periodes_par_an)
        pools_idx, infos_compo = composer_pools_contrainte_k(rng, profils, m, cfg.structure.k_max_meme_secteur)
        # ne garder que les pools complets
        pools_idx = [p for p in pools_idx if len(p) == m]
        n_pools_eff = len(pools_idx)

        # --- 2. facteur systémique Z commun à tous les pools, par mois ---
        etat_z: dict = {}
        z_par_mois = [tirer_z(rng, cfg.stress, t, etat_z) for t in range(total_tours)]

        # --- 3. états : on instancie un MoteurPool par pool, mais on pilote la boucle
        #         tour par tour ici pour permettre la mutualisation inter-pools ---
        moteurs = [MoteurPool(cfg, pid) for pid in range(n_pools_eff)]
        profils_par_pool = [[profils[i] for i in pools_idx[pid]] for pid in range(n_pools_eff)]

        # États membres par pool (on réplique l'init du moteur ici pour garder la main
        # sur la synchronisation tour par tour et la cascade inter-pools).
        etats_pools: List[List[EtatMembre]] = []
        for pid in range(n_pools_eff):
            membres = []
            for i, pr in enumerate(profils_par_pool[pid]):
                tp, urg, pdm = moteurs[pid]._assigner_preference(rng)
                membres.append(EtatMembre(idx=i, profil=pr, type_pref=tp, urgence=urg, pd_mult=pdm))
            moteurs[pid]._appliquer_stress_preferences(membres, rng)
            etats_pools.append(membres)

        # Pré-calcul VECTORISÉ des paramètres de défaut par pool (seuils, rhos, pd_mult)
        # pour tirer tous les défauts d'un pool en un seul appel numpy (perf).
        from risque import pd_conditionnelle_vec
        seuils_pool = [np.array([mb.profil.pd_seuil for mb in membres]) for membres in etats_pools]
        rhos_pool = [np.array([mb.profil.rho for mb in membres]) for membres in etats_pools]
        choc_pd = cfg.stress.choc_pd if cfg.stress.comportemental_actif else 0.0

        reserves = [0.0] * n_pools_eff       # niveau 1 par pool
        provision_differe = [0.0] * n_pools_eff  # surplus de collecte mis de côté pour le différé
        tracker = TrackerCascade()
        r_epargne = cfg.sfd.r_epargne

        # ligne contingente SFD : plafond par pool, pré-provisionné
        plafond_ligne = cfg.sfd.plafond_ligne_par_pool(cfg.structure.c, cfg.mecanisme.part_immediate, m)
        ligne_utilisee = [0.0] * n_pools_eff

        # agrégats économiques
        commission_op_par_pool = [0.0] * n_pools_eff
        consignations_par_pool = [0.0] * n_pools_eff
        confiscations_par_pool = [0.0] * n_pools_eff
        total_bids_par_pool = [0.0] * n_pools_eff
        n_defauts_par_pool = [0] * n_pools_eff

        journal = []
        # agrégats à la volée (pour mode_leger : évitent de reparcourir un DataFrame ensuite)
        agg_interets_credit = 0.0      # somme decaisse * r_credit * duree
        agg_reserve_par_tour = [0.0] * total_tours   # somme des réserves par tour
        agg_collecte_par_tour = [0.0] * total_tours  # somme des collectes par tour
        agg_pools_actifs_mois = 0.0
        r_credit = cfg.sfd.r_credit
        part_imm = cfg.mecanisme.part_immediate
        alpha = cfg.mecanisme.alpha
        pool_mensuel = cfg.structure.pool_mensuel
        immediat = part_imm * pool_mensuel
        differe_nominal = (1.0 - part_imm) * pool_mensuel
        beta = cfg.bonus.beta

        fonds_bonus = 0.0

        # ============ boucle synchronisée tour par tour ============
        for t in range(1, total_tours + 1):
            z = z_par_mois[t - 1]
            slot = (t - 1) % m
            cycle = (t - 1) // m + 1

            if slot == 0:
                for membres in etats_pools:
                    for mb in membres:
                        mb.a_recu_pot = False

            # rémunération des réserves (SFD)
            for pid in range(n_pools_eff):
                reserves[pid] *= (1.0 + r_epargne)

            # --- phase A : défauts + contributions + enchères + besoins, par pool ---
            besoins = [0.0] * n_pools_eff
            infos_tour = [None] * n_pools_eff

            for pid in range(n_pools_eff):
                membres = etats_pools[pid]
                mot = moteurs[pid]

                # défauts (PD conditionnelle au Z commun) — VECTORISÉ par pool
                pd_base = pd_conditionnelle_vec(seuils_pool[pid], rhos_pool[pid], z)
                pd_mults = np.array([mb.pd_mult for mb in membres])
                pd_eff = np.clip(pd_base * pd_mults + choc_pd, 0.0, 1.0)
                tirages = rng.random(len(membres))
                for j, mb in enumerate(membres):
                    if not mb.en_defaut and tirages[j] < pd_eff[j]:
                        mb.en_defaut = True
                        mb.tour_defaut = t
                        n_defauts_par_pool[pid] += 1
                        conf_cons = mb.consignation_versee
                        mb.consignation_confisquee += conf_cons
                        mb.consignation_versee = 0.0
                        confiscations_par_pool[pid] += conf_cons
                        # répartition des confiscations : (1-beta) -> réserve, beta -> fonds bonus
                        reserves[pid] += (1.0 - beta) * conf_cons
                        fonds_bonus += beta * conf_cons

                actifs = [mb for mb in membres if not mb.en_defaut]
                collecte = 0.0
                for mb in actifs:
                    mb.contribue += cfg.structure.c
                    collecte += cfg.structure.c

                # bénéficiaire (ordre résiduel)
                candidats = [mb for mb in membres if (not mb.en_defaut and not mb.a_recu_pot)]
                beneficiaire = None
                if candidats:
                    candidats.sort(key=lambda mb: mb.idx)
                    beneficiaire = candidats[0]
                    beneficiaire.a_recu_pot = True

                # enchère sur le différé
                eligibles = [mb for mb in membres if (not mb.en_defaut and not mb.a_recu_pot)]
                gagnant, bid = mot._enchere(rng, eligibles, slot)
                commission = alpha * bid
                consignation = (1.0 - alpha) * bid
                total_bids_par_pool[pid] += bid
                commission_op_par_pool[pid] += commission
                consignations_par_pool[pid] += consignation
                if gagnant is not None and bid > 0:
                    gagnant.commission_payee += commission
                    gagnant.consignation_versee += consignation
                    reserves[pid] += consignation

                # Besoin de financement PAR TOUR = part IMMÉDIATE seulement (seul décaissement
                # réel du tour). Le surplus de collecte (collecte - immediat) est mis de côté
                # pour provisionner le différé, restitué en fin de cycle (voir phase C).
                # Le différé n'est PAS réclamé chaque tour (sinon on le multiplie par le nb de tours).
                besoin = 0.0
                decaisse = 0.0
                niv = 0
                if beneficiaire is not None:
                    decaisse = immediat
                    beneficiaire.recu_immediat += immediat
                    surplus = collecte - immediat
                    if surplus >= 0:
                        # la collecte couvre l'immédiat ; le surplus provisionne le différé du pool
                        provision_differe[pid] += surplus
                    else:
                        # collecte insuffisante même pour l'immédiat -> réserve (niv1) puis cascade
                        besoin_imm = -surplus
                        pris = min(reserves[pid], besoin_imm)
                        reserves[pid] -= pris
                        if pris > 0:
                            tracker.niv1_reserve_pool += pris
                            tracker.freq_niv1 += 1
                            niv = 1
                        besoin = besoin_imm - pris

                besoins[pid] = besoin
                infos_tour[pid] = dict(
                    collecte=collecte, beneficiaire=(beneficiaire.idx if beneficiaire else None),
                    decaisse=decaisse, bid=bid, commission=commission, consignation=consignation,
                    n_actifs=len(actifs), niv=niv,
                )

            # --- phase B : cascade inter-pools pour les besoins résiduels ---
            # Niveau 2 : méta-réserve = somme des réserves disponibles des AUTRES pools.
            besoin_total = sum(besoins)
            if besoin_total > 1e-9 and cfg.cascade.meta_reserve_active:
                meta_dispo = sum(reserves)  # toutes réserves mutualisables
                for pid in range(n_pools_eff):
                    if besoins[pid] <= 1e-9:
                        continue
                    # puiser dans les autres réserves au prorata de leur disponibilité
                    autres = [(q, reserves[q]) for q in range(n_pools_eff) if q != pid and reserves[q] > 0]
                    dispo_autres = sum(r for _, r in autres)
                    a_couvrir = min(besoins[pid], dispo_autres)
                    if a_couvrir > 0:
                        for q, rq in autres:
                            part = a_couvrir * (rq / dispo_autres) if dispo_autres > 0 else 0.0
                            reserves[q] -= part
                        besoins[pid] -= a_couvrir
                        tracker.niv2_meta_reserve += a_couvrir
                        tracker.freq_niv2 += 1
                        infos_tour[pid]["niv"] = 2

            # Niveau 3 : ligne contingente SFD (plafonnée par pool)
            if cfg.cascade.ligne_sfd_active:
                for pid in range(n_pools_eff):
                    if besoins[pid] <= 1e-9:
                        continue
                    dispo_ligne = max(0.0, plafond_ligne - ligne_utilisee[pid])
                    tire = min(besoins[pid], dispo_ligne)
                    if tire > 0:
                        ligne_utilisee[pid] += tire
                        besoins[pid] -= tire
                        tracker.niv3_ligne_sfd += tire
                        tracker.freq_niv3 += 1
                        tracker.perte_sfd += tire   # tirage sur la ligne = coût du risque SFD
                        infos_tour[pid]["niv"] = 3

            # Niveau 4 : pro-rata + report (solde non couvert)
            for pid in range(n_pools_eff):
                if besoins[pid] > 1e-9:
                    tracker.niv4_prorata_report += besoins[pid]
                    tracker.report_total += besoins[pid]
                    tracker.freq_niv4 += 1
                    infos_tour[pid]["niv"] = 4
                    besoins[pid] = 0.0  # reporté / pro-rata appliqué

            # --- phase C : agrégats (toujours) + journal (mode complet seulement) ---
            duree_credit = max(1, m - slot)  # mois jusqu'à fin de cycle
            for pid in range(n_pools_eff):
                it = infos_tour[pid]
                # agrégats économiques à la volée
                if it["decaisse"] > 0:
                    agg_interets_credit += it["decaisse"] * r_credit * duree_credit
                agg_reserve_par_tour[t - 1] += reserves[pid]
                agg_collecte_par_tour[t - 1] += it["collecte"]
                if it["n_actifs"] > 0:
                    agg_pools_actifs_mois += 1
                if not mode_leger:
                    journal.append(dict(
                        tour=t, cycle=cycle, pool=pid, z=z,
                        n_actifs=it["n_actifs"], collecte=it["collecte"],
                        beneficiaire=it["beneficiaire"], decaisse=it["decaisse"],
                        bid=it["bid"], commission=it["commission"],
                        niveau_cascade=it["niv"], reserve_fin=reserves[pid],
                    ))

            if slot == m - 1:  # fin de cycle : restitution des différés (garantie pot complet)
                besoins_dif = [0.0] * n_pools_eff
                for pid in range(n_pools_eff):
                    # différé dû = somme des différés des bénéficiaires DISCIPLINÉS du cycle
                    benefs_disc = [mb for mb in etats_pools[pid]
                                   if (not mb.en_defaut and mb.a_recu_pot)]
                    du = len(benefs_disc) * differe_nominal
                    # financé par la provision accumulée + la réserve du pool
                    dispo = provision_differe[pid] + reserves[pid]
                    if du <= dispo:
                        # prendre d'abord la provision puis la réserve
                        reste = du - provision_differe[pid]
                        provision_differe[pid] = max(0.0, provision_differe[pid] - du)
                        if reste > 0:
                            reserves[pid] -= reste
                        for mb in benefs_disc:
                            mb.recu_differe += differe_nominal
                    else:
                        # provision + réserve épuisées -> cascade (niv2-4) pour le solde
                        provision_differe[pid] = 0.0
                        reserves[pid] = 0.0
                        besoins_dif[pid] = du - dispo
                        # crédit partiel pro-rata aux bénéficiaires (le solde passera en cascade)
                        couvert = dispo
                        for mb in benefs_disc:
                            mb.recu_differe += differe_nominal * (couvert / du) if du > 0 else 0.0

                # cascade pour les besoins de différé non couverts (niv2 méta, niv3 ligne, niv4 report)
                if sum(besoins_dif) > 1e-9:
                    # niv2 méta-réserve
                    if cfg.cascade.meta_reserve_active:
                        for pid in range(n_pools_eff):
                            if besoins_dif[pid] <= 1e-9:
                                continue
                            autres = [(q, reserves[q]) for q in range(n_pools_eff) if q != pid and reserves[q] > 0]
                            dispo_autres = sum(r for _, r in autres)
                            a = min(besoins_dif[pid], dispo_autres)
                            if a > 0:
                                for q, rq in autres:
                                    reserves[q] -= a * (rq / dispo_autres)
                                besoins_dif[pid] -= a
                                tracker.niv2_meta_reserve += a; tracker.freq_niv2 += 1
                    # niv3 ligne SFD
                    if cfg.cascade.ligne_sfd_active:
                        for pid in range(n_pools_eff):
                            if besoins_dif[pid] <= 1e-9:
                                continue
                            dispo_ligne = max(0.0, plafond_ligne - ligne_utilisee[pid])
                            tire = min(besoins_dif[pid], dispo_ligne)
                            if tire > 0:
                                ligne_utilisee[pid] += tire; besoins_dif[pid] -= tire
                                tracker.niv3_ligne_sfd += tire; tracker.freq_niv3 += 1
                                tracker.perte_sfd += tire
                    # niv4 report
                    for pid in range(n_pools_eff):
                        if besoins_dif[pid] > 1e-9:
                            tracker.niv4_prorata_report += besoins_dif[pid]
                            tracker.report_total += besoins_dif[pid]
                            tracker.freq_niv4 += 1

                for pid in range(n_pools_eff):
                    for mb in etats_pools[pid]:
                        if not mb.en_defaut:
                            mb.cycles_sans_defaut += 1

        # ============ clôture : bonus inter-cycle ============
        # intérêts générés par les consignations (déjà capitalisés dans les réserves)
        # fonds_bonus = beta*confiscations + (les consignations restituées + intérêts
        # sont rendues aux membres ; le bonus vient des confiscations + intérêts nets).
        # Ici : bonus = fonds_bonus (part beta des confiscations) réparti au prorata des
        # consignations des membres éligibles (cycles_sans_defaut >= cycles_requis).
        eligibles = []
        for pid in range(n_pools_eff):
            for mb in etats_pools[pid]:
                if (not mb.en_defaut) and mb.cycles_sans_defaut >= cfg.bonus.cycles_requis:
                    eligibles.append(mb)
        total_cons_elig = sum(mb.consignation_versee for mb in eligibles)
        bonus_distribue = 0.0
        if eligibles and fonds_bonus > 0:
            for mb in eligibles:
                if total_cons_elig > 0:
                    part = fonds_bonus * (mb.consignation_versee / total_cons_elig)
                else:
                    part = fonds_bonus / len(eligibles)
                mb.bonus_recu += part
                bonus_distribue += part

        # construire le résultat (journal vide en mode léger)
        df = pd.DataFrame(journal) if not mode_leger else None
        res = ResultatPortefeuille(
            cfg=cfg, tracker=tracker, infos_composition=infos_compo, journal_tours=df,
            fonds_bonus=fonds_bonus, bonus_distribue=bonus_distribue,
            n_membres_eligibles_bonus=len(eligibles),
        )
        # rattacher les états pools (pour les P&L et KPIs)
        res.resultats_pools = etats_pools
        # dépôts moyens = réserve moyenne + collecte moyenne (séquestre), calculés à la volée
        reserve_moy = sum(agg_reserve_par_tour) / total_tours if total_tours else 0.0
        collecte_moy = sum(agg_collecte_par_tour) / total_tours if total_tours else 0.0
        res._meta = dict(
            n_pools_eff=n_pools_eff,
            commission_op_total=sum(commission_op_par_pool),
            consignations_total=sum(consignations_par_pool),
            confiscations_total=sum(confiscations_par_pool),
            total_bids=sum(total_bids_par_pool),
            n_defauts_total=sum(n_defauts_par_pool),
            ligne_utilisee_total=sum(ligne_utilisee),
            plafond_ligne_par_pool=plafond_ligne,
            reserves_finales=list(reserves),
            z_par_mois=z_par_mois,
            # agrégats pré-calculés (utilisés par les P&L, évitent iterrows)
            interets_credit_total=agg_interets_credit,
            depots_moyens=reserve_moy + collecte_moy,
            pools_actifs_mois=agg_pools_actifs_mois,
        )
        return res
