"""moteur.py — Moteur de simulation d'UN pool (un cycle après l'autre).

Adapté du simulateur d'enchères existant au modèle « tontine structurée avec SFD
dépositaire ». Couvre, par tour :
  - collecte des contributions sur le séquestre SFD,
  - allocation : le bénéficiaire reçoit la part immédiate (décaissement ligne de
    crédit SFD), le reste est différé,
  - enchère endogène (valeur-temps) : qui reçoit son différé par anticipation,
  - décomposition du bid : commission ferme alpha (revenu Opérateur, non remboursable)
    + consignation (1 - alpha) détenue par la SFD, restituée en fin de cycle requis,
  - défauts (PD conditionnelle au facteur systémique Z) : confiscation du différé et
    de la consignation,
  - cascade NIVEAU 1 (réserve du pool) : alimentée par les commissions retenues* et les
    confiscations ; couvre en priorité les écarts de contribution.

* Note : la commission ferme alpha est un revenu Opérateur. La part du bid qui alimente
  la réserve du pool est la consignation tant qu'elle n'est pas restituée + les
  confiscations. La réserve est donc endogène au volume d'enchères (d'où le scénario
  « bids faibles » comme test de viabilité).

Les niveaux 2-4 de la cascade (méta-réserve inter-pools, ligne contingente SFD, pro-rata
+ report) sont gérés par l'orchestrateur : le moteur expose, à chaque tour, le BESOIN
non couvert par la réserve du pool (champ `besoin_residuel`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from risque import ProfilRisque, pd_conditionnelle


# ---------------------------------------------------------------------------
# État d'un membre dans un pool
# ---------------------------------------------------------------------------

@dataclass
class EtatMembre:
    idx: int                       # index global (dans la liste de profils)
    profil: ProfilRisque
    type_pref: str                 # urgent / modere / epargnant
    urgence: float                 # multiplicateur d'enchère
    pd_mult: float                 # multiplicateur de PD (préférence de liquidité)

    en_defaut: bool = False
    tour_defaut: Optional[int] = None
    a_recu_pot: bool = False       # a déjà été bénéficiaire dans le cycle courant

    # suivi économique (cumulé sur tous les cycles)
    contribue: float = 0.0
    recu_immediat: float = 0.0     # parts immédiates reçues
    recu_differe: float = 0.0      # différés effectivement restitués
    consignation_versee: float = 0.0   # total consigné (part 1-alpha des bids)
    commission_payee: float = 0.0      # total commission ferme payée (alpha des bids)
    bonus_recu: float = 0.0
    differe_confisque: float = 0.0
    consignation_confisquee: float = 0.0
    cycles_sans_defaut: int = 0

    @property
    def discipline(self) -> bool:
        return not self.en_defaut


# ---------------------------------------------------------------------------
# Résultat d'un pool
# ---------------------------------------------------------------------------

@dataclass
class ResultatPool:
    pool_id: int
    tours: List[dict] = field(default_factory=list)        # journal par tour
    membres: List[EtatMembre] = field(default_factory=list)
    # agrégats
    total_bids: float = 0.0
    commission_operateur: float = 0.0       # alpha * total_bids (revenu Opérateur)
    consignations_totales: float = 0.0       # (1-alpha) * total_bids
    confiscations_totales: float = 0.0
    reserve_finale: float = 0.0
    n_defauts: int = 0


# ---------------------------------------------------------------------------
# Moteur d'un pool
# ---------------------------------------------------------------------------

class MoteurPool:
    """Simule un pool sur n_cycles. Les besoins de financement non couverts par la
    réserve du pool sont signalés à l'orchestrateur via un callback `demander_cascade`.
    """

    def __init__(self, cfg, pool_id: int):
        self.cfg = cfg
        self.pool_id = pool_id

    # -- préférences de liquidité -------------------------------------------
    def _assigner_preference(self, rng) -> tuple:
        p = self.cfg.preferences
        r = rng.random()
        if r < p.part_urgent:
            return "urgent", p.urgence_urgent, p.pd_mult_urgent
        elif r < p.part_urgent + p.part_modere:
            return "modere", p.urgence_modere, p.pd_mult_modere
        else:
            return "epargnant", p.urgence_epargnant, p.pd_mult_epargnant

    def _appliquer_stress_preferences(self, membres: List[EtatMembre], rng):
        """Stress comportemental : bascule une part de modérés/épargnants en urgents."""
        s = self.cfg.stress
        if not s.comportemental_actif or s.bascule_urgents <= 0:
            return
        p = self.cfg.preferences
        for m in membres:
            if m.type_pref in ("modere", "epargnant") and rng.random() < s.bascule_urgents:
                m.type_pref = "urgent"
                m.urgence = p.urgence_urgent
                m.pd_mult = p.pd_mult_urgent

    # -- enchère endogène ---------------------------------------------------
    def _enchere(self, rng, eligibles: List[EtatMembre], slot_dans_cycle: int) -> tuple:
        """Renvoie (gagnant, montant_bid_xof). Bid = valeur-temps de l'accès anticipé
        au différé, modulée par l'urgence du membre et un bruit idiosyncratique.
        """
        if not eligibles:
            return None, 0.0
        m_taille = self.cfg.structure.m_membres
        differe_pool = (1.0 - self.cfg.mecanisme.part_immediate) * self.cfg.structure.pool_mensuel
        mois_gagnes = max(0, (m_taille - 1) - slot_dans_cycle)
        base_frac = self.cfg.mecanisme.rho_mensuel * mois_gagnes  # fraction du différé
        meilleur, meilleur_bid = None, -1.0
        for mb in eligibles:
            bruit = float(np.exp(rng.normal(0.0, self.cfg.mecanisme.bid_bruit_sigma)))
            wtp_frac = base_frac * mb.urgence * bruit
            wtp = float(np.clip(wtp_frac, 0.0, 1.0)) * differe_pool
            if wtp > meilleur_bid:
                meilleur_bid, meilleur = wtp, mb
        return meilleur, max(0.0, meilleur_bid)

    # -- défaut conditionnel au facteur Z -----------------------------------
    def _tire_defaut(self, rng, m: EtatMembre, z: float) -> bool:
        pd = pd_conditionnelle(m.profil, z) * m.pd_mult
        if self.cfg.stress.comportemental_actif:
            pd += self.cfg.stress.choc_pd
        return rng.random() < float(np.clip(pd, 0.0, 1.0))

    # -- simulation complète d'un pool --------------------------------------
    def simuler(self, rng, profils_pool: List[ProfilRisque], z_par_mois: List[float],
                demander_cascade=None) -> ResultatPool:
        cfg = self.cfg
        m_taille = cfg.structure.m_membres
        part_imm = cfg.mecanisme.part_immediate
        alpha = cfg.mecanisme.alpha
        pool_mensuel = cfg.structure.pool_mensuel
        immediat = part_imm * pool_mensuel
        differe_nominal = (1.0 - part_imm) * pool_mensuel

        # membres
        membres: List[EtatMembre] = []
        for i, pr in enumerate(profils_pool):
            tp, urg, pdm = self._assigner_preference(rng)
            membres.append(EtatMembre(idx=i, profil=pr, type_pref=tp, urgence=urg, pd_mult=pdm))
        self._appliquer_stress_preferences(membres, rng)

        res = ResultatPool(pool_id=self.pool_id, membres=membres)
        reserve = 0.0                      # NIVEAU 1 : réserve du pool
        r_epargne = cfg.sfd.r_epargne

        total_tours = cfg.structure.total_tours
        ordre_residuel = list(range(m_taille))  # ordre de passage par défaut

        for t in range(1, total_tours + 1):
            cycle = (t - 1) // m_taille + 1
            slot = (t - 1) % m_taille
            z = z_par_mois[t - 1] if t - 1 < len(z_par_mois) else 0.0

            if slot == 0:  # nouveau cycle : réinitialiser l'éligibilité au pot
                for mb in membres:
                    mb.a_recu_pot = False

            # rémunération de la réserve (déposée à la SFD)
            reserve *= (1.0 + r_epargne)

            # --- 1. défauts du tour (PD conditionnelle à Z) ---
            for mb in membres:
                if not mb.en_defaut and self._tire_defaut(rng, mb, z):
                    mb.en_defaut = True
                    mb.tour_defaut = t
                    res.n_defauts += 1
                    # confiscation du différé accumulé non restitué + consignation
                    conf_dif = mb.recu_differe * 0.0  # le différé non encore versé est dans le pool
                    # consignation déjà versée est confisquée
                    conf_cons = mb.consignation_versee
                    mb.consignation_confisquee += conf_cons
                    mb.consignation_versee = 0.0
                    res.confiscations_totales += conf_cons
                    # alimentation : réserve en priorité (béta géré à la restitution/bonus)
                    reserve += conf_cons

            # --- 2. contributions (les non-défaillants paient) ---
            actifs = [mb for mb in membres if not mb.en_defaut]
            collecte = 0.0
            for mb in actifs:
                mb.contribue += cfg.structure.c
                collecte += cfg.structure.c
            n_actifs = len(actifs)

            # --- 3. bénéficiaire du tour : ordre résiduel parmi les non encore servis ---
            candidats = [mb for mb in membres if (not mb.en_defaut and not mb.a_recu_pot)]
            beneficiaire = None
            if candidats:
                # ordre résiduel stable (par index) ; l'enchère ci-dessous agit sur le DIFFÉRÉ
                candidats.sort(key=lambda mb: mb.idx)
                beneficiaire = candidats[0]
                beneficiaire.a_recu_pot = True

            # --- 4. enchère pour anticiper le différé (parmi non-défaillants non servis) ---
            eligibles = [mb for mb in membres if (not mb.en_defaut and not mb.a_recu_pot)]
            gagnant, bid = self._enchere(rng, eligibles, slot)
            commission = alpha * bid
            consignation = (1.0 - alpha) * bid
            res.total_bids += bid
            res.commission_operateur += commission
            res.consignations_totales += consignation
            if gagnant is not None and bid > 0:
                gagnant.commission_payee += commission
                gagnant.consignation_versee += consignation
                # la consignation entre dans la réserve (détenue SFD) jusqu'à restitution
                reserve += consignation

            # --- 5. décaissement immédiat au bénéficiaire (ligne de crédit SFD) ---
            besoin_residuel = 0.0
            decaisse = 0.0
            niveau_cascade = 0
            if beneficiaire is not None:
                decaisse = immediat
                beneficiaire.recu_immediat += immediat
                # GARANTIE : le pot COMPLET du bénéficiaire discipliné (immédiat + différé)
                #   doit être honoré. Le besoin de financement du tour est donc le pot
                #   nominal complet (immediat + differe_nominal) moins la collecte réelle.
                #   En nominal (collecte = M×c = pot complet), besoin = 0. Dès qu'un membre
                #   fait défaut, la collecte chute et la cascade comble l'écart pour préserver
                #   le pot des disciplinés.
                besoin_tour = (immediat + differe_nominal) - collecte
                if besoin_tour > 0:
                    pris = min(reserve, besoin_tour)
                    reserve -= pris
                    niveau_cascade = 1 if pris > 0 else 0
                    reste = besoin_tour - pris
                    if reste > 1e-9:
                        besoin_residuel = reste  # à traiter par l'orchestrateur (niv 2-4)
                # si collecte > pot complet (impossible en nominal), le surplus ne gonfle pas
                # la réserve (ce serait du différé d'autrui) — on l'ignore.

            # callback cascade niveaux 2-4 (géré par l'orchestrateur)
            niveau_atteint = niveau_cascade
            couvert_externe = 0.0
            if besoin_residuel > 0 and demander_cascade is not None:
                couvert_externe, niveau_atteint = demander_cascade(
                    self.pool_id, t, besoin_residuel, res.n_defauts)
                besoin_residuel -= couvert_externe

            res.tours.append({
                "pool_id": self.pool_id, "tour": t, "cycle": cycle, "z": z,
                "n_actifs": n_actifs, "collecte": collecte,
                "beneficiaire": beneficiaire.idx if beneficiaire else None,
                "decaisse_immediat": decaisse,
                "bid": bid, "commission": commission, "consignation": consignation,
                "gagnant_enchere": gagnant.idx if gagnant else None,
                "defauts_cumules": res.n_defauts,
                "besoin_residuel": besoin_residuel,
                "couvert_externe": couvert_externe,
                "niveau_cascade": niveau_atteint,
                "reserve_fin": reserve,
            })

            # --- 6. fin de cycle : restitution des différés + consignations aux disciplinés ---
            if slot == m_taille - 1:
                self._fin_de_cycle(membres, cycle, reserve_ref=lambda v=None: None)
                # mise à jour cycles_sans_defaut
                for mb in membres:
                    if not mb.en_defaut:
                        mb.cycles_sans_defaut += 1

        res.reserve_finale = reserve
        return res

    def _fin_de_cycle(self, membres, cycle, reserve_ref):
        """Restitution du différé nominal aux membres disciplinés en fin de cycle.
        (La restitution des consignations + bonus inter-cycle est gérée à la clôture
        finale par l'orchestrateur, qui connaît le fonds de bonus mutualisé.)
        """
        # Le différé nominal (part non immédiate du pot de chaque bénéficiaire discipliné)
        # est réputé restitué en fin de cycle ; on le crédite ici pour le suivi.
        differe_par_benef = (1.0 - self.cfg.mecanisme.part_immediate) * self.cfg.structure.pool_mensuel
        for mb in membres:
            if not mb.en_defaut and mb.a_recu_pot:
                mb.recu_differe += differe_par_benef
