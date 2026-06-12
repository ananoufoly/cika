"""config.py — Source unique de tous les paramètres du simulateur.

Tontine rotative structurée avec SFD dépositaire.
Aucune valeur n'est codée en dur ailleurs : tout passe par ces dataclasses.

Acteurs (nommage neutre et générique) :
  - Operateur : la fintech qui compose et opère les pools (perçoit les commissions)
  - SFD       : l'institution de microfinance dépositaire (séquestre, consignations,
                ligne de crédit, ligne contingente)
  - Membre    : participant à un pool
  - Pool      : un groupe rotatif de M membres
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 1. Structure des pools
# ---------------------------------------------------------------------------

@dataclass
class ParamsStructure:
    n_pools: int = 50              # nombre de pools en parallèle
    m_membres: int = 10            # membres par pool
    c: float = 10_000.0            # contribution mensuelle par membre (XOF)
                                   # (10K : seuil de viabilité de l'économie unitaire Opérateur)
    n_cycles: int = 2              # nombre de cycles (un cycle = m_membres tours)
    k_max_meme_secteur: int = 3    # contrainte de composition : max K membres
                                   # d'un même secteur par pool (non-corrélation)

    @property
    def pool_mensuel(self) -> float:
        """Pool collecté chaque tour = M × c."""
        return self.m_membres * self.c

    @property
    def tours_par_cycle(self) -> int:
        return self.m_membres

    @property
    def total_tours(self) -> int:
        return self.m_membres * self.n_cycles


# ---------------------------------------------------------------------------
# 2. Mécanisme par tour (split, enchère, bid)
# ---------------------------------------------------------------------------

@dataclass
class ParamsMecanisme:
    part_immediate: float = 0.50   # fraction du pool reçue immédiatement (décaissement
                                   # ligne de crédit SFD). Le reste est différé.
    alpha: float = 0.50            # part du bid = commission ferme Opérateur
                                   # (non remboursable). (1 - alpha) = consignation SFD.
                                   # (0.50 : niveau standard fermant l'économie unitaire)

    # --- enchère endogène (valeur-temps de l'accès anticipé) ---
    rho_mensuel: float = 0.02      # coût d'opportunité mensuel de la liquidité du membre.
                                   # bid ~ rho * mois_gagnes * montant_differe
    bid_bruit_sigma: float = 0.25  # dispersion idiosyncratique des bids (lognormal)


# ---------------------------------------------------------------------------
# 3. Préférences de liquidité (comportement d'enchère endogène)
# ---------------------------------------------------------------------------

@dataclass
class ParamsPreferences:
    # proportions des trois types (doivent sommer ~1)
    part_urgent: float = 0.20
    part_modere: float = 0.50
    part_epargnant: float = 0.30

    # multiplicateur d'urgence d'enchère par type (les urgents enchérissent plus)
    urgence_urgent: float = 1.60
    urgence_modere: float = 1.00
    urgence_epargnant: float = 0.55

    # multiplicateur de discipline de paiement par type (PD relatif)
    # un urgent est plus fragile (PD plus élevé), un épargnant plus fiable
    pd_mult_urgent: float = 1.35
    pd_mult_modere: float = 1.00
    pd_mult_epargnant: float = 0.70


# ---------------------------------------------------------------------------
# 4. Risque de crédit : facteur systémique unique + secteurs porteurs de charge
# ---------------------------------------------------------------------------

@dataclass
class ParamsRisque:
    pd_base_annuel: float = 0.08   # probabilité de défaut annuelle de base (moyenne)
    pd_base_sigma: float = 0.04    # dispersion de la PD de base entre membres

    # Secteurs : chacun fixe une CHARGE rho sur le facteur systémique unique Z.
    # Charge élevée => membres très corrélés (s'effondrent ensemble quand Z mauvais).
    # (nom_secteur, part_population, charge_rho)
    secteurs: List[Tuple[str, float, float]] = field(default_factory=lambda: [
        ("commerce",     0.30, 0.30),
        ("agriculture",  0.20, 0.45),   # plus exposé aux chocs systémiques
        ("transport",    0.15, 0.25),
        ("services",     0.20, 0.15),   # défensif
        ("artisanat",    0.15, 0.20),
    ])

    def secteur_noms(self) -> List[str]:
        return [s[0] for s in self.secteurs]


# ---------------------------------------------------------------------------
# 5. Taux SFD (dépôts, crédit, replacement)
# ---------------------------------------------------------------------------

@dataclass
class ParamsSFD:
    r_epargne_annuel: float = 0.04    # rémunération de l'épargne (séquestre + consignations)
    r_credit_annuel: float = 0.24     # taux des lignes de crédit SFD (décaissements immédiats)
    r_replacement_annuel: float = 0.08  # taux interne de replacement des dépôts par la SFD
    periodes_par_an: int = 12

    # Ligne contingente SFD (niveau 3 de la cascade), exprimée en "nb de défauts/pool"
    ligne_contingente_defauts_pilote: int = 4   # phase pilote
    ligne_contingente_defauts_croisiere: int = 2  # régime de croisière
    phase_pilote: bool = True

    @property
    def r_epargne(self) -> float:
        return (1.0 + self.r_epargne_annuel) ** (1.0 / self.periodes_par_an) - 1.0

    @property
    def r_credit(self) -> float:
        return (1.0 + self.r_credit_annuel) ** (1.0 / self.periodes_par_an) - 1.0

    @property
    def r_replacement(self) -> float:
        return (1.0 + self.r_replacement_annuel) ** (1.0 / self.periodes_par_an) - 1.0

    def plafond_ligne_par_pool(self, c: float, part_immediate: float, m: float) -> float:
        """Plafond de la ligne contingente par pool, en XOF.
        = nb_defauts_couverts × exposition d'un défaut.
        L'exposition d'un défaut ~ contribution manquée cumulée + part immédiate non couverte.
        Approche conservatrice : nb_defauts × part_immediate_du_pool / m.
        """
        nb = (self.ligne_contingente_defauts_pilote if self.phase_pilote
              else self.ligne_contingente_defauts_croisiere)
        exposition_unitaire = part_immediate * (m * c) / m  # ~ part immédiate par bénéficiaire
        return nb * exposition_unitaire


# ---------------------------------------------------------------------------
# 6. Bonus inter-cycle
# ---------------------------------------------------------------------------

@dataclass
class ParamsBonus:
    cycles_requis: int = 2         # cycles complets sans défaut pour être éligible
    beta: float = 0.50             # part des confiscations allouée au fonds de bonus
                                   # (le reste -> réserve). 0.50 par défaut.


# ---------------------------------------------------------------------------
# 7. Cascade de garanties (waterfall)
# ---------------------------------------------------------------------------

@dataclass
class ParamsCascade:
    meta_reserve_active: bool = True   # niveau 2 : mutualisation inter-pools
    ligne_sfd_active: bool = True      # niveau 3 : ligne contingente SFD
    # niveau 4 (pro-rata + report) toujours actif en dernier recours


# ---------------------------------------------------------------------------
# 8. Modules de stress (comportemental + macro), combinables
# ---------------------------------------------------------------------------

@dataclass
class ParamsStress:
    # --- stress comportemental ---
    comportemental_actif: bool = False
    choc_pd: float = 0.0           # choc additif sur la PD individuelle [0,1]
    bascule_urgents: float = 0.0   # part de modérés/épargnants basculés en "urgent"
                                   # (plus d'urgents -> plus de bids OU défauts précoces)

    # --- stress macro (facteur systémique Z) ---
    macro_actif: bool = False
    z_choc: float = 0.0            # niveau du choc systémique (en écarts-types de Z,
                                   # valeur NÉGATIVE = mauvaise conjoncture). Ex. -2.0
    z_persistance: int = 0         # nb de mois pendant lesquels le choc Z persiste
                                   # (0 = tirage i.i.d. chaque mois)


# ---------------------------------------------------------------------------
# 9. P&L Opérateur
# ---------------------------------------------------------------------------

@dataclass
class ParamsPnLOperateur:
    cout_acquisition_membre: float = 2_000.0     # XOF par membre recruté (marketing + KYC)

    # Coût opérationnel par pool/mois — DÉCOMPOSÉ (plateforme digitale automatisée).
    # Hypothèse : coût marginal faible, dominé par les frais de transaction mobile money,
    # les notifications et une fraction amortie du support. Total ~800 XOF/pool/mois,
    # bien en deçà d'un suivi manuel par agent de terrain.
    cout_mobile_money_taux: float = 0.01         # % appliqué aux flux (contributions + décaissement)
    cout_notifications_membre_mois: float = 20.0 # XOF par membre par mois (SMS/push)
    cout_support_pool_mois: float = 200.0        # XOF par pool par mois (support/litiges amorti)

    couts_fixes_mensuels: float = 500_000.0      # XOF par mois (équipe, conformité, tech)
    retrocession_sfd: float = 0.0                # % des intérêts de crédit SFD rétrocédés
                                                 # à l'Opérateur (0 = conservateur)

    def cout_ops_pool_mois(self, pool_mensuel: float, m_membres: int,
                           part_immediate: float) -> float:
        """Coût opérationnel d'un pool sur un mois, décomposé et transparent."""
        # flux mensuels = contributions entrantes + décaissement immédiat sortant
        flux = pool_mensuel + part_immediate * pool_mensuel
        cout_mm = self.cout_mobile_money_taux * flux
        cout_notif = self.cout_notifications_membre_mois * m_membres
        return cout_mm + cout_notif + self.cout_support_pool_mois


# ---------------------------------------------------------------------------
# 10. Monte Carlo
# ---------------------------------------------------------------------------

@dataclass
class ParamsMonteCarlo:
    n_runs: int = 1_000
    graine_base: int = 12345


# ---------------------------------------------------------------------------
# Conteneur global
# ---------------------------------------------------------------------------

@dataclass
class Config:
    structure: ParamsStructure = field(default_factory=ParamsStructure)
    mecanisme: ParamsMecanisme = field(default_factory=ParamsMecanisme)
    preferences: ParamsPreferences = field(default_factory=ParamsPreferences)
    risque: ParamsRisque = field(default_factory=ParamsRisque)
    sfd: ParamsSFD = field(default_factory=ParamsSFD)
    bonus: ParamsBonus = field(default_factory=ParamsBonus)
    cascade: ParamsCascade = field(default_factory=ParamsCascade)
    stress: ParamsStress = field(default_factory=ParamsStress)
    pnl_operateur: ParamsPnLOperateur = field(default_factory=ParamsPnLOperateur)
    monte_carlo: ParamsMonteCarlo = field(default_factory=ParamsMonteCarlo)


def config_par_defaut() -> Config:
    return Config()
