"""config.py — Source unique des paramètres. Tontine SFD v2 (avance endogène + promesse).

Acteurs (nommage neutre) :
  - Operateur : la fintech qui compose et opère les pools. NE DÉTIENT JAMAIS LES FONDS.
                N'apporte AUCUN capital propre.
  - SFD       : institution de microfinance dépositaire. Détient le compte tontine (nanti),
                consent les avances, perçoit les frais. Avance TOUJOURS (le membre est servi).
  - Membre    : participant à un pool, avec un sous-compte nominatif.
  - Pool      : groupe rotatif de M membres.
  - FGE       : Fonds de Garantie Endogène. Constitué SANS capital de la fintech, depuis une
                part des primes + les garanties d'enchère saisies sur les fuyards. Absorbe les
                avances SFD non récupérées AVANT que la SFD ne perde.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 1. Structure
# ---------------------------------------------------------------------------
@dataclass
class ParamsStructure:
    n_pools: int = 50
    m_membres: int = 10
    c: float = 100_000.0          # cotisation mensuelle (XOF)
    n_cycles: int = 2
    k_max_meme_secteur: int = 3

    @property
    def pool_mensuel(self) -> float:
        return self.m_membres * self.c

    @property
    def total_tours(self) -> int:
        return self.m_membres * self.n_cycles


# ---------------------------------------------------------------------------
# 2. Compte tontine, avance SFD, frais
# ---------------------------------------------------------------------------
@dataclass
class ParamsCompte:
    r_sfd_annuel: float = 0.18      # taux des avances SFD (frais = avance × r × (M−t)/12)
    periodes_par_an: int = 12
    # remboursement pro-rata des frais prépayés si arrêt anticipé du cycle
    remboursement_prorata_arret: bool = False

    @property
    def r_sfd_mensuel(self) -> float:
        return self.r_sfd_annuel / 12.0


# ---------------------------------------------------------------------------
# 3. Enchère (plancher = frais SFD du tour + prime ; marge endogène au-dessus)
# ---------------------------------------------------------------------------
@dataclass
class ParamsProduit:
    """Tontine NUE (sans garantie, sans bid, risque porté par le participant) vs
    GARANTIE (prime de garantie obligatoire ∝ avance + bid optionnel + couverture SFD)."""
    mode: str = "garantie"          # "garantie" | "nue"
    # prime de garantie : facteur de prudence (1.0 = actuarielle, >1 = majorée)
    prime_facteur_prudence: float = 1.0


@dataclass
class ParamsEnchere:
    prime_operateur_taux: float = 0.015  # marge Opérateur, en fraction du pot (revenu fintech)
    marge_min_fixe: float = 0.0     # marge minimale fixe (défaut 0)
    rho_mensuel: float = 0.02       # coût-temps du membre (valeur de l'accès anticipé)
    bid_bruit_sigma: float = 0.25   # dispersion idiosyncratique des bids
    bid_plafond_frac_pot: float = 0.12   # plafond du SURPLUS de bid (compétitivité/usure)


# ---------------------------------------------------------------------------
# 4. Préférences de liquidité (comportement d'enchère endogène)
# ---------------------------------------------------------------------------
@dataclass
class ParamsPreferences:
    part_urgent: float = 0.20
    part_modere: float = 0.50
    part_epargnant: float = 0.30
    urgence_urgent: float = 1.60
    urgence_modere: float = 1.00
    urgence_epargnant: float = 0.55


# ---------------------------------------------------------------------------
# 5. Fuite délibérée + friction (échec temporaire récupérable)
# ---------------------------------------------------------------------------
@dataclass
class ParamsFuite:
    # Fuite délibérée : probabilité conditionnelle d'avoir encaissé son tour.
    # p_fuite_base = proba TOTALE de fuir sur toute la durée post-encaissement (nominal),
    # interprétable : « X% des bénéficiaires fuient un jour ». Convertie en proba mensuelle
    # dans fuite.py. Ajustée par tour pris (précoce = plus tentant) et macro (Z).
    p_fuite_base: float = 0.06      # 6% des bénéficiaires fuient (nominal)
    # plus on a encaissé tôt (et plus il reste de cotisations à devoir), plus la tentation est forte
    fuite_mult_tour_precoce: float = 1.8  # multiplicateur appliqué au tour 1, décroît vers 1 au dernier tour
    charge_z_fuite: float = 0.35    # sensibilité de la fuite au facteur systémique Z (corrélation)

    # Échec de friction : prélèvement automatique qui échoue temporairement (récupérable).
    taux_echec_friction: float = 0.03   # proba qu'une cotisation échoue un mois donné
    taux_recuperation_friction: float = 0.90  # part récupérée le mois suivant


# ---------------------------------------------------------------------------
# 6. Couverture — trois étages activables séparément (effet marginal)
# ---------------------------------------------------------------------------
@dataclass
class ParamsCouverture:
    # ÉTAGE 1 — MITIGATION
    mitigation_active: bool = True
    acces_sequence_active: bool = True
    t_restreint: int = 3            # tours 1..t_restreint réservés aux membres avec historique
    part_avec_historique: float = 0.50  # part de la population ayant ≥1 cycle d'historique
    garantie_enchere_active: bool = True
    g_cotisations_consignees: int = 1   # consignation (en nb de cotisations) pour bider tôt
    prelevement_auto_efficacite: float = 0.7  # réduit le taux d'échec de friction (0..1)

    # ÉTAGE 2 — PRIME (gérée dans ParamsEnchere ; ici l'activation)
    prime_active: bool = True

    # ÉTAGE 3 — FGE (Fonds de Garantie Endogène) + tranche SFD junior
    fge_actif: bool = True
    # La SFD porte une TRANCHE JUNIOR après épuisement du FGE (peau dans le jeu, garantie
    # crédible). Plafond exprimé en fraction des avances cumulées du portefeuille.
    # Au-delà de ce plafond = événement résiduel (doit être ~0 pour signer).
    tranche_sfd_active: bool = True
    plafond_tranche_sfd_frac: float = 0.05   # 5% des avances cumulées


# ---------------------------------------------------------------------------
# 7. Risque de crédit (réutilisé du v1 : facteur unique + secteurs)
# ---------------------------------------------------------------------------
@dataclass
class ParamsRisque:
    pd_base_annuel: float = 0.08    # ici sert de base à la corrélation sectorielle (charge Z)
    pd_base_sigma: float = 0.04
    secteurs: List[Tuple[str, float, float]] = field(default_factory=lambda: [
        ("commerce", 0.30, 0.30), ("agriculture", 0.20, 0.45),
        ("transport", 0.15, 0.25), ("services", 0.20, 0.15), ("artisanat", 0.15, 0.20),
    ])


# ---------------------------------------------------------------------------
# 8. Stress (conservés du v1)
# ---------------------------------------------------------------------------
@dataclass
class ParamsStress:
    comportemental_actif: bool = False
    choc_fuite: float = 0.0         # choc additif sur la proba de fuite
    bascule_urgents: float = 0.0    # part de modérés/épargnants basculés en urgents
    macro_actif: bool = False
    z_choc: float = 0.0             # niveau du choc systémique (négatif = mauvaise conjoncture)
    z_persistance: int = 0


# ---------------------------------------------------------------------------
# 9. P&L Opérateur
# ---------------------------------------------------------------------------
@dataclass
class ParamsPnLOperateur:
    cout_acquisition_membre: float = 2_000.0
    cout_ops_pool_mois: float = 5_000.0
    couts_fixes_mensuels: float = 0.0   # SUPPRIMÉ : on isole l'économie unitaire (marge par
                                        # pool). Les frais de structure relèvent du financement,
                                        # hors du périmètre de ce modèle.
    cout_capital_annuel: float = 0.10   # rémunération théorique du FGE immobilisé (pour ROE honnête)


# ---------------------------------------------------------------------------
# 10. Compétitivité / conformité
# ---------------------------------------------------------------------------
@dataclass
class ParamsConformite:
    taux_usure_annuel: float = 0.24     # seuil d'usure UEMOA — flag si coût membre > ce taux
    cout_membre_tour1_max: float = 0.15 # compétitivité : coût tour 1 < 15% du pot (paramétrable)


# ---------------------------------------------------------------------------
# 11. Monte Carlo
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
    produit: ParamsProduit = field(default_factory=ParamsProduit)
    compte: ParamsCompte = field(default_factory=ParamsCompte)
    enchere: ParamsEnchere = field(default_factory=ParamsEnchere)
    preferences: ParamsPreferences = field(default_factory=ParamsPreferences)
    fuite: ParamsFuite = field(default_factory=ParamsFuite)
    couverture: ParamsCouverture = field(default_factory=ParamsCouverture)
    risque: ParamsRisque = field(default_factory=ParamsRisque)
    stress: ParamsStress = field(default_factory=ParamsStress)
    pnl_operateur: ParamsPnLOperateur = field(default_factory=ParamsPnLOperateur)
    conformite: ParamsConformite = field(default_factory=ParamsConformite)
    monte_carlo: ParamsMonteCarlo = field(default_factory=ParamsMonteCarlo)


def config_par_defaut() -> Config:
    return Config()
