# Hypothèses du simulateur — Tontine structurée avec SFD dépositaire

Ce document recense les hypothèses du modèle, leur justification et leur sensibilité.
Objectif : pouvoir les **défendre en présentation** et savoir lesquelles déplacent les
conclusions. Tous les paramètres sont dans `config.py` (source unique, aucune valeur en dur).

Acteurs : **Opérateur** (la fintech), **SFD** (institution de microfinance dépositaire),
**Membre**, **Pool**.

---

## 1. Structure et mécanisme

| Hypothèse | Valeur par défaut | Justification | Sensibilité |
|---|---|---|---|
| Pools en parallèle (N) | 50 | Taille de portefeuille pilote | Le break-even Opérateur dépend du nombre total de pools (≈ centaines). |
| Membres par pool (M) | 10 | Taille de tontine usuelle en UEMOA | **Fort levier** : la marge Opérateur croît plus vite que M (le volume de bids ∝ pool mensuel). |
| Contribution mensuelle (c) | 10 000 XOF | **Seuil de viabilité** de l'économie unitaire Opérateur | À 5 000 XOF, la marge par pool est négative — l'Opérateur ne ferme pas son économie. |
| Cycles | 2 | Horizon minimal pour le bonus inter-cycle | Plus de cycles amortissent le coût d'acquisition (ponctuel). |
| Part immédiate | 50 % | Décaissement « crédit » au bénéficiaire ; le reste différé | Pilote la taille de la ligne de crédit SFD et l'exposition au risque. |

---

## 2. Comportement d'enchère (endogène)

- **Le prix des enchères n'est pas fixé** : il émerge de la valeur-temps de l'accès anticipé
  au différé. Un membre enchérit à hauteur de `rho_mensuel × mois_gagnés × montant_différé`,
  modulé par son **urgence** (préférence de liquidité) et un bruit idiosyncratique lognormal.
- **`rho_mensuel = 2 %/mois`** : coût d'opportunité de la liquidité du membre, cohérent avec
  le crédit semi-formel UEMOA. C'est ce qui rend les bids des premiers tours plus élevés.
- **Préférences de liquidité** (urgent / modéré / épargnant : 20 / 50 / 30 %) :
  - urgents : enchérissent plus (urgence 1.6×) et sont plus fragiles (PD ×1.35) ;
  - épargnants : enchérissent peu (0.55×) et sont plus fiables (PD ×0.70).
- **Scénario « bids faibles »** : si la population bascule vers les épargnants et `rho` baisse,
  la **réserve endogène** (alimentée par les bids) se réduit. C'est un **test de viabilité clé** :
  la cascade doit tenir même quand peu de membres enchérissent.

> ⚠️ Hypothèse forte : les membres enchérissent rationnellement selon la valeur-temps. En
> réalité, des biais comportementaux (sur-enchère par urgence, sous-enchère par méfiance)
> existent ; le bruit lognormal en capture une partie, mais pas tout.

---

## 3. Risque de crédit : facteur systémique unique + secteurs (Vasicek mono-facteur)

- **Un unique facteur systémique Z** ~ N(0,1) représente la conjoncture commune à toute
  l'économie (et donc à tous les pools simultanément — source de la corrélation inter-pools).
- **Chaque membre appartient à un secteur** qui fixe sa **charge `rho` sur Z** : un secteur
  exposé (agriculture, `rho=0.45`) voit ses membres défaillir ensemble en mauvaise conjoncture ;
  un secteur défensif (services, `rho=0.15`) est peu corrélé.
- **PD conditionnelle** : `PD_i(Z) = Phi( (Phi⁻¹(PD_i) − √rho_i · Z) / √(1−rho_i) )`,
  convention Z>0 = bonne conjoncture (PD baisse), Z<0 = mauvaise (PD monte).
- **PD de base annuelle = 8 %** (moyenne, dispersion 4 %) : ordre de grandeur du risque
  microfinance UEMOA. Convertie en PD mensuelle.

> Le choix mono-facteur réconcilie rigueur quantitative et lisibilité : les **secteurs**
> portent la charge, ce qui rend la **contrainte K** (ci-dessous) littérale et interprétable.

---

## 4. Contrainte de composition K (non-corrélation)

- Objectif : **au plus K=3 membres d'un même secteur par pool**, pour borner la corrélation
  intra-pool.
- **Hypothèse réaliste assumée** : la contrainte est parfois **mathématiquement infaisable**
  (un secteur sur-représenté dépasse `K × n_pools / M` places). Dans ce cas, l'algorithme
  place **au mieux** et **mesure** le résiduel via le KPI **taux de respect K**, plutôt que
  de déformer la population réelle.
- Exemple : un secteur « commerce » à 30 % de la population force un K minimal faisable > 3,
  d'où un taux de respect ~94 % et non 100 %. C'est l'argument honnête : *« voici la
  diversification réellement atteignable »*.

---

## 5. Cascade de garanties (waterfall)

Ordre strict, à chaque besoin de financement (garantie du **pot complet** des disciplinés) :

1. **Réserve du pool** (niveau 1) : consignations du pool + confiscations.
2. **Méta-réserve inter-pools** (niveau 2) : mutualisation des réserves disponibles des
   autres pools, au prorata. *Limite connue* : en stress **systémique**, tous les pools
   souffrent ensemble, la méta-réserve s'épuise — c'est précisément ce que la contrainte K
   doit atténuer.
3. **Ligne contingente SFD** (niveau 3) : plafonnée à **X défauts/pool** (4 en pilote,
   2 en croisière), pré-provisionnée. **Borne la perte SFD** : c'est l'argument central du
   pitch partenaire — la perte maximale de la SFD est connue d'avance.
4. **Pro-rata + report** (niveau 4) : versement partiel + report du solde au cycle suivant.

**Garantie modélisée** : le **pot complet (immédiat + différé) des membres disciplinés** est
honoré en priorité ; la part immédiate est financée tour par tour, le différé en fin de cycle.

---

## 6. Bonus inter-cycle

- **Éligibilité** : 2 cycles complets sans défaut.
- **Financement** : `beta = 50 %` des confiscations (le reste → réserve) + intérêts générés
  par les consignations déposées à la SFD.
- **Distribution** : au prorata des consignations de chaque membre discipliné.

> Hypothèse : le bonus incite à la discipline et à la rétention. Son ampleur dépend du volume
> de confiscations (donc du taux de défaut) — paradoxalement, plus de défauts ⇒ bonus plus
> élevé pour les survivants. À surveiller pour éviter un aléa moral.

---

## 7. Économie SFD

| Hypothèse | Valeur | Justification |
|---|---|---|
| Taux ligne de crédit | 24 %/an | Taux microcrédit UEMOA standard |
| Rémunération épargne | 4 %/an | Compte épargne SFD |
| Taux de replacement interne | 8 %/an | Placement des dépôts par la SFD (spread = 8 % − 4 %) |
| Ligne contingente | 4 défauts/pool (pilote) | Pré-provisionnée ; borne la perte SFD |

**Intérêts de crédit** : chaque décaissement immédiat est un crédit court terme rémunéré à
`r_credit` sur la durée jusqu'à fin de cycle (approximation conservatrice).

---

## 8. Économie Opérateur

| Poste | Valeur | Justification |
|---|---|---|
| Commission ferme (alpha) | **50 % du bid** | Niveau fermant l'économie unitaire ; non remboursable |
| Acquisition / membre | 2 000 XOF | Marketing digital + KYC (ponctuel) |
| **Coût ops / pool / mois** | **~1 900 XOF** (décomposé) | Plateforme **automatisée** : mobile money (1 % des flux) + notifications (20 XOF/membre) + support (200 XOF/pool) |
| Coûts fixes | 500 000 XOF/mois | Équipe, conformité, tech — absorbés à l'échelle (~centaines de pools) |

> ⚠️ Hypothèse structurante : le coût ops par pool est celui d'une **plateforme digitale
> automatisée** (~1 900 XOF/pool/mois), et **non** d'un suivi manuel par agent (qui serait
> ~5 000 XOF et rendrait l'économie unitaire non viable à cette échelle de contribution).

---

## 9. Modules de stress

- **Stress comportemental** : choc additif sur la PD individuelle (+3 pts) + bascule d'une part
  (30 %) des modérés/épargnants vers « urgent » (plus de fragilité et d'enchères précoces).
- **Stress macro** : choc systémique `Z = −2.5` persistant 4 mois. Frappe **tous les membres
  via leur charge sectorielle** simultanément → défauts corrélés intra- et inter-pools. C'est
  le test de la contrainte K et de la cascade.
- **Combinable** : les deux stress peuvent s'appliquer ensemble (scénario « combiné »).

> Le niveau `Z = −2.5` est un stress **sévère** (≈ choc à 2,5 écarts-types). À calibrer selon
> la sévérité voulue pour le pitch : c'est un scénario « crise », pas un ralentissement léger.

---

## 10. Limites connues

1. **Pas de remplacement intra-cycle** : un pool défaillant continue à effectif réduit
   (conforme à la spec). Le remplacement inter-cycle n'est pas modélisé.
2. **Rationalité des enchères** : supposée ; les biais comportementaux ne sont que
   partiellement capturés par le bruit.
3. **Mono-facteur** : un seul facteur systémique. Des chocs sectoriels **idiosyncratiques**
   (un seul secteur touché, pas toute l'économie) ne sont pas modélisés séparément.
4. **Intérêts de crédit** : approximés sur la durée jusqu'à fin de cycle, sans calendrier de
   remboursement détaillé.
5. **Aléa moral du bonus** : plus de défauts ⇒ bonus plus élevé pour les survivants ; effet
   non régulé dans le modèle actuel.
