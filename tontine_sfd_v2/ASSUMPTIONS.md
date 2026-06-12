# Hypothèses du simulateur — Tontine SFD v2 (crédit-relais + promesse de continuité)

Document de référence pour défendre le modèle en présentation. Tous les paramètres sont
dans `config.py` (source unique). Nommage neutre : **Opérateur** (fintech), **SFD**
(institution dépositaire), **Membre**, **Pool**, **FGE** (Fonds de Garantie Endogène).

---

## 1. Mécanique financière — le crédit-relais

**Validé au chiffre près.** Pool de M membres, cotisation c, pot collecté par tour = (M−1)·c
(le bénéficiaire du tour ne cotise pas pour lui-même ce mois-là).

- Le gagnant du tour reçoit **net = pot − bid**. La SFD **avance** ce net.
- C'est un **prêt** : la SFD le récupère **linéairement** sur la durée restante d = (M−t)
  tours, mensualité = net/d, par prélèvement sur le compte tontine (alimenté par les
  cotisations courantes).
- Le **bid** est retenu à la source : il contient les intérêts SFD + la prime de garantie +
  la marge Opérateur (+ surplus de position). C'est le **prix du crédit-relais**, payé
  d'avance.

> Exemple (M=10, c=100k, bid=150k) : pot=900k, net=750k, mensualité=750k/9≈83 333,
> récupéré sur 9 mois. Le bid 150k = intérêts + prime + marge.

---

## 2. Deux produits : NUE vs GARANTIE

| | Tontine NUE | Tontine GARANTIE |
|---|---|---|
| Bid autorisé | Non (ordre pur) | Oui (optionnel, pour la position) |
| Prime de garantie | Aucune | Obligatoire (∝ avance) |
| Risque de fuite | Porté par le **groupe** | Couvert par **FGE → tranche SFD** |
| Coût pour l'épargnant pur | 0 | ≈ 0 (avance nulle au dernier tour) |

**Principe central** : on ne fait jamais payer l'assurance à l'épargnant. La prime de
garantie est proportionnelle à l'**avance encaissée** (= pot reçu − épargne déjà versée),
qui est exactement le risque que le membre fait porter au système. Le dernier tour (avance
≈ 0) ne paie rien.

---

## 3. Coût d'un encaisseur (tontine garantie) — décomposition

```
coût total (bid) = intérêts SFD        (∝ durée de l'avance — rémunère la SFD)
                 + prime de garantie    (∝ avance — OBLIGATOIRE — alimente le FGE)
                 + marge Opérateur      (∝ pot — revenu fintech)
                 + surplus de position  (OPTIONNEL — uniquement si on bide pour passer devant)
```

- La prime de garantie **taxe le risque** (l'avance) ; le bid **taxe la priorité** (la position).
- Deux variantes de prime comparées dans les sorties : **actuarielle** (facteur 1.0) vs
  **majorée** (facteur > 1, robuste au stress corrélé).

---

## 4. Fuite et continuité

- **Fuite délibérée** (le cas critique) : un membre ayant **encaissé** cesse de contribuer
  au remboursement. `p_fuite_base` = probabilité **TOTALE** de fuir sur toute la durée
  post-encaissement (interprétable : « X% des bénéficiaires fuient »), convertie en proba
  mensuelle. Ajustée par : tour pris (précoce = plus tentant), macro (Z), choc comportemental.
- **Échec de friction** (non critique) : prélèvement automatique qui échoue temporairement,
  récupérable. Réduit par l'efficacité du prélèvement auto.
- **Remplacement** : un membre n'ayant PAS encaissé et qui s'arrête est remplacé (mise rendue
  moins pénalité). Le remplaçant cotise pour SON tour, il ne comble PAS le trou d'un fuyard.

**Le trou** d'une fuite = principal restant de l'avance (vue prêt). C'est ce que la SFD a
réellement décaissé et ne récupère pas. La promesse aux autres reste tenue tant que ce trou
est comblé.

---

## 5. Cascade de couverture (ordre strict)

1. **FGE** (Fonds de Garantie Endogène) : primes de garantie accumulées + garanties d'enchère
   saisies sur les fuyards. **Aucun capital de la fintech.** Première perte.
2. **Tranche SFD junior** : la SFD absorbe ce qui dépasse le FGE, jusqu'à un plafond
   (`plafond_tranche_sfd_frac` des avances). **Son skin in the game** — c'est ce qui rend la
   garantie crédible et aligne ses intérêts. Rémunérée par ses intérêts + spread.
3. **Résiduel** : au-delà du plafond SFD. Doit être ~0 (sinon la promesse casse). Le pitch
   montre que cet événement est rarissime sous mitigations.

> La SFD avance **toujours** : le membre est **toujours servi**. La « casse de promesse » est
> un événement de résiduel > 0, pas un membre non payé.

---

## 6. Mitigations (étage 1, activables séparément pour mesurer l'effet marginal)

- **Accès séquencé** : les tours 1..T_restreint réservés aux membres avec ≥1 cycle d'historique.
- **Garantie d'enchère** : consignation de g cotisations pour bider tôt, saisie si fuite,
  restituée sinon. Filtre anti-fuyard ET alimente le FGE.
- **Prélèvement automatique** : réduit le taux d'échec de friction.

---

## 7. Risque corrélé (réutilisé de la v1)

Facteur systémique unique Z + secteurs porteurs de charge ρ (Vasicek mono-facteur). La fuite
est corrélée via la charge sectorielle : un choc macro (Z<0) augmente les fuites de tous les
membres simultanément → teste la contrainte de composition K et la cascade.

---

## 8. Économie SFD / Opérateur

| Paramètre | Défaut | Justification |
|---|---|---|
| Taux SFD (avances) | 18 %/an | Coût du crédit-relais court terme UEMOA |
| Marge Opérateur | 1.5 % du pot | Revenu fintech par encaissement |
| Acquisition / membre | 2 000 XOF | Marketing + KYC |
| Ops / pool / mois | 5 000 XOF | À revoir (plateforme automatisée → plus bas) |
| Fixes mensuels | 500 000 XOF | Structure ; absorbés à l'échelle |
| Coût du capital FGE | 10 %/an | Rémunération théorique du FGE immobilisé (ROE honnête) |

---

## 9. Conformité / compétitivité

- **Seuil d'usure UEMOA** : 24 %/an. Flag si le coût membre annualisé dépasse.
- **Compétitivité** : coût du tour 1 < 15 % du pot (paramétrable). Critère du pitch.

---

## 10. Limites connues & anti-sélection

1. **Anti-sélection par le nom** : promettre explicitement « garanti » attire les mauvais
   profils (ceux qui comptent fuir) et crée de l'aléa moral. **Le produit ne doit pas être
   commercialisé comme une assurance.** La continuité est un effet de la plomberie, pas une
   promesse vendue. Les filtres (prime ∝ avance, accès séquencé, garantie consignée) sont les
   barrières anti-mauvais-clients.
2. **Dépendance aux bideurs pour le P&L Opérateur** : en scénario « bids faibles », le surplus
   d'enchère s'effondre et le P&L Opérateur peut devenir négatif. La promesse tient (la prime
   de garantie reste collectée), mais la rentabilité fintech suppose un minimum de bideurs.
3. **Rationalité d'enchère** supposée ; biais comportementaux partiellement capturés par le bruit.
4. **Mono-facteur** : un seul facteur systémique ; chocs purement sectoriels non isolés.
5. **Coût du tour 1** : dominé par les intérêts SFD sur la durée longue ; à calibrer pour
   rester sous le seuil de compétitivité.
