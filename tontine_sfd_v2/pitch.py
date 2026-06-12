"""pitch.py — Génère les 6 graphiques du pitch.

  1. Break-even en nombre de pools (P&L Opérateur)
  2. Taux de fuite de break-even (jusqu'où le FGE absorbe)
  3. Taux de continuité par scénario (la promesse tenue)
  4. Exposition SFD mois par mois (profil + max)
  5. Coût membre par tour vs alternatives (compétitivité / usure)
  6. Surface des triplets admissibles (prime, mitigations, capital)

Usage : python pitch.py [--rapide]   (--rapide = 150 runs au lieu de 1000)
"""

from __future__ import annotations
import sys, copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import config_par_defaut
from moteur import simuler_run
from pnl import calculer_pnl_operateur, taux_fuite_break_even
from promesse import agreger_promesse
from scenarios import SCENARIOS, simuler_scenario

COUL = {"nominal": "#0f4c4a", "comportemental": "#b18a3a", "macro": "#b91c1c",
        "combine": "#5b6b87", "bids_faibles": "#14706c"}
LBL = {"nominal": "Nominal", "comportemental": "Comportemental", "macro": "Macro",
       "combine": "Combiné", "bids_faibles": "Bids faibles"}


# 1. Break-even en nombre de pools
def graph_break_even(cfg, fichier="fig1_break_even.png"):
    r = simuler_run(cfg, graine=cfg.monte_carlo.graine_base)
    op = calculer_pnl_operateur(r, cfg)
    marge = op.marge_par_pool; fixes = op.couts_fixes
    be = fixes / marge if marge > 0 else np.inf
    pools = np.arange(0, 200, 2); pnl = marge * pools - fixes
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pools, pnl / 1e6, color="#0f4c4a", lw=2.2, label="P&L Opérateur")
    ax.axhline(0, color="#5b6b87", ls="--", lw=1)
    if np.isfinite(be) and be <= 200:
        ax.axvline(be, color="#b18a3a", ls="--", lw=1.5, label=f"Break-even ≈ {be:.0f} pools")
    ax.set_xlabel("Nombre de pools"); ax.set_ylabel("P&L Opérateur (M XOF)")
    ax.set_title("1 · Break-even Opérateur en nombre de pools")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(fichier, dpi=130); plt.close(fig)
    return be


# 2. Taux de fuite de break-even
def graph_taux_fuite_be(cfg, fichier="fig2_taux_fuite_be.png", graines=6):
    taux = np.linspace(0.02, 0.50, 13)
    perte_sfd = []; residuel = []
    for p in taux:
        ps = []; rs = []
        for g in range(graines):
            c = copy.deepcopy(cfg); c.fuite.p_fuite_base = p
            r = simuler_run(c, graine=g)
            ps.append(r.perte_sfd); rs.append(r.residuel_non_couvert)
        perte_sfd.append(np.mean(ps) / 1e6); residuel.append(np.mean(rs) / 1e6)
    tfbe = taux_fuite_break_even(cfg, graines=graines)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(taux * 100, perte_sfd, color="#b18a3a", lw=2, marker="o", ms=4, label="Recours tranche SFD")
    ax.plot(taux * 100, residuel, color="#b91c1c", lw=2, marker="s", ms=4, label="Résiduel (promesse menacée)")
    ax.axvline(tfbe * 100, color="#0f4c4a", ls="--", lw=1.5, label=f"Seuil FGE ≈ {tfbe*100:.0f}%")
    ax.set_xlabel("Taux de fuite (% des bénéficiaires)"); ax.set_ylabel("Montant (M XOF)")
    ax.set_title("2 · Taux de fuite de break-even — jusqu'où le FGE absorbe")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(fichier, dpi=130); plt.close(fig)
    return tfbe


# 3. Taux de continuité par scénario
def graph_continuite(resultats, fichier="fig3_continuite.png"):
    noms = list(resultats.keys())
    cont = [resultats[n]["taux_continuite"] * 100 for n in noms]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([LBL[n] for n in noms], cont, color=[COUL[n] for n in noms])
    for b, v in zip(bars, cont):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 105); ax.set_ylabel("Taux de continuité (%)")
    ax.set_title("3 · La promesse tenue : tous les tours servis, par scénario")
    ax.grid(alpha=.2, axis="y"); fig.tight_layout(); fig.savefig(fichier, dpi=130); plt.close(fig)


# 4. Exposition SFD mois par mois
def graph_exposition(cfg, fichier="fig4_exposition.png", n_runs=150):
    runs = [simuler_run(_apply("macro", cfg), graine=cfg.monte_carlo.graine_base + i) for i in range(n_runs)]
    ag = agreger_promesse(runs)
    prof = ag["exposition_profil"]
    if not prof: return
    mois = np.arange(1, len(prof["moyenne"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mois, np.array(prof["moyenne"]) / 1e6, color="#0f4c4a", lw=2, label="Exposition moyenne")
    ax.fill_between(mois, 0, np.array(prof["p95"]) / 1e6, color="#0f4c4a", alpha=.12, label="P95")
    ax.set_xlabel("Mois"); ax.set_ylabel("Avances SFD en cours (M XOF)")
    ax.set_title("4 · Exposition SFD mois par mois (scénario macro)")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(fichier, dpi=130); plt.close(fig)


# 5. Coût membre par tour vs alternatives
def graph_cout_membre(cfg, fichier="fig5_cout_membre.png"):
    from compte import prime_garantie
    m = cfg.structure.m_membres; c = cfg.structure.c; pot = (m-1)*c
    r_sfd = cfg.compte.r_sfd_mensuel
    tours = list(range(1, m+1)); couts = []
    for slot in range(m):
        duree = max(1, m-(slot+1)); avance = pot - slot*c  # avance décroît avec l'épargne déjà versée
        avance = max(0, avance)
        interets = pot * r_sfd * duree
        prime = prime_garantie(avance, duree, cfg.fuite.p_fuite_base, m-1, cfg.produit.prime_facteur_prudence)
        marge = cfg.enchere.prime_operateur_taux * pot
        couts.append((interets + prime + marge) / pot * 100)
    usure = cfg.conformite.taux_usure_annuel  # annuel
    seuil_compet = cfg.conformite.cout_membre_tour1_max * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(tours, couts, color="#0f4c4a", label="Coût total du tour (% du pot)")
    ax.axhline(seuil_compet, color="#b18a3a", ls="--", lw=1.5, label=f"Seuil compétitivité ({seuil_compet:.0f}%)")
    ax.set_xlabel("Tour pris"); ax.set_ylabel("Coût (% du pot)")
    ax.set_title("5 · Coût membre par tour — l'épargnant (dernier tour) paie ~0")
    ax.legend(); ax.grid(alpha=.2, axis="y"); fig.tight_layout(); fig.savefig(fichier, dpi=130); plt.close(fig)


# 6. Surface des triplets admissibles (prime × mitigation × résultat)
def graph_surface(cfg, fichier="fig6_surface.png", graines=4):
    primes = np.linspace(0.8, 2.0, 7)      # facteur de prudence de la prime
    mitig = [False, True]                   # mitigations off / on
    grille = np.zeros((len(mitig), len(primes)))
    for i, mit in enumerate(mitig):
        for j, pf in enumerate(primes):
            ok = 0
            for g in range(graines):
                c = copy.deepcopy(cfg)
                c.produit.prime_facteur_prudence = pf
                c.couverture.mitigation_active = mit
                # stress combiné pour tester la robustesse
                c.stress.comportemental_actif = True; c.stress.choc_fuite = 0.06; c.stress.bascule_urgents = 0.3
                c.stress.macro_actif = True; c.stress.z_choc = -2.5; c.stress.z_persistance = 4
                r = simuler_run(c, graine=g)
                op = calculer_pnl_operateur(r, c)
                pot = (c.structure.m_membres-1)*c.structure.c
                # triplet admissible : promesse tenue (résiduel≈0) ET P&L>0 ET coût tour1 < seuil
                promesse_ok = r.residuel_non_couvert <= 1e-6
                pnl_ok = op.resultat_net > 0
                compet_ok = r.cout_membre_tour1 / pot < c.conformite.cout_membre_tour1_max * 1.5
                if promesse_ok and pnl_ok and compet_ok:
                    ok += 1
            grille[i, j] = ok / graines
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(grille, aspect="auto", cmap="Greens", vmin=0, vmax=1,
                   extent=[primes[0], primes[-1], -0.5, 1.5])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Sans mitigation", "Avec mitigation"])
    ax.set_xlabel("Facteur de prudence de la prime")
    ax.set_title("6 · Surface admissible (promesse tenue ∩ P&L>0 ∩ compétitif), stress combiné")
    fig.colorbar(im, ax=ax, label="Part des runs admissibles")
    fig.tight_layout(); fig.savefig(fichier, dpi=130); plt.close(fig)


def _apply(nom, cfg):
    return SCENARIOS[nom](copy.deepcopy(cfg))


def main():
    rapide = "--rapide" in sys.argv
    n_runs = 150 if rapide else 1000
    cfg = config_par_defaut()
    print("Génération des graphiques du pitch...")
    print("  Lancement des scénarios (pour la continuité)...")
    resultats = {n: simuler_scenario(n, cfg, n_runs=n_runs) for n in SCENARIOS}
    be = graph_break_even(cfg); print(f"  ✓ fig1 break-even ≈ {be:.0f} pools")
    tfbe = graph_taux_fuite_be(cfg); print(f"  ✓ fig2 taux fuite BE ≈ {tfbe*100:.0f}%")
    graph_continuite(resultats); print("  ✓ fig3 continuité")
    graph_exposition(cfg); print("  ✓ fig4 exposition SFD")
    graph_cout_membre(cfg); print("  ✓ fig5 coût membre")
    graph_surface(cfg); print("  ✓ fig6 surface admissible")
    print("Terminé.")


if __name__ == "__main__":
    main()
