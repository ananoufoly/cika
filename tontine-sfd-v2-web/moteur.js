/* moteur.js — Port JavaScript du modèle « tontine SFD v2 » (crédit-relais + promesse).
 *
 * Fidèle au moteur Python (config.py / compte.py / fuite.py / moteur.py / pnl.py / promesse.py).
 * Nommage neutre : Operateur, SFD, Membre, Pool, FGE.
 *
 * Mécanique : crédit-relais (gagnant reçoit pot − bid, SFD avance, récup linéaire), prime de
 * garantie obligatoire ∝ avance (mode garanti), bid optionnel pour la position, fuite
 * conditionnelle post-encaissement, cascade de couverture FGE → tranche SFD → résiduel.
 * Modes : "nue" (sans garantie, sans bid) vs "garantie".
 */

// ---- RNG + normale ----
function mulberry32(seed) { let a = seed >>> 0; return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gaussian(rng) { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function erf(x) { const s = Math.sign(x), ax = Math.abs(x); const t = 1 / (1 + 0.3275911 * ax); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax); return s * y; }
function Phi(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
function PhiInv(p) { p = Math.min(Math.max(p, 1e-12), 1 - 1e-12); const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00], b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01], c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00], d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]; const pl = 0.02425, ph = 1 - pl; let q, r; if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); } if (p <= ph) { q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); } q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
function quantile(s, q) { if (!s.length) return 0; const pos = (s.length - 1) * q, b = Math.floor(pos), r = pos - b; return s[b + 1] !== undefined ? s[b] + r * (s[b + 1] - s[b]) : s[b]; }
const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;

// ---- Config par défaut (miroir de config.py, coût fixe = 0) ----
export const DEFAUTS = {
  // structure
  n_pools: 50, m_membres: 10, c: 100000, n_cycles: 2, k_max: 3,
  // produit : "nue" | "garantie"
  mode: "garantie", prime_facteur_prudence: 1.0,
  // compte / SFD
  r_sfd_annuel: 0.18,
  // enchère
  prime_operateur_taux: 0.015, rho_mensuel: 0.02, bid_bruit_sigma: 0.25, bid_plafond_frac_pot: 0.12,
  // préférences
  part_urgent: 0.20, part_modere: 0.50, part_epargnant: 0.30,
  urg_urgent: 1.60, urg_modere: 1.00, urg_epargnant: 0.55,
  // fuite
  p_fuite_base: 0.06, fuite_mult_tour_precoce: 1.8, charge_z_fuite: 0.35,
  taux_echec_friction: 0.03, prelevement_auto_efficacite: 0.7,
  // couverture (3 étages)
  mitigation_active: true, acces_sequence_active: true, t_restreint: 3, part_avec_historique: 0.50,
  garantie_enchere_active: true, g_cotisations: 1,
  prime_active: true, fge_actif: true,
  tranche_sfd_active: true, plafond_tranche_sfd_frac: 0.05,
  // risque
  pd_base_annuel: 0.08, pd_base_sigma: 0.04,
  secteurs: [["commerce", 0.30, 0.30], ["agriculture", 0.20, 0.45], ["transport", 0.15, 0.25], ["services", 0.20, 0.15], ["artisanat", 0.15, 0.20]],
  // stress
  comportemental_actif: false, choc_fuite: 0.0, bascule_urgents: 0.0,
  macro_actif: false, z_choc: 0.0, z_persistance: 0,
  // P&L BRUT : on n'inclut aucun coût (acquisition, ops, fixes). Les coûts relèvent de la
  // structure et du financement, hors du périmètre. Le P&L = revenus bruts du mécanisme.
  cout_acquisition_membre: 0, cout_ops_pool_mois: 0, couts_fixes_mensuels: 0, cout_capital_annuel: 0.10,
};

const rSfdMensuel = p => p.r_sfd_annuel / 12;

// ---- prime de garantie (∝ avance, décroît avec le tour) ----
export function primeGarantie(avance, duree, p_fuite, dureeMax, facteurPrudence) {
  if (avance <= 0) return 0;
  return facteurPrudence * p_fuite * (avance / 2) * (duree / Math.max(1, dureeMax));
}

// ---- proba de fuite mensuelle (depuis proba totale) ----
function probaFuite(pTotal, tEnc, m, z, moisRestants, chargeZ, multPrecoce, choc) {
  const frac = m > 1 ? (tEnc - 1) / (m - 1) : 0;
  const mult = multPrecoce + (1 - multPrecoce) * frac;
  let p = pTotal * mult;
  if (chargeZ > 0) { p = Math.min(Math.max(p, 1e-6), 1 - 1e-6); const logit = Math.log(p / (1 - p)) - chargeZ * z; p = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, logit)))); }
  p = Math.min(Math.max(p + choc, 0), 1);
  const d = Math.max(1, moisRestants);
  return Math.min(Math.max(1 - Math.pow(1 - p, 1 / d), 0), 1);
}

// ---- profils sectoriels (charge Vasicek) ----
function tirerProfils(rng, n, p) {
  const noms = p.secteurs.map(s => s[0]); let parts = p.secteurs.map(s => s[1]); const sp = parts.reduce((a, b) => a + b, 0); parts = parts.map(x => x / sp);
  const cum = []; let acc = 0; for (const x of parts) { acc += x; cum.push(acc); }
  const charges = {}; p.secteurs.forEach(s => charges[s[0]] = s[2]);
  const out = [];
  for (let i = 0; i < n; i++) { const r = rng(); let si = 0; while (si < cum.length - 1 && r > cum[si]) si++; const sec = noms[si]; let pa = Math.min(Math.max(p.pd_base_annuel + gaussian(rng) * p.pd_base_sigma, 0.005), 0.60); const pm = 1 - Math.pow(1 - pa, 1 / 12); out.push({ secteur: sec, rho: charges[sec], seuil: PhiInv(pm) }); }
  return out;
}
function composerPools(rng, profils, m, kMax) {
  const n = profils.length, nP = Math.floor(n / m); const pools = Array.from({ length: nP }, () => []); const cnt = Array.from({ length: nP }, () => ({}));
  const parts = {}; profils.forEach(pr => parts[pr.secteur] = (parts[pr.secteur] || 0) + 1);
  const ordre = profils.map((_, i) => i).sort((i, j) => { const pi = parts[profils[i].secteur], pj = parts[profils[j].secteur]; if (pi !== pj) return pj - pi; return profils[j].rho - profils[i].rho; });
  for (const i of ordre) { const sec = profils[i].secteur; const ouverts = []; for (let pp = 0; pp < nP; pp++) if (pools[pp].length < m) ouverts.push(pp); if (!ouverts.length) continue; const conf = ouverts.filter(pp => (cnt[pp][sec] || 0) < kMax); let pc = conf.length ? conf.reduce((b, pp) => pools[pp].length < pools[b].length ? pp : b, conf[0]) : ouverts.reduce((b, pp) => (cnt[pp][sec] || 0) < (cnt[b][sec] || 0) ? pp : b, ouverts[0]); pools[pc].push(i); cnt[pc][sec] = (cnt[pc][sec] || 0) + 1; }
  return pools.filter(pl => pl.length === m);
}
function pdCond(seuil, rho, z) { return Phi((seuil - Math.sqrt(rho) * z) / Math.sqrt(Math.max(1e-9, 1 - rho))); }
function tirerZ(rng, p, etat) { if (!p.macro_actif || p.z_choc === 0) return gaussian(rng); if (p.z_persistance > 0 && etat.restant > 0) { etat.restant--; return etat.courant; } const z = p.z_choc + gaussian(rng) * 0.6; if (p.z_persistance > 0) { etat.courant = z; etat.restant = p.z_persistance - 1; } return z; }

// ---- simulation d'un run (portefeuille) ----
export function simulerRun(p, graine) {
  const rng = mulberry32(graine);
  const m = p.m_membres, totalTours = m * p.n_cycles, vie = totalTours;
  const pot = (m - 1) * p.c, rSfd = rSfdMensuel(p);
  const profils = tirerProfils(rng, p.n_pools * m, p);
  const poolsIdx = composerPools(rng, profils, m, p.k_max);
  const nPools = poolsIdx.length;
  const etatZ = {}; const zMois = []; for (let t = 0; t < totalTours; t++) zMois.push(tirerZ(rng, p, etatZ));

  function pref() { const r = rng(); if (r < p.part_urgent) return ["urgent", p.urg_urgent]; if (r < p.part_urgent + p.part_modere) return ["modere", p.urg_modere]; return ["epargnant", p.urg_epargnant]; }
  const pools = []; const comptes = [];
  for (let pid = 0; pid < nPools; pid++) {
    const membres = poolsIdx[pid].map((gi, i) => { const pr = profils[gi]; let [tp, urg] = pref(); if (p.comportemental_actif && (tp === "modere" || tp === "epargnant") && rng() < p.bascule_urgents) { tp = "urgent"; urg = p.urg_urgent; } return { i, seuil: pr.seuil, rho: pr.rho, type: tp, urg, aHist: rng() < p.part_avec_historique, aEncaisse: false, tEnc: null, aFui: false, consign: 0, cotise: 0, recu: 0 }; });
    pools.push(membres); comptes.push({ prets: [], decaisseCumule: 0 });
  }

  let fge = 0, trancheSfdUtilisee = 0, fgeProvisions = 0, fgeSaisies = 0;
  let primes = 0, surplusEnchere = 0, interetsSfd = 0, avanceCumulee = 0;
  let couvertFge = 0, couvertSfd = 0, residuel = 0, perteSfd = 0, nFuites = 0, nGratuits = 0;
  let coutTour1 = 0, continuiteOk = true;
  const exposMois = new Array(totalTours).fill(0);
  const chocFuite = p.comportemental_actif ? p.choc_fuite : 0;

  for (let t = 1; t <= totalTours; t++) {
    const z = zMois[t - 1], slot = (t - 1) % m;
    if (slot === 0) for (const mb of pools.flat()) if (!mb.aFui) mb.aEncaisse = false;

    for (let pid = 0; pid < nPools; pid++) {
      const membres = pools[pid], cpt = comptes[pid];
      // 1. fuites
      for (const mb of membres) {
        if (mb.aEncaisse && !mb.aFui) {
          const moisR = Math.max(1, vie - mb.tEnc);
          const pf = probaFuite(p.p_fuite_base, mb.tEnc, m, z, moisR, p.charge_z_fuite, p.fuite_mult_tour_precoce, chocFuite);
          if (rng() < pf) {
            mb.aFui = true; nFuites++;
            let trou = 0;
            for (const pr of cpt.prets) if (pr.membre === mb.i && pr.actif && pr.restant > 1e-9) { pr.actif = false; trou += pr.restant; }
            if (p.garantie_enchere_active && mb.consign > 0) { fge += mb.consign; fgeSaisies += mb.consign; mb.consign = 0; }
            let reste = trou;
            const pf2 = p.fge_actif ? Math.min(fge, reste) : 0; fge -= pf2; reste -= pf2; couvertFge += pf2;
            if (reste > 1e-9 && p.tranche_sfd_active) { const plaf = p.plafond_tranche_sfd_frac * Math.max(cpt.decaisseCumule, 1) * nPools; const dispo = Math.max(0, plaf - trancheSfdUtilisee); const ps = Math.min(dispo, reste); trancheSfdUtilisee += ps; reste -= ps; couvertSfd += ps; perteSfd += ps; }
            if (reste > 1e-9) { residuel += reste; continuiteOk = false; }
          }
        }
      }
      // 2. collecte
      const actifs = membres.filter(mb => !mb.aFui);
      let potColl = 0;
      for (const mb of actifs) if (!mb.aEncaisse) { let taux = p.taux_echec_friction; if (p.mitigation_active) taux *= (1 - p.prelevement_auto_efficacite); potColl += (rng() < taux ? p.c * 0.9 : p.c); mb.cotise += p.c; }
      // 3. attribution
      const elig = actifs.filter(mb => !mb.aEncaisse);
      let eligBid = elig;
      if (p.mode === "garantie" && p.mitigation_active && p.acces_sequence_active && slot < p.t_restreint) eligBid = elig.filter(mb => mb.aHist);
      const dureePret = Math.max(1, m - (slot + 1));
      let gagnant = null, bideur = false, bidSurplusWtp = 0;
      if (p.mode === "garantie") {
        let best = null, bestW = -1; for (const mb of eligBid) { const mg = Math.max(0, (m - 1) - slot); const wtp = p.rho_mensuel * mg * pot * mb.urg * Math.exp(gaussian(rng) * p.bid_bruit_sigma); if (wtp > bestW) { bestW = wtp; best = mb; } }
        if (best && bestW > 0.01 * pot) { gagnant = best; bideur = true; bidSurplusWtp = bestW; if (p.mitigation_active && p.garantie_enchere_active && slot < p.t_restreint && gagnant.consign === 0) gagnant.consign = p.g_cotisations * p.c; }
        else if (elig.length) { gagnant = elig.reduce((b, mb) => mb.i < b.i ? mb : b, elig[0]); bideur = false; }
      } else { if (elig.length) { gagnant = elig.reduce((b, mb) => mb.i < b.i ? mb : b, elig[0]); bideur = false; } }
      // 4. décaissement
      if (gagnant) {
        const avance = Math.max(0, pot - gagnant.cotise);
        let bid = 0, net = pot;
        if (p.mode === "garantie") {
          const primeGar = p.prime_active ? primeGarantie(avance, dureePret, p.p_fuite_base, m - 1, p.prime_facteur_prudence) : 0;
          const interets = pot * rSfd * dureePret, margeOp = p.prime_operateur_taux * pot;
          const coutObl = interets + primeGar + margeOp;
          // fidèle au Python : la wtp est d'abord plafonnée, PUIS on retire le coût obligatoire
          const bidSurplusPlaf = Math.min(bidSurplusWtp, p.bid_plafond_frac_pot * pot);
          let surplus = bideur ? Math.max(0, Math.min(bidSurplusPlaf - coutObl, p.bid_plafond_frac_pot * pot)) : 0;
          bid = coutObl + surplus; net = Math.max(0, pot - bid);
          interetsSfd += interets; primes += margeOp; surplusEnchere += surplus; fgeProvisions += primeGar; fge += primeGar; avanceCumulee += net;
        } else { bid = 0; net = pot; avanceCumulee += pot; }
        cpt.prets.push({ membre: gagnant.i, restant: net, mensualite: net / dureePret, actif: true }); cpt.decaisseCumule += net;
        gagnant.aEncaisse = true; gagnant.tEnc = t; gagnant.recu += net;
        if (!bideur) nGratuits++;
        if (t === 1 && coutTour1 === 0) coutTour1 = bid;
      }
      // 5. récupération
      for (const pr of cpt.prets) if (pr.actif && pr.restant > 1e-9) { const pa = Math.min(pr.mensualite, pr.restant); pr.restant -= pa; }
      exposMois[t - 1] += cpt.prets.filter(pr => pr.actif).reduce((s, pr) => s + pr.restant, 0);
    }
  }

  const nMembres = nPools * m, mois = totalTours;
  // P&L Opérateur
  const revenus = primes + surplusEnchere;
  const coutAcq = p.cout_acquisition_membre * nMembres, coutOps = p.cout_ops_pool_mois * nPools * mois;
  const coutsFixes = p.couts_fixes_mensuels * mois;
  const pnlOp = revenus - (coutAcq + coutOps + coutsFixes);
  const margePool = nPools ? (revenus - coutAcq - coutOps) / nPools : 0;
  const breakEven = coutsFixes <= 0 ? (margePool > 1e-9 ? 0 : Infinity) : (margePool > 1e-9 ? coutsFixes / margePool : Infinity);

  return {
    nPools, nFuites, continuiteOk, residuel, perteSfd, coutTour1, nGratuits,
    couvertFge, couvertSfd, fgeProvisions, fgeSaisies, primes, surplusEnchere, interetsSfd, avanceCumulee,
    expoMois: exposMois, expoMax: Math.max(...exposMois),
    pnlOp, margePool, breakEven, revenus, coutAcq, coutOps,
  };
}

// ---- simulation d'UN pool avec journal détaillé (pour l'animation pédagogique) ----
export function simulerPoolDetail(p, graine) {
  const rng = mulberry32(graine);
  const m = p.m_membres, totalTours = m * p.n_cycles, vie = totalTours;
  const pot = (m - 1) * p.c, rSfd = rSfdMensuel(p);
  // noms simples pour l'animation
  const NOMS = ["Awa", "Koffi", "Mariam", "Ibrahim", "Fanta", "Sékou", "Aïcha", "Moussa", "Rama", "Yao", "Bintou", "Diallo", "Nana", "Oumar", "Salif"];
  function pref() { const r = rng(); if (r < p.part_urgent) return ["urgent", p.urg_urgent]; if (r < p.part_urgent + p.part_modere) return ["modere", p.urg_modere]; return ["epargnant", p.urg_epargnant]; }
  const profs = tirerProfils(rng, m, p);
  const membres = profs.map((pr, i) => { let [tp, urg] = pref(); return { i, nom: NOMS[i % NOMS.length], seuil: pr.seuil, rho: pr.rho, type: tp, urg, aHist: rng() < p.part_avec_historique, aEncaisse: false, tEnc: null, aFui: false, consign: 0, cotise: 0, recu: 0, bidPaye: 0 }; });
  const cpt = { prets: [], decaisseCumule: 0 };
  let fge = 0, trancheUtil = 0;
  const etatZ = {};
  const journal = [];
  const chocFuite = p.comportemental_actif ? p.choc_fuite : 0;

  for (let t = 1; t <= totalTours; t++) {
    const z = tirerZ(rng, p, etatZ), slot = (t - 1) % m;
    if (slot === 0) for (const mb of membres) if (!mb.aFui) mb.aEncaisse = false;
    const ev = { tour: t, fuite: null, garantie: null };

    // fuites
    for (const mb of membres) {
      if (mb.aEncaisse && !mb.aFui) {
        const moisR = Math.max(1, vie - mb.tEnc);
        const pf = probaFuite(p.p_fuite_base, mb.tEnc, m, z, moisR, p.charge_z_fuite, p.fuite_mult_tour_precoce, chocFuite);
        if (rng() < pf) {
          mb.aFui = true; let trou = 0;
          for (const pr of cpt.prets) if (pr.membre === mb.i && pr.actif && pr.restant > 1e-9) { pr.actif = false; trou += pr.restant; }
          if (p.garantie_enchere_active && mb.consign > 0) { fge += mb.consign; mb.consign = 0; }
          ev.fuite = { membre: mb.i, nom: mb.nom, trou };
          if (p.mode === "garantie") {
            let reste = trou;
            const pf2 = p.fge_actif ? Math.min(fge, reste) : 0; fge -= pf2; reste -= pf2;
            let prisSfd = 0;
            if (reste > 1e-9 && p.tranche_sfd_active) { const plaf = p.plafond_tranche_sfd_frac * Math.max(cpt.decaisseCumule, 1); const dispo = Math.max(0, plaf - trancheUtil); prisSfd = Math.min(dispo, reste); trancheUtil += prisSfd; reste -= prisSfd; }
            ev.garantie = { parFge: pf2, parSfd: prisSfd, residuel: Math.max(0, reste) };
          } else {
            ev.garantie = { parFge: 0, parSfd: 0, residuel: trou };  // nue : le groupe perd
          }
        }
      }
    }
    // collecte
    const actifs = membres.filter(mb => !mb.aFui);
    for (const mb of actifs) if (!mb.aEncaisse) mb.cotise += p.c;
    // attribution
    const elig = actifs.filter(mb => !mb.aEncaisse);
    let eligBid = elig;
    if (p.mode === "garantie" && p.mitigation_active && p.acces_sequence_active && slot < p.t_restreint) eligBid = elig.filter(mb => mb.aHist);
    const dureePret = Math.max(1, m - (slot + 1));
    let gagnant = null, bideur = false, bidW = 0;
    if (p.mode === "garantie") {
      let best = null, bestW = -1; for (const mb of eligBid) { const mg = Math.max(0, (m - 1) - slot); const wtp = p.rho_mensuel * mg * pot * mb.urg * Math.exp(gaussian(rng) * p.bid_bruit_sigma); if (wtp > bestW) { bestW = wtp; best = mb; } }
      if (best && bestW > 0.01 * pot) { gagnant = best; bideur = true; bidW = bestW; if (p.mitigation_active && p.garantie_enchere_active && slot < p.t_restreint) gagnant.consign = p.g_cotisations * p.c; }
      else if (elig.length) { gagnant = elig.reduce((b, mb) => mb.i < b.i ? mb : b, elig[0]); }
    } else if (elig.length) { gagnant = elig.reduce((b, mb) => mb.i < b.i ? mb : b, elig[0]); }

    let benefInfo = null;
    if (gagnant) {
      const avance = Math.max(0, pot - gagnant.cotise);
      let bid = 0, net = pot;
      if (p.mode === "garantie") {
        const primeGar = p.prime_active ? primeGarantie(avance, dureePret, p.p_fuite_base, m - 1, p.prime_facteur_prudence) : 0;
        const interets = pot * rSfd * dureePret, margeOp = p.prime_operateur_taux * pot;
        const coutObl = interets + primeGar + margeOp;
        const bidPlaf = Math.min(bidW, p.bid_plafond_frac_pot * pot);
        const surplus = bideur ? Math.max(0, Math.min(bidPlaf - coutObl, p.bid_plafond_frac_pot * pot)) : 0;
        bid = coutObl + surplus; net = Math.max(0, pot - bid);
        cpt.prets.push({ membre: gagnant.i, restant: net, mensualite: net / dureePret, actif: true }); cpt.decaisseCumule += net; fge += primeGar;
        benefInfo = { membre: gagnant.i, nom: gagnant.nom, recu: net, bid, bideur, prime: primeGar, interets, surplus };
      } else {
        net = pot;
        cpt.prets.push({ membre: gagnant.i, restant: net, mensualite: net / dureePret, actif: true }); cpt.decaisseCumule += net;
        benefInfo = { membre: gagnant.i, nom: gagnant.nom, recu: net, bid: 0, bideur: false, prime: 0, interets: 0, surplus: 0 };
      }
      gagnant.aEncaisse = true; gagnant.tEnc = t; gagnant.recu += net; gagnant.bidPaye += bid;
    }
    for (const pr of cpt.prets) if (pr.actif && pr.restant > 1e-9) { const pa = Math.min(pr.mensualite, pr.restant); pr.restant -= pa; }

    // snapshot des états membres pour l'affichage
    const snap = membres.map(mb => ({ nom: mb.nom, type: mb.type, aEncaisse: mb.aEncaisse, aFui: mb.aFui, estBenef: benefInfo && benefInfo.membre === mb.i }));
    journal.push({ ...ev, benef: benefInfo, pot, snap });
  }
  return { m, pot, journal };
}

// ---- Monte Carlo ----
export function monteCarlo(p, nRuns, graineBase) {
  const acc = { pnlOp: [], continuite: [], residuel: [], perteSfd: [], expoMax: [], fuites: [], coutTour1: [], margePool: [], nGratuits: [] };
  let expoProfil = null;
  for (let i = 0; i < nRuns; i++) {
    const r = simulerRun(p, graineBase + i);
    acc.pnlOp.push(r.pnlOp); acc.continuite.push(r.continuiteOk ? 1 : 0); acc.residuel.push(r.residuel);
    acc.perteSfd.push(r.perteSfd); acc.expoMax.push(r.expoMax); acc.fuites.push(r.nFuites);
    acc.coutTour1.push(r.coutTour1); acc.margePool.push(r.margePool); acc.nGratuits.push(r.nGratuits);
    if (!expoProfil) expoProfil = r.expoMois.map(() => 0);
    r.expoMois.forEach((v, j) => expoProfil[j] += v / nRuns);
  }
  const ag = {};
  for (const k of Object.keys(acc)) { const s = acc[k].slice().sort((a, b) => a - b); ag[k] = { moy: mean(acc[k]), p5: quantile(s, 0.05), p95: quantile(s, 0.95) }; }
  ag.taux_continuite = mean(acc.continuite);
  ag.p_promesse_cassee = acc.residuel.filter(x => x > 1e-6).length / nRuns;
  ag.expoProfil = expoProfil;
  ag._pertes = acc.perteSfd;
  return ag;
}
